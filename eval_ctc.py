# ~/src/eval_ctc.py
# -*- coding: utf-8 -*-
import argparse, os, csv, math, random, statistics as st
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import SignDataset, collate_ctc
from models import SignCTCModel

# ------------------------------
# 재현성
# ------------------------------
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ------------------------------
# 하이브리드(TCN stride=2 두 번) 길이 보정
#   T' = ceil( ceil(T/2) / 2 )
# ------------------------------
def ds_len_tensor(xlen: torch.Tensor) -> torch.Tensor:
    return ((xlen + 1) // 2 + 1) // 2


# ------------------------------
# Greedy CTC collapse (배치)
# logits_btv: [B,T',V], blank=0
# ------------------------------
@torch.no_grad()
def greedy_decode_batch(logits_btv: torch.Tensor, blank: int = 0):
    ids_bT = logits_btv.argmax(dim=-1)  # [B,T']
    outs = []
    for ids in ids_bT.tolist():
        prev = blank
        seq = []
        for i in ids:
            if i != blank and i != prev:
                seq.append(i)
            prev = i
        outs.append(seq)
    return outs


# ------------------------------
# CTC Prefix Beam Search (no LM)
#  - logits_btv: [B,T,V], blank=0
#  - returns: List[B] of List[topK seq(ids)]
#  - 간단 프루닝: 각 시점 top-k 심벌, 각 단계 상위 beam_k 유지
# ------------------------------
@torch.no_grad()
def beam_search_ctc_batch(logits_btv: torch.Tensor, beam_k: int, blank: int = 0):
    # logits_btv: [B,T,V]
    B, T, V = logits_btv.shape
    # CPU에서 처리(스칼라 텐서 생성 비용↓, 디바이스 혼동 방지)
    logp = logits_btv.log_softmax(-1).detach().to('cpu')
    results = []
    NEG_INF = -1e30

    def logadd(a: float, b: float) -> float:
        # torch.logaddexp는 Tensor를 요구 → 감싸서 사용
        return torch.logaddexp(torch.tensor(a), torch.tensor(b)).item()

    for b in range(B):
        # beams: { prefix(tuple[int]) : (p_b, p_nb) }  (log-domain floats)
        beams = {(): (0.0, NEG_INF)}  # empty prefix: p_b=log(1), p_nb=-inf

        for t in range(T):
            lp_t = logp[b, t]  # [V] on CPU
            # 현재 beams 상위 beam_k만 유지 (점수=logaddexp(p_b, p_nb))
            scored = sorted(
                beams.items(),
                key=lambda kv: max(kv[1][0], kv[1][1]),
                reverse=True
            )[:beam_k]

            next_beams = {}
            # 심벌 프루닝: 이 시점에서 상위 몇 개만 확장
            topk_sym = torch.topk(lp_t, k=min(max(beam_k * 2, beam_k), V)).indices.tolist()

            for prefix, (p_b, p_nb) in scored:
                # 1) blank 확장: prefix 유지
                pb_merge = logadd(p_b, p_nb)            # log(p_b + p_nb)
                pb_new = pb_merge + lp_t[blank].item()  # log(...) + log P(blank)
                ob_b, ob_nb = next_beams.get(prefix, (NEG_INF, NEG_INF))
                ob_b = logadd(ob_b, pb_new)
                next_beams[prefix] = (ob_b, ob_nb)

                # 2) non-blank 확장
                for v in topk_sym:
                    if v == blank:
                        continue
                    v_lp = lp_t[v].item()
                    if len(prefix) > 0 and v == prefix[-1]:
                        # 같은 토큰 반복은 직전이 blank였을 때만(CTC 규칙)
                        pnb_new = p_b + v_lp
                    else:
                        pnb_new = pb_merge + v_lp  # log(p_b + p_nb) + log P(v)

                    new_pref = prefix + (v,)
                    ob_b, ob_nb = next_beams.get(new_pref, (NEG_INF, NEG_INF))
                    ob_nb = logadd(ob_nb, pnb_new)
                    next_beams[new_pref] = (ob_b, ob_nb)

            beams = next_beams

        # 최종 점수 = logadd(p_b, p_nb)
        scored_final = []
        for pref, (p_b, p_nb) in beams.items():
            scored_final.append((pref, logadd(p_b, p_nb)))
        scored_final.sort(key=lambda x: x[1], reverse=True)

        topk = [list(pref) for pref, _ in scored_final[:beam_k]]
        if not topk:
            topk = [[]]
        results.append(topk)

    return results


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-root', default=str(Path(__file__).parent.parent / 'datasets'))
    # 하이픈 표기 지원 (사용자가 --npz-subdir 사용하므로 이걸 기본으로)
    pa.add_argument('--npz-subdir', dest='npz_subdir', default='npz/eco')
    # 필요시 언더스코어도 받기
    pa.add_argument('--npz_subdir', dest='npz_subdir', default='npz/eco')
    pa.add_argument('--csv-name', default='labels.csv')
    pa.add_argument('--vocab-json', default='vocab.json')

    pa.add_argument('--split', choices=['train', 'valid', 'all'], default='valid')
    pa.add_argument('--split-ratio', type=float, default=0.9)
    pa.add_argument('--stratify', action='store_true', default=False)
    pa.add_argument('--augment', action='store_true', default=False)  # 평가 기본 OFF

    pa.add_argument('--backend', choices=['tcn', 'tfm', 'tcn_tfm'], default='tcn')
    pa.add_argument('--in_dim', type=int, default=126)
    pa.add_argument('--hid', type=int, default=256)
    pa.add_argument('--depth', type=int, default=6)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--drop', type=float, default=0.1)   # models.py는 p 로 받음
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    pa.add_argument('--ckpt', required=True, help='train_word.py로 저장한 .pt')
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--num-workers', type=int, default=2)
    pa.add_argument('--beam-k', type=int, default=0, help='0이면 greedy, >0이면 prefix beam')
    pa.add_argument('--save', default=None, help='CSV 경로 (예: runs/valid.csv)')

    pa.add_argument('--seed', type=int, default=42)
    args = pa.parse_args()

    set_all_seeds(args.seed)
    pin = (args.device == 'cuda')

    # --------------------------
    # 데이터셋
    # --------------------------
    ds = SignDataset(
        data_root=args.data_root,
        split=args.split,
        csv_name=args.csv_name,
        vocab_json=args.vocab_json,
        npz_subdir=args.npz_subdir,
        in_dim_override=args.in_dim,
        split_ratio=args.split_ratio,
        stratify=args.stratify,
        augment=False,                 # 평가에서는 증강 OFF 고정
        verbose=True,
    )

    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, collate_fn=collate_ctc, pin_memory=pin)

    # --------------------------
    # 체크포인트 로드
    # --------------------------
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    ck_vocab = ckpt.get('vocab', None) if isinstance(ckpt, dict) else None
    if ck_vocab is None:
        raise RuntimeError("checkpoint에 'vocab'이 없습니다. train_word.py로 저장한 ckpt를 사용하세요.")

    # vocab 일치 확인 (라벨 매핑 불일치 방지)
    if list(ds.itos) != list(ck_vocab):
        raise RuntimeError(
            "현재 datasets의 vocab.json과 ckpt의 vocab이 다릅니다.\n"
            "→ 학습시 사용한 vocab.json으로 평가하거나, 동일한 데이터 루트를 지정하세요."
        )

    V = len(ds.itos)  # <blank>=0 포함

    # --------------------------
    # 모델 구성 & 가중치 로딩
    # --------------------------
    model = SignCTCModel(
        in_dim=args.in_dim,
        vocab_size=V,
        backend=args.backend,
        hid=args.hid,
        depth=args.depth,
        p=args.drop,
        nhead=args.nhead,
    ).to(args.device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict 불일치 - missing={missing}, unexpected={unexpected}")

    model.eval()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    # --------------------------
    # 평가 루프
    # --------------------------
    all_losses = []
    num_blank_only = 0
    pred_lengths: List[int] = []
    top1_presence = 0
    topk_presence = 0
    n_samples = 0

    rows_to_save = []  # CSV (필요시 확장 가능)

    with torch.no_grad():
        for batch in dl:
            # collate_ctc 반환: xpad[B,T,F], ycat[sumL], x_lens[B], y_lens[B], pad_mask[B,T]
            x, y, x_len, y_len, pad_mask = batch
            B = x.size(0)
            n_samples += B

            x = x.to(args.device, non_blocking=True)              # [B,T,F]
            pad_mask = pad_mask.to(args.device, non_blocking=True)

            # 모델 forward (트랜스포머 계열만 마스크 사용)
            use_mask = (args.backend in ('tfm', 'tcn_tfm'))
            logits = model(x, src_key_padding_mask=pad_mask if use_mask else None)  # [B,T' ,V]

            # CTC loss 계산을 위해 [T',B,V], length 텐서는 CPU int64
            log_probs = logits.log_softmax(-1).transpose(0, 1).contiguous()  # [T',B,V]
            if args.backend == 'tcn_tfm':
                in_lens = ds_len_tensor(x_len)
            else:
                in_lens = x_len
            loss = ctc(
                log_probs,
                y,                                  # 1D target
                in_lens.to('cpu', torch.int64),
                y_len.to('cpu', torch.int64)
            )
            all_losses.append(loss.item())

            # ---- 디코딩 ----
            if args.beam_k and args.beam_k > 0:
                topk_preds = beam_search_ctc_batch(logits, beam_k=args.beam_k, blank=0)  # List[B][K][L]
                preds = [ (klist[0] if len(klist) > 0 else []) for klist in topk_preds ] # Top-1
            else:
                preds = greedy_decode_batch(logits, blank=0)                              # List[B][L]
                topk_preds = [ [p] for p in preds ]                                       # 호환용

            # ---- 지표 집계: blank-only, 길이 ----
            for p in preds:
                if len(p) == 0:
                    num_blank_only += 1
                else:
                    pred_lengths.append(len(p))

            # ---- gold 복원 ----
            cursor = 0
            gold_ids = []
            for L in y_len.tolist():
                gold_ids.append(y[cursor:cursor+L].tolist())
                cursor += L

            # ---- Presence@1 / Presence@K ----
            for g, pk in zip(gold_ids, topk_preds):         # pk: List[K] sequences
                # @1
                p1 = pk[0] if len(pk) > 0 else []
                if all((tok in p1) for tok in g):
                    top1_presence += 1
                # @K
                if args.beam_k and args.beam_k > 1:
                    hit = False
                    for cand in pk:
                        if all((tok in cand) for tok in g):
                            hit = True
                            break
                    if hit:
                        topk_presence += 1

            # ---- CSV 저장용(옵션, Top-1만 기록) ----
            for p in preds:
                pred_tokens = [ds.itos[i] for i in p] if len(p) else []
                rows_to_save.append({'pred_tokens': ' '.join(pred_tokens)})

    # --------------------------
    # 최종 리포트
    # --------------------------
    avg_loss = float(st.mean(all_losses)) if all_losses else float('nan')
    blank_only_ratio = (num_blank_only / n_samples * 100.0) if n_samples else 0.0
    avg_pred_len = float(st.mean(pred_lengths)) if pred_lengths else 0.0
    top1_presence_ratio = (top1_presence / n_samples * 100.0) if n_samples else 0.0

    # 랜덤 베이스라인(ln V) 참고치
    rand_baseline = math.log(V)

    print("===== Validation quick metrics =====")
    print(f"CTC loss : {avg_loss:.4f} (random baseline ~ ln(V) ≈ {rand_baseline:.2f})")
    print(f"Top-1 presence : {top1_presence_ratio:.2f}% (gold ∈ predicted sequence)")
    if args.beam_k and args.beam_k > 1:
        topk_presence_ratio = (topk_presence / n_samples * 100.0) if n_samples else 0.0
        print(f"Presence@{args.beam_k} : {topk_presence_ratio:.2f}% (gold ∈ any of top-{args.beam_k})")
    print(f"Blank-only ratio : {blank_only_ratio:.2f}%")
    print(f"Avg pred length : {avg_pred_len:.2f} tokens (excluding blank-only)")

    # --------------------------
    # CSV 저장 (옵션)
    # --------------------------
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, fieldnames=['pred_tokens'])
            wr.writeheader()
            for row in rows_to_save:
                wr.writerow(row)
        print(f"✓ Saved predictions to {out_path}")


if __name__ == '__main__':
    main()
