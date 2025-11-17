# ~/src/eval_sent.py
# -*- coding: utf-8 -*-
"""
Sentence CTC 평가 스크립트
- GlossSeqDataset + SentenceCTC용
- Metrics:
  - avg CTC loss (valid)
  - sequence exact match accuracy
  - token-level CER (edit distance 기반)
  - avg ref/pred length
  - blank-only ratio
  - unique token recall / precision
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_gloss import GlossSeqDataset, collate_ctc
from models import HybridBackbone, SentenceCTC, down_len


def ctc_greedy_decode(logits, in_len, blank_id=0):
    """
    logits: [B,T',V] (raw logits)
    in_len: [B]  downsample 후 길이
    return: List[List[int]]  (blank/중복 제거된 토큰 id 시퀀스)
    """
    logp = logits.log_softmax(-1)          # [B,T',V]
    ids = logp.argmax(-1)                  # [B,T']
    B, T = ids.size()
    preds = []
    for b in range(B):
        T_eff = int(in_len[b])
        prev = blank_id
        seq = []
        for t in range(T_eff):
            v = int(ids[b, t])
            if v == blank_id:
                prev = blank_id
                continue
            if v == prev:  # collapse repeats
                continue
            seq.append(v)
            prev = v
        preds.append(seq)
    return preds


def edit_distance(a, b):
    """Levenshtein distance for 1D int sequences."""
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # delete
                dp[i][j - 1] + 1,          # insert
                dp[i - 1][j - 1] + cost,   # replace
            )
    return dp[la][lb]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--root', required=True,
                    help='datasets/sentences/BP2 같은 루트')
    pa.add_argument('--index-csv', default='index.csv')
    pa.add_argument('--vocab-json', default='vocab.json')

    pa.add_argument('--ckpt', required=True,
                    help='train_sent.py로 저장한 문장 CTC 체크포인트 경로')
    pa.add_argument('--split', choices=['train', 'valid', 'all'],
                    default='valid')

    # 모델 구조 (train_sent.py와 동일하게!)
    pa.add_argument('--in-dim', type=int, default=126)
    pa.add_argument('--hid', type=int, default=256)
    pa.add_argument('--depth', type=int, default=6)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--subsample-stages', type=int, default=1)

    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = pa.parse_args()

    device = torch.device(args.device)
    pin = args.device.startswith('cuda')

    # Dataset (예: valid split 평가용)
    ds = GlossSeqDataset(
        root=args.root,
        index_csv=args.index_csv,
        vocab_json=args.vocab_json,
        split=args.split,
        split_ratio=0.9,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=collate_ctc,
    )

    # id ↔ token 매핑
    token2id = ds.token2id           # 1..V-1
    id2token = {i: t for t, i in token2id.items()}
    blank_id = 0
    V = len(token2id) + 1

    # 모델 생성 + ckpt 로드
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    # in_dim, hid, subsample_stages는 ckpt에 있으면 그걸 우선 사용
    if isinstance(ckpt, dict):
        in_dim = ckpt.get('in_dim', args.in_dim)
        hid = ckpt.get('hid', args.hid)
        subs = ckpt.get('subsample_stages', args.subsample_stages)
    else:
        in_dim = args.in_dim
        hid = args.hid
        subs = args.subsample_stages

    backbone = HybridBackbone(
        in_dim=in_dim,
        hid=hid,
        depth=args.depth,
        nhead=args.nhead,
        p=args.dropout,
        subsample_stages=subs,
    ).to(device)
    model = SentenceCTC(backbone, vocab_size=V).to(device)
    model.load_state_dict(state)
    model.eval()

    ctc = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    # ---------- 누적 변수들 ----------
    total_loss = 0.0        # CTC loss 합 (배치 평균 * 배치크기)
    total_n = 0             # 샘플 수
    total_seq = 0           # 문장 수
    correct_seq = 0         # 완전 일치 문장 수
    total_tokens = 0        # Ref 토큰 총 수 (CER 분모)
    total_edit = 0          # 편집거리 총합 (CER 분자)

    total_ref_len = 0       # ref 문장 길이 총합
    total_pred_len = 0      # pred 문장 길이 총합
    blank_only = 0          # 예측이 빈 문장인 케이스 수

    total_ref_unique = 0    # ref 고유 토큰 수 합
    total_pred_unique = 0   # pred 고유 토큰 수 합
    total_intersection = 0  # ref/pred 고유 토큰 교집합 합

    with torch.no_grad():
        for X, x_len, Y, y_len in dl:
            X = X.to(device)
            x_len = x_len.to(device)

            pad = torch.arange(X.size(1), device=device)[None, :] >= x_len[:, None]
            logits = model(X, src_key_padding_mask=pad)  # [B,T',V]
            logp = logits.log_softmax(-1).transpose(0, 1)  # [T',B,V]
            in_len = down_len(x_len.to('cpu', torch.int64), subs)  # [B]

            # CTCLoss 계산 (train_sent.py와 동일한 방식)
            loss = ctc(
                logp,
                Y.to(device),
                in_len,
                y_len.to('cpu', torch.int64),
            )
            bs = X.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            # greedy decode
            preds = ctc_greedy_decode(logits, in_len, blank_id=blank_id)

            # 정답 시퀀스(id 리스트) 꺼내기
            gts = []
            for b in range(Y.size(0)):
                L = int(y_len[b])
                gts.append(Y[b, :L].tolist())

            # 시퀀스 정확도 & CER & 추가 지표들
            for p, g in zip(preds, gts):
                total_seq += 1

                # seq 정확도
                if p == g:
                    correct_seq += 1

                # CER (token-level edit distance)
                ed = edit_distance(p, g)
                total_edit += ed
                total_tokens += len(g)

                # 길이 관련
                total_ref_len += len(g)
                total_pred_len += len(p)
                if len(p) == 0:
                    blank_only += 1

                # unique 토큰 기반 recall / precision
                ref_set = set(g)
                pred_set = set(p)
                total_intersection += len(ref_set & pred_set)
                total_ref_unique += len(ref_set)
                total_pred_unique += len(pred_set)

    # ---------- 최종 지표 계산 ----------
    avg_loss = total_loss / max(1, total_n)
    seq_acc = correct_seq / max(1, total_seq)
    cer = total_edit / max(1, total_tokens)

    avg_ref_len = total_ref_len / max(1, total_seq)
    avg_pred_len = total_pred_len / max(1, total_seq)
    blank_ratio = blank_only / max(1, total_seq)

    recall_unique = (
        total_intersection / total_ref_unique if total_ref_unique > 0 else float("nan")
    )
    precision_unique = (
        total_intersection / total_pred_unique if total_pred_unique > 0 else float("nan")
    )

    print(f"[Eval] split={args.split}")
    print(f"  avg CTC loss          : {avg_loss:.4f}")
    print(f"  seq accuracy          : {seq_acc*100:.2f}%")
    print(f"  token CER             : {cer*100:.2f}%")
    print(f"  Avg ref length        : {avg_ref_len:.2f} tokens")
    print(f"  Avg pred length       : {avg_pred_len:.2f} tokens")
    print(f"  Blank-only ratio      : {blank_ratio*100:.2f}%")
    print(f"  Unique token recall   : {recall_unique*100:.2f}%")
    print(f"  Unique token precision: {precision_unique*100:.2f}%")

    # 샘플 몇 개 출력 (토큰→글로스 문자열)
    print("\n[Samples]")
    with torch.no_grad():
        # 첫 배치만 다시 뽑아서 예시 보여주기
        dl_iter = iter(dl)
        X, x_len, Y, y_len = next(dl_iter)
        X = X.to(device)
        x_len = x_len.to(device)
        pad = torch.arange(X.size(1), device=device)[None, :] >= x_len[:, None]
        logits = model(X, src_key_padding_mask=pad)
        in_len = down_len(x_len.to('cpu', torch.int64), subs)
        preds = ctc_greedy_decode(logits, in_len, blank_id=blank_id)

        for i in range(min(5, X.size(0))):
            gt_ids = Y[i, : int(y_len[i])].tolist()
            pr_ids = preds[i]

            gt_tokens = [id2token[j] for j in gt_ids if j in id2token]
            pr_tokens = [id2token[j] for j in pr_ids if j in id2token]

            print(f"- #{i}")
            print("  GT :", " ".join(gt_tokens))
            print("  PR :", " ".join(pr_tokens))


if __name__ == "__main__":
    main()
