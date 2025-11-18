# ~/src/eval_sent.py
# -*- coding: utf-8 -*-

"""
Sentence CTC 평가 스크립트 (print-examples 옵션 추가 버전)
- GlossSeqDataset + SentenceCTC용
- Metrics:
  - avg CTC loss (valid)
  - sequence exact match accuracy
  - token-level CER (edit distance 기반)
  - avg ref/pred length
  - blank-only ratio
  - unique token recall / precision
Sentence CTC 평가 스크립트 (print-examples 옵션 추가 버전)
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
    logp = logits.log_softmax(-1)      # [B,T',V]
    ids = logp.argmax(-1)              # [B,T']
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
            if v == prev:
                continue
            seq.append(v)
            prev = v
        preds.append(seq)
    return preds


def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = [[0] * (lb + 1) for _ in range(la + 1)]

    for i in range(la + 1): dp[i][0] = i
    for j in range(lb + 1): dp[0][j] = j

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb]


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--root', required=True)
    pa.add_argument('--index-csv', default='index.csv')
    pa.add_argument('--vocab-json', default='vocab.json')
    pa.add_argument('--ckpt', required=True)
    pa.add_argument('--split', choices=['train', 'valid', 'all'], default='valid')

    # 모델 설정 (train_sent.py와 동일하게!)
    pa.add_argument('--in-dim', type=int, default=126)
    pa.add_argument('--hid', type=int, default=256)
    pa.add_argument('--depth', type=int, default=6)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--subsample-stages', type=int, default=1)

    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    # 🔥 추가 옵션
    pa.add_argument('--print-examples', type=int, default=5,
                    help='평가 후 출력할 샘플 개수')

    args = pa.parse_args()

    device = torch.device(args.device)
    pin = args.device.startswith('cuda')

    # Dataset
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

    # Token mappings
    token2id = ds.token2id
    id2token = {i: t for t, i in token2id.items()}
    blank_id = 0
    V = len(token2id) + 1

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt

    in_dim = ckpt.get('in_dim', args.in_dim) if isinstance(ckpt, dict) else args.in_dim
    hid = ckpt.get('hid', args.hid)
    subs = ckpt.get('subsample_stages', args.subsample_stages)

    # Build model
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

    # Accumulators
    total_loss = 0.0
    total_n = 0
    total_seq = 0
    correct_seq = 0
    total_tokens = 0
    total_edit = 0
    total_ref_len = 0
    total_pred_len = 0
    blank_only = 0
    total_ref_unique = 0
    total_pred_unique = 0
    total_intersection = 0

    with torch.no_grad():
        for X, x_len, Y, y_len in dl:
            X = X.to(device)
            x_len = x_len.to(device)

            pad = torch.arange(X.size(1), device=device)[None, :] >= x_len[:, None]
            logits = model(X, src_key_padding_mask=pad)
            logp = logits.log_softmax(-1).transpose(0, 1)

            in_len = down_len(x_len.to('cpu', torch.int64), subs)

            loss = ctc(logp, Y.to(device), in_len, y_len.to('cpu', torch.int64))
            bs = X.size(0)
            total_loss += loss.item() * bs
            total_n += bs

            preds = ctc_greedy_decode(logits, in_len, blank_id=blank_id)

            # Extract gold labels
            gts = []
            for b in range(bs):
                L = int(y_len[b])
                gts.append(Y[b, :L].tolist())

            # Metrics
            for p, g in zip(preds, gts):
                total_seq += 1
                if p == g:
                    correct_seq += 1

                ed = edit_distance(p, g)
                total_edit += ed
                total_tokens += len(g)

                total_ref_len += len(g)
                total_pred_len += len(p)
                if len(p) == 0:
                    blank_only += 1

                ref_set = set(g)
                pred_set = set(p)
                total_intersection += len(ref_set & pred_set)
                total_ref_unique += len(ref_set)
                total_pred_unique += len(pred_set)

    # Final metrics
    avg_loss = total_loss / max(1, total_n)
    seq_acc = correct_seq / max(1, total_seq)
    cer = total_edit / max(1, total_tokens)
    avg_ref_len = total_ref_len / total_seq
    avg_pred_len = total_pred_len / total_seq
    blank_ratio = blank_only / total_seq
    recall_unique = total_intersection / total_ref_unique if total_ref_unique > 0 else 0
    precision_unique = total_intersection / total_pred_unique if total_pred_unique > 0 else 0

    print(f"[Eval] split={args.split}")
    print(f"  avg CTC loss          : {avg_loss:.4f}")
    print(f"  seq accuracy          : {seq_acc*100:.2f}%")
    print(f"  token CER             : {cer*100:.2f}%")
    print(f"  Avg ref length        : {avg_ref_len:.2f}")
    print(f"  Avg pred length       : {avg_pred_len:.2f}")
    print(f"  Blank-only ratio      : {blank_ratio*100:.2f}%")
    print(f"  Unique token recall   : {recall_unique*100:.2f}%")
    print(f"  Unique token precision: {precision_unique*100:.2f}%")

    # ======================================================
    # 🔥 여기: print-examples N 개 출력 (추가)
    # ======================================================
    print(f"\n[Samples: showing {args.print_examples}]")

    with torch.no_grad():
        dl_iter = iter(dl)
        X, x_len, Y, y_len = next(dl_iter)

        X = X.to(device)
        x_len = x_len.to(device)
        pad = torch.arange(X.size(1), device=device)[None, :] >= x_len[:, None]
        logits = model(X, src_key_padding_mask=pad)
        in_len = down_len(x_len.to('cpu', torch.int64), subs)
        preds = ctc_greedy_decode(logits, in_len, blank_id=blank_id)

        for i in range(min(args.print_examples, X.size(0))):
            gt_ids = Y[i, : int(y_len[i])].tolist()
            pr_ids = preds[i]
            gt_tokens = [id2token[j] for j in gt_ids if j in id2token]
            pr_tokens = [id2token[j] for j in pr_ids if j in id2token]

            print(f"- #{i}")
            print("  GT :", " ".join(gt_tokens))
            print("  PR :", " ".join(pr_tokens))


if __name__ == "__main__":
    main()
