# ~/src/train_sent.py
# -*- coding: utf-8 -*-
"""
(문장) CTC 학습: HybridBackbone → Linear(H,V) → CTCLoss
- 디코딩: greedy/beam은 eval/decode 스크립트에서 수행
- 전이학습: --init-backbone 로 단어 학습 백본 가중치만 로드
- 기본 subsample_stages=1 (T -> T/2), 필요시 2(T -> T/4)
"""
import argparse, random, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from dataset_gloss import GlossSeqDataset, collate_ctc
from models import (
    HybridBackbone, SentenceCTC, load_backbone_from_word_ckpt,
    down_len
)

def set_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--root', default=str(Path(__file__).parent.parent / 'datasets/sentences'))
    pa.add_argument('--index-csv', default='index.csv')
    pa.add_argument('--vocab-json', default='vocab.json')
    pa.add_argument('--split', choices=['train','valid','all'], default='train')
    pa.add_argument('--split-ratio', type=float, default=0.9)
    pa.add_argument('--seed', type=int, default=42)

    pa.add_argument('--in-dim', type=int, default=126)
    pa.add_argument('--hid', type=int, default=256)
    pa.add_argument('--depth', type=int, default=6)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--subsample-stages', type=int, default=1)  # 1=1/2T, 2=1/4T

    pa.add_argument('--epochs', type=int, default=20)
    pa.add_argument('--batch', type=int, default=16)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--save', default='runs/sent_best.pt')

    # 전이학습(선택): 단어 학습 ckpt에서 backbone만 로드
    pa.add_argument('--init-backbone', default='', help='ex) runs/word_best.pt')
    args = pa.parse_args()

    set_seeds(args.seed)
    pin = args.device.startswith('cuda')

    ds_tr = GlossSeqDataset(root=args.root, index_csv=args.index_csv, vocab_json=args.vocab_json,
                            split='train' if args.split != 'all' else 'all', split_ratio=args.split_ratio)
    ds_va = GlossSeqDataset(root=args.root, index_csv=args.index_csv, vocab_json=args.vocab_json,
                            split='valid' if args.split != 'all' else 'all', split_ratio=args.split_ratio)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True,  num_workers=args.num_workers,
                       pin_memory=pin, collate_fn=collate_ctc)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                       pin_memory=pin, collate_fn=collate_ctc)

    V = len(ds_tr.token2id) + 1  # 0(blank) + 1..N

    backbone = HybridBackbone(
        in_dim=args.in_dim, hid=args.hid, depth=args.depth,
        nhead=args.nhead, p=args.dropout, subsample_stages=args.subsample_stages
    ).to(args.device)

    if args.init_backbone:
        missing, unexpected = load_backbone_from_word_ckpt(backbone, args.init_backbone)
        print(f"[transfer] loaded backbone from {args.init_backbone}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    model = SentenceCTC(backbone, vocab_size=V).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def run_epoch(loader, train=True):
        model.train(train)
        tot, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for X, x_len, Y, y_len in loader:
                # X: [B,T,F], x_len: [B], Y: [sumL], y_len: [B]
                X = X.to(args.device)
                pad = torch.arange(X.size(1), device=X.device)[None, :] >= x_len.to(X.device)[:, None]  # [B,T]
                logits = model(X, src_key_padding_mask=pad)    # [B,T',V]
                logp = logits.log_softmax(-1).transpose(0, 1)  # [T',B,V]
                in_len = down_len(x_len.to('cpu', torch.int64), args.subsample_stages)  # [B]
                loss = ctc(logp, Y.to(args.device), in_len, y_len.to('cpu', torch.int64))
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                tot += loss.item() * X.size(0); n += X.size(0)
        return tot / max(n, 1)

    best = float('inf')
    for ep in range(1, args.epochs+1):
        tr = run_epoch(dl_tr, True)
        va = run_epoch(dl_va, False)
        print(f"[ep {ep:02d}] train {tr:.4f}  valid {va:.4f}")
        if va < best:
            best = va
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "kind": "sentence",
                "in_dim": args.in_dim,
                "hid": args.hid,
                "subsample_stages": args.subsample_stages,
                "vocab_json": args.vocab_json,
            }, args.save)
            print(f"  ✓ saved {args.save} (best={best:.4f})")

if __name__ == "__main__":
    main()
