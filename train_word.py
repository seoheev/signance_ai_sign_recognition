# ~/src/train_word.py
# -*- coding: utf-8 -*-
"""
(단어) 분류 학습: StatsPooling(Mean⊕Std) → Linear → CE
- 백본: HybridBackbone(TCN + ConvSubsample + Transformer)
- 기본 subsample_stages=1 (T -> T/2), 필요시 2(T -> T/4)
"""
import argparse, random, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader

from data_word import WordDataset, collate_word
from data import load_vocab
from models import HybridBackbone, WordClassifier

def set_seeds(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-root', default=str(Path(__file__).parent.parent / 'datasets'))
    pa.add_argument('--npz_subdir', default='npz/eco')
    pa.add_argument('--csv-name', default='labels.csv')
    pa.add_argument('--vocab-json', default='vocab.json')

    pa.add_argument('--in-dim', type=int, default=126)
    pa.add_argument('--hid', type=int, default=256)
    pa.add_argument('--depth', type=int, default=6)
    pa.add_argument('--nhead', type=int, default=4)
    pa.add_argument('--dropout', type=float, default=0.1)
    pa.add_argument('--subsample-stages', type=int, default=1)  # 1=1/2T, 2=1/4T

    pa.add_argument('--epochs', type=int, default=20)
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--lr', type=float, default=3e-4)
    pa.add_argument('--augment', action='store_true')
    pa.add_argument('--closed-world', action='store_true')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--save', default='runs/word_best.pt')
    args = pa.parse_args()

    set_seeds(args.seed)
    pin = args.device.startswith('cuda')

    if args.closed_world:
        tr = WordDataset(args.data_root, args.csv_name, args.vocab_json, args.npz_subdir,
                         split='all', stratify=False, augment=args.augment, in_dim_override=args.in_dim)
        va = WordDataset(args.data_root, args.csv_name, args.vocab_json, args.npz_subdir,
                         split='all', stratify=False, augment=False, in_dim_override=args.in_dim)
    else:
        tr = WordDataset(args.data_root, args.csv_name, args.vocab_json, args.npz_subdir,
                         split='train', stratify=True, augment=args.augment, in_dim_override=args.in_dim)
        va = WordDataset(args.data_root, args.csv_name, args.vocab_json, args.npz_subdir,
                         split='valid', stratify=True, augment=False, in_dim_override=args.in_dim)

    itos, _ = load_vocab(Path(args.data_root)/args.vocab_json)
    V = len(itos)  # <blank>=0 포함일 수 있음 → CE에서는 ignore_index=0로 무시

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.num_workers,
                       pin_memory=pin, collate_fn=collate_word)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.num_workers,
                       pin_memory=pin, collate_fn=collate_word)

    backbone = HybridBackbone(
        in_dim=args.in_dim, hid=args.hid, depth=args.depth,
        nhead=args.nhead, p=args.dropout, subsample_stages=args.subsample_stages
    ).to(args.device)

    model = WordClassifier(backbone, vocab_size=V).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    ce  = nn.CrossEntropyLoss(ignore_index=0)  # 0(<blank>)은 무시

    def evaluate(loader):
        model.eval()
        total, correct, top5 = 0, 0, 0
        with torch.no_grad():
            for x, y, pad in loader:
                x = x.to(args.device); y = y.to(args.device)
                logits = model(x, src_key_padding_mask=pad.to(args.device))  # [B,V]
                total += y.numel()
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                k = min(5, logits.size(1))
                topk = torch.topk(logits, k=k, dim=-1).indices
                top5 += (topk == y.unsqueeze(-1)).any(dim=-1).sum().item()
        acc = correct / max(total, 1)
        acc5 = top5 / max(total, 1)
        return acc, acc5

    best = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        tot, n = 0.0, 0
        for x, y, pad in dl_tr:
            x = x.to(args.device); y = y.to(args.device)
            logits = model(x, src_key_padding_mask=pad.to(args.device))    # [B,V]
            loss = ce(logits, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()*x.size(0); n += x.size(0)
        tr_loss = tot / max(n, 1)
        va_acc, va_acc5 = evaluate(dl_va)
        print(f"[ep {ep:02d}] train {tr_loss:.4f}  valid@1 {va_acc*100:5.2f}%  valid@5 {va_acc5*100:5.2f}%")

        if va_acc > best:
            best = va_acc
            Path(args.save).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "vocab": itos,
                "in_dim": args.in_dim,
                "kind": "word",
                "npz_subdir": args.npz_subdir,
                "backbone": "hybrid",
                "subsample_stages": args.subsample_stages,
                "hid": args.hid,
            }, args.save)
            print(f"  ✓ saved {args.save} (valid@1={va_acc*100:.2f}%)")

if __name__ == "__main__":
    main()
