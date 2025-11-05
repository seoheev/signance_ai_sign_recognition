# ~/src/eval_word.py
# -*- coding: utf-8 -*-
"""
단어(분류) 평가 스크립트
- Metrics: Top-1 Acc, Top-K Acc, macro-F1, confusion matrix
- 저장 옵션: 혼동행렬 CSV/PNG, 분류 리포트 TXT/CSV

예시:
python eval_word.py \
  --data-root ../datasets --npz_subdir npz/eco \
  --csv-name labels.csv --vocab-json vocab.json \
  --in-dim 126 --batch 64 \
  --ckpt runs/word_best.pt \
  --split valid --topk 5 \
  --save-cm-csv runs/word_cm.csv \
  --save-cm-png runs/word_cm.png \
  --save-report runs/word_report.txt
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 내부 모듈
from data_word import WordDataset, collate_word
from data import load_vocab
from models import HybridBackbone, WordClassifier

def try_import_sklearn():
    try:
        from sklearn.metrics import f1_score, confusion_matrix, classification_report
        return f1_score, confusion_matrix, classification_report
    except Exception:
        return None, None, None

f1_score, confusion_matrix, classification_report = try_import_sklearn()

def build_model(ckpt_path: str, V: int, in_dim_cli: int, device: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

    # 하이퍼파라미터 복원(없으면 기본값 사용)
    subsample_stages = ck.get("subsample_stages", 1)
    hid = ck.get("hid", 256)
    in_dim_ck = ck.get("in_dim", in_dim_cli)

    backbone = HybridBackbone(
        in_dim=in_dim_ck, hid=hid, depth=6, nhead=4, p=0.1,
        subsample_stages=subsample_stages
    ).to(device)
    model = WordClassifier(backbone, vocab_size=V).to(device)
    model.load_state(state)
    return model, in_dim_ck, subsample_stages, hid

def plot_confusion_matrix(cm: np.ndarray, labels: list, out_png: str):
    fig = plt.figure(figsize=(max(6, len(labels)*0.25), max(5, len(labels)*0.25)))
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    fig.colorbar(im)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-root', default=str(Path(__file__).parent.parent / 'datasets'))
    pa.add_argument('--npz_subdir', default='npz/eco')
    pa.add_argument('--csv-name', default='labels.csv')
    pa.add_argument('--vocab-json', default='vocab.json')
    pa.add_argument('--in-dim', type=int, default=126)

    pa.add_argument('--batch', type=int, default=64)
    pa.add_argument('--num-workers', type=int, default=4)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    pa.add_argument('--ckpt', required=True)
    pa.add_argument('--split', choices=['train','valid','all'], default='valid')
    pa.add_argument('--split-ratio', type=float, default=0.9)
    pa.add_argument('--augment', action='store_true', help='평가 시 증강은 일반적으로 OFF 권장')

    pa.add_argument('--topk', type=int, default=5)
    pa.add_argument('--save-cm-csv', default='', help='혼동행렬 CSV 저장 경로')
    pa.add_argument('--save-cm-png', default='', help='혼동행렬 PNG 저장 경로')
    pa.add_argument('--save-report', default='', help='classification report 저장(.txt/.csv)')
    args = pa.parse_args()

    device = args.device
    pin = device.startswith('cuda')

    # vocab 로드
    itos, _ = load_vocab(Path(args.data_root)/args.vocab_json)
    V = len(itos)

    # 데이터셋/로더 (분류용)
    ds = WordDataset(args.data_root, args.csv_name, args.vocab_json, args.npz_subdir,
                     split=args.split, split_ratio=args.split_ratio, stratify=True,
                     augment=args.augment, in_dim_override=args.in_dim, verbose=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.num_workers, pin_memory=pin, collate_fn=collate_word)

    # 모델 구성 & 로드
    model, in_dim_ck, subsample_stages, hid = build_model(args.ckpt, V, args.in_dim, device)
    model.eval()
    print(f"[info] ckpt loaded: in_dim={in_dim_ck}, hid={hid}, subsample_stages={subsample_stages}")

    # 평가 루프
    y_true, y_pred, topk_hits, total = [], [], 0, 0
    with torch.no_grad():
        for x, y, pad in dl:
            x = x.to(device); y = y.to(device)
            logits = model(x, src_key_padding_mask=pad.to(device))  # [B,V]
            pred = logits.argmax(dim=-1)
            k = min(args.topk, logits.size(1))
            topk = torch.topk(logits, k=k, dim=-1).indices
            hits = (topk == y.unsqueeze(-1)).any(dim=-1)

            # label==0(<blank>) 샘플은 지표에서 제외 (있을 경우)
            mask_valid = (y != 0)

            y_true.extend(y[mask_valid].cpu().tolist())
            y_pred.extend(pred[mask_valid].cpu().tolist())
            topk_hits += hits[mask_valid].sum().item()
            total += mask_valid.sum().item()

    if total == 0:
        print("[warn] 유효한 샘플( y != 0 )이 없습니다.")
        return

    acc = (np.array(y_true) == np.array(y_pred)).mean()
    acck = topk_hits / total
    print(f"Top-1 Accuracy : {acc*100:.2f}%")
    print(f"Top-{args.topk} Accuracy : {acck*100:.2f}%")

    # Macro-F1 & Confusion Matrix
    class_ids = sorted(set(y_true))  # 사용된 클래스만
    class_names = [itos[c] if c < len(itos) else f"id{c}" for c in class_ids]

    if f1_score is not None:
        f1_macro = f1_score(y_true, y_pred, labels=class_ids, average='macro', zero_division=0)
        print(f"Macro-F1 : {f1_macro:.4f}")
    else:
        f1_macro = None
        print("[info] scikit-learn 미설치 → Macro-F1 계산 생략 (pip install scikit-learn)")

    if confusion_matrix is not None:
        cm = confusion_matrix(y_true, y_pred, labels=class_ids)
        # 저장 옵션
        if args.save_cm_csv:
            out = Path(args.save_cm_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            import csv
            with out.open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow([''] + class_names)  # header
                for i, row in enumerate(cm):
                    w.writerow([class_names[i]] + row.tolist())
            print(f"✓ saved confusion matrix CSV → {out}")

        if args.save_cm_png:
            try:
                plot_confusion_matrix(cm, class_names, args.save_cm_png)
                print(f"✓ saved confusion matrix PNG → {args.save_cm_png}")
            except Exception as e:
                print(f"[warn] CM PNG 저장 실패: {e}")

        if args.save_report and classification_report is not None:
            rep_txt = classification_report(y_true, y_pred, labels=class_ids,
                                            target_names=class_names, digits=4, zero_division=0)
            out = Path(args.save_report)
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.suffix.lower() == '.csv':
                # class별 precision/recall/f1/support를 CSV로 내보내기
                from sklearn.metrics import precision_recall_fscore_support
                import csv
                p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=class_ids, zero_division=0)
                with out.open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['label','precision','recall','f1','support'])
                    for name, pi, ri, fi, si in zip(class_names, p, r, f, s):
                        w.writerow([name, f"{pi:.6f}", f"{ri:.6f}", f"{fi:.6f}", int(si)])
                print(f"✓ saved classification report CSV → {out}")
            else:
                out.write_text(rep_txt, encoding='utf-8')
                print(f"✓ saved classification report TXT → {out}")
    else:
        print("[info] scikit-learn 미설치 → 혼동행렬/리포트 저장 생략 (pip install scikit-learn)")

if __name__ == "__main__":
    main()
