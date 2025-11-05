# ~/src/dump_word_preds.py
# -*- coding: utf-8 -*-
"""
단어(분류) 모델의 샘플별 예측 결과 CSV 덤프 스크립트

예시:
python dump_word_preds.py \
  --data-root ../datasets --npz_subdir npz/eco \
  --csv-name labels.csv --vocab-json vocab.json \
  --in-dim 126 --ckpt runs/word_best.pt \
  --batch 64 --topk 5 \
  --out runs/word_preds.csv
"""
import argparse
from pathlib import Path
import csv
import numpy as np
import torch
import torch.nn.functional as F

from data import load_vocab, norm_name, _to_numpy_2d, normalize_hands_frame, augment_seq
from models import HybridBackbone, WordClassifier

def load_npz_array(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=True) as z:
        if "seq" in z: raw = z["seq"]
        elif "x" in z: raw = z["x"]
        else:
            keys = [k for k in z.files if isinstance(z[k], np.ndarray)]
            if not keys:
                raise ValueError(f"{npz_path.name}: no ndarray keys")
            raw = z[keys[0]]
    return _to_numpy_2d(raw)

def build_model(ckpt_path: str, V: int, in_dim_cli: int, device: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    subsample_stages = ck.get("subsample_stages", 1)
    hid = ck.get("hid", 256)
    in_dim_ck = ck.get("in_dim", in_dim_cli)

    backbone = HybridBackbone(
        in_dim=in_dim_ck, hid=hid, depth=6, nhead=4, p=0.1,
        subsample_stages=subsample_stages
    ).to(device)
    model = WordClassifier(backbone, vocab_size=V).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, in_dim_ck

def pad_batch(xs: list[np.ndarray], device: str):
    """xs: list of [T,F] numpy → returns xpad[B,Tmax,F], pad[B,Tmax] (True=pad)"""
    T = [x.shape[0] for x in xs]
    F = xs[0].shape[1]
    B = len(xs)
    Tmax = max(T)
    xpad = torch.zeros(B, Tmax, F, dtype=torch.float32, device=device)
    pad  = torch.ones (B, Tmax,      dtype=torch.bool,   device=device)
    for i, arr in enumerate(xs):
        t = arr.shape[0]
        xpad[i, :t] = torch.from_numpy(arr).to(device)
        pad [i, :t] = False
    return xpad, pad

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data-root', default=str(Path(__file__).parent.parent / 'datasets'))
    pa.add_argument('--npz_subdir', default='npz/eco')
    pa.add_argument('--csv-name', default='labels.csv')
    pa.add_argument('--vocab-json', default='vocab.json')
    pa.add_argument('--in-dim', type=int, default=126)

    pa.add_argument('--ckpt', required=True)
    pa.add_argument('--batch', type=int, default=64)
    pa.add_argument('--topk', type=int, default=5)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--out', default='runs/word_preds.csv')
    pa.add_argument('--augment', action='store_true', help='(보통 OFF 권장) 덤프 시 증강 적용')
    args = pa.parse_args()

    device = args.device
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # --- 데이터 로드(그냥 DataFrame 쓰지 않고 직접 순회: None 필터링/정렬 보존을 위해) ---
    import pandas as pd
    root = Path(args.data_root)
    per_dir = root / args.npz_subdir
    df = pd.read_csv(root / args.csv_name)
    df.columns = [c.strip().lower() for c in df.columns]
    assert "file" in df.columns and "label" in df.columns
    df["file_norm"] = df["file"].map(norm_name)
    df["label"] = df["label"].astype(str).map(str.strip)

    # vocab
    itos, stoi = load_vocab(root / args.vocab_json)
    V = len(itos)

    # 모델
    model, in_dim_ck = build_model(args.ckpt, V, args.in_dim, device)
    if in_dim_ck != args.in_dim:
        print(f"[warn] ckpt in_dim={in_dim_ck} ≠ CLI in_dim={args.in_dim} → ckpt 설정 사용")

    # 존재하는 샘플만 대상
    def exists(name): return (per_dir / f"{name}.npz").exists()
    df = df[df["file_norm"].map(exists)].reset_index(drop=True)

    # 순회 → 미니배치 처리
    out_rows = []
    names_buf, labels_buf, arr_buf = [], [], []

    def flush_batch():
        nonlocal out_rows, names_buf, labels_buf, arr_buf
        if not arr_buf: return
        xpad, pad = pad_batch(arr_buf, device)
        with torch.no_grad():
            logits = model(xpad, src_key_padding_mask=pad)  # [B,V]
            probs = F.softmax(logits, dim=-1)               # [B,V]
            topk = min(args.topk, probs.size(1))
            topv, topi = torch.topk(probs, k=topk, dim=-1)  # [B,K], [B,K]

        for i, (nm, y_str) in enumerate(zip(names_buf, labels_buf)):
            yi = stoi.get(str(y_str), None)
            yi = 1 if (yi is None or yi == 0) else yi      # <blank>/없으면 <unk>=1 가정
            pred_id = int(topi[i, 0].item())
            pred_name = itos[pred_id] if pred_id < len(itos) else f"id{pred_id}"
            pred_prob = float(topv[i, 0].item())

            k_ids = [int(x) for x in topi[i].tolist()]
            k_names = [itos[j] if j < len(itos) else f"id{j}" for j in k_ids]
            k_probs = [float(x) for x in topv[i].tolist()]

            out_rows.append({
                "file": nm + ".npz",
                "true_label": str(y_str),
                "true_id": yi,
                "pred_id": pred_id,
                "pred_name": pred_name,
                "pred_prob": f"{pred_prob:.6f}",
                "topk_ids": "|".join(map(str, k_ids)),
                "topk_names": "|".join(k_names),
                "topk_probs": "|".join(f"{p:.6f}" for p in k_probs),
            })
        # reset buffers
        names_buf, labels_buf, arr_buf = [], [], []

    for _, row in df.iterrows():
        name = row["file_norm"]
        label_str = str(row["label"])
        npz_path = per_dir / f"{name}.npz"

        try:
            arr = load_npz_array(npz_path)                 # [T,F]
            # 프레임 정규화
            try: arr = np.apply_along_axis(normalize_hands_frame, 1, arr)
            except Exception: pass
            if args.augment:
                try: arr = augment_seq(arr)
                except Exception: pass

            if arr.shape[1] != in_dim_ck:
                # 차원 불일치 → 스킵
                print(f"[skip] in_dim mismatch {npz_path.name}: {arr.shape[1]} != {in_dim_ck}")
                continue

            names_buf.append(name)
            labels_buf.append(label_str)
            arr_buf.append(arr.astype(np.float32))

            if len(arr_buf) >= args.batch:
                flush_batch()

        except Exception as e:
            print(f"[warn] failed to load {npz_path.name}: {e}")

    flush_batch()

    # CSV 저장
    out = Path(args.out)
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "true_label", "true_id",
            "pred_id", "pred_name", "pred_prob",
            "topk_ids", "topk_names", "topk_probs",
        ])
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"✓ saved per-sample predictions → {out}  (rows={len(out_rows)})")

if __name__ == "__main__":
    main()
