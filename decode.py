# ~/src/decode.py
# 추론 테스트 / 디버깅
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from models import SignCTCModel
from data import normalize_hands_frame  # 학습과 동일 정규화

def load_ckpt(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    # 다양한 포맷 호환
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        backend = ck.get("backend", "tcn")
        in_dim = ck.get("in_dim", None)
        itos = ck.get("vocab", None)
    else:
        state, backend, in_dim, itos = ck, "tcn", None, None
    return {"state": state, "backend": backend, "in_dim": in_dim, "itos": itos}

def greedy_batch(logits, blank=0):
    ids = logits.argmax(-1)  # B,T
    outs = []
    for seq in ids:
        out, prev = [], blank
        for i in seq.tolist():
            if i != blank and i != prev:
                out.append(i)
            prev = i
        outs.append(out)
    return outs

def load_npz_as_seq(path, do_norm=True):
    with np.load(path, allow_pickle=True) as z:
        if 'seq' in z: arr = z['seq']
        elif 'x' in z: arr = z['x']
        else:
            keys = [k for k in z.files if isinstance(z[k], np.ndarray)]
            if not keys:
                raise ValueError(f'{path}: no ndarray keys')
            arr = z[keys[0]]
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'{path}: expected 2D (T,F), got shape {arr.shape}')
    if do_norm:
        try:
            arr = np.apply_along_axis(normalize_hands_frame, 1, arr)
        except Exception:
            pass
    return arr  # [T,F]

def iter_npz_inputs(arg):
    p = Path(arg)
    if p.is_dir():
        for f in sorted(p.rglob("*.npz")):
            yield f
    elif "*" in p.name or "?" in p.name or "[" in p.name:
        for f in sorted(p.parent.glob(p.name)):
            if f.suffix == ".npz": yield f
    else:
        yield p

def frame_avg_topk(logp_btV, k=5, temp=1.0):
    # logp_btV: [B,T,V] (logits softmax 후 log)
    score_bv = (logp_btV / max(1e-6, temp)).mean(dim=1).softmax(dim=-1)  # [B,V]
    vals, idx = torch.topk(score_bv, k=min(k, score_bv.size(1)), dim=-1)
    return idx, vals  # [B,K]

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--ckpt', required=True)
    pa.add_argument('--npz', required=True, help='파일/디렉터리/패턴 가능 (ex: samples/*.npz)')
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--normalize', choices=['on','off'], default='on')
    pa.add_argument('--dtype', choices=['fp32','fp16','bf16'], default='fp32')
    pa.add_argument('--blank', type=int, default=0)
    pa.add_argument('--topk', type=int, default=5)
    pa.add_argument('--temp', type=float, default=1.0)
    pa.add_argument('--out', default='', help='결과를 JSON Lines(.jsonl)로 저장')
    args = pa.parse_args()

    ck = load_ckpt(args.ckpt)
    itos = ck["itos"]
    V = len(itos) if itos is not None else None

    if V is None:
        raise RuntimeError("ckpt에 vocab(itos)이 없습니다. 학습 저장 시 'vocab' 키를 포함하세요.")

    model = SignCTCModel(
        in_dim=(ck["in_dim"] if ck["in_dim"] else None),
        vocab_size=V,
        backend=ck["backend"]
    )
    model.load_state_dict(ck["state"])
    model.to(args.device)
    model.eval()

    # dtype 설정
    if args.dtype == 'fp16':
        model = model.half()
    elif args.dtype == 'bf16':
        model = model.bfloat16()

    out_fp = open(args.out, "w", encoding="utf-8") if args.out else None
    normalize_on = (args.normalize == 'on')

    with torch.inference_mode():
        for fpath in iter_npz_inputs(args.npz):
            x_np = load_npz_as_seq(str(fpath), do_norm=normalize_on)  # [T,F]
            if ck["in_dim"] and x_np.shape[1] != ck["in_dim"]:
                print(f"[WARN] {fpath.name}: feature dim {x_np.shape[1]} != ckpt.in_dim {ck['in_dim']}")

            x = torch.from_numpy(x_np).unsqueeze(0)  # [1,T,F]
            if args.dtype == 'fp16': x = x.half()
            if args.dtype == 'bf16': x = x.bfloat16()
            x = x.to(args.device)

            t0 = time.time()
            logits = model(x)                     # [1,T,V]
            dt = time.time() - t0

            ids = greedy_batch(logits, blank=args.blank)[0]
            toks = [itos[i] for i in ids if 0 <= i < V]

            logp = logits.float().log_softmax(-1)  # 안전을 위해 FP32로
            top_idx, top_val = frame_avg_topk(logp, k=args.topk, temp=args.temp)
            top = [(itos[i], float(p)) for i, p in zip(top_idx[0].tolist(), top_val[0].tolist())]

            msg = {
                "file": str(fpath),
                "T": int(x_np.shape[0]),
                "F": int(x_np.shape[1]),
                "backend": ck["backend"],
                "in_dim_ckpt": ck["in_dim"],
                "decode_ids": ids,
                "decode_tokens": toks,
                "topk_frameavg": top,
                "time_sec": round(dt, 4)
            }

            print(f"DECODE [{fpath.name}] → {toks}  | topK={top[:3]}  | {dt*1000:.1f} ms")
            if out_fp:
                out_fp.write(json.dumps(msg, ensure_ascii=False) + "\n")

    if out_fp:
        out_fp.close()
        print(f"[Saved] JSONL → {args.out}")

if __name__ == '__main__':
    main()