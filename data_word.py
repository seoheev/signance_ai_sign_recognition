# ~/src/data_word.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

from data import load_vocab, norm_name, _to_numpy_2d, normalize_hands_frame, augment_seq  # 재사용

class WordDataset(Dataset):
    def __init__(self, data_root, csv_name="labels.csv", vocab_json="vocab.json",
                 npz_subdir="npz/eco", split="train", split_ratio=0.9,
                 seed=42, stratify=True, augment=False, in_dim_override=None, verbose=True):
        self.root = Path(data_root)
        self.per_sample_dir = self.root / npz_subdir
        self.df = pd.read_csv(self.root / csv_name)
        self.df.columns = [c.strip().lower() for c in self.df.columns]
        assert "file" in self.df.columns and "label" in self.df.columns

        self.df["file_norm"] = self.df["file"].map(norm_name)
        self.df["label"] = self.df["label"].astype(str).map(str.strip)

        self.itos, self.stoi = load_vocab(self.root / vocab_json)

        def exists(name): return (self.per_sample_dir / f"{name}.npz").exists()
        before = len(self.df)
        self.df = self.df[self.df["file_norm"].map(exists)].reset_index(drop=True)
        after = len(self.df)
        if verbose:
            print(f"[WordDataset] kept {after}/{before} rows after existence check")

        idx = np.arange(len(self.df))
        rng = np.random.default_rng(seed); rng.shuffle(idx)

        if split == "all":
            use_idx = idx
        elif stratify and split in ("train","valid"):
            groups = {}
            for i in idx: groups.setdefault(self.df.iloc[i]["label"], []).append(i)
            train_idx, valid_idx = [], []
            for gidx in groups.values():
                rng.shuffle(gidx)
                cut = int(len(gidx) * float(split_ratio))
                if len(gidx) <= 1:
                    train_idx.extend(gidx)
                else:
                    train_idx.extend(gidx[:cut]); valid_idx.extend(gidx[cut:])
            use_idx = train_idx if split == "train" else valid_idx
        else:
            cut = int(len(self.df) * float(split_ratio))
            use_idx = idx[:cut] if split=="train" else idx[cut:]

        self.df = self.df.iloc[list(use_idx)].reset_index(drop=True)
        self.augment = augment
        self.in_dim_override = in_dim_override
        self.verbose = verbose

    def __len__(self): return len(self.df)

    def _load_npz(self, name):
        path = self.per_sample_dir / f"{name}.npz"
        with np.load(path, allow_pickle=True) as z:
            if "seq" in z: raw = z["seq"]
            elif "x" in z: raw = z["x"]
            else:
                keys = [k for k in z.files if isinstance(z[k], np.ndarray)]
                if not keys: raise ValueError(f"{path.name}: no ndarray keys")
                raw = z[keys[0]]
        arr = _to_numpy_2d(raw)
        return arr

    def __getitem__(self, i):
        row = self.df.iloc[i]
        name, label_str = row["file_norm"], str(row["label"])
        arr = self._load_npz(name)
        # 프레임별 좌표 정규화
        try: arr = np.apply_along_axis(normalize_hands_frame, 1, arr)
        except Exception: pass
        # (옵션) 약한 증강
        if self.augment: 
            try: arr = augment_seq(arr)
            except Exception: pass

        if self.in_dim_override and arr.shape[1] != self.in_dim_override:
            # 차원 불일치 샘플은 스킵 → 호출측 collate에서 필터링이 쉽도록 None 반환
            return None

        # 단일 라벨 id
        yi = self.stoi.get(label_str, None)
        if yi is None or yi == 0:  # blank나 미정규 라벨은 <unk>로
            yi = self.stoi.get("<unk>", 1)
        return arr.astype(np.float32), int(yi)

def collate_word(batch):
    # None 제거
    items = [(x,y) for x,y in batch if x is not None]
    if not items: raise RuntimeError("empty batch after filtering")
    xs, ys = zip(*items)
    T = [x.shape[0] for x in xs]; F = xs[0].shape[1]; B = len(xs)
    maxT = max(T)
    xpad = torch.zeros(B, maxT, F, dtype=torch.float32)
    pad = torch.ones (B, maxT, dtype=torch.bool)
    for i,x in enumerate(xs):
        t = x.shape[0]
        xpad[i,:t] = torch.from_numpy(x)
        pad [i,:t] = False
    y = torch.tensor(ys, dtype=torch.long)  # [B]
    return xpad, y, pad  # pad: True=pad
