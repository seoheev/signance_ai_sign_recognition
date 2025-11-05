# 말뭉치(문장) 데이터 정의 파일
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class GlossSeqDataset(Dataset):
    def __init__(self, root="processed", index_csv="index.csv", vocab_json="vocab.json",
                 split="train", split_ratio=0.9, seed=42, shuffle=True):
        self.root = Path(root)

        # 1) index.csv 읽기
        with open(self.root / index_csv, "r", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()[1:]  # 헤더 스킵

        rng = np.random.default_rng(seed)
        idx = np.arange(len(lines))
        if shuffle:
            rng.shuffle(idx)

        # 2) 분할
        if split == "all":
            use_idx = idx
        else:
            cut = int(len(lines) * float(split_ratio))
            if split == "train":
                use_idx = idx[:cut]
            elif split == "valid":
                use_idx = idx[cut:]
            else:
                raise ValueError(f"unknown split: {split}")

        # 3) 라인 파싱 (file, label 시퀀스)
        items = []
        for i in use_idx:
            line = lines[i]
            # 예: something,file,label
            parts = line.split(",", 2)
            if len(parts) < 3:
                # 예외: index.csv 형식이 다르면 여기 맞춰 수정
                # 예를 들어 file,label 두 칼럼만 있다면: stem, label = parts[0], parts[1]
                _, file, label = "", parts[0], parts[1]
            else:
                _, file, label = parts
            stem = Path(file).stem
            items.append((stem, label.strip().split()))
        self.items = items

        # 4) vocab 로드 (blank=0 예약)
        with open(self.root / vocab_json, "r", encoding="utf-8-sig") as f:
            vocab = json.load(f)["tokens"]
        self.token2id = {tok: i+1 for i, tok in enumerate(vocab)}  # 0은 blank
        self.blank_id = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, tokens = self.items[idx]
        npz = np.load(self.root / f"{stem}.npz", allow_pickle=True)
        # 키 이름 가드
        if "seq" in npz.files:
            seq = npz["seq"].astype(np.float32)
        elif "x" in npz.files:
            seq = npz["x"].astype(np.float32)
        else:
            # 첫 ndarray 키
            keys = [k for k in npz.files if isinstance(npz[k], np.ndarray)]
            if not keys:
                raise ValueError(f"{stem}.npz: no ndarray keys")
            seq = npz[keys[0]].astype(np.float32)

        target = np.array([self.token2id[t] for t in tokens], dtype=np.int64) if tokens else np.array([], dtype=np.int64)
        return torch.from_numpy(seq), torch.from_numpy(target)

def collate_ctc(batch):
    # batch: List[(seq[T,F], target[L])]
    xs, ys = zip(*batch)
    lens_x = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
    lens_y = torch.tensor([y.shape[0] for y in ys], dtype=torch.int32)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)   # [B, Tmax, F]
    Y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-1)
    return X, lens_x, Y, lens_y
