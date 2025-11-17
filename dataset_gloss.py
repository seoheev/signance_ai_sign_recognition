# 말뭉치(문장) 데이터 정의 파일
import json
import csv   # ✅ 추가
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class GlossSeqDataset(Dataset):
    def __init__(
        self,
        root: str = "processed",
        index_csv: str = "index.csv",
        vocab_json: str = "vocab.json",
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.root = Path(root)

        # -----------------------------
        # 1) index.csv 정석 파싱 (csv 모듈)
        #    컬럼: id, file, label 가정
        # -----------------------------
        index_path = self.root / index_csv
        rows = []
        with index_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            raise ValueError(f"{index_path} 에 유효한 행이 없습니다.")

        # (file, 토큰 리스트) 튜플 리스트로 변환
        items = []
        for row in rows:
            file_name = row.get("file") or row.get("video") or row.get("json")
            if not file_name:
                # file 컬럼이 없는 행은 스킵
                continue

            label_str = row.get("label", "")
            tokens = [t for t in label_str.strip().split() if t]

            stem = Path(file_name).stem  # 확장자 제거 (npz와 매칭용)
            items.append((stem, tokens))

        if not items:
            raise ValueError(f"{index_path}: file/label 정보가 없습니다.")

        # -----------------------------
        # 2) train/valid/all 스플릿
        # -----------------------------
        rng = np.random.default_rng(seed)
        idx_all = np.arange(len(items))
        if shuffle:
            rng.shuffle(idx_all)

        if split == "all":
            idx_sel = idx_all
        else:
            cut = int(len(items) * float(split_ratio))
            if split == "train":
                idx_sel = idx_all[:cut]
            elif split == "valid":
                idx_sel = idx_all[cut:]
            else:
                raise ValueError(f"unknown split: {split}")

        self.items = [items[i] for i in idx_sel]

        # -----------------------------
        # 3) vocab 로드 (blank = 0)
        # -----------------------------
        vocab_path = self.root / vocab_json
        with vocab_path.open("r", encoding="utf-8-sig") as f:
            vocab = json.load(f)["tokens"]

        self.token2id = {tok: i + 1 for i, tok in enumerate(vocab)}  # 0은 CTC blank
        self.blank_id = 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, tokens = self.items[idx]

        # npz 로드
        npz_path = self.root / f"{stem}.npz"
        with np.load(npz_path, allow_pickle=True) as z:
            if "seq" in z:
                seq = z["seq"].astype(np.float32)
            elif "x" in z:
                seq = z["x"].astype(np.float32)
            else:
                # 첫 번째 ndarray 키를 사용
                keys = [k for k in z.files if isinstance(z[k], np.ndarray)]
                if not keys:
                    raise ValueError(f"{stem}.npz: no ndarray keys")
                seq = z[keys[0]].astype(np.float32)

        # 토큰 → id 시퀀스로 매핑
        target = (
            np.array([self.token2id[t] for t in tokens], dtype=np.int64)
            if tokens
            else np.array([], dtype=np.int64)
        )

        return torch.from_numpy(seq), torch.from_numpy(target)


def collate_ctc(batch):
    # batch: List[(seq[T,F], target[L])]
    xs, ys = zip(*batch)
    lens_x = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
    lens_y = torch.tensor([y.shape[0] for y in ys], dtype=torch.int32)
    X = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)  # [B, Tmax, F]
    Y = torch.nn.utils.rnn.pad_sequence(
        ys, batch_first=True, padding_value=-1
    )  # [B, Lmax]
    return X, lens_x, Y, lens_y
