from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import FeatureSpec, sequence_to_fixed_tensor


class SWLLSEDataset(Dataset):
    def __init__(
        self,
        split_csv: str | Path,
        mediapipe_dir: str | Path,
        feature_spec: FeatureSpec,
        augment: bool = False,
        strict_missing: bool = True,
    ) -> None:
        self.split_csv = Path(split_csv)
        self.mediapipe_dir = Path(mediapipe_dir)
        self.feature_spec = feature_spec
        self.augment = augment
        self.strict_missing = strict_missing

        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV not found: {self.split_csv}")
        if not self.mediapipe_dir.exists():
            raise FileNotFoundError(f"Mediapipe dir not found: {self.mediapipe_dir}")

        df = pd.read_csv(self.split_csv, header=None, names=["FILENAME", "CLASS_ID"])
        df["FILENAME"] = df["FILENAME"].astype(str)
        df["CLASS_ID"] = df["CLASS_ID"].astype(int)
        self.records = df.to_dict("records")

    def __len__(self) -> int:
        return len(self.records)

    def _load_sequence(self, filename: str):
        p = self.mediapipe_dir / f"{filename}.pkl"
        if not p.exists():
            if self.strict_missing:
                raise FileNotFoundError(f"Missing pkl: {p}")
            return []
        try:
            with p.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                return data
            return []
        except Exception:
            if self.strict_missing:
                raise
            return []

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        seq = self._load_sequence(rec["FILENAME"])
        x = sequence_to_fixed_tensor(
            seq,
            t_fixed=self.feature_spec.t_fixed,
            augment=self.augment,
        )
        y = int(rec["CLASS_ID"])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
