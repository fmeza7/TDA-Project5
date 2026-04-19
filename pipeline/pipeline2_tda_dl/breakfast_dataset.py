from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class WindowData:
    X: np.ndarray
    y: np.ndarray
    valid_mask: np.ndarray


def load_window_npz(path: Path) -> WindowData:
    with np.load(path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        if "valid_mask" in data.files:
            valid_mask = data["valid_mask"].astype(np.uint8)
        else:
            valid_mask = np.ones_like(y, dtype=np.uint8)
    return WindowData(X=X, y=y, valid_mask=valid_mask)


class BreakfastWindowDataset(Dataset):
    def __init__(self, npz_path: Path):
        self.path = npz_path
        self.data = load_window_npz(npz_path)

    def __len__(self) -> int:
        return int(self.data.X.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = torch.from_numpy(self.data.X[idx])
        y = torch.from_numpy(self.data.y[idx])
        mask = torch.from_numpy(self.data.valid_mask[idx])
        return {
            "x": x,
            "y": y,
            "valid_mask": mask,
        }


def summarize_windows(npz_path: Path) -> Dict:
    data = load_window_npz(npz_path)
    labels, counts = np.unique(data.y.reshape(-1), return_counts=True)
    return {
        "path": str(npz_path),
        "num_windows": int(data.X.shape[0]),
        "window_size": int(data.X.shape[1]) if data.X.ndim == 3 else 0,
        "feature_dim": int(data.X.shape[2]) if data.X.ndim == 3 else 0,
        "label_histogram": {str(int(k)): int(v) for k, v in zip(labels, counts)},
        "valid_ratio": float(data.valid_mask.mean()) if data.valid_mask.size else 0.0,
    }


def make_dataloader(
    npz_path: Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    dataset = BreakfastWindowDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
