from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .io_utils import load_npz


class FrameDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        item = self.samples[idx]
        img = torch.from_numpy(item["image"]).unsqueeze(0)
        meta = {
            "video_name": item["video_name"],
            "category": item["category"],
            "timestamp": item["timestamp"],
        }
        return img, meta


def load_frame_samples(frames_dir: Path) -> List[Dict]:
    samples: List[Dict] = []
    for category in ("tv", "commercials"):
        for npz_path in sorted((frames_dir / category).glob("*_frames.npz")):
            payload = load_npz(npz_path)
            frames = payload["frames"]
            timestamps = payload["timestamps_sec"]
            video_name = payload["video_name"]
            for frame, ts in zip(frames, timestamps):
                samples.append(
                    {
                        "image": frame.astype(np.float32),
                        "timestamp": float(ts),
                        "video_name": str(video_name),
                        "category": category,
                    }
                )
    return samples


def build_dataloaders(
    frames_dir: Path,
    batch_size: int = 128,
    num_workers: int = 0,
    train_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    samples = load_frame_samples(frames_dir)
    random.Random(42).shuffle(samples)
    split_idx = int(len(samples) * train_ratio)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    train_loader = DataLoader(FrameDataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(FrameDataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
