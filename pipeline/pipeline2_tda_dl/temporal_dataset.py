from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import BACKGROUND_ID


@dataclass
class SequenceRecord:
    seq: np.ndarray
    label_id: int
    label_name: str
    video_name: str
    center_time: float
    start_time: float
    end_time: float


class TemporalSequenceDataset(Dataset):
    def __init__(self, records: List[SequenceRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        seq = torch.from_numpy(rec.seq.astype(np.float32))
        label = torch.tensor(rec.label_id, dtype=torch.long)
        return seq, label

    def video_names(self) -> np.ndarray:
        return np.array([rec.video_name for rec in self.records])


def _load_latent_file(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    order = np.argsort(data["center_times"])
    payload = {k: data[k][order] for k in data.files if k not in {"video_name", "source_type"}}
    payload["video_name"] = str(data["video_name"]) if np.ndim(data["video_name"]) == 0 else str(data["video_name"][0])
    payload["source_type"] = str(data["source_type"]) if np.ndim(data["source_type"]) == 0 else str(data["source_type"][0])
    return payload


def build_sequence_records(
    latents_dir: Path,
    seq_len: int,
    min_label_id: int = 0,
) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    for path in sorted(latents_dir.glob("tv/*_latents.npz")):
        payload = _load_latent_file(path)
        z = payload["z_latent"]
        label_ids = payload["label_id"]
        label_names = payload["label_name"]
        center_times = payload["center_times"]
        start_times = payload["start_times"]
        end_times = payload["end_times"]
        video_name = payload["video_name"]
        if z.shape[0] < seq_len:
            continue
        for start in range(0, z.shape[0] - seq_len + 1):
            end_idx = start + seq_len
            mid = start + seq_len // 2
            label_id = int(label_ids[mid])
            if label_id < min_label_id:
                continue
            label_name = str(label_names[mid])
            records.append(
                SequenceRecord(
                    seq=z[start:end_idx],
                    label_id=label_id,
                    label_name=label_name,
                    video_name=video_name,
                    center_time=float(center_times[mid]),
                    start_time=float(start_times[mid]),
                    end_time=float(end_times[mid]),
                )
            )
    return records


def split_by_videos(
    records: List[SequenceRecord],
    train_videos: Sequence[str],
    val_videos: Sequence[str],
) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
    train_set = set(train_videos)
    val_set = set(val_videos)
    if not train_set:
        train_set = {rec.video_name for rec in records if rec.label_id != BACKGROUND_ID}
    if not val_set:
        val_candidates = [rec.video_name for rec in records if rec.video_name not in train_set]
        if val_candidates:
            val_set = {val_candidates[0]}
    train_records = [rec for rec in records if rec.video_name in train_set]
    val_records = [rec for rec in records if rec.video_name in val_set]
    return train_records, val_records
