from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import BACKGROUND_ID


def _norm_name(x) -> str:
    if isinstance(x, bytes):
        x = x.decode("utf-8")
    x = str(x)
    stem = Path(x).stem
    stem = stem.replace("_curves", "").replace("_latents", "")
    return stem


@dataclass
class SequenceRecord:
    seq: np.ndarray
    label_id: int
    label_name: str
    video_name: str
    center_time: float
    start_time: float
    end_time: float
    source_type: str


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
    video_name_raw = data["video_name"] if np.ndim(data["video_name"]) == 0 else data["video_name"][0]
    source_type_raw = data["source_type"] if np.ndim(data["source_type"]) == 0 else data["source_type"][0]
    payload["video_name"] = _norm_name(video_name_raw)
    payload["source_type"] = _norm_name(source_type_raw)
    return payload


def _pad_sequence(z: np.ndarray, target_len: int) -> np.ndarray:
    if z.shape[0] >= target_len:
        return z[:target_len]
    pad_count = target_len - z.shape[0]
    pad_block = np.repeat(z[-1:, :], pad_count, axis=0)
    return np.concatenate([z, pad_block], axis=0)


def build_sequence_records(
    latents_dir: Path,
    seq_len: int,
    min_label_id: int = 0,
    include_commercials: bool = True,
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
                    source_type="tv",
                )
            )
    if include_commercials:
        for path in sorted(latents_dir.glob("commercials/*_latents.npz")):
            payload = _load_latent_file(path)
            z = payload["z_latent"]
            if z.size == 0:
                continue
            label_ids = payload["label_id"]
            label_names = payload["label_name"]
            center_times = payload["center_times"]
            start_times = payload["start_times"]
            end_times = payload["end_times"]
            video_name = payload["video_name"]
            label_id = int(label_ids[0]) if label_ids.size else BACKGROUND_ID
            if label_id < min_label_id:
                continue
            label_name = str(label_names[0]) if label_names.size else ""
            if z.shape[0] < seq_len:
                seq = _pad_sequence(z, seq_len)
                if len(center_times):
                    idx = min(len(center_times) - 1, max(0, len(center_times) // 2))
                    center_time = float(center_times[idx])
                else:
                    center_time = 0.0
                start_time = float(start_times[0]) if len(start_times) else 0.0
                end_time = float(end_times[-1]) if len(end_times) else start_time
                records.append(
                    SequenceRecord(
                        seq=seq,
                        label_id=label_id,
                        label_name=label_name,
                        video_name=video_name,
                        center_time=center_time,
                        start_time=start_time,
                        end_time=end_time,
                        source_type="commercials",
                    )
                )
                continue
            for start in range(0, z.shape[0] - seq_len + 1):
                end_idx = start + seq_len
                mid = start + seq_len // 2
                records.append(
                    SequenceRecord(
                        seq=z[start:end_idx],
                        label_id=label_id,
                        label_name=label_name,
                        video_name=video_name,
                        center_time=float(center_times[mid]),
                        start_time=float(start_times[mid]),
                        end_time=float(end_times[mid]),
                        source_type="commercials",
                    )
                )
    return records


def split_by_videos(
    records: List[SequenceRecord],
    train_videos: Sequence[str],
    val_videos: Sequence[str],
) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
    if not records:
        raise RuntimeError("No hay records para hacer split")

    all_videos = sorted({_norm_name(rec.video_name) for rec in records})

    train_set = {_norm_name(x) for x in train_videos} if train_videos else set()
    val_set = {_norm_name(x) for x in val_videos} if val_videos else set()

    if not train_set and not val_set:
        if len(all_videos) < 2:
            raise RuntimeError(f"No hay suficientes videos para split automático. all_videos={all_videos}")
        val_set = {all_videos[-1]}
        train_set = set(all_videos[:-1])
    elif train_set and not val_set:
        remaining = [v for v in all_videos if v not in train_set]
        if not remaining:
            raise RuntimeError(
                f"No quedan videos para validación. all_videos={all_videos}, train_set={sorted(train_set)}"
            )
        val_set = {remaining[0]}
    elif val_set and not train_set:
        remaining = [v for v in all_videos if v not in val_set]
        if not remaining:
            raise RuntimeError(
                f"No quedan videos para entrenamiento. all_videos={all_videos}, val_set={sorted(val_set)}"
            )
        train_set = set(remaining)

    commercial_records = [rec for rec in records if rec.source_type == "commercials"]
    tv_records = [rec for rec in records if rec.source_type != "commercials"]

    train_records = commercial_records + [rec for rec in tv_records if _norm_name(rec.video_name) in train_set]
    val_records = [rec for rec in tv_records if _norm_name(rec.video_name) in val_set]

    if not train_records or not val_records:
        raise RuntimeError(
            "Split vacío después de filtrar records. "
            f"all_videos={all_videos}, train_set={sorted(train_set)}, val_set={sorted(val_set)}, "
            f"n_train={len(train_records)}, n_val={len(val_records)}"
        )

    return train_records, val_records


def summarize_records(records: List[SequenceRecord]) -> Dict:
    by_video = Counter(rec.video_name for rec in records)
    by_label = Counter(rec.label_id for rec in records)
    by_source = Counter(rec.source_type for rec in records)
    return {
        "num_records": len(records),
        "videos": dict(sorted(by_video.items())),
        "labels": dict(sorted(by_label.items())),
        "source_types": dict(sorted(by_source.items())),
    }
