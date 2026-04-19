from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class SequenceRecord:
    seq: np.ndarray
    label_id: np.ndarray
    label_name: List[str]
    video_name: str
    center_time: float
    start_time: float
    end_time: float
    source_type: str

class TemporalSequenceDataset(Dataset):
    def __init__(self, records: List[SequenceRecord]): self.records = records
    def __len__(self) -> int: return len(self.records)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]
        return torch.from_numpy(rec.seq.astype(np.float32)), torch.from_numpy(rec.label_id.astype(np.int64))

def _load_latent_file(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "center_times" in data:
        order = np.argsort(data["center_times"])
        payload = {k: data[k][order] for k in data.files if k not in {"video_name", "source_type"}}
    else:
        payload = {k: data[k] for k in data.files if k not in {"video_name", "source_type"}}
    payload["video_name"] = str(data["video_name"] if np.ndim(data["video_name"]) == 0 else data["video_name"][0])
    return payload

def build_sequence_records(latents_dir: Path, seq_len: int, min_label_id: int = 0) -> List[SequenceRecord]:
    records: List[SequenceRecord] = []
    # Busca recursivamente sin importar las carpetas intermedias
    for path in sorted(latents_dir.rglob("*_latents.npz")):
        payload = _load_latent_file(path)
        z, label_ids, label_names = payload["z_latent"], payload["label_id"], payload["label_name"]
        video_name = payload["video_name"]
        
        # Generar timestamps ficticios si no están (para compatibilidad con scripts antiguos)
        center_times = payload.get("center_times", np.zeros(len(z)))
        start_times = payload.get("start_times", np.zeros(len(z)))
        end_times = payload.get("end_times", np.zeros(len(z)))

        if z.shape[0] < seq_len: continue

        for start in range(0, z.shape[0] - seq_len + 1):
            end_idx = start + seq_len
            mid = start + seq_len // 2
            
            # Filtro opcional: ignorar ventanas donde el centro es menor a min_label_id (ej. ignorar puro SIL)
            if int(label_ids[mid]) < min_label_id: continue

            records.append(SequenceRecord(
                seq=z[start:end_idx],
                label_id=label_ids[start:end_idx],
                label_name=[str(n) for n in label_names[start:end_idx]],
                video_name=video_name,
                center_time=float(center_times[mid]),
                start_time=float(start_times[mid]),
                end_time=float(end_times[mid]),
                source_type=video_name.split("_")[0] # Guarda PXX como source
            ))
    return records

def split_by_videos(records: List[SequenceRecord], train_prefixes: Sequence[str], val_prefixes: Sequence[str]) -> Tuple[List[SequenceRecord], List[SequenceRecord]]:
    if not records: raise RuntimeError("No hay records para hacer split")
    
    train_records, val_records = [], []
    for rec in records:
        # Se evalúa si el nombre del video empieza con alguno de los prefijos dados (ej. P03, P04...)
        if any(rec.video_name.startswith(p) for p in train_prefixes):
            train_records.append(rec)
        elif any(rec.video_name.startswith(p) for p in val_prefixes):
            val_records.append(rec)

    return train_records, val_records

def summarize_records(records: List[SequenceRecord]) -> Dict:
    by_video = Counter(rec.video_name for rec in records)
    all_labels = np.concatenate([rec.label_id for rec in records]) if records else []
    return {
        "num_records": len(records),
        "videos": dict(sorted(by_video.items())),
        "labels": dict(sorted(Counter(all_labels).items())),
    }