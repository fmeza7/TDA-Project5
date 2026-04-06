from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .bbc_annotations import load_bbc_boundaries
from .io_utils import load_curve_npz, load_preproc_manifest
from .labels import BACKGROUND_ID, BACKGROUND_NAME

TRANSITION_ID = 1
TRANSITION_NAME = "__transition__"


@dataclass
class WindowRecord:
    flat: np.ndarray
    seq: np.ndarray
    video_name: str
    source_type: str
    center_time: float
    start_time: float
    end_time: float
    center_frame: int
    label_name: str
    label_id: int


def _fps_from_manifest(entry: Dict) -> float:
    fps = float(entry.get("sampled_fps", 0.0))
    if fps <= 0:
        fps = float(entry.get("native_fps", 3.0) or 3.0)
    return max(fps, 1e-3)


def _native_fps_from_manifest(entry: Dict) -> float:
    fps = float(entry.get("native_fps", 0.0))
    if fps <= 0:
        fps = float(entry.get("sampled_fps", 25.0) or 25.0)
    return max(fps, 1e-3)


def _window_times(timestamps: np.ndarray, start: int, end: int) -> Tuple[float, float, float, int]:
    safe_end = min(end - 1, len(timestamps) - 1)
    start_time = float(timestamps[start])
    end_time = float(timestamps[safe_end])
    center_idx = start + (end - start) // 2
    center_idx = min(center_idx, len(timestamps) - 1)
    center_time = float(timestamps[center_idx])
    return start_time, end_time, center_time, center_idx


def _label_transition(center_time: float, boundary_times: np.ndarray, tolerance_sec: float) -> Tuple[str, int]:
    if boundary_times.size == 0:
        return BACKGROUND_NAME, BACKGROUND_ID
    idx = int(np.argmin(np.abs(boundary_times - center_time)))
    min_dist = float(abs(boundary_times[idx] - center_time))
    if min_dist <= tolerance_sec:
        return TRANSITION_NAME, TRANSITION_ID
    return BACKGROUND_NAME, BACKGROUND_ID


def _serialize_dataset(records: List[WindowRecord], output_path: Path) -> None:
    if not records:
        raise RuntimeError(f"No se generaron ventanas para {output_path}")
    X_flat = np.stack([rec.flat for rec in records]).astype(np.float32)
    X_seq = np.stack([rec.seq for rec in records]).astype(np.float32)
    payload = {
        "X_flat": X_flat,
        "X_seq": X_seq,
        "video_name": np.array([rec.video_name for rec in records], dtype=np.str_),
        "source_type": np.array([rec.source_type for rec in records], dtype=np.str_),
        "center_time": np.array([rec.center_time for rec in records], dtype=np.float32),
        "start_time": np.array([rec.start_time for rec in records], dtype=np.float32),
        "end_time": np.array([rec.end_time for rec in records], dtype=np.float32),
        "center_frame": np.array([rec.center_frame for rec in records], dtype=np.int32),
        "label_id": np.array([rec.label_id for rec in records], dtype=np.int32),
        "label_name": np.array([rec.label_name for rec in records], dtype=np.str_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def _summary(records: List[WindowRecord]) -> Dict:
    if not records:
        return {
            "num_records": 0,
            "videos": {},
            "labels": {},
            "label_names": {},
            "positive_ratio": 0.0,
        }
    labels = Counter(rec.label_id for rec in records)
    names = Counter(rec.label_name for rec in records)
    positives = labels.get(TRANSITION_ID, 0)
    total = len(records)
    return {
        "num_records": total,
        "videos": dict(Counter(rec.video_name for rec in records)),
        "labels": dict(labels),
        "label_names": dict(names),
        "positive_ratio": positives / max(1, total),
    }


def _auto_split(videos: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    ordered = sorted(videos)
    if len(ordered) < 5:
        raise RuntimeError(f"Se requieren al menos 5 episodios para split train/val/test. videos={ordered}")
    test = ordered[-2:]
    val = ordered[-4:-2]
    train = ordered[:-4]
    return train, val, test


def _validate_split(
    all_videos: Sequence[str],
    train_videos: Sequence[str],
    val_videos: Sequence[str],
    test_videos: Sequence[str],
) -> Tuple[List[str], List[str], List[str]]:
    all_set = set(all_videos)
    train = sorted(set(train_videos))
    val = sorted(set(val_videos))
    test = sorted(set(test_videos))
    if not train and not val and not test:
        return _auto_split(sorted(all_set))

    if not train or not val or not test:
        raise RuntimeError("Debe entregar --train_videos, --val_videos y --test_videos completos o ninguno")

    split_union = set(train) | set(val) | set(test)
    if split_union != all_set:
        missing = sorted(all_set - split_union)
        extras = sorted(split_union - all_set)
        raise RuntimeError(f"Split inválido. missing={missing}, extras={extras}")
    if set(train) & set(val) or set(train) & set(test) or set(val) & set(test):
        raise RuntimeError("Split inválido: hay videos repetidos entre train/val/test")
    return train, val, test


def build_window_records(
    curves_dir: Path,
    preproc_manifest: Dict[str, Dict],
    boundaries_by_video: Dict[str, np.ndarray],
    window_sec: float,
    stride_frames: int,
    boundary_tolerance_sec: float,
) -> List[WindowRecord]:
    records: List[WindowRecord] = []
    for npz_path in sorted((curves_dir / "tv").glob("*_curves.npz")):
        payload = load_curve_npz(npz_path)
        video_name = npz_path.stem.replace("_curves", "")
        manifest_entry = preproc_manifest.get(video_name)
        if manifest_entry is None:
            print(f"[warn] {video_name}: no existe en manifest de preprocesamiento, se omite")
            continue

        sampled_fps = _fps_from_manifest(manifest_entry)
        native_fps = _native_fps_from_manifest(manifest_entry)
        boundary_frames = boundaries_by_video.get(video_name, np.zeros((0,), dtype=np.int32))
        boundary_times = boundary_frames.astype(np.float32) / native_fps

        window_frames = max(1, int(round(window_sec * sampled_fps)))
        signals = payload.signals
        timestamps = payload.timestamps
        total_frames = signals.shape[0]
        if total_frames < window_frames:
            continue

        for start in range(0, total_frames - window_frames + 1, stride_frames):
            end = start + window_frames
            seq = signals[start:end]
            flat = seq.reshape(-1)
            start_t, end_t, center_t, center_idx = _window_times(timestamps, start, end)
            center_frame_native = int(round(center_t * native_fps))
            label_name, label_id = _label_transition(center_t, boundary_times, boundary_tolerance_sec)
            records.append(
                WindowRecord(
                    flat=flat,
                    seq=seq,
                    video_name=video_name,
                    source_type="tv",
                    center_time=center_t,
                    start_time=start_t,
                    end_time=end_t,
                    center_frame=center_frame_native,
                    label_name=label_name,
                    label_id=label_id,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir datasets de transición de tomas para BBC Planet Earth")
    parser.add_argument("--curves_dir", type=Path, required=True)
    parser.add_argument("--preproc_manifest", type=Path, required=True)
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--annotations_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_sec", type=float, default=2.0)
    parser.add_argument("--stride_frames", type=int, default=1)
    parser.add_argument("--boundary_tolerance_sec", type=float, default=0.5)
    parser.add_argument("--train_videos", nargs="*", default=[])
    parser.add_argument("--val_videos", nargs="*", default=[])
    parser.add_argument("--test_videos", nargs="*", default=[])
    args = parser.parse_args()

    preproc_manifest = load_preproc_manifest(args.preproc_manifest)
    boundaries_by_video = load_bbc_boundaries(args.videos_dir, args.annotations_dir)
    records = build_window_records(
        curves_dir=args.curves_dir,
        preproc_manifest=preproc_manifest,
        boundaries_by_video=boundaries_by_video,
        window_sec=args.window_sec,
        stride_frames=args.stride_frames,
        boundary_tolerance_sec=args.boundary_tolerance_sec,
    )
    if not records:
        raise RuntimeError("No se generaron ventanas BBC")

    all_videos = sorted({rec.video_name for rec in records})
    train_videos, val_videos, test_videos = _validate_split(
        all_videos=all_videos,
        train_videos=args.train_videos,
        val_videos=args.val_videos,
        test_videos=args.test_videos,
    )
    train_set, val_set, test_set = set(train_videos), set(val_videos), set(test_videos)
    train_records = [r for r in records if r.video_name in train_set]
    val_records = [r for r in records if r.video_name in val_set]
    test_records = [r for r in records if r.video_name in test_set]
    trainval_records = [r for r in records if r.video_name in (train_set | val_set)]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _serialize_dataset(trainval_records, output_dir / "topoae_dataset.npz")
    _serialize_dataset(records, output_dir / "temporal_dataset.npz")
    _serialize_dataset(trainval_records, output_dir / "temporal_dataset_trainval.npz")
    _serialize_dataset(test_records, output_dir / "temporal_dataset_test.npz")

    split = {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "test_videos": test_videos,
    }
    (output_dir / "bbc_split.json").write_text(json.dumps(split, indent=2))

    summary = {
        "all": _summary(records),
        "train": _summary(train_records),
        "val": _summary(val_records),
        "test": _summary(test_records),
    }
    (output_dir / "bbc_dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    meta = {
        "window_sec": args.window_sec,
        "stride_frames": args.stride_frames,
        "boundary_tolerance_sec": args.boundary_tolerance_sec,
        "label_map": {
            BACKGROUND_NAME: BACKGROUND_ID,
            TRANSITION_NAME: TRANSITION_ID,
        },
    }
    (output_dir / "bbc_dataset_meta.json").write_text(json.dumps(meta, indent=2))

    print("[bbc_transition_dataset] summary all:", summary["all"])
    print("[bbc_transition_dataset] summary train:", summary["train"])
    print("[bbc_transition_dataset] summary val:", summary["val"])
    print("[bbc_transition_dataset] summary test:", summary["test"])


if __name__ == "__main__":
    main()
