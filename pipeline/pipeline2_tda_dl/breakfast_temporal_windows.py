from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class VideoFiles:
    video_id: str
    split: str
    cubical_npz: Path
    curves_npz: Path
    labels_npz: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construir ventanas temporales many-to-many para Breakfast"
    )
    parser.add_argument("--cubical_manifest", type=Path, required=True)
    parser.add_argument("--curves_manifest", type=Path, required=True)
    parser.add_argument("--frame_labels_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_size", type=int, default=31)
    parser.add_argument("--stride_train", type=int, default=5)
    parser.add_argument("--stride_val", type=int, default=5)
    parser.add_argument("--stride_test", type=int, default=1)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--drop_windows_with_unknown", action="store_true")
    parser.add_argument("--strict_alignment", action="store_true")
    parser.add_argument("--strict_missing", action="store_true")
    parser.add_argument("--label_map", type=Path, default=None)
    return parser.parse_args()


def _read_json_list(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest invalido (se esperaba lista): {path}")
    return payload


def _resolve_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _normalize_video_id(value: str) -> str:
    stem = Path(str(value)).stem
    if stem.endswith("_curves"):
        stem = stem[: -len("_curves")]
    if stem.endswith("_labels"):
        stem = stem[: -len("_labels")]
    return stem.strip().lower()


def _selected_splits(splits_csv: str) -> set[str]:
    splits = {x.strip() for x in splits_csv.split(",") if x.strip()}
    return splits or {"train", "val", "test"}


def _load_label_map(args: argparse.Namespace) -> Dict[str, int]:
    if args.label_map is not None:
        label_map_path = args.label_map
    else:
        label_map_path = args.frame_labels_manifest.parent / "label_map.json"

    if not label_map_path.exists():
        return {}

    payload = json.loads(label_map_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k): int(v) for k, v in payload.items()}


def _index_cubical(cubical_manifest: Path) -> Dict[str, Path]:
    rows = _read_json_list(cubical_manifest)
    index: Dict[str, Path] = {}
    for row in rows:
        video_id = str(
            row.get("video_id")
            or row.get("source_path")
            or row.get("output_path")
            or ""
        ).strip()
        output_path = str(row.get("output_path") or "").strip()
        if not video_id or not output_path:
            continue
        index[_normalize_video_id(video_id)] = _resolve_path(
            output_path, cubical_manifest
        )
    return index


def _index_curves(curves_manifest: Path) -> Dict[str, Path]:
    rows = _read_json_list(curves_manifest)
    index: Dict[str, Path] = {}
    for row in rows:
        video_id = str(
            row.get("video_id")
            or row.get("source_path")
            or row.get("output_path")
            or ""
        ).strip()
        output_path = str(row.get("output_path") or "").strip()
        if not video_id or not output_path:
            continue
        index[_normalize_video_id(video_id)] = _resolve_path(
            output_path, curves_manifest
        )
    return index


def _index_labels(frame_labels_manifest: Path) -> Dict[str, Dict]:
    rows = _read_json_list(frame_labels_manifest)
    index: Dict[str, Dict] = {}
    for row in rows:
        video_id = str(row.get("video_id") or row.get("output_path") or "").strip()
        output_path = str(row.get("output_path") or "").strip()
        split = str(row.get("split") or "train").strip()
        if not video_id or not output_path:
            continue
        key = _normalize_video_id(video_id)
        index[key] = {
            "video_id": Path(video_id).stem,
            "split": split,
            "labels_npz": _resolve_path(output_path, frame_labels_manifest),
        }
    return index


def _stride_for_split(split: str, args: argparse.Namespace) -> int:
    if split == "train":
        return max(1, args.stride_train)
    if split == "val":
        return max(1, args.stride_val)
    return max(1, args.stride_test)


def _build_windows_for_video(
    item: VideoFiles,
    args: argparse.Namespace,
    unknown_id: int | None,
) -> Dict[str, np.ndarray] | None:
    with np.load(item.cubical_npz) as data:
        tda_features = data["tda_features"].astype(np.float32)
        timestamps = data["timestamps_sec"].astype(np.float32)

    with np.load(item.curves_npz) as data:
        curve_signals = data["curve_signals"].astype(np.float32)

    with np.load(item.labels_npz) as data:
        frame_label_ids = data["frame_label_ids"].astype(np.int32)
        if "valid_mask" in data.files:
            valid_mask = data["valid_mask"].astype(np.uint8)
        else:
            valid_mask = np.ones((frame_label_ids.shape[0],), dtype=np.uint8)

    lengths = [
        tda_features.shape[0],
        curve_signals.shape[0],
        frame_label_ids.shape[0],
        valid_mask.shape[0],
        timestamps.shape[0],
    ]
    min_len = int(min(lengths)) if lengths else 0
    if min_len <= 0:
        return None

    if args.strict_alignment and any(length != min_len for length in lengths):
        raise RuntimeError(
            f"Longitudes desalineadas en {item.video_id}: tda={lengths[0]} curves={lengths[1]} labels={lengths[2]} mask={lengths[3]} ts={lengths[4]}"
        )

    tda_features = tda_features[:min_len]
    curve_signals = curve_signals[:min_len]
    frame_label_ids = frame_label_ids[:min_len]
    valid_mask = valid_mask[:min_len]
    timestamps = timestamps[:min_len]

    X = np.concatenate([tda_features, curve_signals], axis=1).astype(np.float32)
    y = frame_label_ids.astype(np.int32)
    m = valid_mask.astype(np.uint8)

    win = args.window_size
    stride = _stride_for_split(item.split, args)
    if min_len < win:
        return None

    X_windows: List[np.ndarray] = []
    y_windows: List[np.ndarray] = []
    m_windows: List[np.ndarray] = []
    video_ids: List[str] = []
    splits: List[str] = []
    starts: List[int] = []
    ends: List[int] = []
    center_times: List[float] = []

    for start in range(0, min_len - win + 1, stride):
        end = start + win
        y_slice = y[start:end]
        if (
            args.drop_windows_with_unknown
            and unknown_id is not None
            and np.any(y_slice == unknown_id)
        ):
            continue

        X_windows.append(X[start:end])
        y_windows.append(y_slice)
        m_windows.append(m[start:end])
        video_ids.append(item.video_id)
        splits.append(item.split)
        starts.append(start)
        ends.append(end)
        center_times.append(float(timestamps[start + (win // 2)]))

    if not X_windows:
        return None

    max_len = max(len(x) for x in video_ids)
    return {
        "X": np.stack(X_windows).astype(np.float32),
        "y": np.stack(y_windows).astype(np.int32),
        "valid_mask": np.stack(m_windows).astype(np.uint8),
        "video_id": np.array(video_ids, dtype=f"<U{max(1, max_len)}"),
        "split": np.array(splits, dtype=np.str_),
        "start_idx": np.array(starts, dtype=np.int32),
        "end_idx": np.array(ends, dtype=np.int32),
        "center_time_sec": np.array(center_times, dtype=np.float32),
    }


def _save_split_windows(path: Path, shards: List[Dict[str, np.ndarray]]) -> int:
    if not shards:
        return 0

    merged: Dict[str, np.ndarray] = {}
    keys = shards[0].keys()
    for key in keys:
        merged[key] = np.concatenate([shard[key] for shard in shards], axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **merged)
    return int(merged["X"].shape[0])


def main() -> None:
    args = parse_args()
    selected_splits = _selected_splits(args.splits)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_map = _load_label_map(args)
    unknown_id = label_map.get("__unk__")

    cubical_index = _index_cubical(args.cubical_manifest)
    curves_index = _index_curves(args.curves_manifest)
    labels_index = _index_labels(args.frame_labels_manifest)

    split_shards: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)
    per_video_summary: List[Dict] = []
    missing_refs: List[str] = []
    class_counter = Counter()

    for key, label_meta in labels_index.items():
        split = str(label_meta["split"])
        if split not in selected_splits:
            continue

        cubical_npz = cubical_index.get(key)
        curves_npz = curves_index.get(key)
        labels_npz = Path(label_meta["labels_npz"])
        if cubical_npz is None or curves_npz is None or not labels_npz.exists():
            msg = f"[missing-ref] video={label_meta['video_id']} split={split} cubical={bool(cubical_npz)} curves={bool(curves_npz)} labels_exists={labels_npz.exists()}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            missing_refs.append(msg)
            continue

        item = VideoFiles(
            video_id=str(label_meta["video_id"]),
            split=split,
            cubical_npz=Path(cubical_npz),
            curves_npz=Path(curves_npz),
            labels_npz=labels_npz,
        )
        shard = _build_windows_for_video(item, args, unknown_id)
        if shard is None:
            per_video_summary.append(
                {
                    "video_id": item.video_id,
                    "split": split,
                    "num_windows": 0,
                }
            )
            continue

        split_shards[split].append(shard)
        per_video_summary.append(
            {
                "video_id": item.video_id,
                "split": split,
                "num_windows": int(shard["X"].shape[0]),
            }
        )
        class_counter.update(shard["y"].reshape(-1).tolist())

    split_counts: Dict[str, int] = {}
    for split in sorted(selected_splits):
        out_path = output_dir / f"{split}_windows.npz"
        split_counts[split] = _save_split_windows(out_path, split_shards.get(split, []))

    windows_manifest = {
        "window_size": args.window_size,
        "strides": {
            "train": args.stride_train,
            "val": args.stride_val,
            "test": args.stride_test,
        },
        "drop_windows_with_unknown": args.drop_windows_with_unknown,
        "selected_splits": sorted(selected_splits),
        "num_videos": len(per_video_summary),
        "split_window_counts": split_counts,
        "label_histogram_ids": {
            str(k): int(v) for k, v in sorted(class_counter.items(), key=lambda x: x[0])
        },
        "missing_references": missing_refs,
        "per_video": per_video_summary,
    }

    manifest_path = output_dir / "windows_manifest.json"
    manifest_path.write_text(
        json.dumps(windows_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[breakfast_temporal_windows] manifest={manifest_path}")
    print(f"[breakfast_temporal_windows] split_window_counts={split_counts}")
    if missing_refs:
        print(f"[breakfast_temporal_windows] missing_references={len(missing_refs)}")


if __name__ == "__main__":
    main()
