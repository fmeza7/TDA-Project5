from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .io_utils import (
    build_commercial_class_map,
    load_curve_npz,
    load_preproc_manifest,
    parse_gt_file,
)
from .labels import BACKGROUND_NAME, BACKGROUND_ID


@dataclass
class WindowRecord:
    flat: np.ndarray
    seq: np.ndarray
    video_name: str
    source_type: str
    center_time: float
    start_time: float
    end_time: float
    label_name: Optional[str]
    label_id: int


def _interval_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    if union <= 0:
        return 0.0
    return inter / union


def _fps_from_manifest(entry: Dict) -> float:
    fps = float(entry.get("sampled_fps", 0.0))
    if fps <= 0:
        # Fallback: assume timestamps evenly spaced
        fps = float(entry.get("native_fps", 3.0) or 3.0)
    return max(fps, 1e-3)


def _window_times(timestamps: np.ndarray, start: int, end: int) -> Tuple[float, float, float]:
    safe_end = min(end - 1, len(timestamps) - 1)
    start_time = float(timestamps[start])
    end_time = float(timestamps[safe_end])
    center_idx = start + (end - start) // 2
    center_idx = min(center_idx, len(timestamps) - 1)
    center_time = float(timestamps[center_idx])
    return start_time, end_time, center_time


def _label_tv_window(
    tv_name: str,
    window_start: float,
    window_end: float,
    center_time: float,
    gt_by_tv: Dict[str, List[Dict]],
    positive_overlap: float,
    negative_overlap: float,
    include_ambiguous_as_background: bool,
    center_inside_positive: bool,
) -> Tuple[Optional[str], int]:
    gt_entries = gt_by_tv.get(tv_name, [])
    best_overlap = 0.0
    best_entry: Optional[Dict] = None
    for entry in gt_entries:
        if center_inside_positive and entry["start_time"] <= center_time <= entry["end_time"]:
            return entry["commercial"], -2
        overlap = _interval_iou(window_start, window_end, entry["start_time"], entry["end_time"])
        if overlap > best_overlap:
            best_overlap = overlap
            best_entry = entry
    if best_entry and best_overlap >= positive_overlap:
        return best_entry["commercial"], -2  # placeholder; actual ID resolved later
    if best_overlap <= negative_overlap:
        return BACKGROUND_NAME, BACKGROUND_ID
    if include_ambiguous_as_background:
        return BACKGROUND_NAME, BACKGROUND_ID
    return None, -1


def _dataset_summary(records: List[WindowRecord]) -> Dict:
    if not records:
        return {"num_records": 0, "source_types": {}, "videos": {}, "labels": {}, "label_names": {}}
    return {
        "num_records": len(records),
        "source_types": dict(Counter(rec.source_type for rec in records)),
        "videos": dict(Counter(rec.video_name for rec in records)),
        "labels": dict(Counter(rec.label_id for rec in records)),
        "label_names": dict(Counter((rec.label_name or "__ignored__") for rec in records)),
    }


def _serialize_dataset(records: List[WindowRecord], output_path: Path) -> None:
    if not records:
        raise RuntimeError(f"No se generaron ventanas para {output_path}")
    X_flat = np.stack([rec.flat for rec in records]).astype(np.float32)
    X_seq = np.stack([rec.seq for rec in records]).astype(np.float32)
    meta = {
        "video_name": np.array([rec.video_name for rec in records], dtype=np.str_),
        "source_type": np.array([rec.source_type for rec in records], dtype=np.str_),
        "center_time": np.array([rec.center_time for rec in records], dtype=np.float32),
        "start_time": np.array([rec.start_time for rec in records], dtype=np.float32),
        "end_time": np.array([rec.end_time for rec in records], dtype=np.float32),
        "label_id": np.array([rec.label_id for rec in records], dtype=np.int32),
        "label_name": np.array([rec.label_name or "" for rec in records], dtype=np.str_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, X_flat=X_flat, X_seq=X_seq, **meta)


def build_window_records(
    curves_dir: Path,
    preproc_manifest: Dict[str, Dict],
    class_map: Dict[str, int],
    gt_by_tv: Dict[str, List[Dict]],
    window_sec: float,
    stride_frames: int,
    positive_overlap: float,
    negative_overlap: float,
    include_ambiguous_as_background: bool,
    center_inside_positive: bool,
) -> List[WindowRecord]:
    records: List[WindowRecord] = []
    for category in ("commercials", "tv"):
        glob_path = curves_dir / category / "*_curves.npz"
        for npz_path in tqdm(list(glob_path.parent.glob(glob_path.name)), desc=f"{category} windows"):
            payload = load_curve_npz(npz_path)
            stem = npz_path.stem.replace("_curves", "")
            manifest_entry = preproc_manifest.get(stem)
            if manifest_entry is None:
                continue
            fps = _fps_from_manifest(manifest_entry)
            window_frames = max(1, int(round(window_sec * fps)))
            signals = payload.signals
            total_frames = signals.shape[0]
            if total_frames < window_frames:
                continue
            timestamps = payload.timestamps
            label_name_default = stem if category == "commercials" else None
            for start in range(0, total_frames - window_frames + 1, stride_frames):
                end = start + window_frames
                window_seq = signals[start:end]
                flat = window_seq.reshape(-1)
                start_time, end_time, center_time = _window_times(timestamps, start, end)
                if category == "commercials":
                    label_name = label_name_default
                    label_id = class_map.get(label_name, -1)
                else:
                    label_name, label_id = _label_tv_window(
                        stem,
                        start_time,
                        end_time,
                        center_time,
                        gt_by_tv,
                        positive_overlap,
                        negative_overlap,
                        include_ambiguous_as_background,
                        center_inside_positive,
                    )
                    if label_name == BACKGROUND_NAME:
                        label_id = BACKGROUND_ID
                    elif label_name:
                        label_id = class_map.get(label_name, -1)
                source_type = "commercial" if category == "commercials" else "tv"
                records.append(
                    WindowRecord(
                        flat=flat,
                        seq=window_seq,
                        video_name=stem,
                        source_type=source_type,
                        center_time=center_time,
                        start_time=start_time,
                        end_time=end_time,
                        label_name=label_name,
                        label_id=label_id,
                    )
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir ventanas TDA para TopoAE y modelo temporal")
    parser.add_argument("--curves_dir", type=Path, required=True)
    parser.add_argument("--preproc_manifest", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_sec", type=float, default=8.0)
    parser.add_argument("--stride_frames", type=int, default=1)
    parser.add_argument("--positive_overlap", type=float, default=0.5)
    parser.add_argument("--negative_overlap", type=float, default=0.1)
    parser.add_argument("--center_inside_positive", action="store_true")
    parser.add_argument("--include_ambiguous_as_background", action="store_true")
    parser.add_argument("--mode", choices=["topoae", "temporal", "both"], default="both")
    args = parser.parse_args()

    preproc_manifest = load_preproc_manifest(args.preproc_manifest)
    gt_entries = parse_gt_file(args.gt)
    gt_by_tv: Dict[str, List[Dict]] = {}
    for entry in gt_entries:
        gt_by_tv.setdefault(entry["television"], []).append(entry)
    class_map = build_commercial_class_map(args.curves_dir)
    class_map.setdefault(BACKGROUND_NAME, BACKGROUND_ID)

    records = build_window_records(
        args.curves_dir,
        preproc_manifest,
        class_map,
        gt_by_tv,
        args.window_sec,
        args.stride_frames,
        args.positive_overlap,
        args.negative_overlap,
        args.include_ambiguous_as_background,
        args.center_inside_positive,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def save_mode(mode: str, filtered: List[WindowRecord]) -> None:
        dataset_path = output_dir / f"{mode}_dataset.npz"
        _serialize_dataset(filtered, dataset_path)
        meta = {
            "num_windows": len(filtered),
            "mode": mode,
            "window_sec": args.window_sec,
            "stride_frames": args.stride_frames,
        }
        (output_dir / f"{mode}_dataset_meta.json").write_text(json.dumps(meta, indent=2))
        summary = _dataset_summary(filtered)
        summary_path = output_dir / f"{mode}_dataset_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[window_dataset] {mode} summary:", summary)

    if args.mode in ("topoae", "both"):
        save_mode("topoae", records)
    if args.mode in ("temporal", "both"):
        temp_records = [rec for rec in records if rec.label_id >= 0]
        save_mode("temporal", temp_records)


if __name__ == "__main__":
    main()
