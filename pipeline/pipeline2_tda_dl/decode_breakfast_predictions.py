from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decodificar predicciones temporales Breakfast"
    )
    parser.add_argument("--raw_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--min_segment_sec", type=float, default=0.5)
    return parser.parse_args()


def _load_label_map(raw_manifest: Path, label_map_path: Path | None) -> Dict[str, int]:
    candidates: List[Path] = []
    if label_map_path is not None:
        candidates.append(label_map_path)
    candidates.append(raw_manifest.parent / "label_map.json")

    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): int(v) for k, v in payload.items()}
    return {}


def _mode_filter(labels: np.ndarray, kernel_size: int) -> np.ndarray:
    if labels.size == 0:
        return labels
    k = int(max(1, kernel_size))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return labels.copy()

    radius = k // 2
    out = labels.copy()
    for i in range(labels.shape[0]):
        s = max(0, i - radius)
        e = min(labels.shape[0], i + radius + 1)
        window = labels[s:e]
        values, counts = np.unique(window, return_counts=True)
        max_count = np.max(counts)
        best = values[counts == max_count]
        if best.shape[0] == 1:
            out[i] = best[0]
        else:
            center = labels[i]
            out[i] = center if center in set(best.tolist()) else best[0]
    return out


def _runs(labels: np.ndarray) -> List[tuple[int, int, int]]:
    if labels.size == 0:
        return []
    runs: List[tuple[int, int, int]] = []
    start = 0
    cur = int(labels[0])
    for i in range(1, labels.shape[0]):
        if int(labels[i]) != cur:
            runs.append((start, i, cur))
            start = i
            cur = int(labels[i])
    runs.append((start, labels.shape[0], cur))
    return runs


def _merge_short_segments(labels: np.ndarray, min_frames: int) -> np.ndarray:
    if labels.size == 0 or min_frames <= 1:
        return labels.copy()

    out = labels.copy()
    changed = True
    while changed:
        changed = False
        runs = _runs(out)
        for idx, (s, e, label) in enumerate(runs):
            length = e - s
            if length >= min_frames:
                continue

            left = runs[idx - 1] if idx > 0 else None
            right = runs[idx + 1] if idx + 1 < len(runs) else None
            if left is None and right is None:
                continue
            if left is None:
                target = right[2]
            elif right is None:
                target = left[2]
            else:
                left_len = left[1] - left[0]
                right_len = right[1] - right[0]
                target = left[2] if left_len >= right_len else right[2]

            if int(target) != int(label):
                out[s:e] = int(target)
                changed = True
        # repeat until convergence
    return out


def _labels_to_segments(
    label_ids: np.ndarray, timestamps_sec: np.ndarray, id_to_name: Dict[int, str]
) -> List[Dict]:
    segments: List[Dict] = []
    if label_ids.size == 0:
        return segments

    runs = _runs(label_ids)
    for s, e, label in runs:
        start_sec = float(timestamps_sec[s])
        end_idx = min(max(0, e - 1), timestamps_sec.shape[0] - 1)
        if e < timestamps_sec.shape[0]:
            end_sec = float(timestamps_sec[e])
        else:
            delta = (
                float(np.median(np.diff(timestamps_sec)))
                if timestamps_sec.shape[0] > 1
                else 0.0
            )
            end_sec = float(timestamps_sec[end_idx] + max(0.0, delta))
        segments.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "label_id": int(label),
                "label": id_to_name.get(int(label), f"class_{int(label)}"),
            }
        )
    return segments


def main() -> None:
    args = parse_args()
    raw_rows = json.loads(args.raw_manifest.read_text(encoding="utf-8"))
    if not isinstance(raw_rows, list):
        raise ValueError("raw_manifest invalido: se esperaba lista")

    output_dir = args.output_dir.resolve()
    decoded_dir = output_dir / "decoded_predictions"
    decoded_dir.mkdir(parents=True, exist_ok=True)

    label_map = _load_label_map(args.raw_manifest, args.label_map)
    id_to_name = {idx: name for name, idx in label_map.items()} if label_map else {}

    manifest_rows: List[Dict] = []
    for row in raw_rows:
        video_id = str(row.get("video_id") or "").strip()
        output_path = str(row.get("output_path") or "").strip()
        if not video_id or not output_path:
            continue

        pred_path = Path(output_path)
        if not pred_path.is_absolute():
            pred_path = (args.raw_manifest.parent / pred_path).resolve()
        if not pred_path.exists():
            continue

        with np.load(pred_path, allow_pickle=True) as data:
            timestamps_sec = data["timestamps_sec"].astype(np.float32)
            pred_label_ids = data["pred_label_ids"].astype(np.int32)

        smoothed = _mode_filter(pred_label_ids, args.kernel_size)
        if timestamps_sec.shape[0] > 1:
            dt = float(np.median(np.diff(timestamps_sec)))
        else:
            dt = 0.0
        min_frames = (
            int(max(1, round(args.min_segment_sec / max(dt, 1e-6)))) if dt > 0 else 1
        )
        decoded_ids = _merge_short_segments(smoothed, min_frames)
        decoded_labels = np.array(
            [id_to_name.get(int(i), f"class_{int(i)}") for i in decoded_ids],
            dtype=np.str_,
        )
        segments = _labels_to_segments(decoded_ids, timestamps_sec, id_to_name)

        out_npz = decoded_dir / f"{video_id}_decoded.npz"
        np.savez_compressed(
            out_npz,
            video_id=np.array([video_id], dtype=np.str_),
            timestamps_sec=timestamps_sec,
            decoded_label_ids=decoded_ids.astype(np.int32),
            decoded_labels=decoded_labels,
            raw_label_ids=pred_label_ids.astype(np.int32),
        )

        out_segments = decoded_dir / f"{video_id}_segments.json"
        out_segments.write_text(
            json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        manifest_rows.append(
            {
                "video_id": video_id,
                "output_path": str(out_npz),
                "segments_path": str(out_segments),
                "num_frames": int(decoded_ids.shape[0]),
                "num_segments": int(len(segments)),
            }
        )

    manifest_path = output_dir / "decoded_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "num_videos": len(manifest_rows),
        "kernel_size": int(args.kernel_size),
        "min_segment_sec": float(args.min_segment_sec),
    }
    summary_path = output_dir / "decoded_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[decode_breakfast_predictions] videos={len(manifest_rows)}")
    print(f"[decode_breakfast_predictions] manifest={manifest_path}")


if __name__ == "__main__":
    main()
