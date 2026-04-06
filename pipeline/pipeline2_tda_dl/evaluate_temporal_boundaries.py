from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .bbc_annotations import load_bbc_boundaries
from .io_utils import load_preproc_manifest


def _load_predictions(path: Path) -> Dict[str, List[float]]:
    by_video: Dict[str, List[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        video = parts[0]
        time_sec = float(parts[1])
        by_video.setdefault(video, []).append(time_sec)
    for video in by_video:
        by_video[video] = sorted(by_video[video])
    return by_video


def _build_gt_times(
    boundaries_by_video: Dict[str, np.ndarray],
    preproc_manifest: Dict[str, Dict],
) -> Dict[str, np.ndarray]:
    gt: Dict[str, np.ndarray] = {}
    for video_name, frames in boundaries_by_video.items():
        entry = preproc_manifest.get(video_name)
        if entry is None:
            continue
        native_fps = float(entry.get("native_fps", 25.0) or 25.0)
        native_fps = max(native_fps, 1e-6)
        gt[video_name] = frames.astype(np.float32) / native_fps
    return gt


def _match_events(pred: Sequence[float], gt: Sequence[float], tolerance_sec: float) -> Tuple[int, int, int, List[float]]:
    if not pred and not gt:
        return 0, 0, 0, []
    if not gt:
        return 0, len(pred), 0, []
    if not pred:
        return 0, 0, len(gt), []

    used_gt = [False] * len(gt)
    tp = 0
    dists: List[float] = []
    for p in pred:
        best_idx = -1
        best_dist = float("inf")
        for idx, g in enumerate(gt):
            if used_gt[idx]:
                continue
            dist = abs(p - g)
            if dist <= tolerance_sec and dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx >= 0:
            used_gt[best_idx] = True
            tp += 1
            dists.append(best_dist)
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, dists


def _metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _target_videos(all_videos: Sequence[str], target_videos: Sequence[str]) -> List[str]:
    if not target_videos:
        return sorted(all_videos)
    return sorted(set(target_videos))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluar detección de transiciones temporales BBC")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--annotations_dir", type=Path, required=True)
    parser.add_argument("--preproc_manifest", type=Path, required=True)
    parser.add_argument("--tolerance_sec", type=float, default=0.5)
    parser.add_argument("--target_videos", nargs="*", default=[])
    parser.add_argument("--output_json", type=Path, default=None)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions)
    boundaries_by_video = load_bbc_boundaries(args.videos_dir, args.annotations_dir)
    preproc_manifest = load_preproc_manifest(args.preproc_manifest)
    gt = _build_gt_times(boundaries_by_video, preproc_manifest)

    videos = _target_videos(sorted(gt.keys()), args.target_videos)
    per_video: Dict[str, Dict] = {}
    tp_total = 0
    fp_total = 0
    fn_total = 0
    all_dists: List[float] = []

    for video in videos:
        pred_times = predictions.get(video, [])
        gt_times = gt.get(video, np.zeros((0,), dtype=np.float32)).tolist()
        tp, fp, fn, dists = _match_events(pred_times, gt_times, args.tolerance_sec)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        all_dists.extend(dists)
        metrics = _metrics(tp, fp, fn)
        per_video[video] = {
            "num_pred": len(pred_times),
            "num_gt": len(gt_times),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            **metrics,
            "matched_mae_sec": float(np.mean(dists)) if dists else None,
        }

    global_metrics = _metrics(tp_total, fp_total, fn_total)
    report = {
        "tolerance_sec": args.tolerance_sec,
        "videos": videos,
        "global": {
            "tp": tp_total,
            "fp": fp_total,
            "fn": fn_total,
            **global_metrics,
            "matched_mae_sec": float(np.mean(all_dists)) if all_dists else None,
        },
        "per_video": per_video,
    }

    print("[evaluate_boundaries] global:", report["global"])
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2))
        print(f"[evaluate_boundaries] report -> {args.output_json}")


if __name__ == "__main__":
    main()
