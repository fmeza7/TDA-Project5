from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .io_utils import ensure_dir, load_npz
from .postprocess import Detection, filter_short, merge_by_gap


def load_curve_map(curve_dir: Path) -> Dict[str, Path]:
    mapping = {}
    for category in ("tv", "commercials"):
        for npz_path in (curve_dir / category).glob("*_curves.npz"):
            payload = np.load(npz_path, allow_pickle=True)
            video_name = str(payload["video_name"]) if payload["video_name"].size == 1 else str(payload["video_name"][0])
            mapping[video_name] = npz_path
    return mapping


def load_durations(manifest_path: Path) -> Dict[str, float]:
    manifest = json.loads(manifest_path.read_text())
    durations = {}
    for entry in manifest:
        name = Path(entry.get("source_path", "")).stem or entry.get("video_name")
        if name:
            durations[name] = float(entry.get("duration_sec", 30.0))
    return durations


def detect_for_video(payload, curve_data, args, durations):
    curve_labels = curve_data["curve_labels"]
    try:
        z_idx = list(curve_labels).index("combined_activity_z")
    except ValueError:
        raise RuntimeError("combined_activity_z no encontrado en curvas")
    z_scores = curve_data["curve_signals"][:, z_idx]
    timestamps = payload["timestamps_sec"]
    winners = []
    for i in range(min(len(z_scores), payload["neighbor_scores"].shape[0])):
        if z_scores[i] < args.curve_threshold:
            continue
        scores = payload["neighbor_scores"][i]
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        if best_score < args.score_threshold:
            continue
        commercial = str(payload["commercial_video_names"][i][best_idx])
        ts = float(timestamps[i])
        winners.append((ts, commercial, best_score))
    detections: List[Detection] = []
    if not winners:
        return detections
    winners.sort(key=lambda x: x[0])
    current = [winners[0]]
    for win in winners[1:]:
        last = current[-1]
        if win[1] == last[1] and win[0] - last[0] <= args.merge_gap_sec:
            current.append(win)
        else:
            detections.append(build_detection(payload["video_name"], current, durations, args))
            current = [win]
    if current:
        detections.append(build_detection(payload["video_name"], current, durations, args))
    detections = merge_by_gap(sorted(detections, key=lambda d: d.start_time), args.merge_gap_sec)
    detections = filter_short(detections, args.min_segment_sec)
    return detections


def build_detection(tv_name, window_group, durations, args):
    tv = str(tv_name)
    commercial = window_group[0][1]
    duration = durations.get(commercial, 30.0)
    center = np.mean([w[0] for w in window_group])
    start_time = max(0.0, center - duration / 2)
    score = float(np.mean([w[2] for w in window_group]))
    return Detection(tv, start_time, duration, commercial, score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detector sobre k-NN de TDA en embeddings")
    parser.add_argument("--knn_dir", type=Path, required=True)
    parser.add_argument("--curve_dir", type=Path, required=True)
    parser.add_argument("--preproc_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--curve_threshold", type=float, default=1.5)
    parser.add_argument("--merge_gap_sec", type=float, default=2.0)
    parser.add_argument("--min_segment_sec", type=float, default=3.0)
    args = parser.parse_args()

    durations = load_durations(args.preproc_manifest)
    curve_map = load_curve_map(args.curve_dir)

    detections: List[Detection] = []
    for npz_path in sorted((args.knn_dir / "tv").glob("*_knn.npz")):
        payload = load_npz(npz_path)
        video_name = str(payload["video_name"]) if np.ndim(payload["video_name"]) == 0 else str(payload["video_name"][0])
        curve_path = curve_map.get(video_name)
        if curve_path is None:
            continue
        curve_data = load_npz(curve_path)
        detections.extend(detect_for_video(payload, curve_data, args, durations))

    detections.sort(key=lambda d: (d.television, d.start_time))
    ensure_dir(args.output.parent)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# television\tinicio_seg\tlargo_seg\tcomercial\tscore\n")
        for det in detections:
            handle.write(
                f"{det.television}\t{det.start_time:.3f}\t{det.duration:.3f}\t{det.commercial}\t{det.score:.4f}\n"
            )
    print(f"[detector_embedding_tda] escrito {len(detections)} detecciones en {args.output}")


if __name__ == "__main__":
    main()
