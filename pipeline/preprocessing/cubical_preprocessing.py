#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

import cv2
import gudhi as gd
from gudhi.representations import PersistenceImage
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov"}

@dataclass
class VideoSummary:
    person_id: str
    source_path: str
    output_path: str
    num_frames: int
    native_fps: float
    sampled_fps: float
    duration_sec: float
    sample_stride_frames: float
    feature_dim: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocesamiento Breakfast Dataset (Flujo Óptico + TDA)")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Ruta a BreakfastII_15fps_qvga_sync")
    parser.add_argument("--output_dir", type=str, default="pipeline_outputs/cubical")
    parser.add_argument("--sample_fps", type=float, default=3.0)
    parser.add_argument("--grid_size", type=int, default=48)
    parser.add_argument("--min_persistence", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def frame_indices_for_sampling(native_fps: float, target_fps: float, total_frames: int) -> Tuple[List[int], List[float]]:
    stride = native_fps / target_fps
    indices, timestamps = [], []
    samples = int(math.floor(total_frames / stride))
    for i in range(samples):
        idx = int(round(i * stride))
        if idx >= total_frames: break
        indices.append(idx)
        timestamps.append(idx / native_fps)
    return indices, timestamps

def diagram_stats(diag: np.ndarray) -> np.ndarray:
    if diag.size == 0: return np.zeros(5, dtype=np.float32)
    lifetimes = np.maximum(diag[:, 1] - diag[:, 0], 0.0)
    if lifetimes.size == 0: return np.zeros(5, dtype=np.float32)
    return np.array([float(len(lifetimes)), float(np.sum(lifetimes)), float(np.max(lifetimes)), float(np.mean(lifetimes)), float(np.std(lifetimes))], dtype=np.float32)

def transform_diagram(diag: np.ndarray, pi_transform: PersistenceImage) -> np.ndarray:
    if diag.size == 0: return np.zeros(pi_transform.resolution[0] * pi_transform.resolution[1], dtype=np.float32)
    return pi_transform.transform([diag])[0].astype(np.float32)

def cubical_descriptor(mag_norm: np.ndarray, grid_size: int, min_persistence: float, pi_transform_h0: PersistenceImage, pi_transform_h1: PersistenceImage) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = cv2.resize(mag_norm, (grid_size, grid_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    
    complex_ = gd.CubicalComplex(dimensions=resized.shape, top_dimensional_cells=resized.ravel())
    diag_raw = complex_.persistence(homology_coeff_field=2, min_persistence=min_persistence)
    
    diag_h0 = np.array([inv for dim, inv in diag_raw if dim == 0 and np.isfinite(inv[1])], dtype=np.float32)
    diag_h1 = np.array([inv for dim, inv in diag_raw if dim == 1 and np.isfinite(inv[1])], dtype=np.float32)
    if diag_h0.ndim == 1: diag_h0 = np.zeros((0, 2), dtype=np.float32)
    if diag_h1.ndim == 1: diag_h1 = np.zeros((0, 2), dtype=np.float32)

    stats_h0, stats_h1 = diagram_stats(diag_h0), diagram_stats(diag_h1)
    brightness = np.array([resized.mean(), resized.std()], dtype=np.float32)

    feature_vec = np.concatenate([stats_h0, stats_h1, brightness, transform_diagram(diag_h0, pi_transform_h0), transform_diagram(diag_h1, pi_transform_h1)], dtype=np.float32)
    return feature_vec, diag_h0, diag_h1

def parse_labels_to_dense(label_filepath: Path, total_frames: int, action_to_id: Dict[str, int]) -> np.ndarray:
    dense_labels = np.zeros(total_frames, dtype=np.int32)
    with label_filepath.open('r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2: continue
            try:
                start_frame, end_frame = map(int, parts[0].split('-'))
            except ValueError: continue
            
            start_idx = max(0, start_frame - 1)
            end_idx = min(total_frames, end_frame)
            dense_labels[start_idx:end_idx] = action_to_id.get(parts[1], 0)
    return dense_labels

def process_video(video_path: Path, label_path: Path, person_id: str, output_root: Path, args: argparse.Namespace, pi_h0: PersistenceImage, pi_h1: PersistenceImage, action_to_id: Dict[str, int], id_to_action: Dict[int, str]) -> VideoSummary | None:
    output_file = output_root / person_id / f"{video_path.stem}.npz"
    if output_file.exists() and not args.overwrite: return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return None
    native_fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    dense_labels = parse_labels_to_dense(label_path, total_frames, action_to_id)
    indices, timestamps = frame_indices_for_sampling(native_fps, args.sample_fps, total_frames)

    valid_indices, valid_timestamps, descriptors, prev_gray = [], [], [], None

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

        feature_vec, _, _ = cubical_descriptor(mag_norm, args.grid_size, args.min_persistence, pi_h0, pi_h1)
        descriptors.append(feature_vec)
        valid_indices.append(idx)
        valid_timestamps.append(timestamps[i])
        prev_gray = gray

    cap.release()
    if not descriptors: return None

    features = np.stack(descriptors).astype(np.float32)
    labels_sampled = dense_labels[valid_indices]
    names_sampled = np.array([id_to_action.get(lid, "SIL") for lid in labels_sampled], dtype=np.str_)

    payload = {
        "timestamps_sec": np.asarray(valid_timestamps, dtype=np.float32),
        "frame_indices": np.asarray(valid_indices, dtype=np.int32),
        "tda_features": features,
        "label_id": labels_sampled,
        "label_name": names_sampled
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)

    return VideoSummary(person_id, str(video_path), str(output_file.relative_to(output_root)), features.shape[0], float(native_fps), float(args.sample_fps), total_frames/native_fps, float(native_fps/args.sample_fps), int(features.shape[1]))

def main() -> None:
    args = parse_args()
    dataset_dir, output_root = Path(args.dataset_dir), Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Construir Vocabulario global dinámico
    action_to_id = {"SIL": 0}
    for label_path in dataset_dir.rglob("cam01/*.labels"):
        with label_path.open('r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2 and parts[1] not in action_to_id:
                    action_to_id[parts[1]] = len(action_to_id)
                    
    id_to_action = {v: k for k, v in action_to_id.items()}
    with open(output_root / "action_map.json", "w") as f: json.dump(id_to_action, f, indent=2)

    pi_transform = PersistenceImage(bandwidth=0.05, weight=lambda b_d: b_d[1]-b_d[0], resolution=[16, 16], im_range=[0.0, 1.0, 0.0, 1.0])
    pi_transform.fit([np.array([[0.0, 1.0]], dtype=np.float32)])

    # 2. Procesar Videos
    summaries = []
    for video_path in dataset_dir.rglob("cam01/*.avi"):
        label_path = video_path.with_name(f"{video_path.name}.labels")
        if not label_path.exists(): continue
        
        person_id = video_path.parent.parent.name # Extrae 'P48'
        print(f"[{person_id}] Procesando {video_path.name}")
        summary = process_video(video_path, label_path, person_id, output_root, args, pi_transform, pi_transform, action_to_id, id_to_action)
        if summary: summaries.append(summary)

    with (output_root / "manifest.json").open("w") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)

if __name__ == "__main__":
    main()