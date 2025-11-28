#!/usr/bin/env python3
"""
Preprocesamiento basado solo en video para el proyecto de detección de comerciales.

Pasos implementados:
  1. Limpieza temporal: remuestrea cada video (TV y comerciales) a un FPS fijo.
  2. Normalización espacial: convierte a escala de grises, reduce a una grilla pequeña y escala a [0, 1].
  3. Extracción de rasgos topológicos por frame con complejos cúbicos (GUDHI).
  4. Vectorización con estadísticas + Persistence Image para obtener vectores fijos aptos para TDA.
  5. Persistencia de los NPZ resultantes y un manifest con metadatos básicos.

Este script no usa audio ni embeddings CNN; toda la información proviene de frames cúbicos.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import gudhi as gd
from gudhi.representations import PersistenceImage
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mpg", ".mpeg", ".avi", ".mkv", ".mov"}


@dataclass
class VideoSummary:
    category: str
    source_path: str
    output_path: str
    num_frames: int
    native_fps: float
    sampled_fps: float
    duration_sec: float
    sample_stride_frames: float
    feature_dim: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento de video mediante complejos cúbicos (sin audio/CNN)"
    )
    parser.add_argument("--tv_dir", type=str, required=True, help="Directorio que contiene los videos de televisión")
    parser.add_argument("--commercials_dir", type=str, required=True, help="Directorio con los comerciales individuales")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_outputs/cubical",
        help="Ruta donde se guardarán los NPZ generados",
    )
    parser.add_argument("--sample_fps", type=float, default=3.0, help="FPS objetivo para todos los videos")
    parser.add_argument("--grid_size", type=int, default=48, help="Lado de la grilla normalizada por frame")
    parser.add_argument("--min_persistence", type=float, default=0.005, help="Filtro mínimo de persistencia en GUDHI")
    parser.add_argument("--overwrite", action="store_true", help="Reprocesa aunque existan NPZ previos")
    return parser.parse_args()


def iter_videos(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def frame_indices_for_sampling(native_fps: float, target_fps: float, total_frames: int) -> Tuple[List[int], List[float]]:
    if native_fps <= 0 or target_fps <= 0:
        raise ValueError("FPS nativo/objetivo inválido")
    stride = native_fps / target_fps
    indices: List[int] = []
    timestamps: List[float] = []
    samples = int(math.floor(total_frames / stride))
    for i in range(samples):
        idx = int(round(i * stride))
        if idx >= total_frames:
            break
        indices.append(idx)
        timestamps.append(idx / native_fps)
    return indices, timestamps


def diagram_stats(diag: np.ndarray) -> np.ndarray:
    if diag.size == 0:
        return np.zeros(5, dtype=np.float32)
    lifetimes = np.maximum(diag[:, 1] - diag[:, 0], 0.0)
    if lifetimes.size == 0:
        return np.zeros(5, dtype=np.float32)
    return np.array(
        [
            float(len(lifetimes)),
            float(np.sum(lifetimes)),
            float(np.max(lifetimes)),
            float(np.mean(lifetimes)),
            float(np.std(lifetimes)),
        ],
        dtype=np.float32,
    )


def cubical_descriptor(
    frame_gray: np.ndarray,
    grid_size: int,
    min_persistence: float,
    pi_transform: PersistenceImage,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resized = cv2.resize(frame_gray, (grid_size, grid_size), interpolation=cv2.INTER_AREA).astype(np.float32)
    normalized = (resized - resized.min()) / (np.ptp(resized) + 1e-6)

    complex_ = gd.CubicalComplex(dimensions=normalized.shape, top_dimensional_cells=normalized.ravel())
    diag_raw = complex_.persistence(homology_coeff_field=2, min_persistence=min_persistence)
    diag_h0, diag_h1 = [], []
    for dim, interval in diag_raw:
        if not np.isfinite(interval[1]):
            continue
        if dim == 0:
            diag_h0.append(interval)
        elif dim == 1:
            diag_h1.append(interval)
    diag_h0 = np.array(diag_h0, dtype=np.float32) if diag_h0 else np.zeros((0, 2), dtype=np.float32)
    diag_h1 = np.array(diag_h1, dtype=np.float32) if diag_h1 else np.zeros((0, 2), dtype=np.float32)

    stats_h0 = diagram_stats(diag_h0)
    stats_h1 = diagram_stats(diag_h1)
    brightness = np.array([normalized.mean(), normalized.std()], dtype=np.float32)

    if diag_h0.size or diag_h1.size:
        stack = (
            np.vstack([diag_h0, diag_h1])
            if (diag_h0.size and diag_h1.size)
            else (diag_h0 if diag_h0.size else diag_h1)
        )
        pi_vec = pi_transform.transform([stack])[0].astype(np.float32)
    else:
        res = pi_transform.resolution
        pi_vec = np.zeros(res[0] * res[1], dtype=np.float32)

    feature_vec = np.concatenate([stats_h0, stats_h1, brightness, pi_vec], dtype=np.float32)
    return feature_vec, diag_h0, diag_h1


def process_video(
    video_path: Path,
    category: str,
    output_root: Path,
    args: argparse.Namespace,
    pi_transform: PersistenceImage,
) -> VideoSummary | None:
    output_file = output_root / category / f"{video_path.stem}.npz"
    if output_file.exists() and not args.overwrite:
        print(f"[{category}] Omitiendo {video_path.name} (ya procesado)")
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir {video_path}")
        return None
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / native_fps if native_fps > 0 else 0.0

    indices, timestamps = frame_indices_for_sampling(native_fps, args.sample_fps, total_frames)
    descriptors: List[np.ndarray] = []
    h0_births: List[float] = []
    h0_deaths: List[float] = []
    h0_offsets: List[int] = [0]
    h1_births: List[float] = []
    h1_deaths: List[float] = []
    h1_offsets: List[int] = [0]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_vec, diag_h0, diag_h1 = cubical_descriptor(gray, args.grid_size, args.min_persistence, pi_transform)
        descriptors.append(feature_vec)
        if diag_h0.size:
            h0_births.extend(diag_h0[:, 0].tolist())
            h0_deaths.extend(diag_h0[:, 1].tolist())
        h0_offsets.append(len(h0_births))
        if diag_h1.size:
            h1_births.extend(diag_h1[:, 0].tolist())
            h1_deaths.extend(diag_h1[:, 1].tolist())
        h1_offsets.append(len(h1_births))

    cap.release()

    if descriptors:
        features = np.stack(descriptors).astype(np.float32)
    else:
        pi_dim = pi_transform.resolution[0] * pi_transform.resolution[1]
        features = np.zeros((0, pi_dim + 12), dtype=np.float32)

    payload = {
        "timestamps_sec": np.asarray(timestamps, dtype=np.float32),
        "frame_indices": np.asarray(indices, dtype=np.int32),
        "tda_features": features,
        "diag_h0_births": np.asarray(h0_births, dtype=np.float32),
        "diag_h0_deaths": np.asarray(h0_deaths, dtype=np.float32),
        "diag_h0_offsets": np.asarray(h0_offsets, dtype=np.int32),
        "diag_h1_births": np.asarray(h1_births, dtype=np.float32),
        "diag_h1_deaths": np.asarray(h1_deaths, dtype=np.float32),
        "diag_h1_offsets": np.asarray(h1_offsets, dtype=np.int32),
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **payload)

    summary = VideoSummary(
        category=category,
        source_path=str(video_path),
        output_path=str(output_file.relative_to(output_root)),
        num_frames=features.shape[0],
        native_fps=float(native_fps),
        sampled_fps=float(args.sample_fps),
        duration_sec=float(duration),
        sample_stride_frames=float(native_fps / args.sample_fps if args.sample_fps > 0 else 0.0),
        feature_dim=int(features.shape[1]) if features.size else int(pi_transform.resolution[0] * pi_transform.resolution[1] + 12),
    )
    return summary


def write_manifest(path: Path, summaries: List[VideoSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(s) for s in summaries], handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    tv_dir = Path(args.tv_dir)
    com_dir = Path(args.commercials_dir)
    if not tv_dir.exists() or not com_dir.exists():
        raise FileNotFoundError("Verifique las rutas --tv_dir y --commercials_dir")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    pi_transform = PersistenceImage(
        bandwidth=0.05,
        weight=lambda birth_death: birth_death[1] - birth_death[0],
        resolution=[16, 16],
        im_range=[0.0, 1.0, 0.0, 1.0],
    )
    pi_transform.fit([np.array([[0.0, 1.0]], dtype=np.float32)])

    summaries: List[VideoSummary] = []
    for category, directory in (("tv", tv_dir), ("commercials", com_dir)):
        for video_path in iter_videos(directory):
            print(f"[{category}] Procesando {video_path.name}")
            summary = process_video(video_path, category, output_root, args, pi_transform)
            if summary:
                summaries.append(summary)

    write_manifest(output_root / "manifest.json", summaries)
    print(f"Preprocesamiento finalizado. Manifest en {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
