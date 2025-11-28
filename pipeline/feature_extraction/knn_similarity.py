#!/usr/bin/env python3
"""
Búsqueda por similitud (k-NN) entre frames de TV y comerciales usando descriptores cúbicos.

Entrada: directorio generado por `pipeline/preprocessing/cubical_preprocessing.py`
Salida: para cada video de TV, un archivo NPZ con los `k` vecinos (comerciales) más
        cercanos por frame según similitud coseno.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

VIDEO_DIRS = ("tv", "commercials")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buscar k vecinos más cercanos entre TV y comerciales (descriptores cúbicos)")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio raíz de los NPZ cúbicos (con subcarpetas tv/ y commercials/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_outputs/knn",
        help="Ruta donde guardar los resultados por video de TV",
    )
    parser.add_argument("--k", type=int, default=5, help="Cantidad de vecinos a conservar por frame de TV")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Aplica normalización L2 a los descriptores antes de calcular similitud coseno",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Procesa los frames de TV en lotes para ahorrar memoria")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza archivos existentes en la carpeta de salida")
    return parser.parse_args()


def iter_npz(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in sorted(root.glob("*.npz")):
        if path.is_file():
            yield path


def load_features(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    feats = data["tda_features"].astype(np.float32)
    timestamps = data["timestamps_sec"].astype(np.float32)
    indices = data["frame_indices"].astype(np.int32)
    return feats, timestamps, indices


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return matrix / norms


def build_commercial_bank(root: Path, normalize: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_list: List[np.ndarray] = []
    video_idx_list: List[int] = []
    frame_idx_list: List[int] = []
    timestamp_list: List[float] = []
    video_names: List[str] = []

    for vid_idx, npz_path in enumerate(iter_npz(root)):
        feats, timestamps, frames = load_features(npz_path)
        if feats.size == 0:
            continue
        video_names.append(npz_path.stem)
        feature_list.append(feats)
        video_idx_list.append(np.full(feats.shape[0], vid_idx, dtype=np.int32))
        frame_idx_list.append(frames.astype(np.int32))
        timestamp_list.append(timestamps.astype(np.float32))

    if not feature_list:
        raise RuntimeError(f"No se encontraron características en {root}")

    feature_matrix = np.vstack(feature_list).astype(np.float32)
    if normalize:
        feature_matrix = normalize_rows(feature_matrix)
    video_indices = np.concatenate(video_idx_list)
    frame_indices = np.concatenate(frame_idx_list)
    timestamps = np.concatenate(timestamp_list)
    return feature_matrix, video_indices, frame_indices, timestamps, video_names


def compute_neighbors(
    tv_feats: np.ndarray,
    bank_matrix: np.ndarray,
    k: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_tv = tv_feats.shape[0]
    neighbor_idx = np.zeros((num_tv, k), dtype=np.int32)
    neighbor_scores = np.zeros((num_tv, k), dtype=np.float32)

    for start in range(0, num_tv, batch_size):
        end = min(start + batch_size, num_tv)
        batch = tv_feats[start:end]
        sims = batch @ bank_matrix.T  # (batch, bank)
        top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        neighbor_idx[start:end] = sorted_idx
        neighbor_scores[start:end] = sorted_scores
    return neighbor_idx, neighbor_scores


def process_tv_video(
    npz_path: Path,
    bank_matrix: np.ndarray,
    bank_meta: Dict[str, np.ndarray],
    video_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, str] | None:
    feats, timestamps, frames = load_features(npz_path)
    if feats.size == 0:
        print(f"[WARN] {npz_path.name} no tiene características; se omite.")
        return None
    tv_feats = normalize_rows(feats) if args.normalize else feats
    neighbor_idx, neighbor_scores = compute_neighbors(tv_feats, bank_matrix, args.k, args.batch_size)

    output_path = Path(args.output_dir) / "tv" / f"{npz_path.stem}_knn.npz"
    if output_path.exists() and not args.overwrite:
        print(f"[skip] {output_path.name} ya existe")
        return None

    meta = {
        "timestamps_sec": timestamps,
        "frame_indices": frames,
        "neighbor_indices": neighbor_idx,
        "neighbor_scores": neighbor_scores,
        "commercial_video_idx": bank_meta["video_idx"],
        "commercial_frame_idx": bank_meta["frame_idx"],
        "commercial_timestamps": bank_meta["timestamps"],
        "commercial_video_names": np.array(video_names, dtype=np.str_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **meta)
    return {"tv_video": npz_path.stem, "output": str(output_path)}


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    bank_matrix, vid_idx, frame_idx, timestamps, video_names = build_commercial_bank(
        input_root / "commercials", args.normalize
    )
    bank_meta = {
        "video_idx": vid_idx,
        "frame_idx": frame_idx,
        "timestamps": timestamps,
    }

    manifest: List[Dict[str, str]] = []
    for tv_npz in iter_npz(input_root / "tv"):
        res = process_tv_video(tv_npz, bank_matrix, bank_meta, video_names, args)
        if res:
            manifest.append(res)

    if manifest:
        manifest_path = output_root / "manifest_knn.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        print(f"KNN finalizado. Manifest en {manifest_path}")
    else:
        print("No se generaron archivos KNN (¿ya existen y no se usó --overwrite?).")


if __name__ == "__main__":
    main()
