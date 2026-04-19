#!/usr/bin/env python3
"""
Búsqueda por similitud (k-NN) entre frames de consulta y un banco de entrenamiento.

Entrada: directorio generado por `pipeline/preprocessing/cubical_preprocessing.py`
Salida: para cada video de validación, un archivo NPZ con los `k` frames de 
        entrenamiento más cercanos según similitud coseno.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Buscar k vecinos más cercanos para clasificación de acciones")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio raíz de los NPZ cúbicos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_outputs/knn",
        help="Ruta donde guardar los resultados por video consultado",
    )
    parser.add_argument(
        "--train_prefixes", 
        nargs="+", 
        required=True, 
        help="Prefijos para construir el banco de entrenamiento (ej. P03 P04 P05)"
    )
    parser.add_argument(
        "--val_prefixes", 
        nargs="+", 
        required=True, 
        help="Prefijos de los videos a evaluar/consultar (ej. P48 P49 P50)"
    )
    parser.add_argument("--k", type=int, default=5, help="Cantidad de vecinos a conservar por frame")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Aplica normalización L2 a los descriptores antes de calcular similitud coseno",
    )
    parser.add_argument(
        "--standardize",
        action="store_true",
        help="Estandariza cada columna usando media/desviación del banco de entrenamiento antes de normalizar",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Procesa en lotes para ahorrar memoria")
    parser.add_argument("--overwrite", action="store_true", help="Reemplaza archivos existentes")
    return parser.parse_args()


def get_files_by_prefix(root: Path, prefixes: List[str]) -> List[Path]:
    if not root.exists():
        return []
    matched_files = []
    for path in sorted(root.rglob("*.npz")):
        person_id = path.parent.name
        if any(person_id.startswith(p) for p in prefixes):
            matched_files.append(path)
    return matched_files


def load_features(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    feats = data["tda_features"].astype(np.float32)
    timestamps = data.get("timestamps_sec", np.zeros(feats.shape[0])).astype(np.float32)
    indices = data.get("frame_indices", np.arange(feats.shape[0])).astype(np.int32)
    labels = data.get("label_id", np.zeros(feats.shape[0])).astype(np.int32)
    return feats, timestamps, indices, labels


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    return matrix / norms


def fit_standardizer(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0, keepdims=True).astype(np.float32)
    std = matrix.std(axis=0, keepdims=True).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_standardizer(matrix: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / std).astype(np.float32)


def build_reference_bank(
    root: Path, prefixes: List[str], normalize: bool, standardize: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray | None, np.ndarray | None]:
    feature_list: List[np.ndarray] = []
    video_idx_list: List[int] = []
    frame_idx_list: List[int] = []
    timestamp_list: List[float] = []
    label_idx_list: List[int] = []
    video_names: List[str] = []

    files = get_files_by_prefix(root, prefixes)
    for vid_idx, npz_path in enumerate(files):
        feats, timestamps, frames, labels = load_features(npz_path)
        if feats.size == 0:
            continue
        video_names.append(npz_path.stem)
        feature_list.append(feats)
        video_idx_list.append(np.full(feats.shape[0], vid_idx, dtype=np.int32))
        frame_idx_list.append(frames)
        timestamp_list.append(timestamps)
        label_idx_list.append(labels)

    if not feature_list:
        raise RuntimeError(f"No se encontraron características para los prefijos de entrenamiento en {root}")

    feature_matrix = np.vstack(feature_list).astype(np.float32)
    scaler_mean: np.ndarray | None = None
    scaler_std: np.ndarray | None = None
    if standardize:
        scaler_mean, scaler_std = fit_standardizer(feature_matrix)
        feature_matrix = apply_standardizer(feature_matrix, scaler_mean, scaler_std)
    if normalize:
        feature_matrix = normalize_rows(feature_matrix)
        
    video_indices = np.concatenate(video_idx_list)
    frame_indices = np.concatenate(frame_idx_list)
    timestamps = np.concatenate(timestamp_list)
    labels_indices = np.concatenate(label_idx_list)
    
    return feature_matrix, video_indices, frame_indices, timestamps, labels_indices, video_names, scaler_mean, scaler_std


def compute_neighbors(
    query_feats: np.ndarray,
    bank_matrix: np.ndarray,
    k: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    k = max(1, min(k, bank_matrix.shape[0]))
    num_queries = query_feats.shape[0]
    neighbor_idx = np.zeros((num_queries, k), dtype=np.int32)
    neighbor_scores = np.zeros((num_queries, k), dtype=np.float32)

    for start in range(0, num_queries, batch_size):
        end = min(start + batch_size, num_queries)
        batch = query_feats[start:end]
        sims = batch @ bank_matrix.T  # (batch, bank)
        top_idx = np.argpartition(sims, -k, axis=1)[:, -k:]
        top_scores = np.take_along_axis(sims, top_idx, axis=1)
        order = np.argsort(-top_scores, axis=1)
        sorted_idx = np.take_along_axis(top_idx, order, axis=1)
        sorted_scores = np.take_along_axis(top_scores, order, axis=1)
        neighbor_idx[start:end] = sorted_idx
        neighbor_scores[start:end] = sorted_scores
    return neighbor_idx, neighbor_scores


def process_query_video(
    npz_path: Path,
    bank_matrix: np.ndarray,
    bank_meta: Dict[str, np.ndarray],
    video_names: List[str],
    args: argparse.Namespace,
) -> Dict[str, str] | None:
    feats, timestamps, frames, true_labels = load_features(npz_path)
    if feats.size == 0:
        print(f"[WARN] {npz_path.name} no tiene características; se omite.")
        return None
        
    query_feats = feats.astype(np.float32)
    if args.standardize and bank_meta.get("scaler_mean") is not None:
        query_feats = apply_standardizer(query_feats, bank_meta["scaler_mean"], bank_meta["scaler_std"])
    if args.normalize:
        query_feats = normalize_rows(query_feats)
        
    neighbor_idx, neighbor_scores = compute_neighbors(query_feats, bank_matrix, args.k, args.batch_size)

    person_id = npz_path.parent.name
    output_path = Path(args.output_dir) / person_id / f"{npz_path.stem}_knn.npz"
    if output_path.exists() and not args.overwrite:
        print(f"[skip] {output_path.name} ya existe")
        return None

    meta = {
        "timestamps_sec": timestamps,
        "frame_indices": frames,
        "true_label_id": true_labels, # Guardamos el GT para métricas
        "neighbor_indices": neighbor_idx,
        "neighbor_scores": neighbor_scores,
        "train_video_idx": bank_meta["video_idx"],
        "train_frame_idx": bank_meta["frame_idx"],
        "train_label_id": bank_meta["label_id"], # La clase de los vecinos
        "train_video_names": np.array(video_names, dtype=np.str_),
        "standardized": np.array([args.standardize], dtype=np.bool_),
        "normalized": np.array([args.normalize], dtype=np.bool_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **meta)
    
    return {"query_video": npz_path.stem, "person_id": person_id, "output": str(output_path)}


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # 1. Construir banco con los prefijos de entrenamiento
    bank_matrix, vid_idx, frame_idx, timestamps, label_idx, video_names, scaler_mean, scaler_std = build_reference_bank(
        input_root, args.train_prefixes, args.normalize, args.standardize
    )
    bank_meta = {
        "video_idx": vid_idx,
        "frame_idx": frame_idx,
        "timestamps": timestamps,
        "label_id": label_idx,
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
    }

    # 2. Consultar con los prefijos de validación
    manifest: List[Dict[str, str]] = []
    query_files = get_files_by_prefix(input_root, args.val_prefixes)
    
    for query_npz in query_files:
        res = process_query_video(query_npz, bank_matrix, bank_meta, video_names, args)
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