#!/usr/bin/env python3
"""
Detección de apariciones basada en vecinos más cercanos (k-NN) y desfase constante.

Usa los archivos generados por pipeline/feature_extraction/knn_similarity.py:
  - Para cada frame de TV se toma el vecino más similar (comercial y timestamp).
  - Se buscan secuencias consecutivas donde el comercial y el desfase se mantienen
    estables. Si la duración de la secuencia cubre una fracción suficiente del
    comercial real, se reporta una detección.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def vote_window(
    start_idx: int,
    end_idx: int,
    tv_timestamps: np.ndarray,
    neighbor_idx: np.ndarray,
    neighbor_scores: np.ndarray,
    video_idx: np.ndarray,
    commercial_ts: np.ndarray,
    commercial_names: List[str],
) -> tuple[str, float, float]:
    idx_slice = neighbor_idx[start_idx:end_idx, 0]
    if idx_slice.size == 0:
        return "", 0.0, 0.0
    scores = neighbor_scores[start_idx:end_idx, 0]
    vid_ids = video_idx[idx_slice]
    names = [commercial_names[int(i)] for i in vid_ids]
    unique, counts = np.unique(names, return_counts=True)
    best_idx = int(np.argmax(counts))
    best_name = str(unique[best_idx])
    mask = np.array(names) == best_name
    if not mask.any():
        return best_name, 0.0, 0.0
    selected_scores = scores[mask]
    avg_score = float(np.mean(selected_scores)) if selected_scores.size else 0.0
    offsets = tv_timestamps[start_idx:end_idx][mask] - commercial_ts[idx_slice][mask]
    avg_offset = float(np.mean(offsets)) if offsets.size else 0.0
    return best_name, avg_score, avg_offset

@dataclass
class Track:
    commercial: str
    start_idx: int
    last_idx: int
    offset: float
    best_score: float
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detector basado en k-NN con desfase constante")
    parser.add_argument(
        "--knn_dir",
        type=str,
        required=True,
        help="Directorio con los archivos *_knn.npz por video de TV",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Manifest JSON del preprocesamiento (para obtener duración de los comerciales)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pipeline/detection/detecciones_knn.txt",
        help="Archivo donde guardar las detecciones",
    )
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score mínimo para iniciar/continuar una pista")
    parser.add_argument("--offset_tolerance", type=float, default=1.0, help="Tolerancia (segundos) en el desfase TV - comercial")
    parser.add_argument(
        "--coverage_ratio",
        type=float,
        default=0.6,
        help="Fracción mínima del largo del comercial que debe cubrir la pista (valor base)",
    )
    parser.add_argument(
        "--coverage_ratio_short",
        type=float,
        default=None,
        help="Cobertura mínima para comerciales más cortos (usa coverage_ratio si no se especifica)",
    )
    parser.add_argument(
        "--coverage_ratio_long",
        type=float,
        default=None,
        help="Cobertura mínima para comerciales largos (usa coverage_ratio si no se especifica)",
    )
    parser.add_argument(
        "--min_gap",
        type=float,
        default=0.0,
        help="Separación mínima (segundos) entre detecciones de un mismo comercial; "
        "si se detectan dos apariciones más cercanas se conserva la de mayor score",
    )
    parser.add_argument("--min_frames", type=int, default=3, help="Frames mínimos base para validar una pista")
    parser.add_argument("--min_frames_short", type=int, default=None, help="Frames mínimos para comerciales cortos")
    parser.add_argument("--min_frames_long", type=int, default=None, help="Frames mínimos para comerciales largos")
    parser.add_argument(
        "--duration_threshold",
        type=float,
        default=20.0,
        help="Umbral (seg) para decidir si un comercial es 'corto' o 'largo' en los parámetros adaptativos",
    )
    parser.add_argument(
        "--window_sec",
        type=float,
        default=1.0,
        help="Tamaño de la ventana (segundos) usada para votar el comercial dominante y suavizar el desfase",
    )
    parser.add_argument(
        "--curve_dir",
        type=str,
        default=None,
        help="Directorio con curvas topológicas (combined_activity) para filtrar picos",
    )
    parser.add_argument(
        "--curve_threshold",
        type=float,
        default=1.5,
        help="Umbral z-score mínimo de la curva para considerar una ventana",
    )
    parser.add_argument("--include_score", action="store_true", help="Incluye la columna de score en el archivo de salida")
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    mapping: Dict[str, Dict] = {}
    for entry in data:
        stem = Path(entry["source_path"]).stem
        mapping[stem] = entry
    return mapping


def finalize_track(
    track: Optional[Track],
    timestamps: np.ndarray,
    commercial_durations: Dict[str, float],
    args: argparse.Namespace,
) -> Optional[Dict]:
    if track is None:
        return None
    start_time = float(timestamps[track.start_idx])
    end_time = float(timestamps[track.last_idx])
    observed_duration = max(end_time - start_time, 0.0)
    expected_duration = commercial_durations.get(track.commercial, observed_duration)
    if expected_duration <= 0:
        expected_duration = observed_duration

    short_ratio = args.coverage_ratio_short if args.coverage_ratio_short is not None else args.coverage_ratio
    long_ratio = args.coverage_ratio_long if args.coverage_ratio_long is not None else args.coverage_ratio
    short_frames = args.min_frames_short if args.min_frames_short is not None else args.min_frames
    long_frames = args.min_frames_long if args.min_frames_long is not None else args.min_frames

    if expected_duration <= args.duration_threshold:
        coverage_target = short_ratio
        min_frames_target = short_frames
    else:
        coverage_target = long_ratio
        min_frames_target = long_frames

    frame_count = track.last_idx - track.start_idx + 1
    if frame_count < min_frames_target:
        print(f"[skip] pista {track.commercial} descartada por min_frames ({frame_count} < {min_frames_target})")
        return None
    if observed_duration < expected_duration * coverage_target:
        ratio = observed_duration / expected_duration if expected_duration else 0.0
        print(
            f"[skip] pista {track.commercial} cobertura insuficiente "
            f"({observed_duration:.2f}s vs {expected_duration:.2f}s; ratio={ratio:.2f})"
        )
        return None
    return {
        "commercial": track.commercial,
        "start_time": start_time,
        "duration": expected_duration,
        "score": track.best_score,
    }


def deduplicate_by_gap(detections: List[Dict], min_gap: float) -> List[Dict]:
    if min_gap <= 0 or not detections:
        return detections
    ordered = sorted(detections, key=lambda det: det["start_time"])
    filtered: List[Dict] = []
    last_kept_idx: Dict[str, int] = {}
    for det in ordered:
        name = det["commercial"]
        idx = last_kept_idx.get(name)
        if idx is None:
            filtered.append(det)
            last_kept_idx[name] = len(filtered) - 1
            continue
        last_det = filtered[idx]
        if det["start_time"] - last_det["start_time"] < min_gap:
            if det["score"] > last_det["score"]:
                filtered[idx] = det
            print(
                f"[dedup] pista {name} descartada por min_gap "
                f"({det['start_time']:.2f}s vs {last_det['start_time']:.2f}s)"
            )
            continue
        filtered.append(det)
        last_kept_idx[name] = len(filtered) - 1
    return filtered


def detect_in_video(
    tv_name: str,
    npz_path: Path,
    manifest: Dict[str, Dict],
    args: argparse.Namespace,
) -> List[str]:
    data = np.load(npz_path)
    timestamps = data["timestamps_sec"].astype(np.float32)
    neighbor_idx = data["neighbor_indices"]
    neighbor_scores = data["neighbor_scores"]
    video_idx = data["commercial_video_idx"]
    frame_idx = data["commercial_frame_idx"]
    commercial_ts = data["commercial_timestamps"]
    commercial_names = [str(name) for name in data["commercial_video_names"]]

    commercial_durations = {
        name: float(manifest.get(name, {}).get("duration_sec", 0.0)) for name in commercial_names
    }

    detections: List[str] = []
    current: Optional[Track] = None

    tv_meta = manifest.get(tv_name, {})
    sample_fps = float(tv_meta.get("sampled_fps", 3.0))
    window_frames = max(int(round(args.window_sec * sample_fps)), 1)
    curve_signal = None
    curve_loaded = False
    curve_blocks = 0
    curve_rejections = 0
    if args.curve_dir:
        curve_manifest = json.loads(Path(args.curve_dir, "manifest_curves.json").read_text())
        curve_map = {
            Path(entry["source_path"]).stem: entry["output_path"]
            for entry in curve_manifest
            if entry["category"] == "tv"
        }
        curve_rel = curve_map.get(tv_name)
        if curve_rel:
            curve_path = Path(curve_rel)
            if not curve_path.is_absolute():
                curve_path = (Path.cwd() / curve_path).resolve()
            if curve_path.exists():
                curve_data = np.load(curve_path)
                labels = [str(x) for x in curve_data["curve_labels"]]
                if "combined_activity" in labels:
                    idx = labels.index("combined_activity")
                    curve_signal = curve_data["curve_signals"][:, idx]
                    curve_loaded = True
            else:
                print(f"[curve] archivo no encontrado: {curve_path} para {tv_name}")
        else:
            print(f"[curve] sin entrada en manifest para {tv_name}")
    if args.curve_dir and not curve_loaded:
        print(f"[curve] sin señal cargada para {tv_name}, no se aplicó filtro topológico")

    i = 0
    curve_used = False
    while i < len(timestamps):
        window_end = min(len(timestamps), i + window_frames)
        if curve_signal is not None:
            z_slice = curve_signal[i:window_end]
            if z_slice.size == 0 or np.max(z_slice) < args.curve_threshold:
                curve_rejections += 1
                result = finalize_track(current, timestamps, commercial_durations, args)
                if result:
                    detections.append(result)
                current = None
                i = window_end
                continue
            curve_used = True
            curve_blocks += 1
        com_name, avg_score, avg_offset = vote_window(
            i, window_end, timestamps, neighbor_idx, neighbor_scores, video_idx, commercial_ts, commercial_names
        )
        score = avg_score
        if score < args.score_threshold:
            result = finalize_track(current, timestamps, commercial_durations, args)
            if result:
                detections.append(result)
            current = None
            i = window_end
            continue

        offset = float(timestamps[i] - avg_offset)

        if (
            current
            and current.commercial == com_name
            and abs(offset - current.offset) <= args.offset_tolerance
            and i == current.last_idx + 1
        ):
            current.last_idx = window_end - 1
            current.offset = 0.5 * (current.offset + offset)
            current.best_score = max(current.best_score, score)
            i = window_end
            continue

        result = finalize_track(current, timestamps, commercial_durations, args)
        if result:
            detections.append(result)

        current = Track(
            commercial=com_name,
            start_idx=i,
            last_idx=window_end - 1,
            offset=offset,
            best_score=score,
        )
        i = window_end

    # finalize last track
    result = finalize_track(current, timestamps, commercial_durations, args)
    if result:
        detections.append(result)

    if curve_signal is not None:
        total_windows = curve_blocks + curve_rejections
        if total_windows:
            print(
                f"[curve] ventanas activas={curve_blocks} filtradas={curve_rejections} "
                f"({curve_rejections/total_windows:.1%} descartadas)"
            )
    detections = deduplicate_by_gap(detections, args.min_gap)

    rows: List[str] = []
    for det in detections:
        if args.include_score:
            rows.append(
                f"{tv_name}\t{round(det['start_time'], 3)}\t{round(det['duration'], 3)}\t{det['commercial']}\t{round(det['score'], 4)}"
            )
        else:
            rows.append(f"{tv_name}\t{round(det['start_time'], 3)}\t{round(det['duration'], 3)}\t{det['commercial']}")
    return rows


def main() -> None:
    args = parse_args()
    manifest = load_manifest(Path(args.manifest))
    knn_root = Path(args.knn_dir)
    detections: List[str] = []
    for npz_path in sorted(knn_root.glob("tv/*_knn.npz")):
        tv_name = npz_path.stem.replace("_knn", "")
        rows = detect_in_video(tv_name, npz_path, manifest, args)
        detections.extend(rows)

    if not detections:
        print("No se generaron detecciones (revise los parámetros).")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        header = "# television\tinicio_seg\tlargo_seg\tcomercial"
        if args.include_score:
            header += "\tscore"
        handle.write(header + "\n")
        for row in detections:
            handle.write(row + "\n")
    print(f"Detecciones generadas: {len(detections)} en {output_path}")


if __name__ == "__main__":
    main()
