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


def rolling_zscore(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0:
        return signal
    if window <= 1 or signal.size <= window:
        mean = signal.mean()
        std = signal.std()
        std = std if std > 1e-6 else 1.0
        return ((signal - mean) / std).astype(np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    mean = np.convolve(signal, kernel, mode="same")
    sq_mean = np.convolve(signal ** 2, kernel, mode="same")
    var = np.maximum(sq_mean - mean ** 2, 1e-6)
    std = np.sqrt(var, dtype=np.float32)
    return ((signal - mean) / std).astype(np.float32)


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
    window_idx = neighbor_idx[start_idx:end_idx]
    window_scores = neighbor_scores[start_idx:end_idx]
    if window_idx.size == 0:
        return "", 0.0, 0.0
    flat_idx = window_idx.reshape(-1)
    flat_scores = window_scores.reshape(-1)
    repeats = window_idx.shape[1]
    tv_times = np.repeat(tv_timestamps[start_idx:end_idx], repeats)
    score_by_name: Dict[str, float] = {}
    count_by_name: Dict[str, int] = {}
    offsets_by_name: Dict[str, List[float]] = {}
    for idx_val, sc, tv_time in zip(flat_idx, flat_scores, tv_times):
        vid_id = int(video_idx[idx_val])
        name = commercial_names[vid_id]
        score_by_name[name] = score_by_name.get(name, 0.0) + float(sc)
        count_by_name[name] = count_by_name.get(name, 0) + 1
        offsets_by_name.setdefault(name, []).append(float(tv_time - commercial_ts[idx_val]))
    if not score_by_name:
        return "", 0.0, 0.0
    best_name = max(score_by_name, key=score_by_name.get)
    avg_score = score_by_name[best_name] / max(count_by_name.get(best_name, 1), 1)
    offset_list = offsets_by_name.get(best_name, [])
    avg_offset = float(np.mean(offset_list)) if offset_list else 0.0
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
    parser.add_argument(
        "--curve_z_window",
        type=int,
        default=15,
        help="Ventana (frames) para calcular z-score si no existe combined_activity_z en las curvas",
    )
    parser.add_argument(
        "--merge_overlap_ratio",
        type=float,
        default=0.5,
        help="Solapamiento mínimo (IoU) para fusionar detecciones superpuestas del mismo comercial",
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
    if timestamps.size > 1:
        median_dt = float(np.median(np.diff(timestamps)))
    else:
        median_dt = 0.0
    observed_duration = max((end_time - start_time) + max(median_dt, 0.0), max(median_dt, 0.0))
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


def merge_overlapping_detections(dets: List[Dict], overlap_ratio: float) -> List[Dict]:
    if not dets:
        return dets
    merged: List[Dict] = []
    dets_sorted = sorted(dets, key=lambda d: (d["commercial"], d["start_time"]))
    for det in dets_sorted:
        if not merged:
            merged.append(det)
            continue
        prev = merged[-1]
        if prev["commercial"] != det["commercial"]:
            merged.append(det)
            continue
        prev_end = prev["start_time"] + prev["duration"]
        det_end = det["start_time"] + det["duration"]
        inter = min(prev_end, det_end) - max(prev["start_time"], det["start_time"])
        union = max(prev_end, det_end) - min(prev["start_time"], det["start_time"])
        iou = inter / union if inter > 0 and union > 0 else 0.0
        if iou >= overlap_ratio:
            if det["score"] > prev["score"]:
                merged[-1] = det
        else:
            merged.append(det)
    return merged


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
        manifest_path = Path(args.curve_dir) / "manifest_curves.json"
        curve_manifest = json.loads(manifest_path.read_text())
        curve_map = {
            Path(entry["source_path"]).stem: entry["output_path"]
            for entry in curve_manifest
            if entry["category"] == "tv"
        }
        curve_rel = curve_map.get(tv_name)
        if curve_rel:
            curve_path = Path(curve_rel)
            if not curve_path.is_absolute():
                curve_path = (Path(args.curve_dir) / curve_path).resolve()
            if curve_path.exists():
                curve_data = np.load(curve_path)
                labels = [str(x) for x in curve_data["curve_labels"]]
                if "combined_activity_z" in labels:
                    idx = labels.index("combined_activity_z")
                    curve_signal = curve_data["curve_signals"][:, idx].astype(np.float32)
                    curve_loaded = True
                elif "combined_activity" in labels:
                    idx = labels.index("combined_activity")
                    raw_signal = curve_data["curve_signals"][:, idx].astype(np.float32)
                    curve_signal = rolling_zscore(raw_signal, max(args.curve_z_window, 1))
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
        if not com_name or score < args.score_threshold:
            result = finalize_track(current, timestamps, commercial_durations, args)
            if result:
                detections.append(result)
            current = None
            i = window_end
            continue

        offset = float(avg_offset)

        if (
            current
            and current.commercial == com_name
            and abs(offset - current.offset) <= args.offset_tolerance
            and i <= current.last_idx + 1
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
    detections = merge_overlapping_detections(detections, args.merge_overlap_ratio)

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
