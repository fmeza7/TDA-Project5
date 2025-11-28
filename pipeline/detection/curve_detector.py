#!/usr/bin/env python3
"""
Detección de comerciales usando curvas topológicas (solo video).

Algoritmo:
  1. Carga las curvas generadas en pipeline/feature_extraction/topological_curves.py.
  2. Selecciona la señal "combined_activity" y la convierte a z-score con media móvil.
  3. Detecta segmentos donde el z-score supera un umbral (picos/anomalías).
  4. Para cada pico, compara la forma de la curva con la de cada comercial mediante
     correlación coseno normalizada y selecciona el mejor match.
  5. Genera detecciones con inicio/duración y las escribe en detecciones.txt.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class CurveEntry:
    name: str
    category: str
    timestamps: np.ndarray
    raw_signal: np.ndarray
    z_signal: np.ndarray
    sample_fps: float
    duration_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detección basada en curvas topológicas (picos + matching)")
    parser.add_argument(
        "--curves_dir",
        type=str,
        required=True,
        help="Directorio raíz de las curvas (contiene tv/ y commercials/)",
    )
    parser.add_argument(
        "--preproc_manifest",
        type=str,
        required=True,
        help="Manifest JSON generado en la etapa de preprocesamiento cúbico",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detecciones.txt",
        help="Archivo de salida con las detecciones (formato requerido por la tarea)",
    )
    parser.add_argument("--smooth_window", type=int, default=30, help="Ventana para media móvil (frames)")
    parser.add_argument("--z_threshold", type=float, default=2.5, help="Umbral de z-score para detectar picos")
    parser.add_argument(
        "--min_peak_sec",
        type=float,
        default=4.0,
        help="Duración mínima (segundos) de un segmento activo para considerarlo candidato",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.6,
        help="Umbral mínimo de correlación coseno para aceptar el match con un comercial",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> Dict[str, Dict]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    mapping: Dict[str, Dict] = {}
    for record in records:
        # key: nombre base del archivo (sin ext) + categoria
        source = Path(record["source_path"]).stem
        key = f"{record['category']}::{source}"
        mapping[key] = record
    return mapping


def iter_curve_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in sorted(root.glob("*_curves.npz")):
        if path.is_file():
            yield path


def extract_signal(npz_path: Path, label: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    curves = data["curve_signals"].astype(np.float32)
    labels = data["curve_labels"]
    timestamps = data["timestamps_sec"].astype(np.float32)
    idx = None
    for i, lbl in enumerate(labels):
        if lbl == label:
            idx = i
            break
    if idx is None:
        raise ValueError(f"No se encontró la señal {label} en {npz_path}")
    return timestamps, curves[:, idx]


def moving_stats(signal: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    if signal.size == 0:
        return signal, signal
    if window <= 1 or signal.size < window:
        std = signal.std()
        std = std if std > 1e-6 else 1.0
        return (
            np.full_like(signal, signal.mean(), dtype=np.float32),
            np.full_like(signal, std, dtype=np.float32),
        )
    kernel = np.ones(window, dtype=np.float32) / window
    mean = np.convolve(signal, kernel, mode="same")
    sq_mean = np.convolve(signal ** 2, kernel, mode="same")
    var = np.maximum(sq_mean - mean ** 2, 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


def zscore_signal(signal: np.ndarray, window: int) -> np.ndarray:
    if signal.size == 0:
        return signal
    mean, std = moving_stats(signal, window)
    return (signal - mean) / std


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    if vec.size == 0:
        return vec
    v = vec.astype(np.float32)
    v -= v.mean()
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return np.zeros_like(v)
    return v / norm


def detect_segments(signal: np.ndarray, z_threshold: float, min_len_frames: int) -> List[Tuple[int, int, int]]:
    active = signal > z_threshold
    segments: List[Tuple[int, int, int]] = []
    i = 0
    n = len(signal)
    while i < n:
        if not active[i]:
            i += 1
            continue
        start = i
        while i < n and active[i]:
            i += 1
        end = i
        if end - start < min_len_frames:
            continue
        peak_rel = np.argmax(signal[start:end])
        peak_idx = start + int(peak_rel)
        segments.append((start, end, peak_idx))
    return segments


def load_curves(args: argparse.Namespace, manifest_map: Dict[str, Dict]) -> Dict[str, Dict[str, CurveEntry]]:
    curves_root = Path(args.curves_dir)
    entries: Dict[str, Dict[str, CurveEntry]] = {"tv": {}, "commercials": {}}
    for category in entries.keys():
        directory = curves_root / category
        for curve_path in iter_curve_files(directory):
            timestamps, signal = extract_signal(curve_path, "combined_activity")
            name = curve_path.stem.replace("_curves", "")
            manifest_key = f"{category}::{name}"
            manifest_entry = manifest_map.get(manifest_key)
            if manifest_entry is None:
                print(f"[WARN] No hay metadatos para {manifest_key}")
                continue
            sample_fps = float(manifest_entry.get("sampled_fps", 3.0))
            duration_sec = float(manifest_entry.get("duration_sec", timestamps[-1] if timestamps.size else 0.0))
            entries[category][name] = CurveEntry(
                name=name,
                category=category,
                timestamps=timestamps,
                raw_signal=signal,
                z_signal=zscore_signal(signal, args.smooth_window),
                sample_fps=sample_fps,
                duration_sec=duration_sec,
            )
    return entries


def match_candidate(
    tv_signal: np.ndarray,
    tv_timestamps: np.ndarray,
    peak_idx: int,
    commercials: Dict[str, CurveEntry],
) -> Tuple[str, float, int] | None:
    best_name = ""
    best_score = -np.inf
    best_start = 0
    for entry in commercials.values():
        com_signal = normalize_vector(entry.z_signal)
        if com_signal.size == 0:
            continue
        length = com_signal.size
        if tv_signal.size < length:
            continue
        start = max(0, min(peak_idx - length // 2, tv_signal.size - length))
        segment = tv_signal[start : start + length]
        norm_segment = normalize_vector(segment)
        score = float(np.dot(norm_segment, com_signal))
        if score > best_score:
            best_score = score
            best_name = entry.name
            best_start = start
    if best_score <= -np.inf:
        return None
    return best_name, best_score, best_start


def main() -> None:
    args = parse_args()
    manifest_map = load_manifest(Path(args.preproc_manifest))
    curves = load_curves(args, manifest_map)
    commercials = curves["commercials"]
    if not curves["tv"] or not commercials:
        raise RuntimeError("No se encontraron curvas para TV o comerciales.")

    detections: List[str] = []
    for tv_entry in curves["tv"].values():
        z_signal = tv_entry.z_signal
        min_len_frames = max(int(args.min_peak_sec * tv_entry.sample_fps), 1)
        segments = detect_segments(z_signal, args.z_threshold, min_len_frames)
        if not segments:
            continue
        for start, end, peak_idx in segments:
            match = match_candidate(z_signal, tv_entry.timestamps, peak_idx, commercials)
            if match is None:
                continue
            com_name, score, best_start = match
            if score < args.match_threshold:
                continue
            com_entry = commercials[com_name]
            start_time = float(tv_entry.timestamps[min(best_start, len(tv_entry.timestamps) - 1)])
            duration = float(com_entry.duration_sec)
            detections.append(
                f"{tv_entry.name}\t{round(start_time, 3)}\t{round(duration, 3)}\t{com_name}\t{round(score, 4)}"
            )
            print(f"[hit] {tv_entry.name} -> {com_name} @ {start_time:.2f}s score={score:.3f}")

    if not detections:
        print("No se generaron detecciones (¿umbrales demasiado altos?).")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# television\tinicio_seg\tlargo_seg\tcomercial\tscore\n")
        for row in detections:
            handle.write(row + "\n")
    print(f"Se escribieron {len(detections)} detecciones en {output_path}")


if __name__ == "__main__":
    main()
