#!/usr/bin/env python3
"""
Generación de curvas topológicas a partir de los descriptores cúbicos por frame.

Entrada: NPZs producidos por pipeline/preprocessing/cubical_preprocessing.py.
Salida: NPZs con señales 1D (curvas) que resumen la dinámica topológica por frame.

Cada curva representa un indicador intuitivo:
  - h0_count, h0_sum, h0_max, h0_std
  - h1_count, h1_sum, h1_max, h1_std
  - brightness_mean, brightness_std
  - pi_h0_energy / pi_h1_energy (normas L2 por homología)
  - combined_activity (h1_sum + pi_h1_energy) y su z-score asociado.

Opcionalmente se puede aplicar un suavizado por ventana móvil para atenuar ruido
frame a frame.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generar curvas topológicas desde descriptores cúbicos por frame")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directorio raíz generado por el preprocesamiento cúbico (contiene subcarpetas tv/ y commercials/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="pipeline_outputs/curves",
        help="Ruta destino para las curvas topológicas",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=0,
        help="Tamaño de ventana (en frames) para suavizar con media móvil (0 = sin suavizado)",
    )
    parser.add_argument(
        "--pi_dim",
        type=int,
        default=256,
        help="Dimensión (número de celdas) de cada Persistence Image por homología",
    )
    parser.add_argument(
        "--z_window",
        type=int,
        default=15,
        help="Ventana (frames) para calcular el z-score rodante de combined_activity",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenera curvas aunque exista el archivo destino")
    return parser.parse_args()


def iter_npz_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in sorted(root.glob("*.npz")):
        if path.is_file():
            yield path


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or signal.size == 0:
        return signal
    kernel = np.ones(window, dtype=np.float32) / float(window)

    def _smooth(col: np.ndarray) -> np.ndarray:
        return np.convolve(col, kernel, mode="same")

    return np.apply_along_axis(_smooth, axis=0, arr=signal).astype(np.float32)


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


CURVE_LABELS = [
    "h0_count",
    "h0_sum_persistence",
    "h0_max_persistence",
    "h0_std_persistence",
    "h1_count",
    "h1_sum_persistence",
    "h1_max_persistence",
    "h1_std_persistence",
    "brightness_mean",
    "brightness_std",
    "pi_h0_energy",
    "pi_h1_energy",
    "combined_activity",
    "combined_activity_delta",
    "brightness_delta",
]


def compute_curves(features: np.ndarray, pi_dim: int) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, len(CURVE_LABELS)), dtype=np.float32)
    h0 = features[:, 0:5]
    h1 = features[:, 5:10]
    brightness = features[:, 10:12]
    pi_h0 = features[:, 12 : 12 + pi_dim]
    pi_h1 = features[:, 12 + pi_dim : 12 + 2 * pi_dim]
    pi_h0_energy = np.linalg.norm(pi_h0, axis=1, keepdims=True)
    pi_h1_energy = np.linalg.norm(pi_h1, axis=1, keepdims=True)
    combined_raw = (h1[:, 1:2] + pi_h1_energy).astype(np.float32)
    combined_delta = np.vstack([np.zeros((1, 1), dtype=np.float32), np.abs(np.diff(combined_raw, axis=0))])
    brightness_delta = np.vstack(
        [np.zeros((1, 1), dtype=np.float32), np.abs(np.diff(brightness[:, 0:1], axis=0))]
    )
    curves = np.hstack(
        [
            h0[:, [0, 1, 2, 4]],  # count, sum, max, std
            h1[:, [0, 1, 2, 4]],
            brightness,
            pi_h0_energy,
            pi_h1_energy,
            combined_raw,
            combined_delta,
            brightness_delta,
        ]
    )
    return curves.astype(np.float32)


def process_video(
    npz_path: Path,
    output_path: Path,
    smooth_window: int,
    z_window: int,
    pi_dim: int,
    overwrite: bool,
) -> Dict:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path.name} existe")
        return {}
    with np.load(npz_path) as data:
        timestamps = data["timestamps_sec"].astype(np.float32)
        features = data["tda_features"].astype(np.float32)
    curves = compute_curves(features, pi_dim)
    if smooth_window > 1 and curves.size:
        curves = moving_average(curves, smooth_window)
    labels = list(CURVE_LABELS)
    if curves.size:
        combined_idx = labels.index("combined_activity")
        combined_z = rolling_zscore(curves[:, combined_idx], max(z_window, 1))
        curves = np.hstack([curves, combined_z[:, None].astype(np.float32)])
    else:
        curves = np.hstack([curves, np.zeros((curves.shape[0], 1), dtype=np.float32)])
    labels.append("combined_activity_z")
    payload = {
        "timestamps_sec": timestamps[: curves.shape[0]],
        "curve_signals": curves,
        "curve_labels": np.array(labels, dtype=np.str_),
        "source_features": np.array([npz_path.name], dtype=np.str_),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    return {
        "source": str(npz_path),
        "output": str(output_path),
        "frames": curves.shape[0],
    }


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict] = []
    for category in ("tv", "commercials"):
        in_dir = input_root / category
        out_dir = output_root / category
        for npz_path in iter_npz_files(in_dir):
            out_path = out_dir / f"{npz_path.stem}_curves.npz"
            res = process_video(
                npz_path,
                out_path,
                args.smooth_window,
                args.z_window,
                args.pi_dim,
                args.overwrite,
            )
            if res:
                res["category"] = category
                summaries.append(res)
    if summaries:
        manifest_path = output_root / "manifest_curves.json"
        json_ready = [
            {"category": s["category"], "source_path": s["source"], "output_path": s["output"], "frames": s["frames"]}
            for s in summaries
        ]
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(json_ready, handle, indent=2, ensure_ascii=False)
        print(f"Curvas generadas. Manifest en {manifest_path}")
    else:
        print("No se generaron curvas (verifique que existan NPZ de entrada).")


if __name__ == "__main__":
    main()
