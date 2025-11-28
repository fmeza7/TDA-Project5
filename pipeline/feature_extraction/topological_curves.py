#!/usr/bin/env python3
"""
Generación de curvas topológicas a partir de los descriptores cúbicos por frame.

Entrada: NPZs producidos por pipeline/preprocessing/cubical_preprocessing.py.
Salida: NPZs con señales 1D (curvas) que resumen la dinámica topológica por frame.

Cada curva representa un indicador intuitivo:
  - h0_count, h0_sum, h0_max, h0_std
  - h1_count, h1_sum, h1_max, h1_std
  - brightness_mean, brightness_std
  - pi_energy (norma L2 de la persistence image)
  - combined_activity (h1_sum + pi_energy) para detectar picos.

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
    parser.add_argument("--overwrite", action="store_true", help="Regenera curvas aunque exista el archivo destino")
    return parser.parse_args()


def iter_npz_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for path in sorted(root.glob("*.npz")):
        if path.is_file():
            yield path


def moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return signal
    pad = window // 2
    padded = np.pad(signal, pad_width=((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones((window, 1), dtype=np.float32) / window
    return np.apply_along_axis(lambda col: np.convolve(col, kernel[:, 0], mode="valid"), axis=0, arr=padded)


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
    "pi_energy",
    "combined_activity",
]


def compute_curves(features: np.ndarray) -> np.ndarray:
    if features.size == 0:
        return np.zeros((0, len(CURVE_LABELS)), dtype=np.float32)
    h0 = features[:, 0:5]
    h1 = features[:, 5:10]
    brightness = features[:, 10:12]
    pi = features[:, 12:]
    pi_energy = np.linalg.norm(pi, axis=1, keepdims=True)
    curves = np.hstack(
        [
            h0[:, [0, 1, 2, 4]],  # count, sum, max, std
            h1[:, [0, 1, 2, 4]],
            brightness,
            pi_energy,
        ]
    )
    combined = (h1[:, 1:2] + pi_energy).astype(np.float32)
    return np.hstack([curves, combined]).astype(np.float32)


def process_video(npz_path: Path, output_path: Path, smooth_window: int, overwrite: bool) -> Dict:
    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path.name} existe")
        return {}
    with np.load(npz_path) as data:
        timestamps = data["timestamps_sec"].astype(np.float32)
        features = data["tda_features"].astype(np.float32)
    curves = compute_curves(features)
    if smooth_window > 1 and curves.size:
        curves = moving_average(curves, smooth_window)
    payload = {
        "timestamps_sec": timestamps[: curves.shape[0]],
        "curve_signals": curves,
        "curve_labels": np.array(CURVE_LABELS, dtype=np.str_),
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
            res = process_video(npz_path, out_path, args.smooth_window, args.overwrite)
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
