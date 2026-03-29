#!/usr/bin/env python3
"""Barrido reproducible de hiperparámetros para knn_detector.py."""
from __future__ import annotations

import argparse
import csv
import itertools
import re
import subprocess
from pathlib import Path
from typing import Dict, List

PARAM_GRID = {
    "score_threshold": [0.20, 0.24, 0.28, 0.32],
    "offset_tolerance": [1.0, 1.5, 2.0],
    "coverage_ratio": [0.05, 0.08, 0.10, 0.15],
    "window_sec": [0.5, 0.8, 1.0],
    "curve_threshold": [1.0, 1.5, 2.0],
    "min_gap": [0.0, 3.0, 5.0],
}


METRIC_PATTERN = {
    "precision": re.compile(r"Precision=(\d+\.\d+|\d+)%"),
    "recall": re.compile(r"Recall=(\d+\.\d+|\d+)%"),
    "f1": re.compile(r"F1=(\d+\.\d+|\d+)"),
    "iou": re.compile(r"IoU=(\d+\.\d+|\d+)"),
    "task_score": re.compile(r"Resultado Tarea=(\d+\.\d+|\d+)%"),
}
COUNT_PATTERN = re.compile(
    r"Correctas=(\d+)\s+Incorrectas=(\d+)\s+Repetidas=(\d+)\s+Total_GT=(\d+)"
)


def run_command(cmd: List[str]) -> str:
    """Run command and return stdout as string."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout + result.stderr


def parse_metrics(output: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, pattern in METRIC_PATTERN.items():
        match = pattern.search(output)
        if match:
            value = float(match.group(1))
            if key in {"precision", "recall", "task_score"}:
                value /= 100.0
            metrics[key] = value
    count_match = COUNT_PATTERN.search(output)
    if count_match:
        metrics["correctas"] = int(count_match.group(1))
        metrics["incorrectas"] = int(count_match.group(2))
        metrics["repetidas"] = int(count_match.group(3))
        metrics["total_gt"] = int(count_match.group(4))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search para knn_detector.py")
    parser.add_argument("--knn_detector", default="pipeline/detection/knn_detector.py", help="Ruta al script detector")
    parser.add_argument("--evaluator", default="evaluar-v2.py", help="Ruta al evaluador oficial")
    parser.add_argument("--knn_dir", required=True, help="Directorio con npz *_knn")
    parser.add_argument("--manifest", required=True, help="Manifest JSON de preprocesamiento")
    parser.add_argument("--curve_dir", required=True, help="Directorio con curvas topológicas")
    parser.add_argument("--gt", required=True, help="Archivo gt.txt")
    parser.add_argument("--results_csv", default="grid_search_results.csv", help="Archivo CSV de salida")
    parser.add_argument(
        "--detections_dir",
        default="pipeline/detection/grid_runs",
        help="Carpeta temporal donde se guardarán las detecciones",
    )
    parser.add_argument("--include_score", action="store_true", help="Propaga --include_score al detector")
    args = parser.parse_args()

    detections_root = Path(args.detections_dir)
    detections_root.mkdir(parents=True, exist_ok=True)

    csv_path = Path(args.results_csv)
    fieldnames = list(PARAM_GRID.keys()) + [
        "precision",
        "recall",
        "f1",
        "iou",
        "task_score",
        "correctas",
        "incorrectas",
        "repetidas",
        "total_gt",
    ]

    combos = list(itertools.product(*PARAM_GRID.values()))
    keys = list(PARAM_GRID.keys())

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx, values in enumerate(combos):
            params = dict(zip(keys, values))
            det_path = detections_root / f"detecciones_{idx}.txt"
            detector_cmd: List[str] = [
                "python",
                args.knn_detector,
                "--knn_dir",
                args.knn_dir,
                "--manifest",
                args.manifest,
                "--output",
                str(det_path),
                "--curve_dir",
                args.curve_dir,
            ]
            if args.include_score:
                detector_cmd.append("--include_score")
            for key, value in params.items():
                detector_cmd.extend([f"--{key}", str(value)])
            print(f"[grid] Ejecutando combinación {idx+1}/{len(combos)}: {params}")
            try:
                run_command(detector_cmd)
            except subprocess.CalledProcessError as exc:
                print(f"[grid] Detector falló con código {exc.returncode}: {exc.stderr}")
                continue

            eval_cmd = ["python", args.evaluator, str(det_path), args.gt]
            try:
                eval_output = run_command(eval_cmd)
            except subprocess.CalledProcessError as exc:
                print(f"[grid] Evaluación falló: {exc.stderr}")
                continue
            metrics = parse_metrics(eval_output)
            row = {**params, **metrics}
            writer.writerow(row)
            csvfile.flush()


if __name__ == "__main__":
    main()
