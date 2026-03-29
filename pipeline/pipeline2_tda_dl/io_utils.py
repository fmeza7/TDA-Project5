from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GT_COLUMNS = ["television", "inicio_seg", "largo_seg", "comercial"]


@dataclass
class CurvePayload:
    timestamps: np.ndarray
    signals: np.ndarray
    labels: List[str]


def load_curve_npz(npz_path: Path) -> CurvePayload:
    data = np.load(npz_path)
    return CurvePayload(
        timestamps=data["timestamps_sec"].astype(np.float32),
        signals=data["curve_signals"].astype(np.float32),
        labels=[str(x) for x in data["curve_labels"]],
    )


def load_curve_manifest(manifest_path: Path) -> Dict[str, str]:
    manifest = json.loads(manifest_path.read_text())
    mapping: Dict[str, str] = {}
    for entry in manifest:
        stem = Path(entry["source_path"]).stem
        mapping[stem] = entry["output_path"]
    return mapping


def load_preproc_manifest(manifest_path: Path) -> Dict[str, Dict]:
    manifest = json.loads(manifest_path.read_text())
    mapping: Dict[str, Dict] = {}
    for entry in manifest:
        stem = Path(entry["source_path"]).stem
        mapping[stem] = entry
    return mapping


def parse_gt_file(gt_path: Path) -> List[Dict]:
    samples: List[Dict] = []
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            tv, start, duration, commercial = parts[:4]
            start_t = float(start)
            duration_t = float(duration)
            samples.append(
                {
                    "television": tv,
                    "start_time": start_t,
                    "duration": duration_t,
                    "end_time": start_t + duration_t,
                    "commercial": commercial,
                }
            )
    return samples


def build_commercial_class_map(curves_dir: Path) -> Dict[str, int]:
    commercials_dir = curves_dir / "commercials"
    class_names = sorted({path.stem.replace("_curves", "") for path in commercials_dir.glob("*_curves.npz")})
    mapping = {"__background__": 0}
    for idx, name in enumerate(class_names, start=1):
        mapping[name] = idx
    return mapping
