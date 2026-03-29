from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    import gudhi as gd
except ImportError as exc:  # pragma: no cover
    raise ImportError("Se requiere gudhi para embedding_tda.py") from exc

from .io_utils import ensure_dir, load_npz


def persistence_stats(diagram, target_dim: int) -> Dict[str, float]:
    pers = [death - birth for dim, (birth, death) in diagram if dim == target_dim and np.isfinite(death)]
    if not pers:
        return {"count": 0.0, "sum": 0.0, "max": 0.0, "std": 0.0}
    pers_arr = np.array(pers, dtype=np.float32)
    return {
        "count": float(len(pers_arr)),
        "sum": float(pers_arr.sum()),
        "max": float(pers_arr.max()),
        "std": float(pers_arr.std()),
    }


def rolling_zscore(signal: np.ndarray, window: int = 25) -> np.ndarray:
    if signal.size == 0:
        return signal
    kernel = np.ones(window, dtype=np.float32) / window
    mean = np.convolve(signal, kernel, mode="same")
    mean_sq = np.convolve(signal ** 2, kernel, mode="same")
    var = np.maximum(mean_sq - mean ** 2, 1e-6)
    return (signal - mean) / np.sqrt(var)


def window_features(points: np.ndarray) -> Dict[str, float]:
    rips = gd.RipsComplex(points=points.tolist())
    st = rips.create_simplex_tree(max_dimension=2)
    diagram = st.persistence(homology_coeff_field=2, min_persistence=1e-4)
    h0 = persistence_stats(diagram, 0)
    h1 = persistence_stats(diagram, 1)
    norms = np.linalg.norm(points, axis=1)
    motion = np.linalg.norm(np.diff(points, axis=0), axis=1) if points.shape[0] > 1 else np.zeros((1,))
    return {
        "h0_count": h0["count"],
        "h0_sum_persistence": h0["sum"],
        "h0_max_persistence": h0["max"],
        "h0_std_persistence": h0["std"],
        "h1_count": h1["count"],
        "h1_sum_persistence": h1["sum"],
        "h1_max_persistence": h1["max"],
        "h1_std_persistence": h1["std"],
        "embedding_energy_mean": float(norms.mean()),
        "embedding_energy_std": float(norms.std()),
        "window_motion": float(motion.mean()) if motion.size else 0.0,
    }


def process_file(npz_path: Path, output_path: Path, window_frames: int, stride: int) -> Dict:
    payload = load_npz(npz_path)
    embeddings = payload["embeddings"]
    timestamps = payload["timestamps_sec"]
    feature_rows: List[List[float]] = []
    feature_labels = [
        "h0_count",
        "h0_sum_persistence",
        "h0_max_persistence",
        "h0_std_persistence",
        "h1_count",
        "h1_sum_persistence",
        "h1_max_persistence",
        "h1_std_persistence",
        "embedding_energy_mean",
        "embedding_energy_std",
        "window_motion",
        "combined_activity",
        "combined_activity_z",
    ]
    centers = []
    combined_values = []
    for start in range(0, max(embeddings.shape[0] - window_frames + 1, 1), stride):
        end = start + window_frames
        if end > embeddings.shape[0]:
            break
        window = embeddings[start:end]
        feats = window_features(window)
        combined = feats["h1_sum_persistence"] + feats["window_motion"]
        combined_values.append(combined)
        center_idx = start + window_frames // 2
        centers.append(timestamps[min(center_idx, len(timestamps) - 1)])
        feature_rows.append([
            feats["h0_count"],
            feats["h0_sum_persistence"],
            feats["h0_max_persistence"],
            feats["h0_std_persistence"],
            feats["h1_count"],
            feats["h1_sum_persistence"],
            feats["h1_max_persistence"],
            feats["h1_std_persistence"],
            feats["embedding_energy_mean"],
            feats["embedding_energy_std"],
            feats["window_motion"],
            combined,
            0.0,
        ])
    features = np.array(feature_rows, dtype=np.float32)
    if features.size == 0:
        features = np.zeros((0, len(feature_labels)), dtype=np.float32)
    combined_z = rolling_zscore(np.array(combined_values, dtype=np.float32), window=25)
    if features.shape[0] == combined_z.shape[0]:
        features[:, -1] = combined_z
    np.savez_compressed(
        output_path,
        timestamps_sec=np.array(centers, dtype=np.float32),
        tda_features=features,
        feature_labels=np.array(feature_labels, dtype=np.str_),
        video_name=payload["video_name"],
        category=payload["category"],
    )
    return {
        "video_name": str(payload["video_name"]),
        "category": str(payload["category"]),
        "num_windows": int(features.shape[0]),
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir descriptores TDA sobre embeddings")
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--window_frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    output_dir = args.output_dir
    tv_out = output_dir / "tv"
    commercials_out = output_dir / "commercials"
    ensure_dir(tv_out)
    ensure_dir(commercials_out)

    manifest = []
    for category in ("tv", "commercials"):
        for npz_path in sorted((args.embeddings_dir / category).glob("*_embeddings.npz")):
            out_root = tv_out if category == "tv" else commercials_out
            out_path = out_root / f"{npz_path.stem}_embedding_tda.npz"
            rec = process_file(npz_path, out_path, args.window_frames, args.stride)
            manifest.append(rec)
    manifest_path = output_dir / "manifest_embedding_tda.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[embedding_tda] manifest en {manifest_path}")


if __name__ == "__main__":
    main()
