from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .io_utils import ensure_dir, load_npz


def build_bank(tda_dir: Path, normalize: bool) -> Dict:
    bank_vectors = []
    timestamps = []
    video_names = []
    video_lookup: List[str] = []
    video_to_idx: Dict[str, int] = {}
    for npz_path in sorted((tda_dir / "commercials").glob("*_embedding_tda.npz")):
        payload = load_npz(npz_path)
        feats = payload["tda_features"].astype(np.float32)
        times = payload["timestamps_sec"]
        name = str(payload["video_name"])
        if name not in video_to_idx:
            video_to_idx[name] = len(video_lookup)
            video_lookup.append(name)
        idx = video_to_idx[name]
        bank_vectors.append(feats)
        timestamps.append(times)
        video_names.append(np.full(shape=(feats.shape[0],), fill_value=idx, dtype=np.int32))
    if not bank_vectors:
        raise RuntimeError("No hay comerciales en el banco de TDA de embeddings")
    bank_matrix = np.vstack(bank_vectors)
    time_array = np.concatenate(timestamps)
    video_idx_array = np.concatenate(video_names)
    if normalize:
        norms = np.linalg.norm(bank_matrix, axis=1, keepdims=True) + 1e-8
        bank_matrix = bank_matrix / norms
    return {
        "vectors": bank_matrix,
        "timestamps": time_array,
        "video_idx": video_idx_array,
        "video_lookup": video_lookup,
    }


def compute_knn(tv_features: np.ndarray, bank: Dict, k: int, normalize: bool):
    feats = tv_features.astype(np.float32)
    if normalize:
        feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    scores = feats @ bank["vectors"].T
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    video_idx = bank["video_idx"][top_idx]
    timestamps = bank["timestamps"][top_idx]
    video_names = np.vectorize(bank["video_lookup"].__getitem__)(video_idx)
    return top_idx, top_scores, video_idx, timestamps, video_names


def main() -> None:
    parser = argparse.ArgumentParser(description="k-NN sobre TDA de embeddings")
    parser.add_argument("--tda_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    ensure_dir(args.output_dir / "tv")
    bank = build_bank(args.tda_dir, args.normalize)
    manifest = []
    for npz_path in sorted((args.tda_dir / "tv").glob("*_embedding_tda.npz")):
        payload = load_npz(npz_path)
        knn_idx, knn_scores, video_idx, timestamps, video_names = compute_knn(
            payload["tda_features"], bank, args.k, args.normalize
        )
        out_path = args.output_dir / "tv" / f"{npz_path.stem}_knn.npz"
        np.savez_compressed(
            out_path,
            neighbor_indices=knn_idx,
            neighbor_scores=knn_scores,
            commercial_video_idx=video_idx,
            commercial_timestamps=timestamps,
            commercial_video_names=video_names,
            timestamps_sec=payload["timestamps_sec"],
            video_name=payload["video_name"],
        )
        manifest.append({
            "video_name": str(payload["video_name"]),
            "path": str(out_path),
            "num_windows": int(payload["tda_features"].shape[0]),
        })
    manifest_path = args.output_dir / "manifest_embedding_knn.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[embedding_knn] manifest en {manifest_path}")


if __name__ == "__main__":
    main()
