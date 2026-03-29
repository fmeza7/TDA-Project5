from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .temporal_dataset import build_sequence_records, summarize_records


def _summarize_npz(npz_path: Path) -> dict:
    if not npz_path.exists():
        return {"error": f"missing {npz_path}"}
    data = np.load(npz_path, allow_pickle=True)
    labels = data.get("label_id")
    videos = data.get("video_name")
    source_types = data.get("source_type")
    summary = {
        "num_rows": int(labels.shape[0]) if labels is not None else 0,
        "labels": {},
        "videos": {},
        "source_types": {},
    }
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        summary["labels"] = {int(k): int(v) for k, v in zip(unique, counts)}
    if videos is not None:
        unique, counts = np.unique(videos, return_counts=True)
        summary["videos"] = {str(k): int(v) for k, v in zip(unique, counts)}
    if source_types is not None:
        unique, counts = np.unique(source_types, return_counts=True)
        summary["source_types"] = {str(k): int(v) for k, v in zip(unique, counts)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditar datasets y latentes del pipeline 2")
    parser.add_argument("--window_data_dir", type=Path, required=True)
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--seq_len", type=int, default=9)
    args = parser.parse_args()

    topoae_npz = args.window_data_dir / "topoae_dataset.npz"
    temporal_npz = args.window_data_dir / "temporal_dataset.npz"
    print("[debug_dataset] topoae_dataset:", _summarize_npz(topoae_npz))
    print("[debug_dataset] temporal_dataset:", _summarize_npz(temporal_npz))

    tv_latents = sorted((args.latents_dir / "tv").glob("*_latents.npz"))
    commercial_latents = sorted((args.latents_dir / "commercials").glob("*_latents.npz"))
    print(f"[debug_dataset] latents tv files={len(tv_latents)} commercials files={len(commercial_latents)}")

    records_tv_only = build_sequence_records(
        args.latents_dir, args.seq_len, min_label_id=0, include_commercials=False
    )
    print("[debug_dataset] records without commercials:", summarize_records(records_tv_only))

    records_with_commercials = build_sequence_records(
        args.latents_dir, args.seq_len, min_label_id=0, include_commercials=True
    )
    print("[debug_dataset] records with commercials:", summarize_records(records_with_commercials))


if __name__ == "__main__":
    main()
