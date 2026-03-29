from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .io_utils import ensure_dir, load_npz


def process_file(npz_path: Path, output_path: Path) -> dict:
    payload = load_npz(npz_path)
    np.savez_compressed(
        output_path,
        curve_signals=payload["tda_features"],
        curve_labels=payload["feature_labels"],
        timestamps_sec=payload["timestamps_sec"],
        video_name=payload["video_name"],
        category=payload["category"],
    )
    return {
        "video_name": str(payload["video_name"]),
        "category": str(payload["category"]),
        "num_points": int(payload["tda_features"].shape[0]),
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Construir curvas 1D sobre TDA de embeddings")
    parser.add_argument("--tda_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir
    tv_out = output_dir / "tv"
    commercials_out = output_dir / "commercials"
    ensure_dir(tv_out)
    ensure_dir(commercials_out)

    manifest = []
    for category in ("tv", "commercials"):
        for npz_path in sorted((args.tda_dir / category).glob("*_embedding_tda.npz")):
            out_root = tv_out if category == "tv" else commercials_out
            out_path = out_root / f"{npz_path.stem}_curves.npz"
            manifest.append(process_file(npz_path, out_path))
    manifest_path = output_dir / "manifest_embedding_curves.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[curve_builder] manifest en {manifest_path}")


if __name__ == "__main__":
    main()
