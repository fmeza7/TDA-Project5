from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.feature_extraction.topological_curves import process_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generar curvas topologicas para Breakfast"
    )
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--smooth_window", type=int, default=0)
    parser.add_argument("--z_window", type=int, default=15)
    parser.add_argument("--pi_dim", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict_missing", action="store_true")
    parser.add_argument("--manifest_out", type=Path, default=None)
    return parser.parse_args()


def _selected_splits(splits_csv: str) -> List[str]:
    splits = [x.strip() for x in splits_csv.split(",") if x.strip()]
    return splits or ["train", "val", "test"]


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = (
        args.manifest_out.resolve()
        if args.manifest_out
        else output_dir / "manifest_curves.json"
    )

    rows: List[Dict] = []
    missing_inputs = 0
    for split in _selected_splits(args.splits):
        split_dir = input_dir / split
        if not split_dir.exists():
            if args.strict_missing:
                raise FileNotFoundError(
                    f"No existe split dir en input_dir: {split_dir}"
                )
            missing_inputs += 1
            continue

        for npz_path in sorted(split_dir.glob("*.npz")):
            out_path = output_dir / split / f"{npz_path.stem}_curves.npz"
            result = process_video(
                npz_path=npz_path,
                output_path=out_path,
                smooth_window=args.smooth_window,
                z_window=args.z_window,
                pi_dim=args.pi_dim,
                overwrite=args.overwrite,
            )

            if result:
                frames = int(result["frames"])
            elif out_path.exists():
                with np.load(out_path) as data:
                    frames = int(data["curve_signals"].shape[0])
            else:
                if args.strict_missing:
                    raise RuntimeError(f"No se pudo generar curvas para {npz_path}")
                continue

            rows.append(
                {
                    "video_id": npz_path.stem,
                    "split": split,
                    "source_path": str(npz_path.resolve()),
                    "output_path": str(out_path.resolve()),
                    "frames": frames,
                }
            )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary = {
        "num_records": len(rows),
        "missing_input_splits": missing_inputs,
        "splits": sorted({row["split"] for row in rows}),
    }
    summary_path = manifest_out.with_name(manifest_out.stem + "_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[breakfast_curves] records={len(rows)}")
    print(f"[breakfast_curves] manifest={manifest_out}")
    print(f"[breakfast_curves] summary={summary_path}")


if __name__ == "__main__":
    main()
