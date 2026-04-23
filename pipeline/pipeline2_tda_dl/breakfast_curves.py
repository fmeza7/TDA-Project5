from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from pipeline.feature_extraction.topological_curves import process_video

from .repro_utils import relpath_str, runtime_metadata, safe_filename, write_json


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
    parser.add_argument(
        "--strict_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--manifest_out", type=Path, default=None)
    parser.add_argument("--metadata_out", type=Path, default=None)
    return parser.parse_args()


def _selected_splits(splits_csv: str) -> List[str]:
    splits = [x.strip() for x in splits_csv.split(",") if x.strip()]
    return splits or ["train", "val", "test"]


def _load_cubical_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest cubical invalido: {path}")
    return payload


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
    cubical_manifest = _load_cubical_manifest(input_dir / "manifest_cubical.json")

    rows: List[Dict] = []
    missing_inputs = 0
    selected_splits = _selected_splits(args.splits)
    if cubical_manifest:
        records_iter = [
            row
            for row in cubical_manifest
            if str(row.get("split") or "").strip() in selected_splits
        ]
    else:
        records_iter = []
        for split in selected_splits:
            split_dir = input_dir / split
            if not split_dir.exists():
                if args.strict_missing:
                    raise FileNotFoundError(
                        f"No existe split dir en input_dir: {split_dir}"
                    )
                missing_inputs += 1
                continue
            for npz_path in sorted(split_dir.glob("*.npz")):
                records_iter.append(
                    {
                        "sample_id": npz_path.stem,
                        "video_id": npz_path.stem,
                        "split": split,
                        "output_path": relpath_str(npz_path, input_dir),
                    }
                )

    for row in records_iter:
        split = str(row.get("split") or "").strip()
        output_raw = str(row.get("output_path") or "").strip()
        sample_id = str(row.get("sample_id") or row.get("video_id") or "").strip()
        if not split or not output_raw or not sample_id:
            if args.strict_missing:
                raise ValueError(f"Fila invalida en manifest cubical: {row}")
            missing_inputs += 1
            continue

        npz_path = (input_dir / output_raw).resolve()
        if not npz_path.exists():
            if args.strict_missing:
                raise FileNotFoundError(f"No existe NPZ cubical: {npz_path}")
            missing_inputs += 1
            continue

        out_path = output_dir / split / f"{safe_filename(sample_id)}_curves.npz"
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
                "sample_id": sample_id,
                "video_id": str(row.get("video_id") or npz_path.stem),
                "split": split,
                "source_path": relpath_str(npz_path, manifest_out.parent),
                "output_path": relpath_str(out_path, manifest_out.parent),
                "frames": frames,
            }
        )

    write_json(manifest_out, rows)

    summary = {
        "num_records": len(rows),
        "missing_input_splits": missing_inputs,
        "splits": sorted({row["split"] for row in rows}),
    }
    summary_path = manifest_out.with_name(manifest_out.stem + "_summary.json")
    write_json(summary_path, summary)

    metadata_path = (
        args.metadata_out.resolve()
        if args.metadata_out is not None
        else manifest_out.with_name(manifest_out.stem + "_metadata.json")
    )
    write_json(
        metadata_path,
        runtime_metadata(
            stage="breakfast_curves",
            args=args,
            extra={"num_records": len(rows)},
        ),
    )

    print(f"[breakfast_curves] records={len(rows)}")
    print(f"[breakfast_curves] manifest={manifest_out}")
    print(f"[breakfast_curves] summary={summary_path}")
    print(f"[breakfast_curves] metadata={metadata_path}")


if __name__ == "__main__":
    main()
