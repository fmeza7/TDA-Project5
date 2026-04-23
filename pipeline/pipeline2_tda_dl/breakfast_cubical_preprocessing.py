from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Dict, List

import numpy as np
from gudhi.representations import PersistenceImage

from pipeline.preprocessing.cubical_preprocessing import process_video

from .repro_utils import relpath_str, runtime_metadata, safe_filename, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesamiento cubical para Breakfast usando manifest"
    )
    parser.add_argument("--dataset_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_fps", type=float, default=3.0)
    parser.add_argument("--grid_size", type=int, default=48)
    parser.add_argument("--min_persistence", type=float, default=0.005)
    parser.add_argument("--splits", type=str, default="train,val,test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict_missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--manifest_out", type=Path, default=None)
    parser.add_argument("--metadata_out", type=Path, default=None)
    return parser.parse_args()


def _load_dataset_manifest(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest invalido (se esperaba lista): {path}")
    return payload


def _resolve_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _runtime_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        sample_fps=args.sample_fps,
        grid_size=args.grid_size,
        min_persistence=args.min_persistence,
        overwrite=args.overwrite,
    )


def _pi_transforms() -> tuple[PersistenceImage, PersistenceImage]:
    pi_transform_h0 = PersistenceImage(
        bandwidth=0.05,
        weight=lambda birth_death: birth_death[1] - birth_death[0],
        resolution=[16, 16],
        im_range=[0.0, 1.0, 0.0, 1.0],
    )
    pi_transform_h1 = PersistenceImage(
        bandwidth=0.05,
        weight=lambda birth_death: birth_death[1] - birth_death[0],
        resolution=[16, 16],
        im_range=[0.0, 1.0, 0.0, 1.0],
    )
    unit_diag = [np.array([[0.0, 1.0]], dtype=np.float32)]
    pi_transform_h0.fit(unit_diag)
    pi_transform_h1.fit(unit_diag)
    return pi_transform_h0, pi_transform_h1


def _selected_splits(splits_csv: str) -> set[str]:
    splits = {x.strip() for x in splits_csv.split(",") if x.strip()}
    return splits or {"train", "val", "test"}


def main() -> None:
    args = parse_args()
    dataset_entries = _load_dataset_manifest(args.dataset_manifest)
    selected_splits = _selected_splits(args.splits)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = (
        args.manifest_out.resolve()
        if args.manifest_out
        else output_dir / "manifest_cubical.json"
    )

    runtime_args = _runtime_args(args)
    pi_h0, pi_h1 = _pi_transforms()

    records: List[Dict] = []
    skipped_split = 0
    missing_paths = 0
    seen_sample_ids: set[str] = set()
    for entry in dataset_entries:
        split = str(entry.get("split") or "train").strip()
        if split not in selected_splits:
            skipped_split += 1
            continue

        sample_id = str(entry.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError(f"Entrada sin sample_id en dataset manifest: {entry}")
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id duplicado en dataset manifest: {sample_id}")
        seen_sample_ids.add(sample_id)

        raw_video_path = str(entry.get("video_path") or "").strip()
        if not raw_video_path:
            if args.strict_missing:
                raise ValueError(f"Entrada sin video_path: {entry}")
            missing_paths += 1
            continue

        video_path = _resolve_path(raw_video_path, args.dataset_manifest)
        if not video_path.exists():
            if args.strict_missing:
                raise FileNotFoundError(f"No existe video: {video_path}")
            missing_paths += 1
            continue

        annotation_raw = str(entry.get("annotation_path") or "").strip()
        annotation_path = (
            relpath_str(
                _resolve_path(annotation_raw, args.dataset_manifest),
                manifest_out.parent,
            )
            if annotation_raw
            else ""
        )

        output_rel = f"{split}/{safe_filename(sample_id)}.npz"
        output_abs = output_dir / output_rel
        legacy_rel = f"{split}/{video_path.stem}.npz"
        legacy_abs = output_dir / legacy_rel

        if output_abs.exists() and not args.overwrite:
            summary = None
        else:
            summary = process_video(
                video_path=video_path,
                category=split,
                output_root=output_dir,
                args=runtime_args,
                pi_transform_h0=pi_h0,
                pi_transform_h1=pi_h1,
            )
            if legacy_abs.exists() and legacy_abs != output_abs:
                output_abs.parent.mkdir(parents=True, exist_ok=True)
                legacy_abs.replace(output_abs)

        if summary is None and not output_abs.exists() and legacy_abs.exists():
            output_abs.parent.mkdir(parents=True, exist_ok=True)
            legacy_abs.replace(output_abs)

        if summary is not None:
            num_frames = summary.num_frames
            native_fps = summary.native_fps
            sampled_fps = summary.sampled_fps
            duration_sec = summary.duration_sec
            feature_dim = summary.feature_dim
        elif output_abs.exists():
            with np.load(output_abs) as data:
                features = data["tda_features"]
                num_frames = int(features.shape[0])
                feature_dim = int(features.shape[1]) if features.ndim == 2 else 0
            native_fps = float(entry.get("native_fps", 0.0) or 0.0)
            sampled_fps = float(
                entry.get("sampled_fps", args.sample_fps) or args.sample_fps
            )
            duration_sec = float(entry.get("duration_sec", 0.0) or 0.0)
        else:
            if args.strict_missing:
                raise RuntimeError(
                    f"No se pudo generar output cubical para {video_path}"
                )
            continue

        records.append(
            {
                "sample_id": sample_id,
                "video_id": str(entry.get("video_id") or video_path.stem),
                "split": split,
                "subject_id": str(entry.get("subject_id") or ""),
                "activity_label": str(entry.get("activity_label") or ""),
                "annotation_path": annotation_path,
                "source_path": relpath_str(video_path, manifest_out.parent),
                "output_path": output_rel,
                "num_frames": int(num_frames),
                "native_fps": float(native_fps),
                "sampled_fps": float(sampled_fps),
                "duration_sec": float(duration_sec),
                "feature_dim": int(feature_dim),
            }
        )

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_out, records)

    summary = {
        "num_records": len(records),
        "selected_splits": sorted(selected_splits),
        "skipped_by_split": skipped_split,
        "missing_paths": missing_paths,
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
            stage="breakfast_cubical_preprocessing",
            args=args,
            extra={
                "num_records": len(records),
                "selected_splits": sorted(selected_splits),
            },
        ),
    )

    print(f"[breakfast_cubical_preprocessing] records={len(records)}")
    print(f"[breakfast_cubical_preprocessing] manifest={manifest_out}")
    print(f"[breakfast_cubical_preprocessing] summary={summary_path}")
    print(f"[breakfast_cubical_preprocessing] metadata={metadata_path}")


if __name__ == "__main__":
    main()
