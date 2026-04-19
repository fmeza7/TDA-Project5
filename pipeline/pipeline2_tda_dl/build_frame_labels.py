from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from .breakfast_annotations import ActionSegment, load_action_segments


@dataclass(frozen=True)
class CubicalVideoMeta:
    key: str
    npz_path: Path
    native_fps: float
    sampled_fps: float
    duration_sec: float


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    video_id: str
    split: str
    subject_id: str
    activity_label: str
    annotation_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construir labels frame-level alineados con timestamps TDA para Breakfast"
    )
    parser.add_argument("--dataset_manifest", type=Path, required=True)
    parser.add_argument("--cubical_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_split_names", type=str, default="train")
    parser.add_argument("--sil_label", type=str, default="__sil__")
    parser.add_argument("--unk_label", type=str, default="__unk__")
    parser.add_argument(
        "--time_units", choices=["auto", "seconds", "frames"], default="auto"
    )
    parser.add_argument("--native_fps_default", type=float, default=15.0)
    parser.add_argument("--lowercase_labels", action="store_true")
    parser.add_argument("--frame_end_exclusive", action="store_true")
    parser.add_argument("--strict_missing", action="store_true")
    return parser.parse_args()


def _read_json_list(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest invalido (se esperaba lista): {path}")
    return payload


def _normalize_key(value: str) -> str:
    return Path(str(value)).stem.strip().lower()


def _safe_str(entry: Dict, keys: Iterable[str], default: str = "") -> str:
    for key in keys:
        value = entry.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _resolve_path(raw_path: str, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _build_cubical_index(cubical_manifest_path: Path) -> Dict[str, CubicalVideoMeta]:
    entries = _read_json_list(cubical_manifest_path)
    index: Dict[str, CubicalVideoMeta] = {}
    for entry in entries:
        output_raw = _safe_str(entry, ["output_path"])
        if not output_raw:
            continue
        npz_path = _resolve_path(output_raw, cubical_manifest_path)

        native_fps = float(entry.get("native_fps", 0.0) or 0.0)
        sampled_fps = float(entry.get("sampled_fps", 0.0) or 0.0)
        duration_sec = float(entry.get("duration_sec", 0.0) or 0.0)

        candidates = [
            _safe_str(entry, ["video_id"]),
            _safe_str(entry, ["source_path"]),
            _safe_str(entry, ["output_path"]),
        ]

        for candidate in candidates:
            if not candidate:
                continue
            key = _normalize_key(candidate)
            if key not in index:
                index[key] = CubicalVideoMeta(
                    key=key,
                    npz_path=npz_path,
                    native_fps=native_fps,
                    sampled_fps=sampled_fps,
                    duration_sec=duration_sec,
                )
    return index


def _dataset_entries(dataset_manifest_path: Path) -> List[DatasetEntry]:
    payload = _read_json_list(dataset_manifest_path)
    records: List[DatasetEntry] = []
    for entry in payload:
        annotation_raw = _safe_str(entry, ["annotation_path", "annotation"])
        if not annotation_raw:
            continue

        video_id = _safe_str(
            entry, ["video_id", "source_path", "video_path", "output_path"]
        )
        if not video_id:
            continue

        split = _safe_str(entry, ["split"], default="train")
        subject_id = _safe_str(entry, ["subject_id", "subject"], default="")
        activity_label = _safe_str(entry, ["activity_label", "activity"], default="")

        records.append(
            DatasetEntry(
                key=_normalize_key(video_id),
                video_id=Path(video_id).stem,
                split=split,
                subject_id=subject_id,
                activity_label=activity_label,
                annotation_path=_resolve_path(annotation_raw, dataset_manifest_path),
            )
        )
    return records


def _is_train_split(split: str, train_names: set[str]) -> bool:
    return split.strip().lower() in train_names


def _collect_train_labels(
    dataset_entries: List[DatasetEntry],
    cubical_index: Dict[str, CubicalVideoMeta],
    train_names: set[str],
    args: argparse.Namespace,
) -> set[str]:
    labels: set[str] = set()
    for entry in dataset_entries:
        if not _is_train_split(entry.split, train_names):
            continue

        cubical_meta = cubical_index.get(entry.key)
        if cubical_meta is None:
            if args.strict_missing:
                raise KeyError(
                    f"No se encontro video en cubical manifest: {entry.video_id}"
                )
            continue

        if not entry.annotation_path.exists():
            if args.strict_missing:
                raise FileNotFoundError(
                    f"No existe annotation_path: {entry.annotation_path}"
                )
            continue

        native_fps = (
            cubical_meta.native_fps
            if cubical_meta.native_fps > 0
            else args.native_fps_default
        )
        segments = load_action_segments(
            annotation_path=entry.annotation_path,
            native_fps=native_fps,
            assume_time_units=args.time_units,
            frame_end_inclusive=(not args.frame_end_exclusive),
            lowercase_labels=args.lowercase_labels,
            duration_sec=cubical_meta.duration_sec,
        )
        labels.update(segment.label for segment in segments)
    return labels


def _build_label_map(
    train_labels: set[str], sil_label: str, unk_label: str
) -> Dict[str, int]:
    ordered: List[str] = []
    for special in (sil_label, unk_label):
        if special not in ordered:
            ordered.append(special)

    for label in sorted(train_labels):
        if label not in ordered:
            ordered.append(label)

    return {label: idx for idx, label in enumerate(ordered)}


def assign_labels_to_timestamps(
    timestamps_sec: np.ndarray,
    segments: List[ActionSegment],
    sil_label: str,
) -> np.ndarray:
    max_label_len = max(
        len(sil_label),
        max((len(segment.label) for segment in segments), default=0),
    )
    labels = np.full(
        shape=(len(timestamps_sec),),
        fill_value=sil_label,
        dtype=f"<U{max(1, max_label_len)}",
    )
    if len(timestamps_sec) == 0 or not segments:
        return labels

    seg_idx = 0
    max_idx = len(segments)
    for frame_idx, timestamp in enumerate(
        timestamps_sec.astype(np.float64, copy=False)
    ):
        while seg_idx < max_idx and timestamp >= segments[seg_idx].end_sec:
            seg_idx += 1
        if seg_idx >= max_idx:
            break
        segment = segments[seg_idx]
        if segment.start_sec <= timestamp < segment.end_sec:
            labels[frame_idx] = segment.label
    return labels


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_names = {
        name.strip().lower()
        for name in args.train_split_names.split(",")
        if name.strip()
    }
    if not train_names:
        train_names = {"train"}

    cubical_index = _build_cubical_index(args.cubical_manifest)
    if not cubical_index:
        raise RuntimeError("No se pudo construir indice cubical desde manifest")

    dataset_entries = _dataset_entries(args.dataset_manifest)
    if not dataset_entries:
        raise RuntimeError(
            "No se encontraron entradas con annotation_path en dataset manifest"
        )

    train_labels = _collect_train_labels(
        dataset_entries, cubical_index, train_names, args
    )
    label_map = _build_label_map(
        train_labels, sil_label=args.sil_label, unk_label=args.unk_label
    )
    label_map_path = output_dir / "label_map.json"
    label_map_path.write_text(
        json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    manifest_rows: List[Dict] = []
    label_counter = Counter()
    split_counter = defaultdict(
        lambda: {"videos": 0, "frames": 0, "unknown_frames": 0, "sil_frames": 0}
    )
    missing_entries: List[str] = []

    unk_id = label_map[args.unk_label]
    sil_id = label_map[args.sil_label]

    for entry in dataset_entries:
        cubical_meta = cubical_index.get(entry.key)
        if cubical_meta is None:
            msg = f"[missing-cubical] {entry.video_id}"
            if args.strict_missing:
                raise KeyError(msg)
            missing_entries.append(msg)
            continue

        if not cubical_meta.npz_path.exists():
            msg = f"[missing-npz] {entry.video_id} -> {cubical_meta.npz_path}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            missing_entries.append(msg)
            continue

        if not entry.annotation_path.exists():
            msg = f"[missing-annotation] {entry.video_id} -> {entry.annotation_path}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            missing_entries.append(msg)
            continue

        native_fps = (
            cubical_meta.native_fps
            if cubical_meta.native_fps > 0
            else args.native_fps_default
        )
        segments = load_action_segments(
            annotation_path=entry.annotation_path,
            native_fps=native_fps,
            assume_time_units=args.time_units,
            frame_end_inclusive=(not args.frame_end_exclusive),
            lowercase_labels=args.lowercase_labels,
            duration_sec=cubical_meta.duration_sec,
        )

        with np.load(cubical_meta.npz_path) as data:
            timestamps_sec = data["timestamps_sec"].astype(np.float32)

        frame_labels = assign_labels_to_timestamps(
            timestamps_sec=timestamps_sec,
            segments=segments,
            sil_label=args.sil_label,
        )
        frame_label_ids = np.array(
            [label_map.get(str(label), unk_id) for label in frame_labels],
            dtype=np.int32,
        )
        valid_mask = np.ones(shape=(frame_label_ids.shape[0],), dtype=np.uint8)

        split_dir = output_dir / entry.split
        split_dir.mkdir(parents=True, exist_ok=True)
        output_npz = split_dir / f"{entry.video_id}_labels.npz"
        np.savez_compressed(
            output_npz,
            timestamps_sec=timestamps_sec,
            frame_labels=frame_labels,
            frame_label_ids=frame_label_ids,
            valid_mask=valid_mask,
            video_id=np.array([entry.video_id], dtype=np.str_),
            split=np.array([entry.split], dtype=np.str_),
            subject_id=np.array([entry.subject_id], dtype=np.str_),
            activity_label=np.array([entry.activity_label], dtype=np.str_),
            annotation_path=np.array([str(entry.annotation_path)], dtype=np.str_),
            cubical_npz_path=np.array([str(cubical_meta.npz_path)], dtype=np.str_),
        )

        split_stats = split_counter[entry.split]
        split_stats["videos"] += 1
        split_stats["frames"] += int(frame_label_ids.shape[0])
        split_stats["unknown_frames"] += int(np.sum(frame_label_ids == unk_id))
        split_stats["sil_frames"] += int(np.sum(frame_label_ids == sil_id))

        label_counter.update(frame_label_ids.tolist())
        manifest_rows.append(
            {
                "video_id": entry.video_id,
                "split": entry.split,
                "subject_id": entry.subject_id,
                "activity_label": entry.activity_label,
                "output_path": str(output_npz),
                "num_frames": int(frame_label_ids.shape[0]),
                "num_unknown_frames": int(np.sum(frame_label_ids == unk_id)),
                "num_sil_frames": int(np.sum(frame_label_ids == sil_id)),
            }
        )

    manifest_path = output_dir / "manifest_frame_labels.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    inv_label_map = {idx: label for label, idx in label_map.items()}
    label_distribution = {
        inv_label_map.get(label_id, f"id_{label_id}"): int(count)
        for label_id, count in sorted(label_counter.items(), key=lambda x: x[0])
    }
    summary_payload = {
        "num_videos": len(manifest_rows),
        "num_missing": len(missing_entries),
        "missing_entries": missing_entries,
        "train_split_names": sorted(train_names),
        "time_units": args.time_units,
        "frame_end_inclusive": not args.frame_end_exclusive,
        "label_map_path": str(label_map_path),
        "label_distribution": label_distribution,
        "splits": split_counter,
    }
    summary_path = output_dir / "summary_frame_labels.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[build_frame_labels] videos_processed={len(manifest_rows)}")
    print(f"[build_frame_labels] label_map={label_map_path}")
    print(f"[build_frame_labels] manifest={manifest_path}")
    print(f"[build_frame_labels] summary={summary_path}")
    if missing_entries:
        print(f"[build_frame_labels] entries_missing={len(missing_entries)}")


if __name__ == "__main__":
    main()
