from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from .repro_utils import build_sample_id, relpath_str, runtime_metadata, write_json


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
KNOWN_ACTIVITIES = [
    "coffee",
    "orange juice",
    "chocolate milk",
    "tea",
    "cereals",
    "fried egg",
    "pancakes",
    "fruit salad",
    "sandwich",
    "scrambled egg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Construir manifest de Breakfast actions"
    )
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--annotations_dir", type=Path, required=True)
    parser.add_argument("--output_manifest", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--default_split", type=str, default="")
    parser.add_argument("--expected_splits", type=str, default="train,val,test")
    parser.add_argument(
        "--max_videos_per_split",
        type=int,
        default=0,
        help="Limita videos por split (0 = sin limite)",
    )
    parser.add_argument(
        "--activities",
        type=str,
        default=",".join(KNOWN_ACTIVITIES),
        help="Lista de actividades separada por coma",
    )
    parser.add_argument(
        "--train_subjects",
        type=str,
        default="",
        help="Subjects para train separados por coma (override)",
    )
    parser.add_argument(
        "--val_subjects",
        type=str,
        default="",
        help="Subjects para val separados por coma (override)",
    )
    parser.add_argument(
        "--test_subjects",
        type=str,
        default="",
        help="Subjects para test separados por coma (override)",
    )
    parser.add_argument(
        "--strict_annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fuzzy_annotations",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--metadata_out", type=Path, default=None)
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _split_tokens(value: str) -> List[str]:
    return [x for x in re.split(r"[^A-Za-z0-9]+", value) if x]


def _normalize_subject(value: str) -> str:
    token = value.strip().upper()
    if re.fullmatch(r"P\d{2,3}", token):
        return token
    return ""


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def _subject_from_path(path: Path) -> str:
    candidates = _split_tokens(path.stem)
    for parent in path.parents:
        if parent.name:
            candidates.extend(_split_tokens(parent.name))
    for token in candidates:
        normalized = _normalize_subject(token)
        if normalized:
            return normalized
    return ""


def _activity_from_path(path: Path, activities: List[str]) -> str:
    composite = " ".join(
        [
            path.stem,
            path.parent.name,
            path.parent.parent.name if path.parent.parent else "",
        ]
    )
    normalized = _normalize_text(composite)

    options = sorted(
        {x.strip() for x in activities if x.strip()}, key=len, reverse=True
    )
    for activity in options:
        if _normalize_text(activity) in normalized:
            return activity

    parent_name = path.parent.name.strip()
    if parent_name and not re.fullmatch(r"P\d{2,3}", parent_name):
        return parent_name
    return ""


def _annotation_index(annotations_dir: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for ann in sorted(annotations_dir.rglob("*.txt")):
        key = _normalize_text(ann.stem)
        index.setdefault(key, []).append(ann)
    return index


def find_annotation(
    video_path: Path, ann_index: Dict[str, List[Path]], fuzzy_match: bool = False
) -> Path | None:
    video_key = _normalize_text(video_path.stem)
    exact = ann_index.get(video_key, [])
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise ValueError(
            f"Multiples anotaciones exactas para {video_path}: {sorted(str(p) for p in exact)}"
        )

    if not fuzzy_match:
        return None

    candidates: List[Path] = []
    for key, paths in ann_index.items():
        if video_key in key or key in video_key:
            candidates.extend(paths)
    if not candidates:
        return None
    unique_candidates = sorted({candidate.resolve() for candidate in candidates})
    if len(unique_candidates) > 1:
        raise ValueError(
            f"Matching difuso ambiguo para {video_path}: {[str(p) for p in unique_candidates]}"
        )
    candidates = sorted(
        candidates,
        key=lambda p: (abs(len(_normalize_text(p.stem)) - len(video_key)), len(str(p))),
    )
    return candidates[0]


def _expected_splits(value: str) -> set[str]:
    return {token.strip().lower() for token in value.split(",") if token.strip()}


def _ensure_explicit_split_configuration(
    split_mapping: Dict[str, str], default_split: str
) -> None:
    if split_mapping:
        return
    if default_split.strip():
        return
    raise ValueError(
        "Debes definir los splits de forma explicita usando --split_file o --train_subjects/--val_subjects/--test_subjects"
    )


def _validate_subject_split_mapping(split_mapping: Dict[str, str]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for subject, split in split_mapping.items():
        subject_key = _normalize_subject(subject)
        split_name = split.strip().lower()
        if not subject_key or not split_name:
            continue
        normalized[subject_key] = split_name
    return normalized


def _load_split_mapping(path: Path | None) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                _normalize_subject(str(k)): str(v)
                for k, v in data.items()
                if _normalize_subject(str(k))
            }
        if isinstance(data, list):
            mapping: Dict[str, str] = {}
            for row in data:
                if not isinstance(row, dict):
                    continue
                subject = str(row.get("subject_id") or row.get("subject") or "").strip()
                split = str(row.get("split") or "").strip()
                subject = _normalize_subject(subject)
                if subject and split:
                    mapping[subject] = split
            return mapping
        return {}

    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row or row.startswith("#"):
            continue
        if "," in row:
            parts = [x.strip() for x in row.split(",")]
        else:
            parts = row.split()
        if len(parts) >= 2:
            subject = _normalize_subject(parts[0])
            if subject:
                mapping[subject] = parts[1]
    return mapping


def _override_split_mapping(
    base: Dict[str, str], split_name: str, subjects_csv: str
) -> Dict[str, str]:
    mapping = dict(base)
    subjects = [x.strip() for x in subjects_csv.split(",") if x.strip()]
    for subject in subjects:
        subject_key = _normalize_subject(subject)
        if subject_key:
            mapping[subject_key] = split_name
    return mapping


def build_manifest(
    videos_dir: Path,
    annotations_dir: Path,
    split_mapping: Dict[str, str],
    default_split: str,
    activities: List[str],
    strict_annotations: bool,
    strict_metadata: bool,
    fuzzy_annotations: bool,
) -> List[Dict]:
    ann_index = _annotation_index(annotations_dir)
    rows: List[Dict] = []
    missing_annotations = 0
    seen_sample_ids: set[str] = set()

    for video_path in iter_video_files(videos_dir):
        subject_id = _subject_from_path(video_path)
        if not subject_id and strict_metadata:
            raise ValueError(f"No se pudo inferir subject_id para {video_path}")

        split = split_mapping.get(subject_id, default_split.strip().lower())
        if not split:
            raise ValueError(
                f"No se encontro split para subject_id={subject_id or '<empty>'} video={video_path}"
            )

        activity_label = _activity_from_path(video_path, activities)
        if not activity_label and strict_metadata:
            raise ValueError(f"No se pudo inferir activity_label para {video_path}")

        annotation = find_annotation(
            video_path, ann_index, fuzzy_match=fuzzy_annotations
        )
        if annotation is None:
            missing_annotations += 1
            if strict_annotations:
                raise FileNotFoundError(f"No se encontro anotacion para {video_path}")
            annotation_path = ""
        else:
            annotation_path = str(annotation.resolve())

        sample_id = build_sample_id(
            split=split,
            subject_id=subject_id,
            video_id=video_path.stem,
        )
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id duplicado detectado: {sample_id}")
        seen_sample_ids.add(sample_id)

        row = {
            "sample_id": sample_id,
            "video_id": video_path.stem,
            "video_path": str(video_path.resolve()),
            "activity_label": activity_label,
            "subject_id": subject_id,
            "split": split.strip().lower(),
            "annotation_path": annotation_path,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x["split"], x["subject_id"], x["video_id"]))
    print(
        f"[breakfast_manifest_builder] videos={len(rows)} missing_annotations={missing_annotations}"
    )
    return rows


def write_manifest(path: Path, rows: List[Dict]) -> None:
    serialized: List[Dict] = []
    for row in rows:
        item = dict(row)
        for key in ("video_path", "annotation_path"):
            raw_value = str(item.get(key) or "").strip()
            if raw_value:
                item[key] = relpath_str(Path(raw_value), path.parent)
        serialized.append(item)
    write_json(path, serialized)


def write_summary(path: Path, rows: List[Dict]) -> None:
    split_counts: Dict[str, int] = {}
    subject_counts: Dict[str, int] = {}
    activity_counts: Dict[str, int] = {}
    with_annotations = 0
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        subject = row.get("subject_id", "") or ""
        if subject:
            subject_counts[subject] = subject_counts.get(subject, 0) + 1
        activity = row.get("activity_label", "") or ""
        if activity:
            activity_counts[activity] = activity_counts.get(activity, 0) + 1
        if row.get("annotation_path"):
            with_annotations += 1

    payload = {
        "num_videos": len(rows),
        "num_with_annotations": with_annotations,
        "num_without_annotations": len(rows) - with_annotations,
        "splits": split_counts,
        "subjects": subject_counts,
        "activities": activity_counts,
    }
    write_json(path, payload)


def _validate_manifest(rows: List[Dict], expected_splits: set[str]) -> None:
    if not rows:
        raise RuntimeError("El manifest de Breakfast quedo vacio")

    seen_sample_ids: set[str] = set()
    present_splits: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "").strip()
        if not sample_id:
            raise ValueError("Todas las filas del manifest deben incluir sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"sample_id duplicado detectado: {sample_id}")
        seen_sample_ids.add(sample_id)
        present_splits.add(str(row.get("split") or "").strip().lower())

    missing_splits = sorted(expected_splits - present_splits)
    if missing_splits:
        raise ValueError(f"Faltan splits esperados en el manifest: {missing_splits}")


def _limit_rows_per_split(rows: List[Dict], max_videos_per_split: int) -> List[Dict]:
    if max_videos_per_split <= 0:
        return rows
    kept: List[Dict] = []
    counters: Dict[str, int] = {}
    for row in rows:
        split = str(row.get("split") or "").strip().lower()
        current = counters.get(split, 0)
        if current >= max_videos_per_split:
            continue
        counters[split] = current + 1
        kept.append(row)
    return kept


def main() -> None:
    args = parse_args()
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"No existe videos_dir: {args.videos_dir}")
    if not args.annotations_dir.exists():
        raise FileNotFoundError(f"No existe annotations_dir: {args.annotations_dir}")

    split_mapping = _load_split_mapping(args.split_file)
    split_mapping = _override_split_mapping(split_mapping, "train", args.train_subjects)
    split_mapping = _override_split_mapping(split_mapping, "val", args.val_subjects)
    split_mapping = _override_split_mapping(split_mapping, "test", args.test_subjects)
    split_mapping = _validate_subject_split_mapping(split_mapping)
    _ensure_explicit_split_configuration(split_mapping, args.default_split)

    activities = [x.strip() for x in args.activities.split(",") if x.strip()]
    expected_splits = _expected_splits(args.expected_splits)
    rows = build_manifest(
        videos_dir=args.videos_dir,
        annotations_dir=args.annotations_dir,
        split_mapping=split_mapping,
        default_split=args.default_split,
        activities=activities,
        strict_annotations=args.strict_annotations,
        strict_metadata=args.strict_metadata,
        fuzzy_annotations=args.fuzzy_annotations,
    )
    rows = _limit_rows_per_split(rows, args.max_videos_per_split)
    _validate_manifest(rows, expected_splits)

    write_manifest(args.output_manifest, rows)
    summary_path = args.output_manifest.with_name(
        args.output_manifest.stem + "_summary.json"
    )
    write_summary(summary_path, rows)
    metadata_path = (
        args.metadata_out.resolve()
        if args.metadata_out is not None
        else args.output_manifest.with_name(
            args.output_manifest.stem + "_metadata.json"
        )
    )
    write_json(
        metadata_path,
        runtime_metadata(
            stage="breakfast_manifest_builder",
            args=args,
            extra={
                "num_videos": len(rows),
                "expected_splits": sorted(expected_splits),
                "max_videos_per_split": int(args.max_videos_per_split),
            },
        ),
    )
    print(f"[breakfast_manifest_builder] manifest={args.output_manifest}")
    print(f"[breakfast_manifest_builder] summary={summary_path}")
    print(f"[breakfast_manifest_builder] metadata={metadata_path}")


if __name__ == "__main__":
    main()
