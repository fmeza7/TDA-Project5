from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
DEFAULT_ANNOTATION_SUFFIXES = [".labels", ".txt"]
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
    parser.add_argument("--default_split", type=str, default="train")
    parser.add_argument(
        "--activities",
        type=str,
        default=",".join(KNOWN_ACTIVITIES),
        help="Lista de actividades separada por coma",
    )
    parser.add_argument(
        "--camera_folders",
        type=str,
        default="",
        help="Filtrar videos por nombre de carpeta padre, separado por coma (ej: cam01)",
    )
    parser.add_argument(
        "--annotation_suffixes",
        type=str,
        default=",".join(DEFAULT_ANNOTATION_SUFFIXES),
        help="Sufijos de anotaciones aceptados, separados por coma (ej: .labels,.txt)",
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
    parser.add_argument("--strict_annotations", action="store_true")
    return parser.parse_args()


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _split_tokens(value: str) -> List[str]:
    return [x for x in re.split(r"[^A-Za-z0-9]+", value) if x]


def _csv_tokens(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_subject(value: str) -> str:
    token = value.strip().upper()
    if re.fullmatch(r"P\d{2,3}", token):
        return token
    return ""


def _is_camera_folder(value: str) -> bool:
    token = value.strip().lower()
    return bool(re.fullmatch(r"(cam\d+|stereo|webcam\d*|ch\d+)", token))


def _camera_folders(csv_value: str) -> set[str]:
    return {token.lower() for token in _csv_tokens(csv_value)}


def _annotation_suffix_list(csv_value: str) -> List[str]:
    suffixes: List[str] = []
    for raw in _csv_tokens(csv_value):
        token = raw.lower()
        if not token.startswith("."):
            token = "." + token
        suffixes.append(token)
    return sorted(set(suffixes), key=len, reverse=True)


def iter_video_files(root: Path, camera_folders: set[str]) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if camera_folders and not any(
            parent.name.strip().lower() in camera_folders for parent in path.parents
        ):
            continue
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

    stem_tokens = []
    for token in _split_tokens(path.stem):
        if _normalize_subject(token):
            continue
        if _is_camera_folder(token) or re.fullmatch(r"ch\d+", token.lower()):
            continue
        stem_tokens.append(token.lower())
    if stem_tokens:
        return " ".join(stem_tokens)

    parent_name = path.parent.name.strip()
    if (
        parent_name
        and not re.fullmatch(r"P\d{2,3}", parent_name)
        and not _is_camera_folder(parent_name)
    ):
        return parent_name
    return ""


def _strip_annotation_suffix(
    filename: str, annotation_suffixes: List[str]
) -> str:
    lower = filename.lower()
    for suffix in annotation_suffixes:
        if lower.endswith(suffix):
            return filename[: len(filename) - len(suffix)]
    return Path(filename).stem


def _annotation_index(
    annotations_dir: Path, annotation_suffixes: List[str]
) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for ann in sorted(annotations_dir.rglob("*")):
        if not ann.is_file():
            continue
        if not any(ann.name.lower().endswith(suffix) for suffix in annotation_suffixes):
            continue
        key = _normalize_text(_strip_annotation_suffix(ann.name, annotation_suffixes))
        index.setdefault(key, []).append(ann)
    return index


def _find_sibling_annotation(
    video_path: Path, annotation_suffixes: List[str]
) -> Path | None:
    for suffix in annotation_suffixes:
        candidate = video_path.with_name(video_path.name + suffix)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def find_annotation(
    video_path: Path,
    ann_index: Dict[str, List[Path]],
    annotation_suffixes: List[str],
) -> Path | None:
    sibling = _find_sibling_annotation(video_path, annotation_suffixes)
    if sibling is not None:
        return sibling

    video_key = _normalize_text(video_path.stem)
    exact = ann_index.get(video_key, [])
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        return sorted(exact, key=lambda p: len(str(p)))[0]

    candidates: List[Path] = []
    for key, paths in ann_index.items():
        if video_key in key or key in video_key:
            candidates.extend(paths)
    if not candidates:
        return None
    candidates = sorted(
        candidates,
        key=lambda p: (abs(len(_normalize_text(p.stem)) - len(video_key)), len(str(p))),
    )
    return candidates[0]


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
    camera_folders: set[str],
    annotation_suffixes: List[str],
) -> List[Dict]:
    ann_index = _annotation_index(annotations_dir, annotation_suffixes)
    rows: List[Dict] = []
    missing_annotations = 0

    for video_path in iter_video_files(videos_dir, camera_folders):
        subject_id = _subject_from_path(video_path)
        split = split_mapping.get(subject_id, default_split)
        annotation = find_annotation(video_path, ann_index, annotation_suffixes)
        if annotation is None:
            missing_annotations += 1
            if strict_annotations:
                raise FileNotFoundError(f"No se encontro anotacion para {video_path}")
            annotation_path = ""
        else:
            annotation_path = str(annotation.resolve())

        row = {
            "video_id": video_path.stem,
            "video_path": str(video_path.resolve()),
            "activity_label": _activity_from_path(video_path, activities),
            "subject_id": subject_id,
            "split": split,
            "annotation_path": annotation_path,
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: (x["split"], x["subject_id"], x["video_id"]))
    print(
        f"[breakfast_manifest_builder] videos={len(rows)} missing_annotations={missing_annotations}"
    )
    return rows


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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

    activities = [x.strip() for x in args.activities.split(",") if x.strip()]
    camera_folders = _camera_folders(args.camera_folders)
    annotation_suffixes = _annotation_suffix_list(args.annotation_suffixes)
    rows = build_manifest(
        videos_dir=args.videos_dir,
        annotations_dir=args.annotations_dir,
        split_mapping=split_mapping,
        default_split=args.default_split,
        activities=activities,
        strict_annotations=args.strict_annotations,
        camera_folders=camera_folders,
        annotation_suffixes=annotation_suffixes,
    )

    write_manifest(args.output_manifest, rows)
    summary_path = args.output_manifest.with_name(
        args.output_manifest.stem + "_summary.json"
    )
    write_summary(summary_path, rows)
    print(f"[breakfast_manifest_builder] manifest={args.output_manifest}")
    print(f"[breakfast_manifest_builder] summary={summary_path}")


if __name__ == "__main__":
    main()
