from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import re


_NUMBER_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class ActionSegment:
    start_sec: float
    end_sec: float
    label: str


@dataclass(frozen=True)
class RawSegment:
    start_value: float
    end_value: float
    label: str
    has_decimal: bool


def _is_number(value: str) -> bool:
    return bool(_NUMBER_RE.match(value.strip()))


def _clean_label(raw_label: str, lowercase: bool) -> str:
    label = re.sub(r"\s+", " ", raw_label.strip())
    if lowercase:
        label = label.lower()
    return label


def _parse_line(line: str, lowercase_labels: bool) -> RawSegment | None:
    row = line.strip()
    if not row or row.startswith("#"):
        return None

    parts_tab = row.split("\t")
    if len(parts_tab) >= 3 and _is_number(parts_tab[0]) and _is_number(parts_tab[1]):
        start_token = parts_tab[0].strip()
        end_token = parts_tab[1].strip()
        label = _clean_label(" ".join(parts_tab[2:]), lowercase_labels)
        return RawSegment(
            start_value=float(start_token),
            end_value=float(end_token),
            label=label,
            has_decimal=("." in start_token or "." in end_token),
        )

    parts_space = row.split(maxsplit=2)
    if (
        len(parts_space) >= 3
        and _is_number(parts_space[0])
        and _is_number(parts_space[1])
    ):
        start_token = parts_space[0].strip()
        end_token = parts_space[1].strip()
        label = _clean_label(parts_space[2], lowercase_labels)
        return RawSegment(
            start_value=float(start_token),
            end_value=float(end_token),
            label=label,
            has_decimal=("." in start_token or "." in end_token),
        )

    return None


def _infer_units(
    rows: Sequence[RawSegment],
    native_fps: float,
    duration_sec: float | None,
) -> str:
    if not rows:
        return "seconds"
    if any(row.has_decimal for row in rows):
        return "seconds"

    max_end = max(row.end_value for row in rows)
    if duration_sec is not None and duration_sec > 0 and max_end <= duration_sec * 1.25:
        return "seconds"

    if native_fps > 0 and max_end >= max(1000.0, native_fps * 120.0):
        return "frames"

    return "seconds"


def _to_seconds(
    row: RawSegment,
    units: str,
    native_fps: float,
    frame_end_inclusive: bool,
) -> ActionSegment:
    if units == "seconds":
        start_sec = float(row.start_value)
        end_sec = float(row.end_value)
    elif units == "frames":
        if native_fps <= 0:
            raise ValueError(
                "native_fps debe ser > 0 cuando las anotaciones estan en frames"
            )
        start_sec = float(row.start_value) / native_fps
        end_frame = float(row.end_value) + (1.0 if frame_end_inclusive else 0.0)
        end_sec = end_frame / native_fps
    else:
        raise ValueError(f"Unidad temporal desconocida: {units}")

    return ActionSegment(start_sec=start_sec, end_sec=end_sec, label=row.label)


def _sanitize_segments(segments: Iterable[ActionSegment]) -> List[ActionSegment]:
    ordered = sorted(segments, key=lambda s: (s.start_sec, s.end_sec, s.label))
    clean: List[ActionSegment] = []
    for segment in ordered:
        start = max(0.0, float(segment.start_sec))
        end = max(0.0, float(segment.end_sec))
        if end <= start:
            continue

        if clean and start < clean[-1].end_sec:
            start = clean[-1].end_sec
        if end <= start:
            continue

        clean.append(ActionSegment(start_sec=start, end_sec=end, label=segment.label))
    return clean


def load_action_segments(
    annotation_path: Path,
    native_fps: float,
    assume_time_units: str = "auto",
    frame_end_inclusive: bool = True,
    lowercase_labels: bool = False,
    duration_sec: float | None = None,
) -> List[ActionSegment]:
    if assume_time_units not in {"auto", "seconds", "frames"}:
        raise ValueError("assume_time_units debe ser one of: auto|seconds|frames")

    rows: List[RawSegment] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_line(line, lowercase_labels)
        if parsed is not None and parsed.label:
            rows.append(parsed)

    if not rows:
        return []

    units = assume_time_units
    if units == "auto":
        units = _infer_units(
            rows=rows, native_fps=native_fps, duration_sec=duration_sec
        )

    segments = [
        _to_seconds(row, units, native_fps, frame_end_inclusive) for row in rows
    ]
    return _sanitize_segments(segments)
