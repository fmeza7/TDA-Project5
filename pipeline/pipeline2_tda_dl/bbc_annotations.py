from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _episode_idx_from_name(name: str) -> int | None:
    match = re.search(r"(\d{2})", name)
    if not match:
        return None
    return int(match.group(1))


def discover_episode_mapping(videos_dir: Path, annotations_dir: Path) -> Dict[str, Path]:
    videos_by_idx: Dict[int, str] = {}
    for video_path in sorted(videos_dir.glob("*.mp4")):
        idx = _episode_idx_from_name(video_path.stem)
        if idx is None:
            continue
        videos_by_idx[idx] = video_path.stem

    annotations_by_idx: Dict[int, Path] = {}
    for ann_path in sorted(annotations_dir.glob("*.txt")):
        idx = _episode_idx_from_name(ann_path.stem)
        if idx is None:
            continue
        annotations_by_idx[idx] = ann_path

    mapping: Dict[str, Path] = {}
    for idx, video_stem in videos_by_idx.items():
        ann_path = annotations_by_idx.get(idx)
        if ann_path is None:
            continue
        mapping[video_stem] = ann_path
    return mapping


def parse_shot_intervals(ann_path: Path) -> List[Tuple[int, int]]:
    intervals: List[Tuple[int, int]] = []
    for line in ann_path.read_text(encoding="utf-8").splitlines():
        row = line.strip()
        if not row:
            continue
        parts = row.split()
        if len(parts) < 2:
            continue
        start = int(parts[0])
        end = int(parts[1])
        if end < start:
            continue
        intervals.append((start, end))
    intervals.sort(key=lambda x: x[0])
    return intervals


def shot_boundaries_from_intervals(intervals: List[Tuple[int, int]]) -> np.ndarray:
    if len(intervals) <= 1:
        return np.zeros((0,), dtype=np.int32)
    boundaries = [start for start, _ in intervals[1:]]
    return np.array(sorted(set(boundaries)), dtype=np.int32)


def load_bbc_boundaries(videos_dir: Path, annotations_dir: Path) -> Dict[str, np.ndarray]:
    mapping = discover_episode_mapping(videos_dir, annotations_dir)
    boundaries: Dict[str, np.ndarray] = {}
    for video_stem, ann_path in mapping.items():
        intervals = parse_shot_intervals(ann_path)
        boundaries[video_stem] = shot_boundaries_from_intervals(intervals)
    return boundaries
