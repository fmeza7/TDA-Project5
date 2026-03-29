from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    television: str
    start_time: float
    duration: float
    commercial: str
    score: float


def merge_by_gap(detections: List[Detection], merge_gap_sec: float) -> List[Detection]:
    if not detections:
        return []
    merged = [detections[0]]
    for det in detections[1:]:
        prev = merged[-1]
        if det.television == prev.television and det.commercial == prev.commercial:
            gap = det.start_time - (prev.start_time + prev.duration)
            if gap <= merge_gap_sec:
                total_dur = max(prev.start_time + prev.duration, det.start_time + det.duration) - prev.start_time
                merged[-1] = Detection(prev.television, prev.start_time, total_dur, prev.commercial, max(prev.score, det.score))
                continue
        merged.append(det)
    return merged


def filter_short(detections: List[Detection], min_segment_sec: float) -> List[Detection]:
    return [det for det in detections if det.duration >= min_segment_sec]
