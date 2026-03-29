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

    @property
    def end_time(self) -> float:
        return self.start_time + self.duration


def merge_adjacent(dets: List[Detection], merge_gap_sec: float) -> List[Detection]:
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: (d.television, d.commercial, d.start_time))
    merged: List[Detection] = []
    for det in dets:
        if not merged:
            merged.append(det)
            continue
        last = merged[-1]
        if (
            det.television == last.television
            and det.commercial == last.commercial
            and det.start_time - last.end_time <= merge_gap_sec
        ):
            duration = max(det.end_time, last.end_time) - last.start_time
            score = (last.score + det.score) / 2.0
            merged[-1] = Detection(last.television, last.start_time, duration, last.commercial, score)
        else:
            merged.append(det)
    return merged


def filter_short(dets: List[Detection], min_segment_sec: float) -> List[Detection]:
    return [det for det in dets if det.duration >= min_segment_sec]


def dedup_close(dets: List[Detection], dedup_gap_sec: float) -> List[Detection]:
    dets = sorted(dets, key=lambda d: (d.television, d.commercial, d.start_time))
    kept: List[Detection] = []
    for det in dets:
        if not kept:
            kept.append(det)
            continue
        last = kept[-1]
        if det.television == last.television and det.commercial == last.commercial:
            if det.start_time - last.start_time < dedup_gap_sec:
                if det.score > last.score:
                    kept[-1] = det
                continue
        kept.append(det)
    return kept
