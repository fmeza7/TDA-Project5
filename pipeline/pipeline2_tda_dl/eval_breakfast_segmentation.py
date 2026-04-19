from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluar segmentacion temporal en Breakfast"
    )
    parser.add_argument("--decoded_manifest", type=Path, required=True)
    parser.add_argument("--frame_labels_manifest", type=Path, required=True)
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--splits", type=str, default="test")
    parser.add_argument("--ignore_labels", type=str, default="")
    parser.add_argument("--ignore_ids", type=str, default="")
    parser.add_argument("--output_json", type=Path, default=None)
    return parser.parse_args()


def _read_json_list(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Manifest invalido (se esperaba lista): {path}")
    return payload


def _resolve_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _selected_splits(splits_csv: str) -> set[str]:
    splits = {x.strip() for x in splits_csv.split(",") if x.strip()}
    return splits or {"test"}


def _load_label_map(args: argparse.Namespace) -> Dict[str, int]:
    candidates: List[Path] = []
    if args.label_map is not None:
        candidates.append(args.label_map)
    candidates.append(args.frame_labels_manifest.parent / "label_map.json")
    candidates.append(args.decoded_manifest.parent / "label_map.json")
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return {str(k): int(v) for k, v in payload.items()}
    return {}


def _build_ignore_ids(
    ignore_labels_csv: str, ignore_ids_csv: str, label_map: Dict[str, int]
) -> set[int]:
    ignore_ids: set[int] = set()
    for token in [x.strip() for x in ignore_ids_csv.split(",") if x.strip()]:
        try:
            ignore_ids.add(int(token))
        except ValueError:
            continue
    for label in [x.strip() for x in ignore_labels_csv.split(",") if x.strip()]:
        if label in label_map:
            ignore_ids.add(int(label_map[label]))
    return ignore_ids


def _collapse_labels(seq: Sequence[int]) -> List[int]:
    collapsed: List[int] = []
    for value in seq:
        v = int(value)
        if not collapsed or collapsed[-1] != v:
            collapsed.append(v)
    return collapsed


def _levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[len(a), len(b)])


def _segments(
    label_ids: np.ndarray, ignore_ids: set[int]
) -> List[Tuple[int, int, int]]:
    if label_ids.size == 0:
        return []
    segs: List[Tuple[int, int, int]] = []
    start = 0
    cur = int(label_ids[0])
    for i in range(1, label_ids.shape[0]):
        if int(label_ids[i]) != cur:
            if cur not in ignore_ids:
                segs.append((start, i, cur))
            start = i
            cur = int(label_ids[i])
    if cur not in ignore_ids:
        segs.append((start, label_ids.shape[0], cur))
    return segs


def _segment_iou(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    a_s, a_e, _ = a
    b_s, b_e, _ = b
    inter = max(0, min(a_e, b_e) - max(a_s, b_s))
    union = max(a_e, b_e) - min(a_s, b_s)
    if union <= 0:
        return 0.0
    return float(inter / union)


def _segment_f1_counts(
    pred: List[Tuple[int, int, int]], gt: List[Tuple[int, int, int]], threshold: float
) -> Tuple[int, int, int]:
    if not pred and not gt:
        return 0, 0, 0
    if not gt:
        return 0, len(pred), 0
    if not pred:
        return 0, 0, len(gt)

    gt_used = [False] * len(gt)
    tp = 0
    for p in pred:
        best_idx = -1
        best_iou = 0.0
        for idx, g in enumerate(gt):
            if gt_used[idx] or p[2] != g[2]:
                continue
            iou = _segment_iou(p, g)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= threshold:
            gt_used[best_idx] = True
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float((2 * tp) / denom) if denom > 0 else 0.0


def main() -> None:
    args = parse_args()
    selected_splits = _selected_splits(args.splits)
    label_map = _load_label_map(args)
    ignore_ids = _build_ignore_ids(args.ignore_labels, args.ignore_ids, label_map)

    decoded_rows = _read_json_list(args.decoded_manifest)
    frame_rows = _read_json_list(args.frame_labels_manifest)

    decoded_index: Dict[str, Path] = {}
    for row in decoded_rows:
        video_id = str(row.get("video_id") or "").strip()
        out_path = str(row.get("output_path") or "").strip()
        if not video_id or not out_path:
            continue
        decoded_index[Path(video_id).stem.lower()] = _resolve_path(
            out_path, args.decoded_manifest
        )

    per_video: List[Dict] = []
    frame_correct = 0
    frame_total = 0
    edit_scores: List[float] = []
    f1_counts = {10: [0, 0, 0], 25: [0, 0, 0], 50: [0, 0, 0]}

    for row in frame_rows:
        split = str(row.get("split") or "").strip()
        if split not in selected_splits:
            continue

        video_id = str(row.get("video_id") or "").strip()
        gt_path_raw = str(row.get("output_path") or "").strip()
        if not video_id or not gt_path_raw:
            continue

        key = Path(video_id).stem.lower()
        pred_path = decoded_index.get(key)
        if pred_path is None or not pred_path.exists():
            continue

        gt_path = _resolve_path(gt_path_raw, args.frame_labels_manifest)
        if not gt_path.exists():
            continue

        with np.load(gt_path) as data:
            gt_ids = data["frame_label_ids"].astype(np.int32)

        with np.load(pred_path) as data:
            pred_ids = data["decoded_label_ids"].astype(np.int32)

        length = min(gt_ids.shape[0], pred_ids.shape[0])
        gt_ids = gt_ids[:length]
        pred_ids = pred_ids[:length]

        mask = np.ones((length,), dtype=bool)
        if ignore_ids:
            for iid in ignore_ids:
                mask &= gt_ids != iid

        gt_eval = gt_ids[mask]
        pred_eval = pred_ids[mask]
        if gt_eval.size == 0:
            continue

        correct = int(np.sum(gt_eval == pred_eval))
        total = int(gt_eval.size)
        frame_correct += correct
        frame_total += total

        gt_collapsed = _collapse_labels(gt_eval.tolist())
        pred_collapsed = _collapse_labels(pred_eval.tolist())
        if len(gt_collapsed) == 0:
            edit = 0.0
        else:
            dist = _levenshtein(pred_collapsed, gt_collapsed)
            edit = float((1.0 - dist / max(1, len(gt_collapsed))) * 100.0)
        edit_scores.append(edit)

        pred_segments = _segments(pred_eval, ignore_ids=set())
        gt_segments = _segments(gt_eval, ignore_ids=set())
        for t in (10, 25, 50):
            tp, fp, fn = _segment_f1_counts(
                pred_segments, gt_segments, threshold=t / 100.0
            )
            f1_counts[t][0] += tp
            f1_counts[t][1] += fp
            f1_counts[t][2] += fn

        per_video.append(
            {
                "video_id": Path(video_id).stem,
                "split": split,
                "num_frames_eval": total,
                "frame_acc": float(correct / max(1, total)),
                "edit_score": edit,
                "num_pred_segments": int(len(pred_segments)),
                "num_gt_segments": int(len(gt_segments)),
            }
        )

    frame_acc = float(frame_correct / max(1, frame_total))
    mean_edit = float(np.mean(edit_scores)) if edit_scores else 0.0
    segmental_f1 = {
        f"F1@{t}": _f1_from_counts(*f1_counts[t]) * 100.0 for t in (10, 25, 50)
    }

    report = {
        "num_videos": len(per_video),
        "splits": sorted(selected_splits),
        "ignore_ids": sorted(ignore_ids),
        "frame_accuracy": frame_acc,
        "edit_score": mean_edit,
        **segmental_f1,
        "per_video": per_video,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    print(
        "[eval_breakfast_segmentation]",
        json.dumps(
            {
                k: report[k]
                for k in [
                    "num_videos",
                    "frame_accuracy",
                    "edit_score",
                    "F1@10",
                    "F1@25",
                    "F1@50",
                ]
            },
            indent=2,
        ),
    )
    if args.output_json is not None:
        print(f"[eval_breakfast_segmentation] report={args.output_json}")


if __name__ == "__main__":
    main()
