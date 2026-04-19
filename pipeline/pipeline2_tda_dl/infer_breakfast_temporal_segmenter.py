from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from .breakfast_temporal_segmenter import TDATemporalSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inferencia many-to-many para segmentacion Breakfast"
    )
    parser.add_argument("--windows_npz", type=Path, required=True)
    parser.add_argument("--model_checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--frame_labels_manifest", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--label_map", type=Path, default=None)
    parser.add_argument("--save_avg_logits", action="store_true")
    return parser.parse_args()


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_label_map(
    args: argparse.Namespace, checkpoint_payload: Dict
) -> Dict[str, int]:
    if args.label_map is not None and args.label_map.exists():
        payload = json.loads(args.label_map.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return {str(k): int(v) for k, v in payload.items()}

    ckpt_map = checkpoint_payload.get("label_map")
    if isinstance(ckpt_map, dict):
        return {str(k): int(v) for k, v in ckpt_map.items()}
    return {}


def _read_frame_labels_manifest(path: Path | None) -> Dict[str, Dict]:
    if path is None or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return {}

    index: Dict[str, Dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        video_id = str(row.get("video_id") or "").strip()
        output_path = str(row.get("output_path") or "").strip()
        if not video_id or not output_path:
            continue
        npz_path = Path(output_path)
        if not npz_path.is_absolute():
            npz_path = (path.parent / npz_path).resolve()
        index[Path(video_id).stem.lower()] = {
            "video_id": Path(video_id).stem,
            "split": str(row.get("split") or ""),
            "npz_path": npz_path,
        }
    return index


def _build_video_buffers(
    video_ids: np.ndarray,
    end_idx: np.ndarray,
    frame_labels_index: Dict[str, Dict],
    num_classes: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    buffers: Dict[str, Dict[str, np.ndarray]] = {}
    for raw_video in video_ids:
        video = Path(str(raw_video)).stem
        key = video.lower()
        if key in buffers:
            continue

        total_len = 0
        timestamps = None
        gt_labels = None
        gt_path = (
            frame_labels_index.get(key, {}).get("npz_path")
            if frame_labels_index
            else None
        )
        if gt_path is not None and Path(gt_path).exists():
            with np.load(gt_path) as data:
                timestamps = data["timestamps_sec"].astype(np.float32)
                total_len = int(timestamps.shape[0])
                if "frame_label_ids" in data.files:
                    gt_labels = data["frame_label_ids"].astype(np.int32)

        if total_len <= 0:
            mask = np.array(
                [Path(str(v)).stem.lower() == key for v in video_ids], dtype=bool
            )
            total_len = int(np.max(end_idx[mask])) if np.any(mask) else 0
            timestamps = np.arange(total_len, dtype=np.float32)

        buffers[key] = {
            "video_id": np.array([video], dtype=np.str_),
            "sum_logits": np.zeros((total_len, num_classes), dtype=np.float32),
            "count": np.zeros((total_len,), dtype=np.float32),
            "timestamps_sec": timestamps.astype(np.float32),
            "gt_label_ids": gt_labels,
        }
    return buffers


def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(f"Using device: {device}")

    if not args.windows_npz.exists():
        raise FileNotFoundError(f"No existe windows_npz: {args.windows_npz}")
    if not args.model_checkpoint.exists():
        raise FileNotFoundError(f"No existe model_checkpoint: {args.model_checkpoint}")

    ckpt = torch.load(args.model_checkpoint, map_location="cpu")
    num_classes = int(ckpt["num_classes"])
    model = TDATemporalSegmenter(
        input_dim=int(ckpt["input_dim"]),
        num_classes=num_classes,
        hidden_dim=int(ckpt["hidden_dim"]),
        num_layers=int(ckpt["num_layers"]),
        dropout=float(ckpt["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    label_map = _load_label_map(args, ckpt)
    frame_labels_index = _read_frame_labels_manifest(args.frame_labels_manifest)

    with np.load(args.windows_npz, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        valid_mask = (
            data["valid_mask"].astype(np.uint8)
            if "valid_mask" in data.files
            else np.ones(data["y"].shape, dtype=np.uint8)
        )
        video_ids = data["video_id"].astype(np.str_)
        start_idx = data["start_idx"].astype(np.int32)
        end_idx = data["end_idx"].astype(np.int32)

    buffers = _build_video_buffers(video_ids, end_idx, frame_labels_index, num_classes)

    n = int(X.shape[0])
    for start in range(0, n, max(1, args.batch_size)):
        end = min(n, start + max(1, args.batch_size))
        batch_x = torch.from_numpy(X[start:end]).to(device)
        with torch.no_grad():
            batch_logits = model(batch_x).detach().cpu().numpy().astype(np.float32)

        batch_mask = valid_mask[start:end]
        batch_videos = video_ids[start:end]
        batch_start = start_idx[start:end]
        batch_end = end_idx[start:end]

        for i in range(batch_logits.shape[0]):
            key = Path(str(batch_videos[i])).stem.lower()
            item = buffers[key]
            s = int(batch_start[i])
            e = int(batch_end[i])
            win_logits = batch_logits[i]
            mask = batch_mask[i].astype(bool)

            if e - s != win_logits.shape[0]:
                length = min(e - s, win_logits.shape[0])
                e = s + length
                win_logits = win_logits[:length]
                mask = mask[:length]

            target_slice = item["sum_logits"][s:e]
            count_slice = item["count"][s:e]
            target_slice[mask] += win_logits[mask]
            count_slice[mask] += 1.0

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict] = []
    inv_label_map = {idx: name for name, idx in label_map.items()} if label_map else {}

    for key, item in sorted(buffers.items(), key=lambda kv: kv[0]):
        count = item["count"]
        sum_logits = item["sum_logits"]
        safe_count = np.where(count > 0, count, 1.0)
        avg_logits = sum_logits / safe_count[:, None]
        if np.any(count == 0):
            avg_logits[count == 0] = 0.0

        pred_ids = np.argmax(avg_logits, axis=1).astype(np.int32)
        pred_labels = np.array(
            [inv_label_map.get(int(i), f"class_{int(i)}") for i in pred_ids],
            dtype=np.str_,
        )

        video_id = str(item["video_id"][0])
        out_path = raw_dir / f"{video_id}_raw_pred.npz"
        payload = {
            "video_id": item["video_id"],
            "timestamps_sec": item["timestamps_sec"],
            "pred_label_ids": pred_ids,
            "pred_labels": pred_labels,
            "count": count.astype(np.float32),
        }
        if args.save_avg_logits:
            payload["avg_logits"] = avg_logits.astype(np.float32)
        if item["gt_label_ids"] is not None:
            payload["gt_label_ids"] = item["gt_label_ids"]
        np.savez_compressed(out_path, **payload)

        manifest_rows.append(
            {
                "video_id": video_id,
                "output_path": str(out_path),
                "num_frames": int(pred_ids.shape[0]),
                "num_uncovered_frames": int(np.sum(count == 0)),
            }
        )

    manifest_path = output_dir / "raw_predictions_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if label_map:
        (output_dir / "label_map.json").write_text(
            json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    summary = {
        "num_videos": len(manifest_rows),
        "windows_npz": str(args.windows_npz),
        "model_checkpoint": str(args.model_checkpoint),
    }
    summary_path = output_dir / "raw_predictions_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[infer_breakfast_temporal_segmenter] videos={len(manifest_rows)}")
    print(f"[infer_breakfast_temporal_segmenter] manifest={manifest_path}")


if __name__ == "__main__":
    main()
