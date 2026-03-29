from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .postprocess import Detection, dedup_close, filter_short, merge_adjacent
from .temporal_dataset import _load_latent_file
from .temporal_model import TemporalClassifier


def _load_class_map(path: Path) -> Dict[int, str]:
    mapping = json.loads(path.read_text())
    return {int(k): v for k, v in mapping.items()}


def _commercial_durations(preproc_manifest: Path) -> Dict[str, float]:
    manifest = json.loads(preproc_manifest.read_text())
    durations: Dict[str, float] = {}
    for entry in manifest:
        stem = Path(entry["source_path"]).stem
        durations[stem] = float(entry.get("duration_sec", 0.0))
    return durations


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia temporal TDA→DL")
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--temporal_model_dir", type=Path, required=True)
    parser.add_argument("--preproc_manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.6)
    parser.add_argument("--merge_gap_sec", type=float, default=2.0)
    parser.add_argument("--min_segment_sec", type=float, default=3.0)
    parser.add_argument("--dedup_gap_sec", type=float, default=5.0)
    parser.add_argument("--window_sec", type=float, default=8.0)
    args = parser.parse_args()

    config_path = args.temporal_model_dir / "temporal_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    seq_len = int(config.get("seq_len", 9))

    class_map = _load_class_map(args.temporal_model_dir / "class_map.json")
    model_ckpt = torch.load(args.temporal_model_dir / "temporal_best.pt", map_location="cpu")
    first_file = next((args.latents_dir / "tv").glob("*_latents.npz"), None)
    if first_file is None:
        raise RuntimeError("No se encontraron latentes de TV; ejecute export_latents primero")
    latent_sample = np.load(first_file, allow_pickle=True)["z_latent"]
    latent_dim = latent_sample.shape[1]
    model = TemporalClassifier(latent_dim=latent_dim, num_classes=len(class_map), seq_len=seq_len)
    model.load_state_dict(model_ckpt["state_dict"])
    model.eval()

    durations = _commercial_durations(args.preproc_manifest)
    detections: List[Detection] = []

    for path in sorted((args.latents_dir / "tv").glob("*_latents.npz")):
        payload = _load_latent_file(path)
        z = payload["z_latent"]
        center_times = payload["center_times"]
        start_times = payload["start_times"]
        video_name = payload["video_name"]
        if z.shape[0] < seq_len:
            continue
        preds = []
        for start in range(0, z.shape[0] - seq_len + 1):
            end_idx = start + seq_len
            mid = start + seq_len // 2
            seq_tensor = torch.from_numpy(z[start:end_idx]).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            score, cls_id = torch.max(probs, dim=0)
            cls_id = int(cls_id.item())
            score = float(score.item())
            if cls_id == 0 or score < args.score_threshold:
                continue
            commercial = class_map.get(cls_id, "")
            if not commercial:
                continue
            preds.append(
                {
                    "video": video_name,
                    "commercial": commercial,
                    "score": score,
                    "center": float(center_times[mid]),
                    "start": float(start_times[mid]),
                }
            )
        if not preds:
            continue
        preds.sort(key=lambda x: x["center"])
        group: List[Dict] = []
        for item in preds:
            if not group:
                group.append(item)
                continue
            last = group[-1]
            if item["commercial"] == last["commercial"] and item["center"] - last["center"] <= args.merge_gap_sec:
                group.append(item)
            else:
                detections.append(_group_to_detection(group, durations, args.window_sec))
                group = [item]
        if group:
            detections.append(_group_to_detection(group, durations, args.window_sec))

    detections = merge_adjacent(detections, args.merge_gap_sec)
    detections = filter_short(detections, args.min_segment_sec)
    detections = dedup_close(detections, args.dedup_gap_sec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# television\tinicio_seg\tlargo_seg\tcomercial\tscore\n")
        for det in detections:
            handle.write(
                f"{det.television}\t{det.start_time:.3f}\t{det.duration:.3f}\t{det.commercial}\t{det.score:.4f}\n"
            )


def _group_to_detection(group: List[Dict], durations: Dict[str, float], window_sec: float) -> Detection:
    video = group[0]["video"]
    commercial = group[0]["commercial"]
    avg_score = float(np.mean([g["score"] for g in group]))
    start_time = max(0.0, group[0]["center"] - window_sec / 2.0)
    duration = durations.get(commercial, window_sec)
    return Detection(video, start_time, duration, commercial, avg_score)


if __name__ == "__main__":
    main()
