from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from .temporal_dataset import _load_latent_file
from .temporal_model import TemporalClassifier


@dataclass
class BoundaryPrediction:
    video_name: str
    time_sec: float
    score: float


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_class_map(path: Path) -> Dict[int, str]:
    data = json.loads(path.read_text())
    return {int(k): str(v) for k, v in data.items()}


def _resolve_positive_id(class_map: Dict[int, str]) -> int:
    for idx, name in class_map.items():
        if "transition" in name.lower():
            return idx
    if 1 in class_map:
        return 1
    raise RuntimeError(f"No se pudo resolver clase positiva en class_map={class_map}")


def _merge_predictions(preds: List[BoundaryPrediction], merge_gap_sec: float) -> List[BoundaryPrediction]:
    if not preds:
        return []
    preds = sorted(preds, key=lambda x: x.time_sec)
    merged: List[BoundaryPrediction] = []
    group: List[BoundaryPrediction] = [preds[0]]

    for item in preds[1:]:
        if item.time_sec - group[-1].time_sec <= merge_gap_sec:
            group.append(item)
            continue
        merged.append(_collapse_group(group))
        group = [item]
    merged.append(_collapse_group(group))
    return merged


def _collapse_group(group: List[BoundaryPrediction]) -> BoundaryPrediction:
    weights = np.array([max(1e-6, x.score) for x in group], dtype=np.float64)
    times = np.array([x.time_sec for x in group], dtype=np.float64)
    merged_time = float(np.sum(times * weights) / np.sum(weights))
    merged_score = float(np.max([x.score for x in group]))
    return BoundaryPrediction(video_name=group[0].video_name, time_sec=merged_time, score=merged_score)


def _matches_target(video_name: str, target_videos: Sequence[str]) -> bool:
    if not target_videos:
        return True
    return video_name in set(target_videos)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferencia temporal para detectar transiciones de tomas")
    parser.add_argument("--latents_dir", type=Path, required=True)
    parser.add_argument("--temporal_model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--merge_gap_sec", type=float, default=0.8)
    parser.add_argument("--target_videos", nargs="*", default=[])
    parser.add_argument("--seq_len", type=int, default=0)
    args = parser.parse_args()

    device = resolve_device()
    print(f"Using device: {device}")

    config_path = args.temporal_model_dir / "temporal_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    seq_len = int(args.seq_len or config.get("seq_len", 9))

    class_map = _load_class_map(args.temporal_model_dir / "class_map.json")
    positive_id = _resolve_positive_id(class_map)
    print(f"[infer_boundaries] positive_id={positive_id} name={class_map.get(positive_id)}")

    ckpt_path = args.temporal_model_dir / "temporal_best.pt"
    if not ckpt_path.exists():
        raise RuntimeError(f"No existe checkpoint temporal: {ckpt_path}")
    model_ckpt = torch.load(ckpt_path, map_location="cpu")

    first_file = next((args.latents_dir / "tv").glob("*_latents.npz"), None)
    if first_file is None:
        raise RuntimeError("No se encontraron latentes en latents/tv")
    latent_sample = np.load(first_file, allow_pickle=True)["z_latent"]
    latent_dim = int(latent_sample.shape[1])

    model = TemporalClassifier(latent_dim=latent_dim, num_classes=len(class_map), seq_len=seq_len).to(device)
    model.load_state_dict(model_ckpt["state_dict"])
    model.eval()

    all_preds: List[BoundaryPrediction] = []
    for path in sorted((args.latents_dir / "tv").glob("*_latents.npz")):
        payload = _load_latent_file(path)
        video_name = payload["video_name"]
        if not _matches_target(video_name, args.target_videos):
            continue

        z = payload["z_latent"]
        center_times = payload["center_times"]
        if z.shape[0] < seq_len:
            continue

        raw_preds: List[BoundaryPrediction] = []
        for start in range(0, z.shape[0] - seq_len + 1):
            end_idx = start + seq_len
            mid = start + seq_len // 2
            seq_tensor = torch.from_numpy(z[start:end_idx]).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1).squeeze(0)
            positive_score = float(probs[positive_id].item())
            if positive_score < args.score_threshold:
                continue
            raw_preds.append(
                BoundaryPrediction(
                    video_name=video_name,
                    time_sec=float(center_times[mid]),
                    score=positive_score,
                )
            )

        merged = _merge_predictions(raw_preds, merge_gap_sec=args.merge_gap_sec)
        print(f"[infer_boundaries] {video_name}: raw={len(raw_preds)} merged={len(merged)}")
        all_preds.extend(merged)

    all_preds = sorted(all_preds, key=lambda x: (x.video_name, x.time_sec))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# television\ttime_sec\tscore\n")
        for pred in all_preds:
            handle.write(f"{pred.video_name}\t{pred.time_sec:.4f}\t{pred.score:.6f}\n")
    print(f"[infer_boundaries] total_predictions={len(all_preds)} -> {args.output}")


if __name__ == "__main__":
    main()
