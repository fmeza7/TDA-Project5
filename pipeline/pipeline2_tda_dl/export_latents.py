from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .topoae_model import TopoAutoencoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar embeddings latentes del TopoAE")
    parser.add_argument("--window_data", type=Path, required=True)
    parser.add_argument("--topoae_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.window_data, allow_pickle=True)
    X = data["X_flat"]
    video_names = data["video_name"]
    source_types = data["source_type"]
    center = data["center_time"]
    start = data["start_time"]
    end = data["end_time"]
    label_ids = data["label_id"]
    label_names = data["label_name"]

    mean = np.load(args.topoae_dir / "scaler_mean.npy")
    std = np.load(args.topoae_dir / "scaler_std.npy")
    input_dim = X.shape[1]

    model = TopoAutoencoder(input_dim=input_dim)
    ckpt = torch.load(args.topoae_dir / "topoae_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    output_dir = args.output_dir
    (output_dir / "tv").mkdir(parents=True, exist_ok=True)
    (output_dir / "commercials").mkdir(parents=True, exist_ok=True)

    grouped: Dict[Tuple[str, str], Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for idx in tqdm(range(X.shape[0]), desc="Export latents"):
        x = torch.from_numpy(((X[idx] - mean) / std).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            z = model.encode(x).squeeze(0).cpu().numpy()
        key = (source_types[idx], video_names[idx])
        target = grouped[key]
        target.setdefault("z_latent", []).append(z)
        target.setdefault("center_times", []).append(float(center[idx]))
        target.setdefault("start_times", []).append(float(start[idx]))
        target.setdefault("end_times", []).append(float(end[idx]))
        target.setdefault("label_id", []).append(int(label_ids[idx]))
        target.setdefault("label_name", []).append(str(label_names[idx]))

    manifest = []
    class_names = set()
    for (source_type, video_name), payload in grouped.items():
        arrays = {k: np.array(v) for k, v in payload.items()}
        path = output_dir / source_type / f"{video_name}_latents.npz"
        np.savez_compressed(path, video_name=video_name, source_type=source_type, **arrays)
        manifest.append({"video_name": video_name, "source_type": source_type, "path": str(path)})
        class_names.update({name for name in payload["label_name"] if name})

    (output_dir / "manifest_latents.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "class_names.json").write_text(json.dumps(sorted(class_names), indent=2))
