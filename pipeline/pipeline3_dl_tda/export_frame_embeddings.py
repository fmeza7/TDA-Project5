from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .io_utils import ensure_dir, load_npz, resolve_device
from .visual_autoencoder import VisualAutoencoder


def embed_frames(model, frames: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    embeddings = []
    model.eval()
    with torch.no_grad():
        for idx in range(0, frames.shape[0], batch_size):
            batch = torch.from_numpy(frames[idx : idx + batch_size]).unsqueeze(1).to(device)
            z = model.encode(batch).cpu().numpy()
            embeddings.append(z)
    return np.concatenate(embeddings, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exportar embeddings por frame con el autoencoder visual")
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--visual_ae_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    args = parser.parse_args()

    ckpt = torch.load(args.visual_ae_dir / "visual_ae_best.pt", map_location="cpu")
    latent_dim = ckpt.get("latent_dim", 64)
    model = VisualAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(ckpt["state_dict"])
    device = resolve_device()
    model.to(device)

    output_dir = args.output_dir
    tv_out = output_dir / "tv"
    commercials_out = output_dir / "commercials"
    ensure_dir(tv_out)
    ensure_dir(commercials_out)

    manifest = []
    for category in ("tv", "commercials"):
        for npz_path in sorted((args.frames_dir / category).glob("*_frames.npz")):
            payload = load_npz(npz_path)
            frames = payload["frames"]
            embeddings = embed_frames(model, frames.astype(np.float32), device, batch_size=args.batch_size)
            out_path = (tv_out if category == "tv" else commercials_out) / f"{npz_path.stem.replace('_frames', '')}_embeddings.npz"
            np.savez_compressed(
                out_path,
                embeddings=embeddings,
                timestamps_sec=payload["timestamps_sec"],
                video_name=payload["video_name"],
                category=category,
            )
            manifest.append({
                "video_name": str(payload["video_name"]),
                "category": category,
                "latent_dim": int(latent_dim),
                "num_embeddings": int(embeddings.shape[0]),
                "path": str(out_path),
            })
    manifest_path = output_dir / "manifest_embeddings.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[export_frame_embeddings] generado manifest en {manifest_path}")


if __name__ == "__main__":
    main()
