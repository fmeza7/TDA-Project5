from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .io_utils import ensure_dir


def sample_video(
    video_path: Path,
    output_path: Path,
    sample_fps: float,
    image_size: int,
    category: str,
) -> Dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(int(round(src_fps / sample_fps)), 1)
    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    frame_indices: List[int] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
            norm = resized.astype(np.float32) / 255.0
            frames.append(norm)
            timestamps.append(idx / src_fps)
            frame_indices.append(idx)
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"No se muestrearon frames para {video_path}")
    frames_arr = np.stack(frames)
    meta = {
        "frames": frames_arr,
        "timestamps_sec": np.array(timestamps, dtype=np.float32),
        "frame_indices": np.array(frame_indices, dtype=np.int32),
        "video_name": video_path.stem,
        "category": category,
    }
    np.savez_compressed(output_path, **meta)
    duration = timestamps[-1] if timestamps else 0.0
    return {
        "video_name": video_path.stem,
        "category": category,
        "num_frames": int(frames_arr.shape[0]),
        "sampled_fps": sample_fps,
        "duration_sec": float(duration),
        "path": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Muestrear frames para el pipeline DL -> TDA")
    parser.add_argument("--tv_dir", type=Path, required=True)
    parser.add_argument("--commercials_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_fps", type=float, default=3.0)
    parser.add_argument("--image_size", type=int, default=48)
    args = parser.parse_args()

    output_dir = args.output_dir
    tv_out = output_dir / "tv"
    commercials_out = output_dir / "commercials"
    ensure_dir(tv_out)
    ensure_dir(commercials_out)

    manifest: List[Dict] = []
    for folder, category, out_root in [
        (args.tv_dir, "tv", tv_out),
        (args.commercials_dir, "commercials", commercials_out),
    ]:
        video_paths: List[Path] = []
        for pattern in ("*.mpg", "*.MPG", "*.mp4", "*.MP4"):
            video_paths.extend(folder.glob(pattern))
        for video_path in sorted(video_paths):
            out_path = out_root / f"{video_path.stem}_frames.npz"
            rec = sample_video(video_path, out_path, args.sample_fps, args.image_size, category)
            manifest.append(rec)
    manifest_path = output_dir / "manifest_frames.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[frame_sampling] procesados {len(manifest)} videos; manifest en {manifest_path}")


if __name__ == "__main__":
    main()
