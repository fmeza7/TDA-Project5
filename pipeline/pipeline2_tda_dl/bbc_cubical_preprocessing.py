from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from gudhi.representations import PersistenceImage

from pipeline.preprocessing.cubical_preprocessing import (
    VideoSummary,
    iter_videos,
    process_video,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocesar BBC Planet Earth (solo videos) con descriptores cubicales"
    )
    parser.add_argument("--videos_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_fps", type=float, default=5.0)
    parser.add_argument("--grid_size", type=int, default=48)
    parser.add_argument("--min_persistence", type=float, default=0.005)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"No existe --videos_dir: {args.videos_dir}")

    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    pi_transform_h0 = PersistenceImage(
        bandwidth=0.05,
        weight=lambda birth_death: birth_death[1] - birth_death[0],
        resolution=[16, 16],
        im_range=[0.0, 1.0, 0.0, 1.0],
    )
    pi_transform_h1 = PersistenceImage(
        bandwidth=0.05,
        weight=lambda birth_death: birth_death[1] - birth_death[0],
        resolution=[16, 16],
        im_range=[0.0, 1.0, 0.0, 1.0],
    )
    pi_transform_h0.fit([np.array([[0.0, 1.0]], dtype=np.float32)])
    pi_transform_h1.fit([np.array([[0.0, 1.0]], dtype=np.float32)])

    summaries: list[VideoSummary] = []
    for video_path in iter_videos(args.videos_dir):
        print(f"[bbc] Procesando {video_path.name}")
        summary = process_video(
            video_path=video_path,
            category="tv",
            output_root=output_root,
            args=args,
            pi_transform_h0=pi_transform_h0,
            pi_transform_h1=pi_transform_h1,
        )
        if summary is not None:
            summaries.append(summary)

    write_manifest(output_root / "manifest.json", summaries)
    print(f"[bbc] finalizado. videos_procesados={len(summaries)}")


if __name__ == "__main__":
    main()
