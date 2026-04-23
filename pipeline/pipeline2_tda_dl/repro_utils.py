from __future__ import annotations

import json
import os
import platform
import random
import re
import hashlib
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION_PACKAGES = (
    "numpy",
    "scipy",
    "opencv-python",
    "gudhi",
    "scikit-learn",
    "torch",
    "matplotlib",
)


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    cleaned = cleaned.strip("._")
    return cleaned or "artifact"


def build_sample_id(split: str, subject_id: str, video_id: str) -> str:
    split_token = split.strip().lower()
    subject_token = subject_id.strip().upper()
    video_token = Path(video_id).stem.strip()
    if not split_token or not subject_token or not video_token:
        raise ValueError(
            "split, subject_id y video_id deben estar presentes para construir sample_id"
        )
    return f"{split_token}__{subject_token}__{video_token}"


def relpath_str(target: Path, base_dir: Path) -> str:
    target = target.resolve()
    base_dir = base_dir.resolve()
    return os.path.relpath(target, start=base_dir)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def version_snapshot(
    packages: Iterable[str] = DEFAULT_VERSION_PACKAGES,
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def serialize_args(args: Any) -> dict[str, Any]:
    if hasattr(args, "__dict__"):
        payload = vars(args)
    elif isinstance(args, dict):
        payload = args
    else:
        raise TypeError(f"No se puede serializar args de tipo {type(args)!r}")

    serialized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            serialized[key] = str(value)
        else:
            serialized[key] = value
    return serialized


def config_hash(args: Any) -> str:
    payload = serialize_args(args)
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def runtime_metadata(
    *, stage: str, args: Any, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "config_hash": config_hash(args),
        "args": serialize_args(args),
        "package_versions": version_snapshot(),
    }
    if extra:
        payload.update(extra)
    return payload


def seed_everything(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    except Exception:
        return
