from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def normalize_name(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return Path(str(value)).stem


def load_manifest(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else {}
