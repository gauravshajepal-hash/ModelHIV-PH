from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path, *, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_tensor_artifact(path: Path) -> np.ndarray:
    artifact = np.load(path)
    if isinstance(artifact, np.ndarray):
        return artifact
    if "values" in artifact.files:
        return np.asarray(artifact["values"])
    if len(artifact.files) == 1:
        return np.asarray(artifact[artifact.files[0]])
    raise ValueError(f"Cannot infer tensor array from artifact: {path}")
