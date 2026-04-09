from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import load_tensor_artifact, read_json


def load_phase2_compatibility_payload(run_dir: Path) -> dict[str, Any]:
    phase2_dir = run_dir / "phase2"
    frozen_payload = read_json(phase2_dir / "phase3_compatibility_payload.json", default={})
    if isinstance(frozen_payload, dict) and frozen_payload:
        payload = dict(frozen_payload)
        for key, path_key in (
            ("core_feature_tensor_array", "core_feature_tensor_path"),
            ("direct_feature_tensor_array", "direct_feature_tensor_path"),
            ("hidden_driver_feature_tensor_array", "hidden_driver_feature_tensor_path"),
            ("multiscale_support_feature_tensor_array", "multiscale_support_feature_tensor_path"),
        ):
            tensor_path = Path(str(payload.get(path_key) or ""))
            if tensor_path.exists():
                payload[key] = np.asarray(load_tensor_artifact(tensor_path), dtype=np.float32)
            else:
                payload[key] = np.zeros((0, 0, 0), dtype=np.float32)
        if "core_feature_tensor_array" not in payload:
            payload["core_feature_tensor_array"] = np.asarray(payload.get("direct_feature_tensor_array"), dtype=np.float32)
        payload["compatibility_payload_source"] = "frozen_artifact"
        return payload
    missing = phase2_dir / "phase3_compatibility_payload.json"
    raise FileNotFoundError(
        f"Frozen Phase 2 compatibility payload is required for historical replay but is missing: {missing}"
    )


__all__ = ["load_phase2_compatibility_payload"]
