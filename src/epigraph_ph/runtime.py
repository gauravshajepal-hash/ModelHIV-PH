from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None


@dataclass(slots=True)
class BackendStatus:
    name: str
    available: bool
    selected: bool = False
    device: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TensorArtifact:
    backend: str
    device: str
    dtype: str
    shape: list[int]
    axis_names: list[str]
    value_path: str
    summary_path: str
    pt_path: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    try:
        path.write_text(text, encoding="utf-8")
    except FileNotFoundError:
        ensure_dir(path.parent)
        path.write_text(text, encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


def detect_backends() -> dict[str, BackendStatus]:
    force_cpu = str(os.getenv("EPIGRAPH_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"}
    torch_device = "cpu"
    torch_available = torch is not None
    if torch_available and not force_cpu:
        try:
            if torch.cuda.is_available():
                torch_device = "cuda"
        except Exception:
            torch_device = "cpu"
    jax_device = "cpu"
    jax_available = jax is not None
    if jax_available and not force_cpu:
        try:
            devices = jax.devices()
            if devices:
                jax_device = devices[0].platform
        except Exception:
            jax_device = "cpu"
    return {
        "torch": BackendStatus("torch", torch_available, torch_available and not force_cpu, torch_device, notes="force_cpu" if force_cpu else ""),
        "jax": BackendStatus("jax", jax_available, jax_available and not force_cpu, jax_device, notes="force_cpu" if force_cpu else ""),
    }


def choose_torch_device(prefer_gpu: bool = True) -> str:
    if torch is None:
        return "cpu"
    if str(os.getenv("EPIGRAPH_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    if prefer_gpu:
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
    return "cpu"


def choose_jax_device(prefer_gpu: bool = True) -> str:
    if jax is None:
        return "cpu"
    if str(os.getenv("EPIGRAPH_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return "cpu"
    try:
        devices = jax.devices()
        if prefer_gpu:
            for device in devices:
                if device.platform in {"gpu", "cuda"}:
                    return device.platform
        return devices[0].platform if devices else "cpu"
    except Exception:
        return "cpu"


def to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if torch is not None and isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    if jax is not None:
        try:
            return np.asarray(array)
        except Exception:
            pass
    return np.asarray(array)


def to_torch_tensor(array: Any, *, device: str | None = None, dtype: Any | None = None):
    if torch is None:
        raise RuntimeError("torch is not available")
    tensor = array if isinstance(array, torch.Tensor) else torch.as_tensor(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def to_jax_array(array: Any):
    if jax is None:
        raise RuntimeError("jax is not available")
    return jnp.asarray(to_numpy(array))


def tensor_summary(array: Any, *, axis_names: list[str], backend: str) -> dict[str, Any]:
    np_array = to_numpy(array)
    summary = {
        "backend": backend,
        "dtype": str(np_array.dtype),
        "shape": list(np_array.shape),
        "axis_names": axis_names,
        "nan_count": int(np.isnan(np_array).sum()) if np.issubdtype(np_array.dtype, np.floating) else 0,
        "min": float(np.nanmin(np_array)) if np_array.size else None,
        "max": float(np.nanmax(np_array)) if np_array.size else None,
        "mean": float(np.nanmean(np_array)) if np_array.size else None,
    }
    return summary


def save_tensor_artifact(
    *,
    array: Any,
    axis_names: list[str],
    artifact_dir: Path,
    stem: str,
    backend: str,
    device: str = "",
    notes: list[str] | None = None,
    save_pt: bool = True,
) -> dict[str, Any]:
    ensure_dir(artifact_dir)
    np_array = to_numpy(array)
    value_path = artifact_dir / f"{stem}.npz"
    np.savez_compressed(value_path, values=np_array)
    pt_path: Path | None = None
    if save_pt and backend == "torch" and torch is not None:
        try:
            pt_path = artifact_dir / f"{stem}.pt"
            torch.save(to_torch_tensor(np_array, device="cpu"), pt_path)
        except Exception:
            pt_path = None
    summary = tensor_summary(np_array, axis_names=axis_names, backend=backend)
    summary["device"] = device
    summary["notes"] = notes or []
    summary_path = artifact_dir / f"{stem}.summary.json"
    write_json(summary_path, summary)
    artifact = TensorArtifact(
        backend=backend,
        device=device,
        dtype=str(np_array.dtype),
        shape=list(np_array.shape),
        axis_names=axis_names,
        value_path=str(value_path),
        summary_path=str(summary_path),
        pt_path=str(pt_path) if pt_path else None,
        notes=notes or [],
    )
    return artifact.to_dict()


def load_tensor_artifact(value_path: str | Path) -> np.ndarray:
    payload = np.load(Path(value_path))
    if isinstance(payload, np.lib.npyio.NpzFile):
        return np.asarray(payload["values"])
    return np.asarray(payload)


def write_ground_truth_package(
    *,
    phase_dir: Path,
    phase_name: str,
    checks: list[dict[str, Any]],
    summary: dict[str, Any] | None = None,
    profile_id: str | None = None,
    truth_sources: list[str] | None = None,
    stage_manifest_path: str | None = None,
) -> dict[str, str]:
    ensure_dir(phase_dir)
    passed = sum(1 for check in checks if bool(check.get("passed")))
    failed = len(checks) - passed
    manifest = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "generated_at": utc_now_iso(),
        "check_count": len(checks),
        "truth_sources": truth_sources or [],
        "stage_manifest_path": stage_manifest_path or "",
        "required_files": [
            str(phase_dir / "ground_truth_manifest.json"),
            str(phase_dir / "ground_truth_checks.json"),
            str(phase_dir / "ground_truth_summary.json"),
        ],
    }
    summary_payload = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "overall_passed": failed == 0,
        "passed_check_count": passed,
        "failed_check_count": failed,
    }
    if summary:
        summary_payload.update(summary)
    manifest_path = phase_dir / "ground_truth_manifest.json"
    checks_path = phase_dir / "ground_truth_checks.json"
    summary_path = phase_dir / "ground_truth_summary.json"
    write_json(manifest_path, manifest)
    write_json(checks_path, checks)
    write_json(summary_path, summary_payload)
    return {
        "ground_truth_manifest": str(manifest_path),
        "ground_truth_checks": str(checks_path),
        "ground_truth_summary": str(summary_path),
    }


def torch_to_numpy_handoff(array: Any) -> np.ndarray:
    return to_numpy(array)


def numpy_to_jax_handoff(array: Any):
    return to_jax_array(array)


@dataclass(slots=True)
class RunContext:
    run_id: str
    plugin_id: str
    run_dir: Path

    @classmethod
    def create(cls, *, run_id: str, plugin_id: str) -> "RunContext":
        return cls(run_id=run_id, plugin_id=plugin_id, run_dir=ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id))

    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def update_manifest(self, **sections: Any) -> None:
        payload = read_json(self.manifest_path(), default={})
        if not isinstance(payload, dict):
            payload = {}
        payload.update(sections)
        write_json(self.manifest_path(), payload)

    def record_stage_outputs(self, stage: str, outputs: list[Path]) -> None:
        payload = read_json(self.manifest_path(), default={})
        if not isinstance(payload, dict):
            payload = {}
        stage_outputs = dict(payload.get("stage_outputs", {}))
        stage_outputs[stage] = [str(path) for path in outputs]
        payload["stage_outputs"] = stage_outputs
        write_json(self.manifest_path(), payload)
