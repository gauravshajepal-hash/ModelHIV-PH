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
    from jax import dlpack as jax_dlpack
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    jax_dlpack = None


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


def _canonical_interop_device_kind(name: str | None) -> str:
    lowered = str(name or "").strip().lower()
    if lowered in {"cuda", "gpu"}:
        return "gpu"
    if lowered in {"cpu"}:
        return "cpu"
    return lowered or "unknown"


def torch_to_jax_handoff(array: Any, *, prefer_dlpack: bool = True) -> tuple[Any, dict[str, Any]]:
    if torch is None or jax is None:
        result = to_jax_array(array)
        return result, {"source": "torch" if torch is not None else "numpy", "target": "jax", "used_dlpack": False, "reason": "backend_unavailable"}
    tensor = array if isinstance(array, torch.Tensor) else to_torch_tensor(array, device="cpu")
    source_device = str(getattr(getattr(tensor, "device", None), "type", "cpu") or "cpu")
    try:
        target_backend = str(jax.default_backend())
    except Exception:
        target_backend = "unknown"
    source_kind = _canonical_interop_device_kind(source_device)
    target_kind = _canonical_interop_device_kind(target_backend)
    dlpack_tensor = tensor.detach().contiguous()
    device_transfer = "none"
    zero_copy_scope = "end_to_end" if source_kind == target_kind else "unknown"
    if prefer_dlpack and source_kind == "gpu" and target_kind == "cpu":
        dlpack_tensor = dlpack_tensor.to("cpu")
        device_transfer = "cuda_to_cpu"
        zero_copy_scope = "host_only_after_device_transfer"
    if prefer_dlpack and jax_dlpack is not None:
        try:
            import torch.utils.dlpack as torch_dlpack

            try:
                jax_array = jax_dlpack.from_dlpack(dlpack_tensor)
            except TypeError:
                jax_array = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(dlpack_tensor))
            reason = "success"
            if device_transfer != "none":
                reason = "cpu_dlpack_after_device_transfer"
            return jax_array, {
                "source": "torch",
                "target": "jax",
                "used_dlpack": True,
                "reason": reason,
                "source_device": source_device,
                "source_device_kind": source_kind,
                "target_backend": target_backend,
                "target_backend_kind": target_kind,
                "device_transfer": device_transfer,
                "zero_copy_scope": zero_copy_scope,
            }
        except Exception as exc:  # pragma: no cover
            result = to_jax_array(tensor)
            return result, {
                "source": "torch",
                "target": "jax",
                "used_dlpack": False,
                "reason": f"fallback:{type(exc).__name__}",
                "source_device": source_device,
                "source_device_kind": source_kind,
                "target_backend": target_backend,
                "target_backend_kind": target_kind,
                "device_transfer": device_transfer,
                "zero_copy_scope": "none",
            }
    result = to_jax_array(tensor)
    return result, {
        "source": "torch",
        "target": "jax",
        "used_dlpack": False,
        "reason": "dlpack_disabled",
        "source_device": source_device,
        "source_device_kind": source_kind,
        "target_backend": target_backend,
        "target_backend_kind": target_kind,
        "device_transfer": device_transfer,
        "zero_copy_scope": "none",
    }


def jax_to_torch_handoff(array: Any, *, prefer_dlpack: bool = True, device: str | None = None):
    if jax is None or torch is None:
        tensor = to_torch_tensor(to_numpy(array), device=device or "cpu")
        return tensor, {"source": "jax" if jax is not None else "numpy", "target": "torch", "used_dlpack": False, "reason": "backend_unavailable"}
    if prefer_dlpack and jax_dlpack is not None:
        try:
            import torch.utils.dlpack as torch_dlpack

            tensor = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(array))
            if device is not None:
                tensor = tensor.to(device)
            return tensor, {"source": "jax", "target": "torch", "used_dlpack": True, "reason": "success"}
        except Exception as exc:  # pragma: no cover
            tensor = to_torch_tensor(to_numpy(array), device=device or "cpu")
            return tensor, {"source": "jax", "target": "torch", "used_dlpack": False, "reason": f"fallback:{type(exc).__name__}"}
    tensor = to_torch_tensor(to_numpy(array), device=device or "cpu")
    return tensor, {"source": "jax", "target": "torch", "used_dlpack": False, "reason": "dlpack_disabled"}


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


def write_gold_standard_package(
    *,
    phase_dir: Path,
    phase_name: str,
    checks: list[dict[str, Any]],
    gold_profile: dict[str, Any],
    summary: dict[str, Any] | None = None,
    profile_id: str | None = None,
    stage_manifest_path: str | None = None,
) -> dict[str, str]:
    ensure_dir(phase_dir)
    passed = sum(1 for check in checks if bool(check.get("passed")))
    failed = len(checks) - passed
    standards = [dict(row) for row in list(gold_profile.get("standards", []))]
    manifest = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "generated_at": utc_now_iso(),
        "gold_standard_mode": str(gold_profile.get("mode") or "unspecified"),
        "truth_claim_level": str(gold_profile.get("truth_claim_level") or "unspecified"),
        "description": str(gold_profile.get("description") or ""),
        "benchmark_policy": str(gold_profile.get("benchmark_policy") or ""),
        "standards": standards,
        "judging_principles": list(gold_profile.get("judging_principles", [])),
        "required_claims": list(gold_profile.get("required_claims", [])),
        "related_layers": list(gold_profile.get("related_layers", [])),
        "stage_manifest_path": stage_manifest_path or "",
        "required_files": [
            str(phase_dir / "gold_standard_manifest.json"),
            str(phase_dir / "gold_standard_checks.json"),
            str(phase_dir / "gold_standard_summary.json"),
        ],
    }
    summary_payload = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "overall_passed": failed == 0,
        "passed_check_count": passed,
        "failed_check_count": failed,
        "gold_standard_mode": manifest["gold_standard_mode"],
        "truth_claim_level": manifest["truth_claim_level"],
        "benchmark_policy": manifest["benchmark_policy"],
        "standard_count": len(standards),
    }
    if summary:
        summary_payload.update(summary)
    manifest_path = phase_dir / "gold_standard_manifest.json"
    checks_path = phase_dir / "gold_standard_checks.json"
    summary_path = phase_dir / "gold_standard_summary.json"
    write_json(manifest_path, manifest)
    write_json(checks_path, checks)
    write_json(summary_path, summary_payload)
    return {
        "gold_standard_manifest": str(manifest_path),
        "gold_standard_checks": str(checks_path),
        "gold_standard_summary": str(summary_path),
    }


def _tensor_summary_path(value_path: str | Path) -> Path:
    path = Path(value_path)
    if path.suffix == ".npz":
        return path.with_suffix(".summary.json")
    return path.parent / f"{path.stem}.summary.json"


def _boundary_shape_check(boundary: dict[str, Any]) -> dict[str, Any]:
    name = str(boundary.get("name") or "")
    kind = str(boundary.get("kind") or "")
    path = Path(str(boundary.get("path") or ""))
    exists = path.exists()
    result: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "path": str(path),
        "exists": exists,
        "passed": False,
    }
    if not exists:
        result["reason"] = "missing_path"
        return result
    if kind == "tensor":
        summary_path = Path(str(boundary.get("summary_path") or _tensor_summary_path(path)))
        summary = read_json(summary_path, default={}) if summary_path.exists() else {}
        shape = list(summary.get("shape", []))
        axis_names = list(summary.get("axis_names", []))
        expected_shape = list(boundary.get("expected_shape", []))
        expected_axis_names = list(boundary.get("expected_axis_names", []))
        min_rank = int(boundary.get("min_rank", 0) or 0)
        finite_required = bool(boundary.get("finite_required", False))
        nan_count = int(summary.get("nan_count", 0) or 0)
        shape_ok = bool(shape) and (not expected_shape or shape == expected_shape)
        axis_ok = (not expected_axis_names) or axis_names == expected_axis_names
        rank_ok = len(shape) >= min_rank
        finite_ok = (not finite_required) or nan_count == 0
        result.update(
            {
                "summary_path": str(summary_path),
                "actual_shape": shape,
                "axis_names": axis_names,
                "expected_shape": expected_shape,
                "expected_axis_names": expected_axis_names,
                "nan_count": nan_count,
                "passed": bool(shape_ok and axis_ok and rank_ok and finite_ok),
            }
        )
        return result
    payload = read_json(path, default=[] if kind == "json_rows" else {})
    if kind == "json_rows":
        row_count = len(payload) if isinstance(payload, list) else 0
        min_rows = int(boundary.get("min_rows", 0) or 0)
        expected_row_count = boundary.get("expected_row_count")
        exact_ok = expected_row_count is None or row_count == int(expected_row_count)
        result.update(
            {
                "row_count": row_count,
                "min_rows": min_rows,
                "expected_row_count": int(expected_row_count) if expected_row_count is not None else None,
                "passed": row_count >= min_rows and exact_ok,
            }
        )
        return result
    if kind == "json_dict":
        expected_keys = list(boundary.get("expected_keys", []))
        key_count = len(payload) if isinstance(payload, dict) else 0
        keys_ok = isinstance(payload, dict) and all(key in payload for key in expected_keys)
        result.update(
            {
                "key_count": key_count,
                "expected_keys": expected_keys,
                "passed": bool(isinstance(payload, dict) and keys_ok),
            }
        )
        return result
    result["reason"] = "unsupported_kind"
    return result


def write_boundary_shape_package(
    *,
    phase_dir: Path,
    phase_name: str,
    boundaries: list[dict[str, Any]],
    profile_id: str | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, str]:
    ensure_dir(phase_dir)
    checks = [_boundary_shape_check(boundary) for boundary in boundaries]
    passed = sum(1 for check in checks if bool(check.get("passed")))
    failed = len(checks) - passed
    manifest = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "generated_at": utc_now_iso(),
        "boundary_count": len(boundaries),
        "boundaries": [
            {
                "name": str(boundary.get("name") or ""),
                "kind": str(boundary.get("kind") or ""),
                "path": str(boundary.get("path") or ""),
            }
            for boundary in boundaries
        ],
        "required_files": [
            str(phase_dir / "boundary_shape_manifest.json"),
            str(phase_dir / "boundary_shape_checks.json"),
            str(phase_dir / "boundary_shape_summary.json"),
        ],
    }
    summary_payload = {
        "phase_name": phase_name,
        "profile_id": profile_id or "legacy",
        "overall_passed": failed == 0,
        "passed_check_count": passed,
        "failed_check_count": failed,
        "boundary_count": len(boundaries),
    }
    if summary:
        summary_payload.update(summary)
    manifest_path = phase_dir / "boundary_shape_manifest.json"
    checks_path = phase_dir / "boundary_shape_checks.json"
    summary_path = phase_dir / "boundary_shape_summary.json"
    write_json(manifest_path, manifest)
    write_json(checks_path, checks)
    write_json(summary_path, summary_payload)
    return {
        "boundary_shape_manifest": str(manifest_path),
        "boundary_shape_checks": str(checks_path),
        "boundary_shape_summary": str(summary_path),
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
