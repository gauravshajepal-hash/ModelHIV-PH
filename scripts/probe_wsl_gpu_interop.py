from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from epigraph_ph.runtime import ensure_dir, torch_to_jax_handoff, utc_now_iso, write_json

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"JAX is required for the WSL GPU interop probe: {exc}") from exc

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Torch is required for the WSL GPU interop probe: {exc}") from exc


def _nvidia_smi_text() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # pragma: no cover
        return f"nvidia-smi unavailable: {type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe WSL Torch-CUDA to JAX-GPU DLPack interop")
    parser.add_argument("--output", required=True, help="Path to write the JSON report")
    parser.add_argument("--matrix-shape", default="1024,256", help="Probe tensor shape as rows,cols")
    args = parser.parse_args()

    rows_text, cols_text = [part.strip() for part in str(args.matrix_shape).split(",", 1)]
    rows = int(rows_text)
    cols = int(cols_text)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    if not torch.cuda.is_available():
        raise SystemExit("Torch CUDA is not available inside WSL")
    if jax.default_backend() not in {"gpu", "cuda"}:
        raise SystemExit(f"JAX default backend is not GPU-backed: {jax.default_backend()}")

    tensor = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
    jax_array, report = torch_to_jax_handoff(tensor, prefer_dlpack=True)
    jax_sum = float(jax.device_get(jnp.sum(jax_array)))
    numpy_sum = float(np.sum(np.arange(rows * cols, dtype=np.float32)))

    payload = {
        "timestamp_utc": utc_now_iso(),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "jax_version": getattr(jax, "__version__", "unknown"),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_device_name": torch.cuda.get_device_name(0),
        "jax_default_backend": str(jax.default_backend()),
        "jax_devices": [str(device) for device in jax.devices()],
        "probe_shape": [rows, cols],
        "interop_report": report,
        "jax_sum": jax_sum,
        "expected_sum": numpy_sum,
        "sum_close": bool(np.isclose(jax_sum, numpy_sum)),
        "nvidia_smi": _nvidia_smi_text(),
    }
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
