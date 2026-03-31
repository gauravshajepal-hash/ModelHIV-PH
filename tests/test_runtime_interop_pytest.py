from __future__ import annotations

import numpy as np
import pytest

from epigraph_ph.runtime import _canonical_interop_device_kind, torch_to_jax_handoff

try:
    import jax
except Exception:  # pragma: no cover
    jax = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


pytestmark = pytest.mark.skipif(jax is None or torch is None, reason="torch+jax required")


def test_canonical_interop_device_kind_aliases_cuda_to_gpu() -> None:
    assert _canonical_interop_device_kind("cuda") == "gpu"
    assert _canonical_interop_device_kind("gpu") == "gpu"
    assert _canonical_interop_device_kind("cpu") == "cpu"


def test_torch_to_jax_handoff_cpu_uses_dlpack() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    jax_array, report = torch_to_jax_handoff(tensor, prefer_dlpack=True)

    assert np.allclose(np.asarray(jax_array), tensor.numpy())
    assert report["used_dlpack"] is True
    assert report["source_device"] == "cpu"
    assert report["source_device_kind"] == "cpu"
    assert report["target_backend"] == jax.default_backend()
    assert report["target_backend_kind"] == _canonical_interop_device_kind(jax.default_backend())
    assert report["device_transfer"] == "none"
    assert report["reason"] == "success"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA tensor path requires CUDA")
def test_torch_to_jax_handoff_cuda_tensor_harmonizes_to_cpu_dlpack_when_jax_is_cpu_only() -> None:
    if jax.default_backend() != "cpu":
        pytest.skip("This test is only meaningful when JAX is CPU-only")

    tensor = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4)
    jax_array, report = torch_to_jax_handoff(tensor, prefer_dlpack=True)

    assert np.allclose(np.asarray(jax_array), tensor.detach().cpu().numpy())
    assert report["used_dlpack"] is True
    assert report["source_device"] == "cuda"
    assert report["source_device_kind"] == "gpu"
    assert report["target_backend"] == "cpu"
    assert report["target_backend_kind"] == "cpu"
    assert report["device_transfer"] == "cuda_to_cpu"
    assert report["zero_copy_scope"] == "host_only_after_device_transfer"
    assert report["reason"] == "cpu_dlpack_after_device_transfer"
