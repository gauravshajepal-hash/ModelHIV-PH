# WSL GPU Bootstrap For Phase 3

This repo now includes a reproducible WSL2 bootstrap path for real Torch-CUDA to JAX-GPU DLPack interop.

## Why this exists

Native Windows can run Torch CUDA, but JAX GPU support is not the target environment here. The supported route on this machine is WSL2 Ubuntu with NVIDIA GPU access. That is the environment where Phase 3 can use a true end-to-end CUDA handoff without the Windows-side `cuda_to_cpu` harmonization step.

## Files

- Bootstrap script: [scripts/bootstrap_wsl_gpu_env.sh](/D:/EpiGraph_PH/scripts/bootstrap_wsl_gpu_env.sh)
- PowerShell wrapper: [scripts/bootstrap_wsl_gpu_env.ps1](/D:/EpiGraph_PH/scripts/bootstrap_wsl_gpu_env.ps1)
- Probe script: [scripts/probe_wsl_gpu_interop.py](/D:/EpiGraph_PH/scripts/probe_wsl_gpu_interop.py)

## Bootstrap from Windows PowerShell

```powershell
Set-Location D:\EpiGraph_PH
.\scripts\bootstrap_wsl_gpu_env.ps1
```

That creates a user-local WSL venv at `/home/gaurav/.venvs/modelhiv-ph-gpu` by default and installs:

- `torch` from the CUDA 12.6 wheel index
- `gpytorch`
- `jax[cuda12]`
- `numpyro`
- repo runtime dependencies
- editable `modelhiv-ph`

## Probe the zero-copy handoff inside WSL

```bash
source /home/gaurav/.venvs/modelhiv-ph-gpu/bin/activate
cd /mnt/d/EpiGraph_PH
python scripts/probe_wsl_gpu_interop.py \
  --output artifacts/runs/manual-wsl-gpu-interop-20260330/phase3/interop_report.json
```

Expected success conditions:

- `torch_cuda_available = true`
- `jax_default_backend = "gpu"` or `"cuda"`
- `interop_report.used_dlpack = true`
- `interop_report.device_transfer = "none"`
- `interop_report.zero_copy_scope = "end_to_end"`

If `device_transfer = "cuda_to_cpu"`, the probe was not actually running against a GPU-backed JAX environment.
