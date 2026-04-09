from __future__ import annotations

from typing import Any


def run_phase2_build(*args: Any, **kwargs: Any) -> dict[str, Any]:
    from .pipeline import run_phase2_build as _run_phase2_build

    return _run_phase2_build(*args, **kwargs)


__all__ = ["run_phase2_build"]
