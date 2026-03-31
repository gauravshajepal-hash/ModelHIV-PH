from __future__ import annotations

from typing import Any

__all__ = [
    "build_deep_pipeline_audit",
    "build_phase0_literature_review",
    "build_gold_standard_report",
    "run_module_test_report",
]


def build_deep_pipeline_audit(*args: Any, **kwargs: Any):
    from .deep_pipeline_audit import build_deep_pipeline_audit as _impl

    return _impl(*args, **kwargs)


def build_phase0_literature_review(*args: Any, **kwargs: Any):
    from .literature_review import build_phase0_literature_review as _impl

    return _impl(*args, **kwargs)


def build_gold_standard_report(*args: Any, **kwargs: Any):
    from .gold_standard_report import build_gold_standard_report as _impl

    return _impl(*args, **kwargs)


def run_module_test_report(*args: Any, **kwargs: Any):
    from .module_test_report import run_module_test_report as _impl

    return _impl(*args, **kwargs)
