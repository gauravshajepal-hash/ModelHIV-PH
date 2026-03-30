from __future__ import annotations


def ensure_builtin_plugins_registered() -> None:
    from . import hiv  # noqa: F401


__all__ = ["ensure_builtin_plugins_registered"]
