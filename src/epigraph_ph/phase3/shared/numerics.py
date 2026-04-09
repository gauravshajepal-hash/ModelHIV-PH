from __future__ import annotations

import numpy as np


FLOAT32_EPS = float(np.finfo(np.float32).eps)
# Use a square-root epsilon scale for division/logit safeguards so the floor is
# above raw machine noise but still negligible relative to modeled shares.
SAFE_DIVISION_EPS = float(np.sqrt(FLOAT32_EPS))


def safe_floor(value: float | None) -> float:
    numeric = float(value) if value is not None else 0.0
    return max(numeric, SAFE_DIVISION_EPS)

