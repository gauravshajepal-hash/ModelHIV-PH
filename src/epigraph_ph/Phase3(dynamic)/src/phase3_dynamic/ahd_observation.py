"""Reported AHD classification with explicitly unknown clinical/CD4 status.

The latent AHD fraction is not identified by incomplete classification alone.
Profile deviance below treats detection in the two disease groups separately.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import xlogy


@dataclass(frozen=True)
class AhdStatus:
    advanced: int
    known_nonadvanced: int
    unknown: int

    def __post_init__(self) -> None:
        for count in self.counts:
            if isinstance(count, bool) or not math.isfinite(count) or int(count) != count or count < 0:
                raise ValueError("AHD status counts must be finite nonnegative integers")
        if self.total <= 0:
            raise ValueError("Empty classification denominator")

    @property
    def counts(self) -> tuple[int, int, int]:
        return self.advanced, self.known_nonadvanced, self.unknown

    @property
    def total(self) -> int:
        return sum(self.counts)

    @property
    def bounds(self) -> tuple[float, float]:
        return self.advanced / self.total, (self.advanced + self.unknown) / self.total

    @property
    def classified_fraction(self) -> float:
        return (self.advanced + self.known_nonadvanced) / self.total

    @property
    def complete_case_fraction(self) -> float | None:
        known = self.advanced + self.known_nonadvanced
        return self.advanced / known if known else None

    @classmethod
    def from_ledger(cls, row: dict) -> AhdStatus:
        if row.get("observation_role") != "auxiliary_likelihood" or "classification_likelihood" not in row.get("allowed_use", []):
            raise ValueError("Row not allowed in the AHD classification likelihood")
        if row.get("status") != "accepted" or row.get("measurement_semantics") != "joint_classification_count":
            raise ValueError("Unaccepted or incorrectly typed classification row")
        result = cls(row["advanced"], row["known_nonadvanced"], row["unknown"])
        if result.total != row["total_diagnoses"]:
            raise ValueError("AHD status partition does not match diagnosis denominator")
        return result


def observation_probabilities(p: float, q_advanced: float, q_nonadvanced: float) -> np.ndarray:
    values = np.array([p, q_advanced, q_nonadvanced], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
        raise ValueError("Fractions must be in [0, 1]")
    a, e = p * q_advanced, (1 - p) * q_nonadvanced
    return np.array([a, e, max(0.0, 1 - a - e)])


def profile_probabilities(status: AhdStatus, p: float) -> np.ndarray:
    """Exact constrained MLE of the observation probabilities for a fixed p."""
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("AHD probability outside [0, 1]")
    a, e, u = np.array(status.counts, dtype=float) / status.total
    lower, upper = status.bounds
    if lower <= p <= upper:
        return np.array([a, e, u])
    if p < lower:
        denominator = e + u
        return np.array([p, (1-p) * e / denominator, (1-p) * u / denominator]) if denominator else np.array([p, 1-p, 0])
    denominator = a + u
    return np.array([p * a / denominator, 1-p, p * u / denominator]) if denominator else np.array([0, 1-p, p])


def classification_deviance(status: AhdStatus, probabilities: np.ndarray | list[float]) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (3,) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Expected a three-category probability simplex")
    counts = np.asarray(status.counts, dtype=float)
    # Twice the likelihood gap to the saturated multinomial model. No pseudocounts.
    return max(0.0, float(2 * np.sum(xlogy(counts, counts / status.total) - xlogy(counts, probabilities))))


def profile_deviance(status: AhdStatus, p: float) -> float:
    return classification_deviance(status, profile_probabilities(status, p))


def mar_deviance(status: AhdStatus, p: float) -> float:
    q = status.classified_fraction
    return classification_deviance(status, observation_probabilities(p, q, q))


def naive_binary_deviance(status: AhdStatus, p: float) -> float:
    """Explicitly misspecified historical control: unknown classified as non-AHD."""
    collapsed = AhdStatus(status.advanced, status.known_nonadvanced + status.unknown, 0)
    return classification_deviance(collapsed, observation_probabilities(p, 1, 1))


def backlog_status_deviance(simulated: dict[int, dict], status: AhdStatus, months: list[int], *, assumption: str = "profile") -> float:
    """Candidate-only adapter for the conserved monthly backlog simulator.

    Requires an explicit assumption equating the late-diagnosis state with AHD.
    Profiles classification, never conditions the hazard on held-out status.
    """
    if len(months) != 3 or sorted(set(months)) != list(range(min(months), min(months) + 3)):
        raise ValueError("A full contiguous quarterly emission is required")
    if any(m not in simulated for m in months):
        raise ValueError("Missing model months; do not pad or divide quarterly evidence")
    total = sum(simulated[m]["new_diagnosed_cases_period"] for m in months)
    late = sum(simulated[m]["late_diagnoses"] for m in months)
    if not math.isfinite(total + late) or total <= 0 or not 0 <= late <= total:
        raise ValueError("Invalid simulated diagnosis emission")
    funcs = {"profile": profile_deviance, "mar": mar_deviance, "naive_binary": naive_binary_deviance}
    if assumption not in funcs:
        raise ValueError("Unknown classification assumption")
    return funcs[assumption](status, late / total)
