from __future__ import annotations

from typing import Any

import numpy as np

from epigraph_ph.runtime import write_json

ALLOWED_SOURCE_TYPES: tuple[str, ...] = (
    "estimated",
    "semi_markov",
    "bayesian_prior",
    "bayesian_posterior",
    "physical_constraint",
    "numerical_guard",
)

REQUIRED_NUMERIC_FIELDS: tuple[str, ...] = (
    "name",
    "value",
    "role",
    "source_type",
    "estimation_data",
    "estimation_method",
    "uncertainty",
    "why_needed",
)


def float32_epsilon() -> float:
    return float(np.finfo(np.float32).eps)


def numerical_guard_entry(*, name: str, role: str, why_needed: str) -> dict[str, Any]:
    return {
        "name": name,
        "value": float32_epsilon(),
        "role": role,
        "source_type": "numerical_guard",
        "estimation_data": "numpy.float32 machine epsilon",
        "estimation_method": "np.finfo(np.float32).eps",
        "uncertainty": "machine-defined constant",
        "why_needed": why_needed,
    }


def validate_numeric_justification(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for entry in entries:
        missing = [field for field in REQUIRED_NUMERIC_FIELDS if field not in entry]
        if missing:
            raise ValueError(f"numeric justification entry is missing fields: {missing}")
        source_type = str(entry["source_type"])
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"unsupported numeric justification source_type: {source_type}")
        validated.append(
            {
                "name": str(entry["name"]),
                "value": entry["value"],
                "role": str(entry["role"]),
                "source_type": source_type,
                "estimation_data": entry["estimation_data"],
                "estimation_method": str(entry["estimation_method"]),
                "uncertainty": entry["uncertainty"],
                "why_needed": str(entry["why_needed"]),
            }
        )
    return validated


def write_numeric_justification(path, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = validate_numeric_justification(entries)
    write_json(path, payload)
    return payload
