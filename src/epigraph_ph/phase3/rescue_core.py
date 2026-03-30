from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.core.province_archetypes import build_synthetic_province_library, infer_province_archetype_priors
from epigraph_ph.geography import infer_region_code, is_national_geo
from epigraph_ph.phase0.models import Phase0BackendStatus, Phase0ManifestArtifact
from epigraph_ph.runtime import (
    RunContext,
    choose_jax_device,
    choose_torch_device,
    detect_backends,
    ensure_dir,
    load_tensor_artifact,
    numpy_to_jax_handoff,
    save_tensor_artifact,
    set_global_seed,
    to_numpy,
    to_torch_tensor,
    utc_now_iso,
    write_json,
    read_json,
    write_ground_truth_package,
)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jax = None
    jnp = None

try:
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import Predictive, SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal
    from numpyro.optim import Adam as NumpyroAdam
except Exception:  # pragma: no cover
    numpyro = None
    dist = None
    Predictive = None
    SVI = None
    Trace_ELBO = None
    AutoNormal = None
    NumpyroAdam = None


RESCUE_PROFILE_ID = "hiv_rescue_v1"
RESCUE_V2_PROFILE_ID = "hiv_rescue_v2"
RESCUE_INFERENCE_FAMILY = "torch_map"
STATE_NAMES = ["U", "D", "A", "V", "L"]
TRANSITION_NAMES = ["U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A"]
CD4_CATALOG = ["lt_200", "200_349", "350_499", "500_plus"]
DURATION_CATALOG = ["0_2", "3_5", "6_11", "12_plus"]
DEFAULT_KP_CATALOG = ["remaining_population", "msm", "tgw", "fsw", "clients_fsw", "pwid", "non_kp_partners"]
DEFAULT_AGE_CATALOG = ["15_24", "25_34", "35_49", "50_plus"]
DEFAULT_SEX_CATALOG = ["male", "female"]
OBSERVATION_ORDER = ["diagnosed_stock", "art_stock", "documented_suppression", "testing_coverage", "deaths"]
_HIV_PLUGIN = get_disease_plugin("hiv")
_PHASE3_PRIORS = dict(_HIV_PLUGIN.prior_hyperparameters.get("phase3", {}))
_PHASE3_STABILIZERS = dict(_HIV_PLUGIN.numerical_stabilizers.get("phase3", {}))
_PHASE3_CONSTRAINTS = dict(_HIV_PLUGIN.constraint_settings.get("phase3", {}))

OBSERVATION_CLASS_WEIGHT = dict(
    _PHASE3_PRIORS.get(
        "observation_class_weight",
        {"direct_observed": 1.0, "bounded_observed": 0.75, "proxy_observed": 0.35, "prior_only": 0.12},
    )
)
TARGET_PATTERNS: dict[str, dict[str, list[str]]] = {
    "diagnosed_stock": {
        "tokens": ["diagnos", "hiv_case", "case_count", "reported_case", "testing", "late diagnosis"],
        "domains": ["cascade", "behavior", "population"],
        "pathways": ["testing_uptake", "linkage_to_care"],
    },
    "art_stock": {
        "tokens": [" art ", "treatment", "retention", "adherence", "arv", "care"],
        "domains": ["biology", "policy", "cascade"],
        "pathways": ["linkage_to_care", "retention_adherence", "suppression_outcomes"],
    },
    "documented_suppression": {
        "tokens": ["suppress", "viral", "viral_load", "vl"],
        "domains": ["biology", "cascade"],
        "pathways": ["suppression_outcomes", "biological_progression"],
    },
    "testing_coverage": {
        "tokens": ["testing", "screening", "uptake", "coverage", "monitor", "knowledge"],
        "domains": ["behavior", "policy", "cascade"],
        "pathways": ["testing_uptake", "health_system_reach", "prevention_access"],
    },
    "deaths": {
        "tokens": ["death", "mortality", "fatal", "survival"],
        "domains": ["biology", "population"],
        "pathways": ["biological_progression"],
    },
}
DEFAULT_KP_PRIOR = np.asarray(_PHASE3_PRIORS.get("kp_prior", [0.62, 0.18, 0.03, 0.05, 0.03, 0.03, 0.06]), dtype=np.float32)
DEFAULT_AGE_PRIOR = np.asarray(_PHASE3_PRIORS.get("age_prior", [0.26, 0.36, 0.24, 0.14]), dtype=np.float32)
DEFAULT_SEX_PRIOR = np.asarray(_PHASE3_PRIORS.get("sex_prior", [0.78, 0.22]), dtype=np.float32)
DEFAULT_CD4_PRIOR = np.asarray(_PHASE3_PRIORS.get("cd4_prior", [0.22, 0.28, 0.25, 0.25]), dtype=np.float32)
TRANSITION_PRIOR = np.asarray(_PHASE3_PRIORS.get("transition_prior", [0.10, 0.13, 0.11, 0.05, 0.08]), dtype=np.float32)
DEFAULT_DURATION_TEMPLATE = np.asarray(
    _PHASE3_PRIORS.get(
        "duration_template",
        [
            [0.38, 0.24, 0.20, 0.18],
            [0.55, 0.23, 0.13, 0.09],
            [0.30, 0.24, 0.23, 0.23],
            [0.12, 0.16, 0.24, 0.48],
            [0.28, 0.24, 0.22, 0.26],
        ],
    ),
    dtype=np.float32,
)
AGE_PROGRESS_RATES = np.asarray(_PHASE3_PRIORS.get("age_progress_rates", [1.0 / 120.0, 1.0 / 120.0, 1.0 / 180.0, 0.0]), dtype=np.float32)
OFFICIAL_REFERENCE_POINTS = [
    {
        "label": "UNAIDS Philippines Executive Summary (Most recent data as of 2022)",
        "month": "2022-01",
        "reference": {"first95": 0.63, "second95": 0.66, "overall_suppressed": 0.407},
        "source_url": "https://sustainability.unaids.org/wp-content/uploads/2024/06/Philippines-Executive-Summary-May-2024.pdf",
    },
    {
        "label": "WHO/UNAIDS Philippines joint release (March 2025 operational snapshot, published June 11 2025)",
        "month": "2025-01",
        "reference": {"first95": 0.55, "second95": 0.66, "documented_suppression_among_art": 0.40},
        "source_url": "https://www.who.int/philippines/news/detail/11-06-2025-unaids--who-support-doh-s-call-for-urgent-action-as-the-philippines-faces-the-fastest-growing-hiv-surge-in-the-asia-pacific-region",
    },
]

HARP_PROGRAM_POINTS = [
    {
        "label": "Philippine HIV Care Cascade as of December 2024",
        "month": "2025-01",
        "estimated_plhiv": 216900.0,
        "diagnosed": 135026.0,
        "on_art": 90854.0,
        "viral_load_tested": 43534.0,
        "suppressed": 41164.0,
        "source_label": "2025 PH HIV Estimates Core Team for WHO, page 13",
    }
]


def _phase3_prior(key: str, default: Any) -> Any:
    return _PHASE3_PRIORS.get(key, default)


def _phase3_constraint(key: str, default: Any) -> Any:
    return _PHASE3_CONSTRAINTS.get(key, default)


def _phase3_stabilizer(key: str, default: Any) -> Any:
    return _PHASE3_STABILIZERS.get(key, default)


@contextmanager
def _temporary_reference_points(
    *,
    official_reference_points: list[dict[str, Any]] | None = None,
    harp_program_points: list[dict[str, Any]] | None = None,
):
    global OFFICIAL_REFERENCE_POINTS, HARP_PROGRAM_POINTS
    previous_official = OFFICIAL_REFERENCE_POINTS
    previous_harp = HARP_PROGRAM_POINTS
    if official_reference_points is not None:
        OFFICIAL_REFERENCE_POINTS = [dict(item) for item in official_reference_points]
    if harp_program_points is not None:
        HARP_PROGRAM_POINTS = [dict(item) for item in harp_program_points]
    try:
        yield
    finally:
        OFFICIAL_REFERENCE_POINTS = previous_official
        HARP_PROGRAM_POINTS = previous_harp


def _sigmoid_np(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _logit_np(value: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    clipped = np.clip(value, eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def _national_reference_indices(province_axis: list[str]) -> list[int]:
    direct = [idx for idx, name in enumerate(province_axis) if is_national_geo(str(name))]
    if direct:
        return direct
    subnational = [
        idx
        for idx, name in enumerate(province_axis)
        if (str(name or "").strip().lower() not in {"", "unknown", "global", "international"})
        and not is_national_geo(str(name))
    ]
    return subnational or list(range(len(province_axis)))


def _national_reference_mask(province_axis: list[str]) -> np.ndarray:
    indices = _national_reference_indices(province_axis)
    mask = np.zeros((len(province_axis),), dtype=np.float32)
    if indices:
        mask[indices] = 1.0 / float(len(indices))
    return mask


def _month_ordinal(month_label: str) -> int | None:
    value = str(month_label or "")
    if len(value) >= 7 and value[:4].isdigit() and value[5:7].isdigit():
        return int(value[:4]) * 12 + int(value[5:7]) - 1
    return None


def _month_label_from_ordinal(ordinal: int) -> str:
    year = ordinal // 12
    month = (ordinal % 12) + 1
    return f"{year:04d}-{month:02d}"


def _month_year(month_label: str) -> int | None:
    ordinal = _month_ordinal(month_label)
    if ordinal is None:
        return None
    return ordinal // 12


def _official_anchor_curves(month_axis: list[str]) -> dict[str, np.ndarray]:
    month_ordinals = np.asarray([_month_ordinal(month) if _month_ordinal(month) is not None else -1 for month in month_axis], dtype=np.int32)
    valid_month_mask = month_ordinals >= 0

    def _interpolate(series: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(month_axis),), dtype=np.float32)
        weights = np.zeros((len(month_axis),), dtype=np.float32)
        if not series or not np.any(valid_month_mask):
            return values, weights
        points_x = np.asarray([item[0] for item in series], dtype=np.float32)
        points_y = np.asarray([item[1] for item in series], dtype=np.float32)
        order = np.argsort(points_x)
        points_x = points_x[order]
        points_y = points_y[order]
        interp_values = np.interp(month_ordinals[valid_month_mask].astype(np.float32), points_x, points_y).astype(np.float32)
        values[valid_month_mask] = interp_values
        nearest_distance = np.min(np.abs(month_ordinals[valid_month_mask][:, None].astype(np.float32) - points_x[None, :]), axis=1)
        weights[valid_month_mask] = np.clip(np.exp(-nearest_distance / 18.0), 0.15, 1.0).astype(np.float32)
        return values, weights

    diag_points: list[tuple[int, float]] = []
    art_points: list[tuple[int, float]] = []
    sup_points: list[tuple[int, float]] = []
    third_points: list[tuple[int, float]] = []
    for point in OFFICIAL_REFERENCE_POINTS:
        ordinal = _month_ordinal(point["month"])
        if ordinal is None:
            continue
        reference = point["reference"]
        first95 = float(reference["first95"]) if "first95" in reference else None
        second95 = float(reference["second95"]) if "second95" in reference else None
        overall = float(reference["overall_suppressed"]) if "overall_suppressed" in reference else None
        third95 = float(reference["documented_suppression_among_art"]) if "documented_suppression_among_art" in reference else None
        if first95 is not None:
            diag_points.append((ordinal, first95))
        if first95 is not None and second95 is not None:
            art_points.append((ordinal, first95 * second95))
        if overall is not None:
            sup_points.append((ordinal, overall))
        elif first95 is not None and second95 is not None and third95 is not None:
            sup_points.append((ordinal, first95 * second95 * third95))
        if third95 is not None:
            third_points.append((ordinal, third95))
        elif first95 is not None and second95 is not None and overall is not None:
            denom = max(first95 * second95, 1e-6)
            third_points.append((ordinal, overall / denom))

    diag_curve, diag_weight = _interpolate(diag_points)
    art_curve, art_weight = _interpolate(art_points)
    sup_curve, sup_weight = _interpolate(sup_points)
    third_curve, third_weight = _interpolate(third_points)
    combined_weight = np.maximum.reduce([diag_weight, art_weight, sup_weight, third_weight]) if len(month_axis) else np.zeros((0,), dtype=np.float32)
    return {
        "diagnosed_stock": diag_curve,
        "art_stock": art_curve,
        "documented_suppression": sup_curve,
        "third95": third_curve,
        "weight": combined_weight.astype(np.float32),
    }


def _harp_program_curves(month_axis: list[str]) -> dict[str, np.ndarray]:
    month_ordinals = np.asarray([_month_ordinal(month) if _month_ordinal(month) is not None else -1 for month in month_axis], dtype=np.int32)
    valid_month_mask = month_ordinals >= 0

    def _interpolate(series: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(month_axis),), dtype=np.float32)
        weights = np.zeros((len(month_axis),), dtype=np.float32)
        if not series or not np.any(valid_month_mask):
            return values, weights
        points_x = np.asarray([item[0] for item in series], dtype=np.float32)
        points_y = np.asarray([item[1] for item in series], dtype=np.float32)
        order = np.argsort(points_x)
        points_x = points_x[order]
        points_y = points_y[order]
        interp_values = np.interp(month_ordinals[valid_month_mask].astype(np.float32), points_x, points_y).astype(np.float32)
        values[valid_month_mask] = interp_values
        nearest_distance = np.min(np.abs(month_ordinals[valid_month_mask][:, None].astype(np.float32) - points_x[None, :]), axis=1)
        weights[valid_month_mask] = np.clip(np.exp(-nearest_distance / 14.0), 0.10, 1.0).astype(np.float32)
        return values, weights

    diag_points: list[tuple[int, float]] = []
    art_points: list[tuple[int, float]] = []
    tested_points: list[tuple[int, float]] = []
    sup_points: list[tuple[int, float]] = []
    tested_among_art_points: list[tuple[int, float]] = []
    suppressed_among_art_points: list[tuple[int, float]] = []
    for point in HARP_PROGRAM_POINTS:
        ordinal = _month_ordinal(point["month"])
        if ordinal is None:
            continue
        plhiv = max(float(point["estimated_plhiv"]), 1.0)
        diagnosed = float(point["diagnosed"]) / plhiv
        on_art = float(point["on_art"]) / plhiv
        tested = float(point["viral_load_tested"]) / plhiv
        suppressed = float(point["suppressed"]) / plhiv
        diag_points.append((ordinal, diagnosed))
        art_points.append((ordinal, on_art))
        tested_points.append((ordinal, tested))
        sup_points.append((ordinal, suppressed))
        tested_among_art_points.append((ordinal, float(point["viral_load_tested"]) / max(float(point["on_art"]), 1.0)))
        suppressed_among_art_points.append((ordinal, float(point["suppressed"]) / max(float(point["on_art"]), 1.0)))

    diag_curve, diag_weight = _interpolate(diag_points)
    art_curve, art_weight = _interpolate(art_points)
    tested_curve, tested_weight = _interpolate(tested_points)
    sup_curve, sup_weight = _interpolate(sup_points)
    tested_art_curve, tested_art_weight = _interpolate(tested_among_art_points)
    suppressed_art_curve, suppressed_art_weight = _interpolate(suppressed_among_art_points)
    combined_weight = np.maximum.reduce([diag_weight, art_weight, tested_weight, sup_weight, tested_art_weight, suppressed_art_weight]) if len(month_axis) else np.zeros((0,), dtype=np.float32)
    return {
        "diagnosed_stock": diag_curve.astype(np.float32),
        "art_stock": art_curve.astype(np.float32),
        "viral_load_tested_stock": tested_curve.astype(np.float32),
        "documented_suppression": sup_curve.astype(np.float32),
        "viral_load_tested_among_art": tested_art_curve.astype(np.float32),
        "suppressed_among_art": suppressed_art_curve.astype(np.float32),
        "weight": combined_weight.astype(np.float32),
    }


def _linkage_anchor_curves(
    *,
    month_axis: list[str],
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
) -> dict[str, np.ndarray]:
    mask = _national_reference_mask(province_axis)
    national_diag = np.tensordot(mask, np.asarray(observation_targets["diagnosed_stock"], dtype=np.float32), axes=(0, 0)).astype(np.float32)
    national_art = np.tensordot(mask, np.asarray(observation_targets["art_stock"], dtype=np.float32), axes=(0, 0)).astype(np.float32)
    observed_second95 = np.clip(national_art / np.clip(national_diag, 1e-6, None), 0.0, 1.0).astype(np.float32)
    observed_weight = (national_diag > 0.02).astype(np.float32) * 0.35

    month_ordinals = [_month_ordinal(month) for month in month_axis]
    growth_proxy = np.zeros((len(month_axis),), dtype=np.float32)
    for idx in range(len(month_axis)):
        next_idx = min(idx + 1, len(month_axis) - 1)
        prev_idx = max(idx - 1, 0)
        use_idx = next_idx if next_idx != idx else prev_idx
        if use_idx == idx:
            growth_proxy[idx] = 0.0
            continue
        delta_ord = month_ordinals[use_idx]
        base_ord = month_ordinals[idx]
        if delta_ord is None or base_ord is None:
            growth_proxy[idx] = 0.0
            continue
        delta_months = max(abs(delta_ord - base_ord), 1)
        if use_idx > idx:
            art_delta = max(float(national_art[use_idx] - national_art[idx]), 0.0)
            diagnosed_gap = max(float(national_diag[idx] - national_art[idx]), 1e-6)
        else:
            art_delta = max(float(national_art[idx] - national_art[use_idx]), 0.0)
            diagnosed_gap = max(float(national_diag[use_idx] - national_art[use_idx]), 1e-6)
        growth_proxy[idx] = float(np.clip((art_delta / float(delta_months)) / diagnosed_gap, 0.02, 0.35))

    official_points: list[tuple[int, float]] = []
    for point in OFFICIAL_REFERENCE_POINTS:
        ordinal = _month_ordinal(point["month"])
        second95 = point.get("reference", {}).get("second95")
        if ordinal is not None and second95 is not None:
            official_points.append((ordinal, float(second95)))

    def _interp_points(points: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(month_axis),), dtype=np.float32)
        weights = np.zeros((len(month_axis),), dtype=np.float32)
        valid = np.asarray([ordinal is not None for ordinal in month_ordinals], dtype=bool)
        if not points or not np.any(valid):
            return values, weights
        points_x = np.asarray([item[0] for item in points], dtype=np.float32)
        points_y = np.asarray([item[1] for item in points], dtype=np.float32)
        order = np.argsort(points_x)
        points_x = points_x[order]
        points_y = points_y[order]
        ordinals = np.asarray([ordinal if ordinal is not None else -1 for ordinal in month_ordinals], dtype=np.float32)
        values[valid] = np.interp(ordinals[valid], points_x, points_y).astype(np.float32)
        nearest_distance = np.min(np.abs(ordinals[valid][:, None] - points_x[None, :]), axis=1)
        weights[valid] = np.clip(np.exp(-nearest_distance / 18.0), 0.10, 1.0).astype(np.float32)
        return values, weights

    official_second95, official_weight = _interp_points(official_points)
    harp_curves = _harp_program_curves(month_axis)
    harp_second95 = np.clip(harp_curves["art_stock"] / np.clip(harp_curves["diagnosed_stock"], 1e-6, None), 0.0, 1.0).astype(np.float32)
    harp_weight = harp_curves["weight"].astype(np.float32)

    total_weight = official_weight + harp_weight + observed_weight
    blended_second95 = np.where(
        total_weight > 0,
        (official_second95 * official_weight + harp_second95 * harp_weight + observed_second95 * observed_weight) / np.clip(total_weight, 1e-6, None),
        observed_second95,
    ).astype(np.float32)
    hazard_from_share = np.clip(0.04 + 0.22 * blended_second95, 0.05, 0.24).astype(np.float32)
    d_to_a_target = np.clip(0.55 * growth_proxy + 0.45 * hazard_from_share, 0.04, 0.28).astype(np.float32)
    linkage_weight = np.clip(np.maximum.reduce([official_weight * 0.9, harp_weight, observed_weight]), 0.05, 1.0).astype(np.float32)
    return {
        "month_axis": np.asarray(month_axis, dtype=object),
        "second95_target": blended_second95,
        "d_to_a_transition_target": d_to_a_target,
        "weight": linkage_weight,
        "observed_second95": observed_second95,
        "growth_proxy": growth_proxy,
        "official_second95": official_second95,
        "harp_second95": harp_second95,
    }


def _suppression_anchor_curves(
    *,
    month_axis: list[str],
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
) -> dict[str, np.ndarray]:
    mask = _national_reference_mask(province_axis)
    national_diag = np.tensordot(mask, np.asarray(observation_targets["diagnosed_stock"], dtype=np.float32), axes=(0, 0)).astype(np.float32)
    national_art = np.tensordot(mask, np.asarray(observation_targets["art_stock"], dtype=np.float32), axes=(0, 0)).astype(np.float32)
    national_sup = np.tensordot(mask, np.asarray(observation_targets["documented_suppression"], dtype=np.float32), axes=(0, 0)).astype(np.float32)
    national_test = np.tensordot(mask, np.asarray(observation_targets["testing_coverage"], dtype=np.float32), axes=(0, 0)).astype(np.float32)

    observed_overall = np.clip(national_sup, 0.0, 1.0).astype(np.float32)
    observed_supp_among_art = np.clip(national_sup / np.clip(national_art, 1e-6, None), 0.0, 1.0).astype(np.float32)
    observed_tested_among_art = np.clip(national_test / np.clip(national_art, 1e-6, None), 0.0, 1.0).astype(np.float32)
    observed_weight = ((national_art > 0.02) & (national_test > 0.005)).astype(np.float32) * 0.30

    month_ordinals = [_month_ordinal(month) for month in month_axis]
    official_overall_points: list[tuple[int, float]] = []
    official_third_points: list[tuple[int, float]] = []
    for point in OFFICIAL_REFERENCE_POINTS:
        ordinal = _month_ordinal(point["month"])
        if ordinal is None:
            continue
        ref = point.get("reference", {})
        overall = ref.get("overall_suppressed")
        third = ref.get("documented_suppression_among_art")
        if overall is not None:
            official_overall_points.append((ordinal, float(overall)))
        if third is not None:
            official_third_points.append((ordinal, float(third)))
        elif ref.get("first95") is not None and ref.get("second95") is not None and overall is not None:
            denom = max(float(ref["first95"]) * float(ref["second95"]), 1e-6)
            official_third_points.append((ordinal, float(overall) / denom))

    def _interp_points(points: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
        values = np.zeros((len(month_axis),), dtype=np.float32)
        weights = np.zeros((len(month_axis),), dtype=np.float32)
        valid = np.asarray([ordinal is not None for ordinal in month_ordinals], dtype=bool)
        if not points or not np.any(valid):
            return values, weights
        xs = np.asarray([item[0] for item in points], dtype=np.float32)
        ys = np.asarray([item[1] for item in points], dtype=np.float32)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        ordinals = np.asarray([ordinal if ordinal is not None else -1 for ordinal in month_ordinals], dtype=np.float32)
        values[valid] = np.interp(ordinals[valid], xs, ys).astype(np.float32)
        nearest_distance = np.min(np.abs(ordinals[valid][:, None] - xs[None, :]), axis=1)
        weights[valid] = np.clip(np.exp(-nearest_distance / 18.0), 0.10, 1.0).astype(np.float32)
        return values, weights

    official_overall, official_overall_weight = _interp_points(official_overall_points)
    official_third, official_third_weight = _interp_points(official_third_points)
    harp_curves = _harp_program_curves(month_axis)
    harp_overall = harp_curves["documented_suppression"].astype(np.float32)
    harp_third = harp_curves["suppressed_among_art"].astype(np.float32)
    harp_tested = harp_curves["viral_load_tested_among_art"].astype(np.float32)
    harp_weight = harp_curves["weight"].astype(np.float32)

    total_overall_weight = official_overall_weight + harp_weight + observed_weight
    overall_target = np.where(
        total_overall_weight > 0,
        (official_overall * official_overall_weight + harp_overall * harp_weight + observed_overall * observed_weight) / np.clip(total_overall_weight, 1e-6, None),
        observed_overall,
    ).astype(np.float32)
    total_third_weight = official_third_weight + harp_weight + observed_weight
    supp_among_art_target = np.where(
        total_third_weight > 0,
        (official_third * official_third_weight + harp_third * harp_weight + observed_supp_among_art * observed_weight) / np.clip(total_third_weight, 1e-6, None),
        observed_supp_among_art,
    ).astype(np.float32)
    tested_among_art_target = np.where(
        harp_weight + observed_weight > 0,
        (harp_tested * harp_weight + observed_tested_among_art * observed_weight) / np.clip(harp_weight + observed_weight, 1e-6, None),
        observed_tested_among_art,
    ).astype(np.float32)
    transition_target = np.clip(0.02 + 0.08 * supp_among_art_target + 0.04 * overall_target, 0.04, 0.16).astype(np.float32)
    suppression_weight = np.clip(np.maximum.reduce([official_overall_weight, official_third_weight, harp_weight, observed_weight]), 0.05, 1.0).astype(np.float32)
    return {
        "month_axis": np.asarray(month_axis, dtype=object),
        "overall_suppression_target": overall_target,
        "suppressed_among_art_target": supp_among_art_target,
        "tested_among_art_target": tested_among_art_target,
        "a_to_v_transition_target": transition_target,
        "weight": suppression_weight,
        "observed_overall_suppression": observed_overall,
        "observed_suppressed_among_art": observed_supp_among_art,
        "observed_tested_among_art": observed_tested_among_art,
        "official_overall_suppression": official_overall,
        "official_suppressed_among_art": official_third,
        "harp_overall_suppression": harp_overall,
        "harp_suppressed_among_art": harp_third,
        "harp_tested_among_art": harp_tested,
    }


def _infer_region_from_name(name: str) -> str:
    return infer_region_code(name) or "region_unknown"


def _region_assignments(province_axis: list[str], normalized_rows: list[dict[str, Any]]) -> tuple[list[str], np.ndarray]:
    province_to_region: dict[str, Counter[str]] = defaultdict(Counter)
    for row in normalized_rows:
        geo = str(row.get("geo") or row.get("province") or "").strip()
        region = str(row.get("region") or "").strip().lower()
        if geo:
            province_to_region[geo][region or _infer_region_from_name(geo)] += 1
    regions = []
    for province in province_axis:
        counter = province_to_region.get(province, Counter())
        if counter:
            regions.append(counter.most_common(1)[0][0])
        else:
            regions.append(_infer_region_from_name(province))
    region_axis = sorted(set(regions)) or ["national"]
    region_index = np.asarray([region_axis.index(region) for region in regions], dtype=np.int32)
    return region_axis, region_index


def _default_axis(axis_catalogs: dict[str, Any], key: str, default: list[str]) -> list[str]:
    values = [str(item) for item in axis_catalogs.get(key, []) if str(item or "").strip()]
    return values or default[:]


def _normalized_counter(counter: Counter[str], labels: list[str], prior: np.ndarray, strength: float = 12.0) -> np.ndarray:
    values = prior.astype(np.float32) * strength
    for idx, label in enumerate(labels):
        values[idx] += float(counter.get(label, 0))
    total = float(values.sum())
    if total <= 0.0:
        return prior.astype(np.float32)
    return (values / total).astype(np.float32)


def _subgroup_row_weight(row: dict[str, Any]) -> float:
    if str(row.get("source_bank") or "") == "phase1_standardized_tensor":
        return 0.0
    weight = float(row.get("evidence_weight") or 0.0)
    weight *= 0.45 + 0.25 * float(row.get("measurement_quality_weight") or 0.0)
    weight *= 0.35 + 0.35 * float(row.get("spatial_relevance_weight") or 0.0) + 0.30 * float(row.get("source_tier_weight") or 0.0)
    geo_resolution = str(row.get("geo_resolution") or "")
    if geo_resolution in {"province", "city"}:
        geo_factor = 1.0
    elif geo_resolution == "region":
        geo_factor = 0.8
    elif geo_resolution == "national":
        geo_factor = 0.45
    else:
        geo_factor = 0.2
    return float(max(0.0, weight * geo_factor))


def _load_subgroup_anchor_pack(run_dir: Any) -> dict[str, Any]:
    path = run_dir / "harp_archive" / "subgroup_anchor_pack.json"
    if path.exists():
        return read_json(path, default={})
    return {}


def _load_network_prior_signals(run_dir: Any, province_axis: list[str]) -> dict[str, dict[str, float]]:
    tensor_path = run_dir / "phase15" / "network_feature_tensor.npz"
    catalog_path = run_dir / "phase15" / "network_feature_catalog.json"
    if not tensor_path.exists() or not catalog_path.exists():
        return {}
    tensor = load_tensor_artifact(tensor_path).astype(np.float32)
    catalog = read_json(catalog_path, default=[])
    if tensor.ndim != 3 or tensor.shape[0] != len(province_axis):
        return {}
    name_to_idx = {str(row.get("factor_name") or ""): idx for idx, row in enumerate(catalog)}

    def _mean_surface(name: str) -> np.ndarray:
        idx = name_to_idx.get(name)
        if idx is None:
            return np.zeros((len(province_axis),), dtype=np.float32)
        return tensor[:, :, idx].mean(axis=1).astype(np.float32)

    def _normalize(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=np.float32)
        lo = float(values[finite].min())
        hi = float(values[finite].max())
        if hi - lo < 1e-6:
            return np.full(values.shape, 0.5, dtype=np.float32)
        out = (values - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    awareness = _normalize(_mean_surface("awareness_propagation_score"))
    accessibility = _normalize(_mean_surface("treatment_hub_accessibility"))
    inbound = _normalize(_mean_surface("mobility_inbound_exposure_flux"))
    fragility = _normalize(_mean_surface("service_giant_component_loss"))
    continuity = _normalize(_mean_surface("continuity_of_care_stress"))
    isolation = _normalize(_mean_surface("community_information_isolation_score"))
    urbanity = _normalize(0.4 * awareness + 0.35 * accessibility + 0.25 * inbound - 0.15 * isolation)
    stress = _normalize(0.55 * fragility + 0.45 * continuity)
    return {
        province_axis[idx]: {
            "urbanity": float(urbanity[idx]),
            "stress": float(stress[idx]),
            "awareness": float(awareness[idx]),
            "accessibility": float(accessibility[idx]),
        }
        for idx in range(len(province_axis))
    }


def _anchor_distribution_for_geo(
    anchor_pack: dict[str, Any],
    *,
    province: str,
    region: str,
    kp_axis: list[str],
) -> Counter[str]:
    counter: Counter[str] = Counter()
    national_profile = anchor_pack.get("national_kp_profile") or {}
    mapped = national_profile.get("mapped_distribution") or {}
    for label in kp_axis:
        counter[label] += float(mapped.get(label, 0.0)) * 18.0
    for anchor in anchor_pack.get("subnational_kp_anchors", []):
        anchor_geo = str(anchor.get("geo") or "")
        anchor_region = str(anchor.get("region") or "")
        if anchor_geo not in {province, region, "Philippines"} and anchor_region not in {region, "national"}:
            continue
        estimate = float(anchor.get("estimated_population_15_plus") or 0.0)
        coverage = float(anchor.get("prevention_coverage") or 0.0)
        strength = min(20.0, max(2.0, estimate / 80_000.0) + 6.0 * coverage)
        counter["msm"] += 0.78 * strength
        if "tgw" in kp_axis:
            counter["tgw"] += 0.22 * strength
    return counter


def _network_adjusted_distribution(
    prior: np.ndarray,
    *,
    labels: list[str],
    signal: dict[str, float],
    kind: str,
) -> np.ndarray:
    adjusted = prior.astype(np.float32).copy()
    urbanity = float(signal.get("urbanity", 0.5))
    stress = float(signal.get("stress", 0.5))
    awareness = float(signal.get("awareness", 0.5))
    adjustment = dict((_phase3_prior("subgroup_network_adjustment", {}) or {}).get(kind, {}))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    if kind == "kp":
        for label, coeffs in adjustment.items():
            if label not in label_to_idx:
                continue
            delta = 0.0
            delta += float(coeffs.get("urbanity", 0.0)) * (urbanity - 0.5)
            delta += float(coeffs.get("awareness", 0.0)) * (awareness - 0.5)
            delta += float(coeffs.get("stress", 0.0)) * (stress - 0.5)
            adjusted[label_to_idx[label]] += delta
    elif kind == "age":
        for label, coeffs in adjustment.items():
            if label not in label_to_idx:
                continue
            delta = 0.0
            delta += float(coeffs.get("urbanity", 0.0)) * (urbanity - 0.5)
            delta += float(coeffs.get("awareness", 0.0)) * (awareness - 0.5)
            adjusted[label_to_idx[label]] += delta
    elif kind == "sex":
        for label, coeffs in adjustment.items():
            if label not in label_to_idx:
                continue
            urbanity_center = float(coeffs.get("urbanity_center", 0.5))
            delta = 0.0
            delta += float(coeffs.get("urbanity", 0.0)) * (urbanity - urbanity_center)
            delta += float(coeffs.get("stress", 0.0)) * (stress - 0.5)
            adjusted[label_to_idx[label]] += delta
    adjusted = np.clip(adjusted, 1e-4, None)
    adjusted = adjusted / np.clip(adjusted.sum(), 1e-6, None)
    return adjusted.astype(np.float32)


def _build_subgroup_weights(
    run_dir: Any,
    normalized_rows: list[dict[str, Any]],
    province_axis: list[str],
    kp_axis: list[str],
    age_axis: list[str],
    sex_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    strength_cfg = dict(_phase3_prior("subgroup_counter_strength", {}) or {})
    national_strength = dict(strength_cfg.get("national", {}) or {})
    region_strength = float(strength_cfg.get("region", 11.0))
    province_strength = float(strength_cfg.get("province", 7.5))
    province_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"kp": Counter(), "age": Counter(), "sex": Counter()})
    region_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: {"kp": Counter(), "age": Counter(), "sex": Counter()})
    national_counts: dict[str, Counter[str]] = {"kp": Counter(), "age": Counter(), "sex": Counter()}
    for row in normalized_rows:
        weight = _subgroup_row_weight(row)
        if weight <= 0.0:
            continue
        province = str(row.get("province") or row.get("geo") or "").strip()
        region = str(row.get("region") or infer_region_code(province) or "region_unknown")
        if province not in province_axis and region != "national":
            province = ""
        if row.get("kp_group"):
            kp_label = str(row["kp_group"])
            kp_weight = weight * (0.40 if kp_label == "remaining_population" else 1.0)
            national_counts["kp"][kp_label] += kp_weight
            if region:
                region_counts[region]["kp"][kp_label] += kp_weight
            if province:
                province_counts[province]["kp"][kp_label] += kp_weight
        if row.get("age_band"):
            age_label = str(row["age_band"])
            national_counts["age"][age_label] += weight
            if region:
                region_counts[region]["age"][age_label] += weight
            if province:
                province_counts[province]["age"][age_label] += weight
        if row.get("sex"):
            sex_label = str(row["sex"])
            national_counts["sex"][sex_label] += weight
            if region:
                region_counts[region]["sex"][sex_label] += weight
            if province:
                province_counts[province]["sex"][sex_label] += weight

    anchor_pack = _load_subgroup_anchor_pack(run_dir)
    network_signals = _load_network_prior_signals(run_dir, province_axis)
    national_anchor_kp = _anchor_distribution_for_geo(anchor_pack, province="Philippines", region="national", kp_axis=kp_axis)
    for label, value in national_anchor_kp.items():
        national_counts["kp"][label] += value

    national_kp = _normalized_counter(national_counts["kp"], kp_axis, DEFAULT_KP_PRIOR[: len(kp_axis)], strength=float(national_strength.get("kp", 18.0)))
    national_age = _normalized_counter(national_counts["age"], age_axis, DEFAULT_AGE_PRIOR[: len(age_axis)], strength=float(national_strength.get("age", 16.0)))
    national_sex = _normalized_counter(national_counts["sex"], sex_axis, DEFAULT_SEX_PRIOR[: len(sex_axis)], strength=float(national_strength.get("sex", 16.0)))

    weights = np.zeros((len(province_axis), len(kp_axis), len(age_axis), len(sex_axis)), dtype=np.float32)
    summary_rows = []
    for province_idx, province in enumerate(province_axis):
        region = infer_region_code(province) or "region_unknown"
        region_anchor_kp = _anchor_distribution_for_geo(anchor_pack, province=province, region=region, kp_axis=kp_axis)
        region_kp_counter = Counter(region_counts.get(region, {}).get("kp", Counter()))
        region_kp_counter.update(region_anchor_kp)
        region_age_counter = Counter(region_counts.get(region, {}).get("age", Counter()))
        region_sex_counter = Counter(region_counts.get(region, {}).get("sex", Counter()))
        region_kp = _normalized_counter(region_kp_counter, kp_axis, national_kp, strength=region_strength)
        region_age = _normalized_counter(region_age_counter, age_axis, national_age, strength=region_strength)
        region_sex = _normalized_counter(region_sex_counter, sex_axis, national_sex, strength=region_strength)

        counts = province_counts.get(province, {})
        kp_probs = _normalized_counter(counts.get("kp", Counter()), kp_axis, region_kp, strength=province_strength)
        age_probs = _normalized_counter(counts.get("age", Counter()), age_axis, region_age, strength=province_strength)
        sex_probs = _normalized_counter(counts.get("sex", Counter()), sex_axis, region_sex, strength=province_strength)

        signal = network_signals.get(province, {})
        joint = kp_probs[:, None, None] * age_probs[None, :, None] * sex_probs[None, None, :]
        joint = joint / np.clip(joint.sum(), 1e-6, None)
        weights[province_idx] = joint.astype(np.float32)
        evidence_strength = float(sum(sum(counts.get(kind, Counter()).values()) for kind in ("kp", "age", "sex")))
        summary_rows.append(
            {
                "province": province,
                "region": region,
                "prior_source": "province_region_anchor_network",
                "evidence_strength": round(evidence_strength, 6),
                "network_signal": {key: round(float(value), 6) for key, value in signal.items()},
                "kp_distribution": {kp_axis[idx]: round(float(kp_probs[idx]), 6) for idx in range(len(kp_axis))},
                "base_kp_distribution": {kp_axis[idx]: round(float(kp_probs[idx]), 6) for idx in range(len(kp_axis))},
                "region_kp_distribution": {kp_axis[idx]: round(float(region_kp[idx]), 6) for idx in range(len(kp_axis))},
                "age_distribution": {age_axis[idx]: round(float(age_probs[idx]), 6) for idx in range(len(age_axis))},
                "base_age_distribution": {age_axis[idx]: round(float(age_probs[idx]), 6) for idx in range(len(age_axis))},
                "sex_distribution": {sex_axis[idx]: round(float(sex_probs[idx]), 6) for idx in range(len(sex_axis))},
                "base_sex_distribution": {sex_axis[idx]: round(float(sex_probs[idx]), 6) for idx in range(len(sex_axis))},
            }
        )
    return weights, {
        "rows": summary_rows,
        "national_kp_distribution": {kp_axis[idx]: round(float(national_kp[idx]), 6) for idx in range(len(kp_axis))},
        "national_age_distribution": {age_axis[idx]: round(float(national_age[idx]), 6) for idx in range(len(age_axis))},
        "national_sex_distribution": {sex_axis[idx]: round(float(national_sex[idx]), 6) for idx in range(len(sex_axis))},
        "anchor_pack_present": bool(anchor_pack),
        "network_signal_count": len(network_signals),
    }


def _subgroup_prior_feature_matrix(
    subgroup_summary: dict[str, Any],
    archetype_bundle: dict[str, Any],
    province_axis: list[str],
) -> tuple[np.ndarray, list[str]]:
    subgroup_rows = {str(row.get("province") or ""): row for row in list(subgroup_summary.get("rows", []) or [])}
    archetype_rows = {str(row.get("province") or ""): row for row in list(archetype_bundle.get("rows", []) or [])}
    feature_names = [
        "urbanity",
        "accessibility",
        "awareness",
        "stress",
        "archetype_urban_high_throughput",
        "archetype_migrant_corridor",
        "archetype_remote_island",
        "archetype_fragile_service_network",
        "archetype_under_reporting_province",
    ]
    matrix = np.zeros((len(province_axis), len(feature_names)), dtype=np.float32)
    for province_idx, province in enumerate(province_axis):
        subgroup_row = subgroup_rows.get(province, {})
        signal = dict(subgroup_row.get("network_signal") or {})
        archetype_mix = dict((archetype_rows.get(province, {}) or {}).get("archetype_mixture", {}) or {})
        matrix[province_idx] = np.asarray(
            [
                float(signal.get("urbanity", 0.5)),
                float(signal.get("accessibility", 0.5)),
                float(signal.get("awareness", 0.5)),
                float(signal.get("stress", 0.5)),
                float(archetype_mix.get("urban_high_throughput", 0.0)),
                float(archetype_mix.get("migrant_corridor", 0.0)),
                float(archetype_mix.get("remote_island", 0.0)),
                float(archetype_mix.get("fragile_service_network", 0.0)),
                float(archetype_mix.get("under_reporting_province", 0.0)),
            ],
            dtype=np.float32,
        )
    return matrix.astype(np.float32), feature_names


def _build_province_archetype_bundle(
    *,
    province_axis: list[str],
    month_axis: list[str],
    subgroup_summary: dict[str, Any],
    observation_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    synthetic_library = build_synthetic_province_library(month_axis=month_axis)
    priors = infer_province_archetype_priors(
        province_axis=province_axis,
        month_axis=month_axis,
        subgroup_summary=subgroup_summary,
        observation_targets=observation_targets,
    )
    return {
        **priors,
        "synthetic_library": synthetic_library,
    }


def _feature_support_weight(profile: dict[str, Any], target_name: str) -> float:
    support = 0.35
    support += 0.20 * min(1.0, float(profile.get("numeric_row_count", 0)) / 4.0)
    support += 0.15 * min(1.0, float(sum(profile.get("source_banks", {}).values())) / 4.0)
    if target_name == "documented_suppression":
        support += 0.10 if "biology" in profile.get("domain_families", {}) else 0.0
    if target_name == "testing_coverage":
        support += 0.10 if "behavior" in profile.get("domain_families", {}) else 0.0
    return float(min(1.0, support))


def _match_target_indices(
    target_name: str,
    canonical_axis: list[str],
    parameter_catalog: list[dict[str, Any]],
) -> list[tuple[int, float, dict[str, Any]]]:
    rules = TARGET_PATTERNS[target_name]
    catalog_by_name = {str(row.get("canonical_name") or ""): row for row in parameter_catalog}
    matches: list[tuple[int, float, dict[str, Any]]] = []
    for index, canonical_name in enumerate(canonical_axis):
        lowered = f" {canonical_name.lower()} "
        profile = catalog_by_name.get(canonical_name, {})
        domain_keys = set((profile.get("domain_families") or {}).keys())
        pathway_keys = set((profile.get("pathway_families") or {}).keys())
        token_hit = any(token in lowered for token in rules["tokens"])
        domain_hit = any(domain in domain_keys for domain in rules["domains"])
        pathway_hit = any(pathway in pathway_keys for pathway in rules["pathways"])
        if not (token_hit or domain_hit or pathway_hit):
            continue
        score = 0.25
        score += 0.35 if token_hit else 0.0
        score += 0.20 if domain_hit else 0.0
        score += 0.20 if pathway_hit else 0.0
        score += _feature_support_weight(profile, target_name)
        matches.append((index, float(score), profile | {"canonical_name": canonical_name}))
    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def _surface_from_matches(standardized_tensor: np.ndarray, matches: list[tuple[int, float, dict[str, Any]]]) -> tuple[np.ndarray, dict[str, Any]]:
    if standardized_tensor.size == 0 or not matches:
        empty = np.zeros(standardized_tensor.shape[:2], dtype=np.float32)
        return empty, {"matched_features": [], "feature_count": 0, "direct_support": 0, "anchor_support": 0}
    indices = [item[0] for item in matches[: min(8, len(matches))]]
    raw_weights = np.asarray([item[1] for item in matches[: min(8, len(matches))]], dtype=np.float32)
    weights = raw_weights / np.clip(raw_weights.sum(), 1e-6, None)
    projected = standardized_tensor[:, :, indices]
    blended = np.tensordot(projected, weights, axes=([2], [0]))
    surface = _sigmoid_np(blended).astype(np.float32)
    direct_support = sum(int(item[2].get("evidence_classes", {}).get("observed_numeric", 0) > 0) for item in matches[: min(8, len(matches))])
    anchor_support = sum(int(item[2].get("evidence_classes", {}).get("numeric_prior", 0) > 0) for item in matches[: min(8, len(matches))])
    return surface, {
        "matched_features": [str(matches[idx][2].get("canonical_name") or "") for idx in range(min(8, len(matches)))],
        "feature_count": len(indices),
        "direct_support": direct_support,
        "anchor_support": anchor_support,
    }


def _month_index_lookup(month_axis: list[str]) -> dict[str, int]:
    lookup: dict[str, int] = {}
    for index, month in enumerate(month_axis):
        month_str = str(month)
        lookup.setdefault(month_str, index)
        if len(month_str) >= 4 and month_str[:4].isdigit():
            lookup.setdefault(month_str[:4], index)
    return lookup


def _matched_support_tensor(
    normalized_rows: list[dict[str, Any]],
    canonical_axis: list[str],
    province_axis: list[str],
    month_axis: list[str],
) -> np.ndarray:
    support = np.zeros((len(province_axis), len(month_axis), len(canonical_axis)), dtype=np.float32)
    province_lookup = {province: idx for idx, province in enumerate(province_axis)}
    month_lookup = _month_index_lookup(month_axis)
    canonical_lookup = {name: idx for idx, name in enumerate(canonical_axis)}
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        province = str(row.get("province") or row.get("geo") or "")
        month = str(row.get("time") or "")
        if canonical_name not in canonical_lookup or province not in province_lookup or month not in month_lookup:
            continue
        value = row.get("model_numeric_value")
        if value in {"", None}:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        weight = float(row.get("evidence_weight") or 0.0)
        quality = float(row.get("measurement_quality_weight") or 0.0)
        spatial = float(row.get("spatial_relevance_weight") or 0.0)
        support[province_lookup[province], month_lookup[month], canonical_lookup[canonical_name]] += max(
            0.0,
            weight * (0.5 + 0.3 * quality + 0.2 * spatial),
        )
    return support


def _fill_sparse_surface(surface: np.ndarray, province_axis: list[str]) -> np.ndarray:
    if surface.size == 0:
        return surface.astype(np.float32)
    filled = np.asarray(surface, dtype=np.float32).copy()

    def _safe_nanmean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
        finite = np.isfinite(values)
        counts = finite.sum(axis=axis)
        totals = np.where(finite, values, 0.0).sum(axis=axis)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = totals / np.clip(counts, 1, None)
        if axis is None:
            return float(mean) if counts > 0 else np.nan
        mean = np.asarray(mean, dtype=np.float32)
        mean = np.where(counts > 0, mean, np.nan)
        return mean

    region_groups: dict[str, list[int]] = defaultdict(list)
    for province_idx, province in enumerate(province_axis):
        region_groups[infer_region_code(province)].append(province_idx)

    province_means = _safe_nanmean(filled, axis=1)
    region_month_means: dict[str, np.ndarray] = {}
    region_means: dict[str, float] = {}
    for region, indices in region_groups.items():
        members = filled[indices]
        region_month_means[region] = np.asarray(_safe_nanmean(members, axis=0), dtype=np.float32)
        region_means[region] = float(_safe_nanmean(members))

    national_month_mean = np.asarray(_safe_nanmean(filled, axis=0), dtype=np.float32)
    national_mean = float(_safe_nanmean(filled))
    if not np.isfinite(national_mean):
        national_mean = 0.5
    national_month_mean = np.where(np.isfinite(national_month_mean), national_month_mean, national_mean)

    for province_idx, province in enumerate(province_axis):
        region = infer_region_code(province)
        for month_idx in range(filled.shape[1]):
            if np.isfinite(filled[province_idx, month_idx]):
                continue
            province_mean = province_means[province_idx]
            region_month_mean = region_month_means.get(region, national_month_mean)[month_idx]
            region_mean = region_means.get(region, national_mean)
            if np.isfinite(province_mean):
                filled[province_idx, month_idx] = province_mean
            elif np.isfinite(region_month_mean):
                filled[province_idx, month_idx] = region_month_mean
            elif np.isfinite(region_mean):
                filled[province_idx, month_idx] = region_mean
            else:
                filled[province_idx, month_idx] = national_month_mean[month_idx]
    return np.clip(np.nan_to_num(filled, nan=national_mean), 0.0, 1.0).astype(np.float32)


def _surface_from_sparse_support(
    standardized_tensor: np.ndarray,
    support_tensor: np.ndarray,
    matches: list[tuple[int, float, dict[str, Any]]],
    province_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    if standardized_tensor.size == 0 or support_tensor.size == 0 or not matches:
        empty = np.zeros(standardized_tensor.shape[:2], dtype=np.float32)
        return empty, {
            "matched_features": [],
            "feature_count": 0,
            "direct_support": 0,
            "anchor_support": 0,
            "observed_cell_fraction": 0.0,
            "observed_mask": np.zeros_like(empty, dtype=np.float32),
            "support_surface": np.zeros_like(empty, dtype=np.float32),
        }
    top_matches = matches[: min(8, len(matches))]
    indices = [item[0] for item in top_matches]
    raw_weights = np.asarray([item[1] for item in top_matches], dtype=np.float32)
    weights = raw_weights / np.clip(raw_weights.sum(), 1e-6, None)
    projected = standardized_tensor[:, :, indices]
    support = support_tensor[:, :, indices]
    availability = support > 0.0
    weighted = projected * weights.reshape(1, 1, -1) * availability
    denom = (weights.reshape(1, 1, -1) * availability).sum(axis=2)
    blended = np.full(projected.shape[:2], np.nan, dtype=np.float32)
    valid = denom > 0
    blended[valid] = (weighted.sum(axis=2)[valid] / np.clip(denom[valid], 1e-6, None)).astype(np.float32)
    surface = _fill_sparse_surface(_sigmoid_np(blended), province_axis)
    direct_support = sum(int(item[2].get("evidence_classes", {}).get("observed_numeric", 0) > 0) for item in top_matches)
    anchor_support = sum(int(item[2].get("evidence_classes", {}).get("numeric_prior", 0) > 0) for item in top_matches)
    observed_fraction = float(np.mean(valid)) if valid.size else 0.0
    support_surface = denom.astype(np.float32)
    return surface, {
        "matched_features": [str(item[2].get("canonical_name") or "") for item in top_matches],
        "feature_count": len(indices),
        "direct_support": direct_support,
        "anchor_support": anchor_support,
        "observed_cell_fraction": round(observed_fraction, 6),
        "observed_mask": valid.astype(np.float32),
        "support_surface": support_surface,
    }


def _row_matches_target(row: dict[str, Any], target_name: str) -> bool:
    rules = TARGET_PATTERNS[target_name]
    blob = " ".join(
        [
            str(row.get("candidate_text") or ""),
            str(row.get("canonical_name") or ""),
            str(row.get("domain_family") or ""),
            str(row.get("pathway_family") or ""),
        ]
    ).lower()
    token_hit = any(token in blob for token in rules["tokens"])
    domain_hit = str(row.get("domain_family") or "") in rules["domains"]
    pathway_hit = str(row.get("pathway_family") or "") in rules["pathways"]
    return token_hit or domain_hit or pathway_hit


def _row_share_value(row: dict[str, Any], target_name: str) -> tuple[float | None, float | None]:
    del target_name
    value = row.get("model_numeric_value")
    if value in {"", None}:
        return None, None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None, None
    if not np.isfinite(numeric_value):
        return None, None
    unit = str(row.get("normalized_unit") or "")
    canonical_name = str(row.get("canonical_name") or "")
    if unit == "percent":
        share = numeric_value if numeric_value <= 1.5 else numeric_value / 100.0
        return float(np.clip(share, 0.0, 1.0)), None
    if unit == "unitless":
        if 0.0 <= numeric_value <= 1.0:
            return float(numeric_value), None
        if canonical_name in {"testing_rate", "viral_load"} and 0.0 <= numeric_value <= 100.0:
            return float(np.clip(numeric_value / 100.0, 0.0, 1.0)), None
    if unit in {"count_cases", "count_people", "count_million", "count_billion"}:
        return None, float(np.log1p(max(0.0, numeric_value)))
    return None, None


def _surface_from_normalized_rows(
    normalized_rows: list[dict[str, Any]],
    target_name: str,
    province_axis: list[str],
    month_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    province_lookup = {province: idx for idx, province in enumerate(province_axis)}
    month_lookup = _month_index_lookup(month_axis)
    share_sum = np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
    share_weight = np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
    count_score = np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
    count_weight = np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)
    matched_features: Counter[str] = Counter()
    direct_support = 0
    anchor_support = 0

    for row in normalized_rows:
        province = str(row.get("province") or row.get("geo") or "")
        month = str(row.get("time") or "")
        if province not in province_lookup or month not in month_lookup:
            continue
        if not _row_matches_target(row, target_name):
            continue
        share_value, count_value = _row_share_value(row, target_name)
        if share_value is None and count_value is None:
            continue
        weight = float(row.get("evidence_weight") or 0.0)
        weight *= 0.4 + 0.3 * float(row.get("measurement_quality_weight") or 0.0) + 0.3 * float(row.get("spatial_relevance_weight") or 0.0)
        if weight <= 0.0:
            continue
        province_idx = province_lookup[province]
        month_idx = month_lookup[month]
        matched_features[str(row.get("canonical_name") or "unknown")] += 1
        if bool(row.get("is_direct_measurement")):
            direct_support += 1
        if bool(row.get("is_anchor_eligible")):
            anchor_support += 1
        if share_value is not None:
            share_sum[province_idx, month_idx] += float(share_value) * weight
            share_weight[province_idx, month_idx] += weight
        if count_value is not None:
            count_score[province_idx, month_idx] += float(count_value) * weight
            count_weight[province_idx, month_idx] += weight

    surface = np.full((len(province_axis), len(month_axis)), np.nan, dtype=np.float32)
    valid_share = share_weight > 0
    surface[valid_share] = share_sum[valid_share] / np.clip(share_weight[valid_share], 1e-6, None)

    valid_count = count_weight > 0
    if np.any(valid_count):
        count_mean = np.full_like(surface, np.nan, dtype=np.float32)
        count_mean[valid_count] = count_score[valid_count] / np.clip(count_weight[valid_count], 1e-6, None)
        for month_idx in range(count_mean.shape[1]):
            month_values = count_mean[:, month_idx]
            mask = np.isfinite(month_values)
            if not np.any(mask):
                continue
            lo = float(np.min(month_values[mask]))
            hi = float(np.max(month_values[mask]))
            if hi - lo < 1e-6:
                normalized = np.full(month_values.shape, 0.5, dtype=np.float32)
            else:
                normalized = np.clip((month_values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
            use_mask = mask & ~np.isfinite(surface[:, month_idx])
            surface[use_mask, month_idx] = normalized[use_mask]

    observed_fraction = float(np.mean(np.isfinite(surface))) if surface.size else 0.0
    filled_surface = _fill_sparse_surface(surface, province_axis)
    support_surface = (share_weight + count_weight).astype(np.float32)
    return filled_surface, {
        "matched_features": [name for name, _ in matched_features.most_common(8)],
        "feature_count": len(matched_features),
        "direct_support": direct_support,
        "anchor_support": anchor_support,
        "observed_cell_fraction": round(observed_fraction, 6),
        "observed_mask": np.isfinite(surface).astype(np.float32),
        "support_surface": support_surface,
    }


def _classify_observation(target_name: str, meta: dict[str, Any]) -> str:
    if meta["feature_count"] == 0:
        return "prior_only"
    if target_name == "documented_suppression":
        return "bounded_observed"
    if meta["direct_support"] > 0:
        return "direct_observed"
    if meta["anchor_support"] > 0:
        return "bounded_observed"
    return "proxy_observed"


def _blend_curve_components(components: list[tuple[np.ndarray, np.ndarray]], *, default_value: float) -> np.ndarray:
    if not components:
        return np.asarray([], dtype=np.float32)
    shape = components[0][0].shape
    weighted_sum = np.zeros(shape, dtype=np.float32)
    total_weight = np.zeros(shape, dtype=np.float32)
    for values, weights in components:
        value_arr = np.asarray(values, dtype=np.float32)
        weight_arr = np.asarray(weights, dtype=np.float32)
        if value_arr.shape != shape or weight_arr.shape != shape:
            continue
        weighted_sum += value_arr * weight_arr
        total_weight += weight_arr
    default_curve = np.full(shape, float(default_value), dtype=np.float32)
    return np.where(total_weight > 0.0, weighted_sum / np.clip(total_weight, 1e-6, None), default_curve).astype(np.float32)


def _observation_prior_curves(month_axis: list[str]) -> dict[str, np.ndarray]:
    fallback_cfg = dict(_phase3_prior("observation_fallback", {}) or {})
    official_curves = _official_anchor_curves(month_axis)
    harp_curves = _harp_program_curves(month_axis)

    diagnosed = _blend_curve_components(
        [
            (official_curves["diagnosed_stock"], official_curves["weight"]),
            (harp_curves["diagnosed_stock"], harp_curves["weight"]),
        ],
        default_value=float(fallback_cfg.get("diagnosed_floor", 0.08)),
    )
    second95_official = np.clip(
        official_curves["art_stock"] / np.clip(official_curves["diagnosed_stock"], 1e-6, None),
        0.0,
        1.0,
    ).astype(np.float32)
    second95_harp = np.clip(
        harp_curves["art_stock"] / np.clip(harp_curves["diagnosed_stock"], 1e-6, None),
        0.0,
        1.0,
    ).astype(np.float32)
    second95 = _blend_curve_components(
        [
            (second95_official, official_curves["weight"]),
            (second95_harp, harp_curves["weight"]),
        ],
        default_value=float(fallback_cfg.get("second95_prior", 0.66)),
    )
    art = np.clip(
        diagnosed * second95,
        float(fallback_cfg.get("art_floor", 0.02)),
        np.clip(diagnosed, 0.0, 1.0),
    ).astype(np.float32)
    tested_among_art = _blend_curve_components(
        [(harp_curves["viral_load_tested_among_art"], harp_curves["weight"])],
        default_value=float(fallback_cfg.get("tested_among_art_prior", 0.42)),
    )
    testing = np.clip(
        art * tested_among_art,
        float(fallback_cfg.get("testing_floor", 0.01)),
        np.clip(art, 0.0, 1.0),
    ).astype(np.float32)
    suppressed_among_art = _blend_curve_components(
        [
            (official_curves["third95"], official_curves["weight"]),
            (harp_curves["suppressed_among_art"], harp_curves["weight"]),
        ],
        default_value=float(fallback_cfg.get("suppressed_among_art_prior", 0.58)),
    )
    suppression = np.clip(
        art * suppressed_among_art,
        float(fallback_cfg.get("suppression_floor", 0.01)),
        np.clip(testing, 0.0, 1.0),
    ).astype(np.float32)
    return {
        "diagnosed_stock": diagnosed,
        "art_stock": art,
        "documented_suppression": suppression,
        "testing_coverage": testing,
    }


def _enforce_cascade_ordering(targets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    bounds = dict(_phase3_prior("observation_bounds", {}) or {})
    diagnosed = np.clip(targets["diagnosed_stock"], *(bounds.get("diagnosed_stock", [0.04, 0.98])))
    art = np.clip(targets["art_stock"], *(bounds.get("art_stock", [0.01, 0.96])))
    suppression = np.clip(targets["documented_suppression"], *(bounds.get("documented_suppression", [0.0, 0.94])))
    testing = np.clip(targets["testing_coverage"], *(bounds.get("testing_coverage", [0.0, 0.96])))
    art = np.minimum(art, diagnosed)
    testing = np.minimum(testing, art)
    suppression = np.minimum(suppression, testing)
    art = np.maximum(art, testing)
    diagnosed = np.maximum(diagnosed, art)
    deaths = np.clip(targets["deaths"], *(bounds.get("deaths", [0.0, 0.12])))
    return {
        "diagnosed_stock": diagnosed.astype(np.float32),
        "art_stock": art.astype(np.float32),
        "documented_suppression": suppression.astype(np.float32),
        "testing_coverage": testing.astype(np.float32),
        "deaths": deaths.astype(np.float32),
    }


def _fallback_observation_targets(targets: dict[str, np.ndarray], *, month_axis: list[str]) -> dict[str, np.ndarray]:
    prior_curves = _observation_prior_curves(month_axis)
    diagnosed = targets["diagnosed_stock"]
    art = targets["art_stock"]
    suppression = targets["documented_suppression"]
    testing = targets["testing_coverage"]
    province_count = int(diagnosed.shape[0]) if diagnosed.ndim == 2 else 0

    def _surface_from_curve(name: str) -> np.ndarray:
        curve = np.asarray(prior_curves[name], dtype=np.float32)
        if province_count <= 0:
            return curve.astype(np.float32)
        return np.broadcast_to(curve.reshape(1, -1), (province_count, curve.shape[0])).astype(np.float32)

    if np.allclose(diagnosed, 0.0):
        diagnosed = _surface_from_curve("diagnosed_stock")
    if np.allclose(art, 0.0):
        art = _surface_from_curve("art_stock")
    if np.allclose(suppression, 0.0):
        suppression = _surface_from_curve("documented_suppression")
    if np.allclose(testing, 0.0):
        testing = _surface_from_curve("testing_coverage")
    return {
        "diagnosed_stock": diagnosed.astype(np.float32),
        "art_stock": art.astype(np.float32),
        "documented_suppression": suppression.astype(np.float32),
        "testing_coverage": testing.astype(np.float32),
        "deaths": targets["deaths"].astype(np.float32),
    }


def _observation_target_rows(targets: dict[str, np.ndarray], province_axis: list[str], month_axis: list[str], observation_ladder: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ladder_map = {row["target_name"]: row for row in observation_ladder}
    rows = []
    for target_name in OBSERVATION_ORDER:
        target = targets[target_name]
        ladder = ladder_map[target_name]
        for province_idx, province in enumerate(province_axis):
            for month_idx, month in enumerate(month_axis):
                rows.append(
                    {
                        "target_name": target_name,
                        "observation_class": ladder["observation_class"],
                        "province": province,
                        "time": month,
                        "value": round(float(target[province_idx, month_idx]), 6),
                    }
                )
    return rows


def build_observation_ladder(
    *,
    standardized_tensor: np.ndarray,
    normalized_rows: list[dict[str, Any]],
    parameter_catalog: list[dict[str, Any]],
    canonical_axis: list[str],
    province_axis: list[str],
    month_axis: list[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]]]:
    target_surfaces: dict[str, np.ndarray] = {}
    ladder_rows: list[dict[str, Any]] = []
    support_tensor = _matched_support_tensor(normalized_rows, canonical_axis, province_axis, month_axis)
    for target_name in OBSERVATION_ORDER:
        matches = _match_target_indices(target_name, canonical_axis, parameter_catalog)
        row_surface, row_meta = _surface_from_normalized_rows(normalized_rows, target_name, province_axis, month_axis)
        sparse_surface, sparse_meta = _surface_from_sparse_support(standardized_tensor, support_tensor, matches, province_axis)
        if float(row_meta.get("observed_cell_fraction") or 0.0) > 0.0:
            surface, meta = row_surface, row_meta
        else:
            surface, meta = sparse_surface, sparse_meta
        target_surfaces[target_name] = surface
        observation_class = _classify_observation(target_name, meta)
        ladder_rows.append(
            {
                "target_name": target_name,
                "observation_class": observation_class,
                "matched_features": meta["matched_features"],
                "feature_count": meta["feature_count"],
                "direct_support": meta["direct_support"],
                "anchor_support": meta["anchor_support"],
                "observed_cell_fraction": meta.get("observed_cell_fraction", 0.0),
                "weight": OBSERVATION_CLASS_WEIGHT[observation_class],
                "lower_bound_semantics": target_name == "documented_suppression",
                "province_count": len(province_axis),
                "month_count": len(month_axis),
            }
        )
    target_surfaces = _fallback_observation_targets(target_surfaces, month_axis=month_axis)
    target_surfaces = _enforce_cascade_ordering(target_surfaces)
    target_rows = _observation_target_rows(target_surfaces, province_axis, month_axis, ladder_rows)
    return ladder_rows, target_surfaces, target_rows


def _inject_harp_program_targets(
    *,
    observation_ladder: list[dict[str, Any]],
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
    month_axis: list[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    mask = _national_reference_mask(province_axis)
    national_indices = [idx for idx, value in enumerate(mask) if value > 0]
    if not national_indices:
        target_rows = _observation_target_rows(observation_targets, province_axis, month_axis, observation_ladder)
        return observation_ladder, observation_targets, target_rows, {"month_axis": month_axis, "reference_points": HARP_PROGRAM_POINTS, "applied": False}

    curves = _harp_program_curves(month_axis)
    if not np.any(curves["weight"] > 0):
        target_rows = _observation_target_rows(observation_targets, province_axis, month_axis, observation_ladder)
        return observation_ladder, observation_targets, target_rows, {"month_axis": month_axis, "reference_points": HARP_PROGRAM_POINTS, "applied": False}

    target_map = {
        "diagnosed_stock": "diagnosed_stock",
        "art_stock": "art_stock",
        "documented_suppression": "documented_suppression",
        "testing_coverage": "viral_load_tested_stock",
    }
    updated_targets = {key: np.asarray(value, dtype=np.float32).copy() for key, value in observation_targets.items()}
    for target_name, curve_name in target_map.items():
        if target_name not in updated_targets:
            continue
        curve = curves.get(curve_name)
        if curve is None or curve.shape[0] != len(month_axis):
            continue
        curve_weight = np.asarray(curves.get("weight", np.zeros((len(month_axis),), dtype=np.float32)), dtype=np.float32)
        for province_idx in national_indices:
            current = updated_targets[target_name][province_idx, :].astype(np.float32)
            updated_targets[target_name][province_idx, :] = ((1.0 - curve_weight) * current + curve_weight * curve.astype(np.float32)).astype(np.float32)
    updated_targets = _enforce_cascade_ordering(_fallback_observation_targets(updated_targets, month_axis=month_axis))

    updated_ladder: list[dict[str, Any]] = []
    for row in observation_ladder:
        row_copy = dict(row)
        if row_copy.get("target_name") in target_map:
            row_copy["observation_class"] = "direct_observed"
            row_copy["weight"] = max(float(row_copy.get("weight", 0.0)), OBSERVATION_CLASS_WEIGHT["direct_observed"])
            row_copy["harp_program_support"] = True
        else:
            row_copy["harp_program_support"] = False
        updated_ladder.append(row_copy)

    target_rows = _observation_target_rows(updated_targets, province_axis, month_axis, updated_ladder)
    harp_summary = {
        "month_axis": month_axis,
        "reference_points": HARP_PROGRAM_POINTS,
        "applied": True,
        "national_reference_mask": mask.round(6).tolist(),
        "target_curves": {key: value.round(6).tolist() for key, value in curves.items()},
        "applied_targets": list(target_map.keys()),
    }
    return updated_ladder, updated_targets, target_rows, harp_summary


def _normalize_support_surface(surface: np.ndarray) -> np.ndarray:
    surface = np.asarray(surface, dtype=np.float32)
    if surface.size == 0:
        return surface.astype(np.float32)
    finite = np.isfinite(surface)
    if not np.any(finite):
        return np.zeros_like(surface, dtype=np.float32)
    max_value = float(np.max(surface[finite]))
    if max_value <= 1e-6:
        return np.zeros_like(surface, dtype=np.float32)
    return np.clip(surface / max_value, 0.0, 1.0).astype(np.float32)


def _build_observation_support_bundle(
    *,
    standardized_tensor: np.ndarray,
    normalized_rows: list[dict[str, Any]],
    parameter_catalog: list[dict[str, Any]],
    canonical_axis: list[str],
    province_axis: list[str],
    month_axis: list[str],
    observation_ladder: list[dict[str, Any]],
) -> dict[str, Any]:
    latent_cfg = dict(_phase3_prior("latent_observation", {}) or {})
    support_floor = float(latent_cfg.get("support_floor", 0.05))
    support_power = float(latent_cfg.get("support_power", 0.85))
    support_tensor = _matched_support_tensor(normalized_rows, canonical_axis, province_axis, month_axis)
    ladder_map = {str(row.get("target_name") or ""): row for row in observation_ladder}
    target_support: dict[str, dict[str, np.ndarray | str | float]] = {}
    summary_rows: list[dict[str, Any]] = []
    for target_name in OBSERVATION_ORDER:
        matches = _match_target_indices(target_name, canonical_axis, parameter_catalog)
        row_surface, row_meta = _surface_from_normalized_rows(normalized_rows, target_name, province_axis, month_axis)
        sparse_surface, sparse_meta = _surface_from_sparse_support(standardized_tensor, support_tensor, matches, province_axis)
        if float(row_meta.get("observed_cell_fraction") or 0.0) > 0.0:
            meta = row_meta
            source_kind = "normalized_rows"
            direct_surface = row_surface
        else:
            meta = sparse_meta
            source_kind = "sparse_tensor"
            direct_surface = sparse_surface
        observed_mask = np.asarray(meta.get("observed_mask", np.zeros((len(province_axis), len(month_axis)), dtype=np.float32)), dtype=np.float32)
        support_surface = _normalize_support_surface(
            np.asarray(meta.get("support_surface", np.zeros_like(observed_mask, dtype=np.float32)), dtype=np.float32)
        )
        support_strength = np.maximum(observed_mask, np.clip(np.power(support_surface, support_power), 0.0, 1.0)).astype(np.float32)
        support_strength = np.maximum(support_strength, support_floor * observed_mask).astype(np.float32)
        latent_weight = np.clip(1.0 - support_strength, 0.0, 1.0).astype(np.float32)
        target_support[target_name] = {
            "observed_mask": observed_mask.astype(np.float32),
            "support_strength": support_strength.astype(np.float32),
            "latent_weight": latent_weight.astype(np.float32),
            "direct_surface": np.asarray(direct_surface, dtype=np.float32),
            "source_kind": source_kind,
        }
        summary_rows.append(
            {
                "target_name": target_name,
                "source_kind": source_kind,
                "observation_class": ladder_map.get(target_name, {}).get("observation_class", "prior_only"),
                "observed_fraction": round(float(np.mean(observed_mask)) if observed_mask.size else 0.0, 6),
                "mean_support_strength": round(float(np.mean(support_strength)) if support_strength.size else 0.0, 6),
                "mean_latent_weight": round(float(np.mean(latent_weight)) if latent_weight.size else 0.0, 6),
            }
        )
    return {"targets": target_support, "rows": summary_rows}


def _apply_harp_support_to_bundle(
    bundle: dict[str, Any],
    *,
    province_axis: list[str],
    month_axis: list[str],
) -> dict[str, Any]:
    out = {
        "targets": {
            target: {
                key: (np.asarray(value, dtype=np.float32).copy() if isinstance(value, np.ndarray) else value)
                for key, value in payload.items()
            }
            for target, payload in dict(bundle.get("targets", {}) or {}).items()
        },
        "rows": [dict(row) for row in list(bundle.get("rows", []) or [])],
    }
    curves = _harp_program_curves(month_axis)
    national_mask = _national_reference_mask(province_axis).reshape(len(province_axis), 1).astype(np.float32)
    month_weight = np.asarray(curves.get("weight", np.zeros((len(month_axis),), dtype=np.float32)), dtype=np.float32).reshape(1, len(month_axis))
    if not np.any(month_weight > 0.0):
        return out
    target_map = {
        "diagnosed_stock": "diagnosed_stock",
        "art_stock": "art_stock",
        "documented_suppression": "documented_suppression",
        "testing_coverage": "viral_load_tested_stock",
    }
    for target_name in OBSERVATION_ORDER:
        payload = out["targets"].get(target_name)
        if not payload:
            continue
        if target_name in target_map:
            applied = np.clip(national_mask * month_weight, 0.0, 1.0).astype(np.float32)
            payload["observed_mask"] = np.maximum(np.asarray(payload["observed_mask"], dtype=np.float32), applied).astype(np.float32)
            payload["support_strength"] = np.maximum(np.asarray(payload["support_strength"], dtype=np.float32), applied).astype(np.float32)
            payload["latent_weight"] = np.clip(1.0 - np.asarray(payload["support_strength"], dtype=np.float32), 0.0, 1.0).astype(np.float32)
    for row in out["rows"]:
        if row.get("target_name") in target_map:
            row["mean_support_strength"] = round(
                float(np.mean(np.asarray(out["targets"][str(row["target_name"])]["support_strength"], dtype=np.float32))),
                6,
            )
            row["observed_fraction"] = round(
                float(np.mean(np.asarray(out["targets"][str(row["target_name"])]["observed_mask"], dtype=np.float32))),
                6,
            )
            row["mean_latent_weight"] = round(
                float(np.mean(np.asarray(out["targets"][str(row["target_name"])]["latent_weight"], dtype=np.float32))),
                6,
            )
    return out


def _state_initialization_prior(
    observation_targets: dict[str, np.ndarray],
    *,
    support_bundle: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    init_cfg = dict(_phase3_prior("state_initialization", {}) or {})
    eps = float(_phase3_stabilizer("clip_floor", 1e-4))
    diagnosed = observation_targets["diagnosed_stock"]
    art = np.minimum(observation_targets["art_stock"], diagnosed)
    testing = np.minimum(observation_targets["testing_coverage"], art)
    suppression = np.minimum(observation_targets["documented_suppression"], testing)
    diagnosed_nonart = np.maximum(diagnosed - art, 0.0)
    linkage_prior = float(TRANSITION_PRIOR[1])
    attrition_prior = float(TRANSITION_PRIOR[3])
    reactivation_prior = float(TRANSITION_PRIOR[4])
    base_loss_share = attrition_prior / max(linkage_prior + attrition_prior + reactivation_prior, eps)
    art_gap = np.clip(diagnosed_nonart / np.clip(diagnosed, eps, None), 0.0, 1.0)
    suppression_gap = np.clip(1.0 - suppression / np.clip(testing, eps, None), 0.0, 1.0)
    loss_share = np.clip(
        base_loss_share
        + float(init_cfg.get("loss_share_art_gap_weight", 0.25)) * art_gap
        + float(init_cfg.get("loss_share_suppression_gap_weight", 0.15)) * suppression_gap,
        float(init_cfg.get("loss_share_lower", 0.05)),
        float(init_cfg.get("loss_share_upper", 0.85)),
    )
    lost = np.minimum(diagnosed_nonart * loss_share, diagnosed_nonart)
    diagnosed_no_art = np.maximum(diagnosed_nonart - lost, eps)
    on_art_unsuppressed = np.maximum(art - suppression, eps)
    undiagnosed = np.maximum(1.0 - diagnosed, eps)
    stacked = np.stack([undiagnosed, diagnosed_no_art, on_art_unsuppressed, suppression, lost], axis=-1).astype(np.float32)
    mean = stacked / np.clip(stacked.sum(axis=-1, keepdims=True), float(_phase3_stabilizer("mass_eps", 1e-6)), None)
    default_state = np.asarray(
        ((_phase3_prior("frozen_backtest", {}) or {}).get("default_initial_state", [0.48, 0.19, 0.18, 0.10, 0.05])),
        dtype=np.float32,
    ).reshape(1, 1, len(STATE_NAMES))
    if support_bundle is not None:
        support_targets = dict(support_bundle.get("targets", {}) or {})
        support_surfaces = [
            np.asarray(payload.get("support_strength", np.zeros_like(diagnosed, dtype=np.float32)), dtype=np.float32)
            for payload in [support_targets.get("diagnosed_stock", {}), support_targets.get("art_stock", {}), support_targets.get("documented_suppression", {}), support_targets.get("testing_coverage", {})]
        ]
        if support_surfaces:
            support_strength = np.clip(np.mean(np.stack(support_surfaces, axis=0), axis=0), 0.0, 1.0).astype(np.float32)
        else:
            support_strength = np.zeros_like(diagnosed, dtype=np.float32)
    else:
        support_strength = np.ones_like(diagnosed, dtype=np.float32) * 0.5
    blended = (support_strength[..., None] * mean + (1.0 - support_strength[..., None]) * default_state).astype(np.float32)
    blended = blended / np.clip(blended.sum(axis=-1, keepdims=True), float(_phase3_stabilizer("mass_eps", 1e-6)), None)
    concentration = np.clip(
        float(init_cfg.get("prior_strength_lower", 6.0))
        + (float(init_cfg.get("prior_strength_upper", 26.0)) - float(init_cfg.get("prior_strength_lower", 6.0))) * support_strength,
        float(init_cfg.get("prior_strength_lower", 6.0)),
        float(init_cfg.get("prior_strength_upper", 26.0)),
    ).astype(np.float32)
    return {
        "mean": blended.astype(np.float32),
        "concentration": concentration.astype(np.float32),
        "support_strength": support_strength.astype(np.float32),
    }


def _province_observed_state_targets(observation_targets: dict[str, np.ndarray], *, support_bundle: dict[str, Any] | None = None) -> np.ndarray:
    return _state_initialization_prior(observation_targets, support_bundle=support_bundle)["mean"]


def _state_rows(state_tensor: np.ndarray, province_axis: list[str], month_axis: list[str]) -> list[dict[str, Any]]:
    rows = []
    for province_idx, province in enumerate(province_axis):
        for month_idx, month in enumerate(month_axis):
            for state_idx, state_name in enumerate(STATE_NAMES):
                rows.append(
                    {
                        "province": province,
                        "time": month,
                        "state": state_name,
                        "value": round(float(state_tensor[province_idx, month_idx, state_idx]), 6),
                    }
                )
    return rows


def _choose_rescue_device(total_elements: int) -> str:
    if torch is None:
        return "cpu"
    if total_elements > 4_500_000:
        return "cpu"
    return choose_torch_device(prefer_gpu=True)


def simulate_transition_step_numpy(current: np.ndarray, transition_probs: np.ndarray, age_progress_rates: np.ndarray | None = None) -> np.ndarray:
    next_state = np.zeros_like(current)
    age_progress = age_progress_rates if age_progress_rates is not None else AGE_PROGRESS_RATES
    for duration_idx in range(current.shape[-1]):
        duration_mass = current[..., duration_idx]
        p_ud = transition_probs[..., duration_idx, 0]
        p_da = transition_probs[..., duration_idx, 1]
        p_av = transition_probs[..., duration_idx, 2]
        p_al = transition_probs[..., duration_idx, 3]
        p_la = transition_probs[..., duration_idx, 4]
        stay_u = duration_mass[..., 0] * (1.0 - p_ud)
        stay_d = duration_mass[..., 1] * (1.0 - p_da)
        stay_a = duration_mass[..., 2] * np.maximum(0.0, 1.0 - p_av - p_al)
        stay_v = duration_mass[..., 3]
        stay_l = duration_mass[..., 4] * (1.0 - p_la)
        flows_ud = duration_mass[..., 0] * p_ud
        flows_da = duration_mass[..., 1] * p_da
        flows_av = duration_mass[..., 2] * p_av
        flows_al = duration_mass[..., 2] * p_al
        flows_la = duration_mass[..., 4] * p_la
        next_duration_idx = min(duration_idx + 1, current.shape[-1] - 1)
        next_state[..., 0, next_duration_idx] += stay_u
        next_state[..., 1, next_duration_idx] += stay_d
        next_state[..., 2, next_duration_idx] += stay_a
        next_state[..., 3, next_duration_idx] += stay_v
        next_state[..., 4, next_duration_idx] += stay_l
        next_state[..., 1, 0] += flows_ud
        next_state[..., 2, 0] += flows_da + flows_la
        next_state[..., 3, 0] += flows_av
        next_state[..., 4, 0] += flows_al
    for age_idx in range(current.shape[2] - 1):
        rate = float(age_progress[age_idx])
        if rate <= 0.0:
            continue
        moved = next_state[:, :, age_idx] * rate
        next_state[:, :, age_idx] -= moved
        next_state[:, :, age_idx + 1] += moved
    total = next_state.sum(axis=(-2, -1), keepdims=True)
    return next_state / np.clip(total, 1e-6, None)


def _torch_transition_step(current: Any, transition_probs: Any, age_progress_rates: Any) -> Any:
    del age_progress_rates
    U = current[..., 0, :]
    D = current[..., 1, :]
    A = current[..., 2, :]
    V = current[..., 3, :]
    L = current[..., 4, :]
    p_ud = transition_probs[..., 0]
    p_da = transition_probs[..., 1]
    p_av = transition_probs[..., 2]
    p_al = transition_probs[..., 3]
    p_la = transition_probs[..., 4]

    stay_u = U * (1.0 - p_ud)
    stay_d = D * (1.0 - p_da)
    stay_a = A * torch.clamp(1.0 - p_av - p_al, min=0.0)
    stay_v = V
    stay_l = L * (1.0 - p_la)
    flows_ud = U * p_ud
    flows_da = D * p_da
    flows_av = A * p_av
    flows_al = A * p_al
    flows_la = L * p_la

    def _shift_capped(values: Any) -> Any:
        shifted = torch.cat([torch.zeros_like(values[..., :1]), values[..., :-1]], dim=-1)
        capped = torch.cat([torch.zeros_like(values[..., :-1]), values[..., -1:]], dim=-1)
        return shifted + capped

    def _bucket_zero(flow: Any) -> Any:
        total_flow = flow.sum(dim=-1, keepdim=True)
        return torch.cat([total_flow, torch.zeros_like(flow[..., 1:])], dim=-1)

    U_next = _shift_capped(stay_u)
    D_next = _shift_capped(stay_d) + _bucket_zero(flows_ud)
    A_next = _shift_capped(stay_a) + _bucket_zero(flows_da + flows_la)
    V_next = _shift_capped(stay_v) + _bucket_zero(flows_av)
    L_next = _shift_capped(stay_l) + _bucket_zero(flows_al)

    next_state = torch.stack([U_next, D_next, A_next, V_next, L_next], dim=-2)
    total = next_state.sum(dim=(-2, -1), keepdim=True)
    return next_state / torch.clamp(total, min=1e-6)


def _apply_kp_coupling_numpy(current: np.ndarray, mixing_matrix: np.ndarray) -> np.ndarray:
    coupled = np.einsum("pkaxsd,km->pmaxsd", current, mixing_matrix, optimize=True)
    total = coupled.sum(axis=(-2, -1), keepdims=True)
    return coupled / np.clip(total, 1e-6, None)


def _apply_kp_coupling_torch(current: Any, mixing_matrix: Any) -> Any:
    coupled = torch.einsum("pkaxsd,km->pmaxsd", current, mixing_matrix)
    total = coupled.sum(dim=(-2, -1), keepdim=True)
    return coupled / torch.clamp(total, min=1e-6)


def _load_network_operator_bundle(run_dir: Any, province_count: int) -> tuple[np.ndarray, list[str]]:
    path = run_dir / "phase15" / "network_operator_tensor.npz"
    if not path.exists():
        return np.zeros((0, province_count, province_count), dtype=np.float32), []
    operators = load_tensor_artifact(path).astype(np.float32)
    if operators.ndim != 3 or operators.shape[1] != province_count or operators.shape[2] != province_count:
        return np.zeros((0, province_count, province_count), dtype=np.float32), []
    catalog = read_json(run_dir / "phase15" / "network_operator_catalog.json", default=[])
    names = [str(row.get("operator_name") or row.get("operator_id") or f"operator_{idx}") for idx, row in enumerate(catalog)]
    if len(names) != operators.shape[0]:
        names = [f"operator_{idx}" for idx in range(operators.shape[0])]
    return operators, names


def _network_family_pressure_np(covariates: np.ndarray, indices: list[int]) -> np.ndarray:
    if not indices:
        return np.zeros(covariates.shape[:2], dtype=np.float32)
    clipped = np.clip(covariates[..., indices], -6.0, 6.0)
    return (1.0 / (1.0 + np.exp(-np.mean(clipped, axis=-1)))).astype(np.float32)


def _province_operator_torch(current: Any, operator: Any) -> Any:
    mixed = torch.einsum("pq,qkaxsd->pkaxsd", operator, current)
    total = mixed.sum(dim=(-2, -1), keepdim=True)
    return mixed / torch.clamp(total, min=1e-6)


def _blend_statewise_torch(base: Any, mixed: Any, alpha: Any, state_weights: Any) -> Any:
    weights = state_weights.view(1, 1, 1, 1, len(STATE_NAMES), 1)
    blended = (1.0 - alpha * weights) * base + (alpha * weights) * mixed
    total = blended.sum(dim=(-2, -1), keepdim=True)
    return blended / torch.clamp(total, min=1e-6)


def _province_operator_jax(state: Any, operator: Any) -> Any:
    mixed = jnp.einsum("pq,qs->ps", operator, state)
    return mixed / jnp.clip(jnp.sum(mixed, axis=-1, keepdims=True), 1e-6, None)


def _blend_statewise_jax(base: Any, mixed: Any, alpha: Any, state_weights: Any) -> Any:
    blended = (1.0 - alpha * state_weights.reshape(1, len(STATE_NAMES))) * base + (alpha * state_weights.reshape(1, len(STATE_NAMES))) * mixed
    return blended / jnp.clip(jnp.sum(blended, axis=-1, keepdims=True), 1e-6, None)


def _build_cd4_overlay(
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
    kp_axis: list[str],
    age_axis: list[str],
    sex_axis: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    cd4_cfg = dict(_phase3_prior("cd4_overlay", {}) or {})
    clip_floor = float(_phase3_stabilizer("clip_floor", 1e-4))
    province_count = len(province_axis)
    month_count = observation_targets["diagnosed_stock"].shape[1]
    overlay = np.zeros((province_count, len(kp_axis), len(age_axis), len(sex_axis), len(CD4_CATALOG), month_count), dtype=np.float32)
    prior_dominant_cells = 0
    for province_idx in range(province_count):
        diagnosed = observation_targets["diagnosed_stock"][province_idx]
        art = observation_targets["art_stock"][province_idx]
        suppression = observation_targets["documented_suppression"][province_idx]
        for month_idx in range(month_count):
            low_shift = float(cd4_cfg.get("low_art_weight", 0.10)) * float(1.0 - art[month_idx]) + float(cd4_cfg.get("low_suppression_weight", 0.08)) * float(1.0 - suppression[month_idx])
            high_shift = float(cd4_cfg.get("high_suppression_weight", 0.12)) * float(suppression[month_idx])
            base = DEFAULT_CD4_PRIOR.copy()
            base[0] += low_shift
            base[1] += float(cd4_cfg.get("mid_low_weight", 0.50)) * low_shift
            base[3] += high_shift
            base = np.clip(base, clip_floor, None)
            base = base / np.clip(base.sum(), 1e-6, None)
            for kp_idx in range(len(kp_axis)):
                for age_idx in range(len(age_axis)):
                    for sex_idx in range(len(sex_axis)):
                        overlay[province_idx, kp_idx, age_idx, sex_idx, :, month_idx] = base
                        if np.allclose(base, DEFAULT_CD4_PRIOR, atol=0.04):
                            prior_dominant_cells += 1
    simplex_error = float(np.abs(overlay.sum(axis=4) - 1.0).max()) if overlay.size else 0.0
    summary = {
        "cd4_catalog": CD4_CATALOG,
        "mean_distribution": {CD4_CATALOG[idx]: round(float(overlay[:, :, :, :, idx, :].mean()), 6) for idx in range(len(CD4_CATALOG))},
        "simplex_max_error": round(simplex_error, 8),
        "prior_dominant_cell_count": int(prior_dominant_cells),
        "prior_dominant": prior_dominant_cells >= int(max(1, overlay.shape[0] * overlay.shape[1])),
    }
    return overlay, summary


def _build_covariates(observation_targets: dict[str, np.ndarray]) -> np.ndarray:
    diagnosed = observation_targets["diagnosed_stock"]
    art = observation_targets["art_stock"]
    suppression = observation_targets["documented_suppression"]
    stacked = np.stack([diagnosed, art, suppression], axis=-1).astype(np.float32)
    centered = stacked - stacked.mean(axis=1, keepdims=True)
    return centered.astype(np.float32)


def _build_modifier_covariates(
    *,
    observation_targets: dict[str, np.ndarray],
    standardized_tensor: np.ndarray,
    canonical_axis: list[str],
    candidate_profiles: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = _build_covariates(observation_targets)
    axis_index = {name: idx for idx, name in enumerate(canonical_axis)}
    selected = []
    used = set()
    block_order = ["economics", "logistics", "behavior", "population", "biology", "policy"]
    for block in block_order:
        eligible = [
            profile
            for profile in candidate_profiles
            if profile.get("primary_block") == block
            and profile.get("canonical_name") in axis_index
            and profile.get("canonical_name") not in used
            and profile.get("curation_status") in {"promoted_candidate", "research_candidate", "review"}
        ]
        if not eligible:
            continue
        eligible.sort(
            key=lambda item: (
                float(item.get("curation_score", 0.0)),
                float(item.get("dag_score", 0.0)),
                int(item.get("numeric_support", 0)),
            ),
            reverse=True,
        )
        champion = eligible[0]
        used.add(str(champion["canonical_name"]))
        selected.append(
            {
                "block_name": block,
                "canonical_name": champion["canonical_name"],
                "curation_status": champion.get("curation_status"),
                "curation_score": champion.get("curation_score"),
                "dag_score": champion.get("dag_score"),
            }
        )

    covariate_parts = [base]
    covariate_names = ["obs_diagnosed", "obs_art", "obs_suppression"]
    hook_masks = [
        [1.0, 0.4, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.5, 0.0],
    ]
    for item in selected:
        feature_idx = axis_index[item["canonical_name"]]
        feature_surface = standardized_tensor[:, :, feature_idx : feature_idx + 1].astype(np.float32)
        centered = feature_surface - feature_surface.mean(axis=1, keepdims=True)
        covariate_parts.append(centered)
        covariate_names.append(f"det::{item['block_name']}::{item['canonical_name']}")
        block_name = str(item["block_name"])
        if block_name in {"behavior", "population"}:
            hook_masks.append(_transition_hook_mask(["diagnosis_transitions", "linkage_transitions"]))
        elif block_name in {"biology"}:
            hook_masks.append(_transition_hook_mask(["suppression_transitions", "retention_attrition_transitions"]))
        elif block_name in {"logistics", "policy", "economics"}:
            hook_masks.append(_transition_hook_mask(["linkage_transitions", "retention_attrition_transitions"]))
        else:
            hook_masks.append(_transition_hook_mask(["diagnosis_transitions"]))
    covariates = np.concatenate(covariate_parts, axis=-1).astype(np.float32)
    return covariates, {
        "selected_determinant_modifiers": selected,
        "covariate_names": covariate_names,
        "transition_hook_masks": hook_masks,
        "coupling_covariate_names": [],
        "network_family_indices": {},
    }


def _transition_hook_mask(hooks: list[str]) -> list[float]:
    mask = np.zeros((len(TRANSITION_NAMES),), dtype=np.float32)
    if "diagnosis_transitions" in hooks:
        mask[0] = 1.0
    if "linkage_transitions" in hooks:
        mask[1] = 1.0
    if "suppression_transitions" in hooks:
        mask[2] = 1.0
    if "retention_attrition_transitions" in hooks:
        mask[1] = max(mask[1], 0.4)
        mask[3] = 1.0
        mask[4] = 1.0
    if not np.any(mask):
        mask[:] = 0.25
    return mask.tolist()


def _build_mesoscopic_modifier_covariates(
    *,
    run_dir: Any,
    observation_targets: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = _build_covariates(observation_targets)
    factor_catalog = read_json(run_dir / "phase15" / "mesoscopic_factor_catalog.json", default=[])
    promoted = read_json(run_dir / "phase2" / "promoted_factor_set.json", default=[])
    supporting = read_json(run_dir / "phase2" / "supporting_factor_set.json", default=[])
    factor_tensor_path = run_dir / "phase15" / "mesoscopic_factor_tensor.npz"
    if not factor_catalog or not factor_tensor_path.exists():
        return base, {
            "selected_determinant_modifiers": [],
            "covariate_names": ["obs_diagnosed", "obs_art", "obs_suppression"],
            "transition_hook_masks": [[1.0, 0.4, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 0.5, 0.0]],
            "coupling_covariate_names": [],
            "network_family_indices": {},
        }

    factor_tensor = load_tensor_artifact(factor_tensor_path).astype(np.float32)
    alignment_notes: list[str] = []
    if factor_tensor.shape[0] != base.shape[0] or factor_tensor.shape[1] != base.shape[1]:
        if factor_tensor.shape[0] < base.shape[0] or factor_tensor.shape[1] < base.shape[1]:
            return base, {
                "selected_determinant_modifiers": [],
                "covariate_names": ["obs_diagnosed", "obs_art", "obs_suppression"],
                "transition_hook_masks": [[1.0, 0.4, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 0.5, 0.0]],
                "coupling_covariate_names": [],
                "network_family_indices": {},
                "alignment_notes": [
                    f"factor_tensor_shape_mismatch:{tuple(int(v) for v in factor_tensor.shape)}:base_shape:{tuple(int(v) for v in base.shape)}",
                    "mesoscopic_factor_tensor_dropped_due_to_short_shape",
                ],
            }
        factor_tensor = factor_tensor[: base.shape[0], : base.shape[1], :]
        alignment_notes.append(
            f"factor_tensor_sliced_to_training_window:{tuple(int(v) for v in factor_tensor.shape)}"
        )
    factor_index = {row["factor_id"]: idx for idx, row in enumerate(factor_catalog)}
    covariate_parts = [base]
    covariate_names = ["obs_diagnosed", "obs_art", "obs_suppression"]
    hook_masks = [
        [1.0, 0.4, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 1.0, 0.5, 0.0],
    ]
    selected_rows = []
    coupling_names = []
    network_family_indices: dict[str, list[int]] = defaultdict(list)
    for row, scale, prefix in [(item, 1.0, "main") for item in promoted] + [(item, 0.55, "support") for item in supporting]:
        factor_id = str(row.get("factor_id") or "")
        if factor_id not in factor_index:
            continue
        surface = factor_tensor[:, :, factor_index[factor_id] : factor_index[factor_id] + 1]
        centered = surface - surface.mean(axis=1, keepdims=True)
        covariate_parts.append((centered * scale).astype(np.float32))
        covariate_names.append(f"{prefix}::{row.get('block_name', 'block')}::{row.get('factor_name', factor_id)}")
        covariate_idx = len(covariate_names) - 1
        hooks = list(row.get("transition_hooks") or [])
        hook_masks.append(_transition_hook_mask(hooks))
        selected_rows.append(
            {
                "factor_id": factor_id,
                "factor_name": row.get("factor_name", factor_id),
                "promotion_class": row.get("promotion_class", prefix),
                "block_name": row.get("block_name", "mixed"),
                "factor_class": row.get("factor_class", "mesoscopic_factor"),
                "transition_hooks": hooks,
                "network_feature_family": row.get("network_feature_family", ""),
                "diagnostic_score": row.get("diagnostic_score", 0.0),
            }
        )
        if row.get("network_feature_family") in {"reaction_diffusion", "information_propagation"} or "subgroup_allocation_priors" in hooks:
            coupling_names.append(covariate_names[-1])
        network_family = str(row.get("network_feature_family") or "")
        if network_family:
            network_family_indices[network_family].append(covariate_idx)
    covariates = np.concatenate(covariate_parts, axis=-1).astype(np.float32)
    return covariates, {
        "selected_determinant_modifiers": selected_rows,
        "covariate_names": covariate_names,
        "transition_hook_masks": hook_masks,
        "coupling_covariate_names": coupling_names,
        "network_family_indices": {key: value for key, value in network_family_indices.items()},
        "alignment_notes": alignment_notes,
    }


def _fit_rescue_core_numpy(
    *,
    run_dir: Any,
    profile_id: str,
    observation_targets: dict[str, np.ndarray],
    observation_support_bundle: dict[str, Any] | None = None,
    standardized_tensor: np.ndarray,
    canonical_axis: list[str],
    candidate_profiles: list[dict[str, Any]],
    subgroup_weights: np.ndarray,
    cd4_overlay: np.ndarray,
    province_axis: list[str],
    kp_axis: list[str],
    age_axis: list[str],
    sex_axis: list[str],
    duration_catalog: list[str],
    calibration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del subgroup_weights, cd4_overlay, province_axis, age_axis, sex_axis, duration_catalog
    _, covariate_meta = (
        _build_mesoscopic_modifier_covariates(run_dir=run_dir, observation_targets=observation_targets)
        if profile_id == RESCUE_V2_PROFILE_ID
        else _build_modifier_covariates(
            observation_targets=observation_targets,
            standardized_tensor=standardized_tensor,
            canonical_axis=canonical_axis,
            candidate_profiles=candidate_profiles,
        )
    )
    state_estimates = _province_observed_state_targets(observation_targets, support_bundle=observation_support_bundle).astype(np.float32)
    base_transition = np.broadcast_to(TRANSITION_PRIOR.reshape(1, 1, -1), (state_estimates.shape[0], state_estimates.shape[1], len(TRANSITION_NAMES))).astype(np.float32)
    kp_coupling_matrix = np.eye(len(kp_axis), dtype=np.float32).tolist()
    return {
        "state_estimates": state_estimates,
        "transition_probs": base_transition[:, None, None, None, None, :, :].transpose(0, 1, 2, 3, 4, 6, 5),
        "prediction_stack": {
            "diagnosed_stock": np.clip(1.0 - state_estimates[..., 0], 0.0, 1.0),
            "art_stock": np.clip(state_estimates[..., 2] + state_estimates[..., 3], 0.0, 1.0),
            "documented_suppression": np.clip(state_estimates[..., 3], 0.0, 1.0),
            "testing_coverage": np.clip(1.0 - state_estimates[..., 0], 0.0, 1.0),
            "deaths": np.clip(state_estimates[..., 4] * 0.18, 0.0, 0.25),
        },
        "loss_breakdown": {
            "observation_fit_loss": 0.0,
            "observation_anchor_penalty": 0.0,
            "lower_bound_suppression_penalty": 0.0,
            "diagnosed_optimism_penalty": 0.0,
            "hierarchy_reconciliation_penalty": 0.0,
            "stock_conservation_penalty": 0.0,
            "transition_plausibility_penalty": 0.0,
            "regularization_penalty": 0.0,
            "total_loss": 0.0,
        },
        "loss_trace": [],
        "parameters_summary": {
            "national_transition": TRANSITION_PRIOR.round(6).tolist(),
            "kp_coupling_matrix": kp_coupling_matrix,
            "determinant_covariates": covariate_meta,
            "metapopulation_engine": {"enabled": False, "operator_names": [], "scale": []},
        },
        "device": "cpu",
        "inference_family": "numpy_map",
    }


def _aggregate_transition_step_jax(state: Any, probs: Any) -> Any:
    p_ud, p_da, p_av, p_al, p_la = [probs[..., idx] for idx in range(probs.shape[-1])]
    U = state[..., 0]
    D = state[..., 1]
    A = state[..., 2]
    V = state[..., 3]
    L = state[..., 4]
    flow_ud = U * p_ud
    flow_da = D * p_da
    flow_av = A * p_av
    flow_al = A * p_al
    flow_la = L * p_la
    next_state = jnp.stack(
        [
            U - flow_ud,
            D + flow_ud - flow_da,
            A + flow_da + flow_la - flow_av - flow_al,
            V + flow_av,
            L + flow_al - flow_la,
        ],
        axis=-1,
    )
    next_state = jnp.clip(next_state, 1e-6, None)
    return next_state / jnp.clip(jnp.sum(next_state, axis=-1, keepdims=True), 1e-6, None)


def _rescue_svi_model(
    covariates: Any,
    region_index: Any,
    region_count: int,
    observed_diag: Any,
    observed_art: Any,
    observed_sup: Any,
    observed_test: Any,
    init_base: Any,
    init_province_base: Any,
    init_prior_strength: Any,
    hook_mask: Any,
    mobility_operator: Any,
    service_operator: Any,
    information_operator: Any,
    mobility_pressure: Any,
    service_pressure: Any,
    information_pressure: Any,
    national_anchor_diag: Any,
    national_anchor_art: Any,
    national_anchor_sup: Any,
    national_anchor_third: Any,
    national_anchor_weight: Any,
    vl_test_prior_logit: Any,
    harp_diag: Any,
    harp_art: Any,
    harp_tested: Any,
    harp_sup: Any,
    harp_tested_among_art: Any,
    harp_suppressed_among_art: Any,
    harp_weight: Any,
    linkage_second95_target: Any,
    linkage_d_to_a_target: Any,
    linkage_weight: Any,
    suppression_overall_target: Any,
    suppression_among_art_target: Any,
    suppression_tested_among_art_target: Any,
    suppression_transition_target: Any,
    suppression_weight: Any,
    suppressed_given_test_prior_logit: Any,
    province_transition_prior: Any,
    province_reporting_weight: Any,
    province_vl_prior: Any,
    province_documentation_prior: Any,
    subgroup_feature_matrix: Any,
    base_kp_log: Any,
    base_age_log: Any,
    base_sex_log: Any,
    cd4_low_base: Any,
    cd4_high_base: Any,
    observed_diag_support: Any,
    observed_art_support: Any,
    observed_sup_support: Any,
    observed_test_support: Any,
    national_mask: Any,
) -> None:
    if numpyro is None or jnp is None:
        raise RuntimeError("NumPyro/JAX backend is unavailable")
    province_count, month_count, cov_count = covariates.shape
    svi_cfg = dict(_phase3_prior("jax_svi_priors", {}) or {})
    latent_cfg = dict(_phase3_prior("latent_observation", {}) or {})
    subgroup_cfg = dict(_phase3_prior("subgroup_hyperpriors", {}) or {})
    cd4_hyper_cfg = dict(_phase3_prior("cd4_hyperpriors", {}) or {})
    archetype_cfg = dict(_phase3_prior("archetype_hyperpriors", {}) or {})
    kp_count = int(base_kp_log.shape[-1])
    age_count = int(base_age_log.shape[-1])
    sex_count = int(base_sex_log.shape[-1])
    feature_count = int(subgroup_feature_matrix.shape[-1])
    prior_logit = jnp.asarray(_logit_np(TRANSITION_PRIOR), dtype=jnp.float32)
    national_logit = numpyro.sample("national_logit", dist.Normal(prior_logit, float(svi_cfg.get("transition_prior_sigma", 0.35))).to_event(1))
    region_offset = numpyro.sample(
        "region_offset",
        dist.Normal(jnp.zeros((region_count, len(TRANSITION_NAMES))), float(svi_cfg.get("province_transition_sigma", 0.12)) * 1.15).to_event(2),
    )
    archetype_transition_scale_log = numpyro.sample(
        "archetype_transition_scale_log",
        dist.Normal(jnp.zeros((len(TRANSITION_NAMES),), dtype=jnp.float32), float(archetype_cfg.get("transition_scale_sigma", 0.25))).to_event(1),
    )
    archetype_transition_scale = jnp.exp(archetype_transition_scale_log)
    province_offset = numpyro.sample(
        "province_offset",
        dist.Normal(province_transition_prior * archetype_transition_scale.reshape(1, len(TRANSITION_NAMES)), float(svi_cfg.get("province_transition_sigma", 0.12))).to_event(2),
    )
    subgroup_feature_sigma = float(subgroup_cfg.get("feature_coef_sigma", 0.18))
    subgroup_state_sigma = float(subgroup_cfg.get("state_effect_sigma", 0.10))
    subgroup_transition_sigma = float(subgroup_cfg.get("transition_effect_sigma", 0.08))
    kp_prior_coef = numpyro.sample("kp_prior_coef", dist.Normal(jnp.zeros((feature_count, kp_count), dtype=jnp.float32), subgroup_feature_sigma).to_event(2))
    age_prior_coef = numpyro.sample("age_prior_coef", dist.Normal(jnp.zeros((feature_count, age_count), dtype=jnp.float32), subgroup_feature_sigma).to_event(2))
    sex_prior_coef = numpyro.sample("sex_prior_coef", dist.Normal(jnp.zeros((feature_count, sex_count), dtype=jnp.float32), subgroup_feature_sigma).to_event(2))
    kp_state_effect = numpyro.sample("kp_state_effect", dist.Normal(jnp.zeros((kp_count, len(STATE_NAMES)), dtype=jnp.float32), subgroup_state_sigma).to_event(2))
    age_state_effect = numpyro.sample("age_state_effect", dist.Normal(jnp.zeros((age_count, len(STATE_NAMES)), dtype=jnp.float32), subgroup_state_sigma).to_event(2))
    sex_state_effect = numpyro.sample("sex_state_effect", dist.Normal(jnp.zeros((sex_count, len(STATE_NAMES)), dtype=jnp.float32), subgroup_state_sigma).to_event(2))
    kp_transition_effect = numpyro.sample("kp_transition_effect", dist.Normal(jnp.zeros((kp_count, len(TRANSITION_NAMES)), dtype=jnp.float32), subgroup_transition_sigma).to_event(2))
    age_transition_effect = numpyro.sample("age_transition_effect", dist.Normal(jnp.zeros((age_count, len(TRANSITION_NAMES)), dtype=jnp.float32), subgroup_transition_sigma).to_event(2))
    sex_transition_effect = numpyro.sample("sex_transition_effect", dist.Normal(jnp.zeros((sex_count, len(TRANSITION_NAMES)), dtype=jnp.float32), subgroup_transition_sigma).to_event(2))
    covariate_weights = numpyro.sample("covariate_weights", dist.Normal(jnp.zeros((cov_count, len(TRANSITION_NAMES))), 0.10).to_event(2))
    initial_logit = numpyro.sample("initial_logit", dist.Normal(init_base, 0.22).to_event(1))
    init_sigma = 0.28 / jnp.sqrt(jnp.clip(init_prior_strength.reshape(province_count, 1), 1.0, None))
    province_initial = numpyro.sample("province_initial", dist.Normal(init_province_base, init_sigma).to_event(2))
    metapopulation_scale_logit = numpyro.sample("metapopulation_scale_logit", dist.Normal(jnp.asarray([-2.2, -2.35, -2.3]), 0.35).to_event(1))
    mobility_weights = jnp.asarray([0.45, 0.35, 0.10, 0.00, 0.10], dtype=jnp.float32)
    service_weights = jnp.asarray([0.05, 0.18, 0.36, 0.21, 0.20], dtype=jnp.float32)
    information_weights = jnp.asarray([0.40, 0.34, 0.12, 0.04, 0.10], dtype=jnp.float32)
    vl_test_logit = numpyro.sample("vl_test_logit", dist.Normal(vl_test_prior_logit, float(svi_cfg.get("vl_test_sigma", 0.20))).to_event(1))
    suppressed_given_test_logit = numpyro.sample(
        "suppressed_given_test_logit",
        dist.Normal(suppressed_given_test_prior_logit, float(svi_cfg.get("suppressed_given_test_sigma", 0.16))).to_event(1),
    )
    archetype_process_scale_log = numpyro.sample(
        "archetype_process_scale_log",
        dist.Normal(jnp.zeros((2,), dtype=jnp.float32), float(archetype_cfg.get("process_scale_sigma", 0.22))).to_event(1),
    )
    archetype_process_scale = jnp.exp(archetype_process_scale_log)
    vl_test_province_offset = numpyro.sample(
        "vl_test_province_offset",
        dist.Normal(province_vl_prior * archetype_process_scale[0], float(svi_cfg.get("vl_province_sigma", 0.18))).to_event(1),
    )
    documented_province_offset = numpyro.sample(
        "documented_province_offset",
        dist.Normal(province_documentation_prior * archetype_process_scale[1], float(svi_cfg.get("documentation_province_sigma", 0.16))).to_event(1),
    )
    obs_province_offset = numpyro.sample(
        "obs_province_offset",
        dist.Normal(jnp.zeros((province_count, 4), dtype=jnp.float32), float(latent_cfg.get("province_offset_sigma", 0.28))).to_event(2),
    )
    obs_month_offset = numpyro.sample(
        "obs_month_offset",
        dist.Normal(jnp.zeros((month_count, 4), dtype=jnp.float32), float(latent_cfg.get("month_offset_sigma", 0.22))).to_event(2),
    )
    archetype_observation_scale_log = numpyro.sample(
        "archetype_observation_scale_log",
        dist.Normal(jnp.zeros((3,), dtype=jnp.float32), float(archetype_cfg.get("observation_scale_sigma", 0.18))).to_event(1),
    )
    archetype_observation_scale = jnp.exp(archetype_observation_scale_log)
    cd4_age_sigma = float(cd4_hyper_cfg.get("age_low_sigma", 0.08))
    cd4_high_sigma = float(cd4_hyper_cfg.get("kp_sex_high_sigma", 0.06))
    cd4_transition_sigma = float(cd4_hyper_cfg.get("transition_coef_sigma", 0.10))
    cd4_age_low_offset = numpyro.sample("cd4_age_low_offset", dist.Normal(jnp.zeros((age_count,), dtype=jnp.float32), cd4_age_sigma).to_event(1))
    cd4_kp_sex_high_offset = numpyro.sample("cd4_kp_sex_high_offset", dist.Normal(jnp.zeros((kp_count, sex_count), dtype=jnp.float32), cd4_high_sigma).to_event(2))
    cd4_low_coef = numpyro.sample("cd4_low_coef", dist.Normal(jnp.zeros((len(TRANSITION_NAMES),), dtype=jnp.float32), cd4_transition_sigma).to_event(1))
    cd4_high_coef = numpyro.sample("cd4_high_coef", dist.Normal(jnp.zeros((len(TRANSITION_NAMES),), dtype=jnp.float32), cd4_transition_sigma).to_event(1))

    kp_probs_dynamic = jax.nn.softmax(base_kp_log + subgroup_feature_matrix @ kp_prior_coef, axis=-1)
    age_probs_dynamic = jax.nn.softmax(base_age_log + subgroup_feature_matrix @ age_prior_coef, axis=-1)
    sex_probs_dynamic = jax.nn.softmax(base_sex_log + subgroup_feature_matrix @ sex_prior_coef, axis=-1)
    subgroup_weights_dynamic = kp_probs_dynamic[:, :, None, None] * age_probs_dynamic[:, None, :, None] * sex_probs_dynamic[:, None, None, :]
    subgroup_weights_dynamic = subgroup_weights_dynamic / jnp.clip(jnp.sum(subgroup_weights_dynamic, axis=(1, 2, 3), keepdims=True), 1e-6, None)
    subgroup_initial_shift = kp_probs_dynamic @ kp_state_effect + age_probs_dynamic @ age_state_effect + sex_probs_dynamic @ sex_state_effect
    subgroup_transition_shift = kp_probs_dynamic @ kp_transition_effect + age_probs_dynamic @ age_transition_effect + sex_probs_dynamic @ sex_transition_effect
    cd4_low_dynamic = jnp.clip(cd4_low_base + cd4_age_low_offset.reshape(1, 1, age_count, 1, 1), 0.0, 1.5)
    cd4_high_dynamic = jnp.clip(cd4_high_base + cd4_kp_sex_high_offset.reshape(1, kp_count, 1, sex_count, 1), 0.0, 1.5)
    cd4_low_summary = jnp.sum(cd4_low_dynamic * subgroup_weights_dynamic[..., None], axis=(1, 2, 3))
    cd4_high_summary = jnp.sum(cd4_high_dynamic * subgroup_weights_dynamic[..., None], axis=(1, 2, 3))
    numpyro.deterministic("kp_probs_dynamic", kp_probs_dynamic)
    numpyro.deterministic("age_probs_dynamic", age_probs_dynamic)
    numpyro.deterministic("sex_probs_dynamic", sex_probs_dynamic)

    init_state = jax.nn.softmax(initial_logit.reshape(1, len(STATE_NAMES)) + province_initial + subgroup_initial_shift, axis=-1)
    logits = (
        national_logit.reshape(1, 1, len(TRANSITION_NAMES))
        + region_offset[region_index][:, None, :]
        + province_offset[:, None, :]
        + subgroup_transition_shift[:, None, :]
        + cd4_low_summary[:, :, None] * cd4_low_coef.reshape(1, 1, len(TRANSITION_NAMES))
        + cd4_high_summary[:, :, None] * cd4_high_coef.reshape(1, 1, len(TRANSITION_NAMES))
        + jnp.einsum("ptc,cr->ptr", covariates, covariate_weights * hook_mask)
    )
    probs = jax.nn.sigmoid(logits)
    numpyro.deterministic("transition_probs", probs)

    def _step(state, inputs):
        prob_t, mobility_p_t, service_p_t, info_p_t = inputs
        next_state = _aggregate_transition_step_jax(state, prob_t)
        mobility_alpha = jax.nn.sigmoid(metapopulation_scale_logit[0]) * mobility_p_t.reshape(province_count, 1)
        service_alpha = jax.nn.sigmoid(metapopulation_scale_logit[1]) * service_p_t.reshape(province_count, 1)
        info_alpha = jax.nn.sigmoid(metapopulation_scale_logit[2]) * info_p_t.reshape(province_count, 1)
        next_state = _blend_statewise_jax(next_state, _province_operator_jax(next_state, mobility_operator), mobility_alpha, mobility_weights)
        next_state = _blend_statewise_jax(next_state, _province_operator_jax(next_state, service_operator), service_alpha, service_weights)
        next_state = _blend_statewise_jax(next_state, _province_operator_jax(next_state, information_operator), info_alpha, information_weights)
        return next_state, next_state

    _, state_path = jax.lax.scan(
        _step,
        init_state,
        (
            jnp.swapaxes(probs, 0, 1),
            jnp.swapaxes(mobility_pressure, 0, 1),
            jnp.swapaxes(service_pressure, 0, 1),
            jnp.swapaxes(information_pressure, 0, 1),
        ),
    )
    state_path = jnp.swapaxes(state_path, 0, 1)
    numpyro.deterministic("state_path", state_path)

    diagnosed_pred = 1.0 - state_path[..., 0]
    art_pred = state_path[..., 2] + state_path[..., 3]
    testing_pred = art_pred * jax.nn.sigmoid(vl_test_logit.reshape(1, month_count) + vl_test_province_offset.reshape(province_count, 1))
    suppression_pred = jnp.minimum(
        state_path[..., 3],
        testing_pred * jax.nn.sigmoid(suppressed_given_test_logit.reshape(1, month_count) + documented_province_offset.reshape(province_count, 1)),
    )
    obs_latent_weight_diag = jnp.clip(1.0 - observed_diag_support, 0.0, 1.0)
    obs_latent_weight_art = jnp.clip(1.0 - observed_art_support, 0.0, 1.0)
    obs_latent_weight_sup = jnp.clip(1.0 - observed_sup_support, 0.0, 1.0)
    obs_latent_weight_test = jnp.clip(1.0 - observed_test_support, 0.0, 1.0)
    diag_base = jnp.log(jnp.clip(observed_diag, 1e-5, 1.0 - 1e-5) / jnp.clip(1.0 - observed_diag, 1e-5, 1.0))
    art_base = jnp.log(jnp.clip(observed_art, 1e-5, 1.0 - 1e-5) / jnp.clip(1.0 - observed_art, 1e-5, 1.0))
    sup_base = jnp.log(jnp.clip(observed_sup, 1e-5, 1.0 - 1e-5) / jnp.clip(1.0 - observed_sup, 1e-5, 1.0))
    test_base = jnp.log(jnp.clip(observed_test, 1e-5, 1.0 - 1e-5) / jnp.clip(1.0 - observed_test, 1e-5, 1.0))
    obs_offset_stack = obs_province_offset[:, None, :] + obs_month_offset[None, :, :]
    latent_diag = jax.nn.sigmoid(diag_base + obs_latent_weight_diag * obs_offset_stack[..., 0])
    latent_art = jax.nn.sigmoid(art_base + obs_latent_weight_art * obs_offset_stack[..., 1])
    latent_test = jax.nn.sigmoid(test_base + obs_latent_weight_test * obs_offset_stack[..., 3])
    latent_sup = jax.nn.sigmoid(sup_base + obs_latent_weight_sup * obs_offset_stack[..., 2])
    latent_art = jnp.minimum(latent_art, latent_diag)
    latent_test = jnp.minimum(latent_test, latent_art)
    latent_sup = jnp.minimum(latent_sup, latent_test)
    numpyro.deterministic("latent_obs_diag", latent_diag)
    numpyro.deterministic("latent_obs_art", latent_art)
    numpyro.deterministic("latent_obs_sup", latent_sup)
    numpyro.deterministic("latent_obs_test", latent_test)
    national_diag = jnp.sum(diagnosed_pred * national_mask.reshape(province_count, 1), axis=0)
    national_art = jnp.sum(art_pred * national_mask.reshape(province_count, 1), axis=0)
    national_tested = jnp.sum(testing_pred * national_mask.reshape(province_count, 1), axis=0)
    national_sup = jnp.sum(suppression_pred * national_mask.reshape(province_count, 1), axis=0)
    national_third = national_sup / jnp.clip(national_art, 1e-6, None)
    national_tested_among_art = national_tested / jnp.clip(national_art, 1e-6, None)
    national_d_to_a = jnp.sum(probs[..., 1] * national_mask.reshape(province_count, 1), axis=0)
    national_a_to_v = jnp.sum(probs[..., 2] * national_mask.reshape(province_count, 1), axis=0)

    sigma_diag = numpyro.sample("sigma_diag", dist.HalfNormal(0.08))
    sigma_art = numpyro.sample("sigma_art", dist.HalfNormal(0.08))
    sigma_sup = numpyro.sample("sigma_sup", dist.HalfNormal(0.06))
    sigma_test = numpyro.sample("sigma_test", dist.HalfNormal(0.08))
    province_weight = jnp.clip(province_reporting_weight.reshape(province_count, 1) * archetype_observation_scale[0], 0.15, 1.25)
    obs_fit = (
        jnp.mean(jnp.square((diagnosed_pred - latent_diag) / (sigma_diag / province_weight)))
        + jnp.mean(jnp.square((art_pred - latent_art) / (sigma_art / province_weight)))
        + jnp.mean(jnp.square((suppression_pred - latent_sup) / (sigma_sup / province_weight)))
        + jnp.mean(jnp.square((testing_pred - latent_test) / (sigma_test / province_weight)))
    )
    numpyro.factor("observation_fit", -0.5 * obs_fit)
    anchor_reversion_scale = float(latent_cfg.get("anchor_reversion_scale", 9.0))
    latent_anchor_penalty = jnp.mean(observed_diag_support * jnp.square(latent_diag - observed_diag))
    latent_anchor_penalty = latent_anchor_penalty + jnp.mean(observed_art_support * jnp.square(latent_art - observed_art))
    latent_anchor_penalty = latent_anchor_penalty + jnp.mean(observed_sup_support * jnp.square(latent_sup - observed_sup))
    latent_anchor_penalty = latent_anchor_penalty + jnp.mean(observed_test_support * jnp.square(latent_test - observed_test))
    numpyro.factor("observation_anchor_penalty", -anchor_reversion_scale * latent_anchor_penalty)
    lower_bound = jnp.maximum(latent_sup + 0.03, latent_test * 0.95)
    numpyro.factor("suppression_lower_bound_penalty", -45.0 * jnp.mean(jnp.square(jax.nn.relu(suppression_pred - lower_bound))))
    anchor_penalty = jnp.mean(
        national_anchor_weight
        * (
            jnp.square(national_diag - national_anchor_diag)
            + jnp.square(national_art - national_anchor_art)
            + jnp.square(national_sup - national_anchor_sup)
        )
    )
    valid_third = (national_anchor_weight > 0) & (national_anchor_third > 0)
    third_penalty = jnp.where(valid_third, national_anchor_weight * jnp.square(national_third - national_anchor_third), 0.0)
    numpyro.factor("national_anchor_penalty", -36.0 * (anchor_penalty + jnp.mean(third_penalty)))
    harp_penalty = jnp.mean(
        harp_weight
        * (
            jnp.square(national_diag - harp_diag)
            + jnp.square(national_art - harp_art)
            + jnp.square(national_tested - harp_tested)
            + jnp.square(national_sup - harp_sup)
        )
    )
    valid_harp_tested = (harp_weight > 0) & (harp_tested_among_art > 0)
    valid_harp_supp = (harp_weight > 0) & (harp_suppressed_among_art > 0)
    harp_ratio_penalty = jnp.where(valid_harp_tested, harp_weight * jnp.square(national_tested_among_art - harp_tested_among_art), 0.0)
    harp_sup_ratio_penalty = jnp.where(valid_harp_supp, harp_weight * jnp.square(national_third - harp_suppressed_among_art), 0.0)
    vl_smooth_penalty = jnp.mean(jnp.square(jax.nn.sigmoid(vl_test_logit[1:]) - jax.nn.sigmoid(vl_test_logit[:-1]))) if month_count > 1 else 0.0
    numpyro.factor("harp_program_penalty", -52.0 * (harp_penalty + jnp.mean(harp_ratio_penalty) + jnp.mean(harp_sup_ratio_penalty) + 0.20 * vl_smooth_penalty))
    linkage_penalty = jnp.mean(linkage_weight * (jnp.square(national_d_to_a - linkage_d_to_a_target) + 0.85 * jnp.square(national_art / jnp.clip(national_diag, 1e-6, None) - linkage_second95_target)))
    numpyro.factor("linkage_penalty", -36.0 * linkage_penalty)
    suppression_penalty = jnp.mean(
        suppression_weight
        * (
            jnp.square(national_a_to_v - suppression_transition_target)
            + 0.90 * jnp.square(national_third - suppression_among_art_target)
            + 0.65 * jnp.square(national_sup - suppression_overall_target)
            + 0.35 * jnp.square(national_tested_among_art - suppression_tested_among_art_target)
        )
    )
    numpyro.factor("suppression_penalty", -16.0 * suppression_penalty)
    numpyro.factor(
        "synthetic_pretraining_penalty",
        -4.0
        * (
            jnp.mean(jnp.square(vl_test_province_offset - province_vl_prior * archetype_process_scale[0]))
            + jnp.mean(jnp.square(documented_province_offset - province_documentation_prior * archetype_process_scale[1]))
            + 0.25 * jnp.mean(jnp.square(archetype_transition_scale - 1.0))
            + 0.20 * jnp.mean(jnp.square(archetype_observation_scale - 1.0))
        ),
    )


def _fit_rescue_core_jax_svi(
    *,
    run_dir: Any,
    profile_id: str,
    observation_targets: dict[str, np.ndarray],
    observation_support_bundle: dict[str, Any],
    standardized_tensor: np.ndarray,
    canonical_axis: list[str],
    candidate_profiles: list[dict[str, Any]],
    subgroup_weights: np.ndarray,
    subgroup_summary: dict[str, Any],
    cd4_overlay: np.ndarray,
    region_index: np.ndarray,
    province_axis: list[str],
    month_axis: list[str],
    kp_axis: list[str],
    age_axis: list[str],
    sex_axis: list[str],
    archetype_bundle: dict[str, Any],
    calibration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if jax is None or numpyro is None or SVI is None or AutoNormal is None or Predictive is None or Trace_ELBO is None or NumpyroAdam is None:
        raise RuntimeError("JAX/NumPyro backend is unavailable for hiv_rescue_v1 jax_svi")
    covariates_np, covariate_meta = (
        _build_mesoscopic_modifier_covariates(run_dir=run_dir, observation_targets=observation_targets)
        if profile_id == RESCUE_V2_PROFILE_ID
        else _build_modifier_covariates(
            observation_targets=observation_targets,
            standardized_tensor=standardized_tensor,
            canonical_axis=canonical_axis,
            candidate_profiles=candidate_profiles,
        )
    )
    covariates = numpy_to_jax_handoff(covariates_np).astype(jnp.float32)
    hook_mask = numpy_to_jax_handoff(np.asarray(covariate_meta["transition_hook_masks"], dtype=np.float32)).astype(jnp.float32)
    network_family_indices = covariate_meta.get("network_family_indices", {})
    mobility_pressure_np = _network_family_pressure_np(covariates_np, list(network_family_indices.get("reaction_diffusion", [])))
    service_pressure_np = _network_family_pressure_np(covariates_np, list(network_family_indices.get("percolation_fragility", [])))
    information_pressure_np = _network_family_pressure_np(covariates_np, list(network_family_indices.get("information_propagation", [])))
    operator_tensor_np, operator_names = _load_network_operator_bundle(run_dir, observation_targets["diagnosed_stock"].shape[0])
    if operator_tensor_np.shape[0] >= 3:
        mobility_operator_np, service_operator_np, information_operator_np = operator_tensor_np[:3]
    else:
        province_count = observation_targets["diagnosed_stock"].shape[0]
        identity = np.eye(province_count, dtype=np.float32)
        mobility_operator_np = service_operator_np = information_operator_np = identity
    region_index_jax = numpy_to_jax_handoff(region_index).astype(jnp.int32)
    region_count = int(region_index.max()) + 1 if region_index.size else 1
    observed_diag = numpy_to_jax_handoff(observation_targets["diagnosed_stock"]).astype(jnp.float32)
    observed_art = numpy_to_jax_handoff(observation_targets["art_stock"]).astype(jnp.float32)
    observed_sup = numpy_to_jax_handoff(observation_targets["documented_suppression"]).astype(jnp.float32)
    observed_test = numpy_to_jax_handoff(observation_targets["testing_coverage"]).astype(jnp.float32)
    support_targets = dict(observation_support_bundle.get("targets", {}) or {})
    observed_diag_support = numpy_to_jax_handoff(
        np.asarray(support_targets.get("diagnosed_stock", {}).get("support_strength", np.ones_like(observation_targets["diagnosed_stock"], dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    observed_art_support = numpy_to_jax_handoff(
        np.asarray(support_targets.get("art_stock", {}).get("support_strength", np.ones_like(observation_targets["art_stock"], dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    observed_sup_support = numpy_to_jax_handoff(
        np.asarray(support_targets.get("documented_suppression", {}).get("support_strength", np.ones_like(observation_targets["documented_suppression"], dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    observed_test_support = numpy_to_jax_handoff(
        np.asarray(support_targets.get("testing_coverage", {}).get("support_strength", np.ones_like(observation_targets["testing_coverage"], dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    init_prior = _state_initialization_prior(observation_targets, support_bundle=observation_support_bundle)
    init_prior_mean_np = np.asarray(init_prior["mean"][:, 0, :], dtype=np.float32)
    init_base = numpy_to_jax_handoff(_logit_np(np.clip(init_prior_mean_np.mean(axis=0), 1e-5, 1.0 - 1e-5))).astype(jnp.float32)
    init_province_base = numpy_to_jax_handoff(
        (_logit_np(np.clip(init_prior_mean_np, 1e-5, 1.0 - 1e-5)) - _logit_np(np.clip(init_prior_mean_np.mean(axis=0, keepdims=True), 1e-5, 1.0 - 1e-5))).astype(np.float32)
    ).astype(jnp.float32)
    init_prior_strength = numpy_to_jax_handoff(np.asarray(init_prior["concentration"][:, 0], dtype=np.float32)).astype(jnp.float32)
    mobility_operator = numpy_to_jax_handoff(mobility_operator_np).astype(jnp.float32)
    service_operator = numpy_to_jax_handoff(service_operator_np).astype(jnp.float32)
    information_operator = numpy_to_jax_handoff(information_operator_np).astype(jnp.float32)
    mobility_pressure = numpy_to_jax_handoff(mobility_pressure_np).astype(jnp.float32)
    service_pressure = numpy_to_jax_handoff(service_pressure_np).astype(jnp.float32)
    information_pressure = numpy_to_jax_handoff(information_pressure_np).astype(jnp.float32)
    archetype_transition_prior = numpy_to_jax_handoff(
        np.asarray(archetype_bundle.get("transition_prior_shift", np.zeros((len(province_axis), len(TRANSITION_NAMES)), dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    archetype_reporting_weight = numpy_to_jax_handoff(
        np.asarray(archetype_bundle.get("observation_weight", np.ones((len(province_axis),), dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    archetype_vl_prior = numpy_to_jax_handoff(
        np.asarray(archetype_bundle.get("vl_testing_prior_shift", np.zeros((len(province_axis),), dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    archetype_documentation_prior = numpy_to_jax_handoff(
        np.asarray(archetype_bundle.get("documentation_prior_shift", np.zeros((len(province_axis),), dtype=np.float32)), dtype=np.float32)
    ).astype(jnp.float32)
    subgroup_feature_matrix_np, subgroup_feature_names = _subgroup_prior_feature_matrix(subgroup_summary, archetype_bundle, province_axis)
    base_kp_np = np.asarray(subgroup_weights.sum(axis=(2, 3)), dtype=np.float32)
    base_age_np = np.asarray(subgroup_weights.sum(axis=(1, 3)), dtype=np.float32)
    base_sex_np = np.asarray(subgroup_weights.sum(axis=(1, 2)), dtype=np.float32)
    cd4_low_score = cd4_overlay[:, :, :, :, 0, :] + 0.5 * cd4_overlay[:, :, :, :, 1, :]
    cd4_high_score = cd4_overlay[:, :, :, :, 3, :]
    subgroup_feature_matrix = numpy_to_jax_handoff(subgroup_feature_matrix_np).astype(jnp.float32)
    base_kp_log = numpy_to_jax_handoff(np.log(np.clip(base_kp_np, 1e-5, None))).astype(jnp.float32)
    base_age_log = numpy_to_jax_handoff(np.log(np.clip(base_age_np, 1e-5, None))).astype(jnp.float32)
    base_sex_log = numpy_to_jax_handoff(np.log(np.clip(base_sex_np, 1e-5, None))).astype(jnp.float32)
    cd4_low_base = numpy_to_jax_handoff(cd4_low_score.astype(np.float32)).astype(jnp.float32)
    cd4_high_base = numpy_to_jax_handoff(cd4_high_score.astype(np.float32)).astype(jnp.float32)
    anchor_curves = _official_anchor_curves(month_axis)
    harp_curves = _harp_program_curves(month_axis)
    linkage_curves = _linkage_anchor_curves(month_axis=month_axis, observation_targets=observation_targets, province_axis=province_axis)
    suppression_curves = _suppression_anchor_curves(month_axis=month_axis, observation_targets=observation_targets, province_axis=province_axis)
    national_anchor_diag = numpy_to_jax_handoff(anchor_curves["diagnosed_stock"]).astype(jnp.float32)
    national_anchor_art = numpy_to_jax_handoff(anchor_curves["art_stock"]).astype(jnp.float32)
    national_anchor_sup = numpy_to_jax_handoff(anchor_curves["documented_suppression"]).astype(jnp.float32)
    national_anchor_third = numpy_to_jax_handoff(anchor_curves["third95"]).astype(jnp.float32)
    national_anchor_weight = numpy_to_jax_handoff(anchor_curves["weight"]).astype(jnp.float32)
    vl_test_prior_logit = numpy_to_jax_handoff(_logit_np(np.clip(harp_curves["viral_load_tested_among_art"], 0.10, 0.95))).astype(jnp.float32)
    harp_diag = numpy_to_jax_handoff(harp_curves["diagnosed_stock"]).astype(jnp.float32)
    harp_art = numpy_to_jax_handoff(harp_curves["art_stock"]).astype(jnp.float32)
    harp_tested = numpy_to_jax_handoff(harp_curves["viral_load_tested_stock"]).astype(jnp.float32)
    harp_sup = numpy_to_jax_handoff(harp_curves["documented_suppression"]).astype(jnp.float32)
    harp_tested_among_art = numpy_to_jax_handoff(harp_curves["viral_load_tested_among_art"]).astype(jnp.float32)
    harp_suppressed_among_art = numpy_to_jax_handoff(harp_curves["suppressed_among_art"]).astype(jnp.float32)
    harp_weight = numpy_to_jax_handoff(harp_curves["weight"]).astype(jnp.float32)
    linkage_second95_target = numpy_to_jax_handoff(linkage_curves["second95_target"]).astype(jnp.float32)
    linkage_d_to_a_target = numpy_to_jax_handoff(linkage_curves["d_to_a_transition_target"]).astype(jnp.float32)
    linkage_weight = numpy_to_jax_handoff(linkage_curves["weight"]).astype(jnp.float32)
    suppression_overall_target = numpy_to_jax_handoff(suppression_curves["overall_suppression_target"]).astype(jnp.float32)
    suppression_among_art_target = numpy_to_jax_handoff(suppression_curves["suppressed_among_art_target"]).astype(jnp.float32)
    suppression_tested_among_art_target = numpy_to_jax_handoff(suppression_curves["tested_among_art_target"]).astype(jnp.float32)
    suppression_transition_target = numpy_to_jax_handoff(suppression_curves["a_to_v_transition_target"]).astype(jnp.float32)
    suppression_weight = numpy_to_jax_handoff(suppression_curves["weight"]).astype(jnp.float32)
    suppressed_given_test_prior_logit = numpy_to_jax_handoff(
        _logit_np(
            np.clip(
                harp_curves["suppressed_among_art"] / np.clip(harp_curves["viral_load_tested_among_art"], 1e-4, None),
                0.50,
                0.995,
            )
        )
    ).astype(jnp.float32)
    national_mask = numpy_to_jax_handoff(_national_reference_mask(province_axis)).astype(jnp.float32)

    guide = AutoNormal(_rescue_svi_model)
    svi = SVI(_rescue_svi_model, guide, NumpyroAdam(0.04), Trace_ELBO())
    rng_key = jax.random.PRNGKey(17)
    svi_result = svi.run(
        rng_key,
        180,
        covariates=covariates,
        region_index=region_index_jax,
        region_count=region_count,
        observed_diag=observed_diag,
        observed_art=observed_art,
        observed_sup=observed_sup,
        observed_test=observed_test,
        init_base=init_base,
        init_province_base=init_province_base,
        init_prior_strength=init_prior_strength,
        hook_mask=hook_mask,
        mobility_operator=mobility_operator,
        service_operator=service_operator,
        information_operator=information_operator,
        mobility_pressure=mobility_pressure,
        service_pressure=service_pressure,
        information_pressure=information_pressure,
        national_anchor_diag=national_anchor_diag,
        national_anchor_art=national_anchor_art,
        national_anchor_sup=national_anchor_sup,
        national_anchor_third=national_anchor_third,
        national_anchor_weight=national_anchor_weight,
        vl_test_prior_logit=vl_test_prior_logit,
        harp_diag=harp_diag,
        harp_art=harp_art,
        harp_tested=harp_tested,
        harp_sup=harp_sup,
        harp_tested_among_art=harp_tested_among_art,
        harp_suppressed_among_art=harp_suppressed_among_art,
        harp_weight=harp_weight,
        linkage_second95_target=linkage_second95_target,
        linkage_d_to_a_target=linkage_d_to_a_target,
        linkage_weight=linkage_weight,
        suppression_overall_target=suppression_overall_target,
        suppression_among_art_target=suppression_among_art_target,
        suppression_tested_among_art_target=suppression_tested_among_art_target,
        suppression_transition_target=suppression_transition_target,
        suppression_weight=suppression_weight,
        suppressed_given_test_prior_logit=suppressed_given_test_prior_logit,
        province_transition_prior=archetype_transition_prior,
        province_reporting_weight=archetype_reporting_weight,
        province_vl_prior=archetype_vl_prior,
        province_documentation_prior=archetype_documentation_prior,
        subgroup_feature_matrix=subgroup_feature_matrix,
        base_kp_log=base_kp_log,
        base_age_log=base_age_log,
        base_sex_log=base_sex_log,
        cd4_low_base=cd4_low_base,
        cd4_high_base=cd4_high_base,
        observed_diag_support=observed_diag_support,
        observed_art_support=observed_art_support,
        observed_sup_support=observed_sup_support,
        observed_test_support=observed_test_support,
        national_mask=national_mask,
        progress_bar=False,
    )
    predictive = Predictive(
        _rescue_svi_model,
        guide=guide,
        params=svi_result.params,
        num_samples=48,
        return_sites=[
            "transition_probs",
            "state_path",
            "national_logit",
            "region_offset",
            "province_offset",
            "archetype_transition_scale_log",
            "archetype_process_scale_log",
            "archetype_observation_scale_log",
            "covariate_weights",
            "metapopulation_scale_logit",
            "vl_test_logit",
            "suppressed_given_test_logit",
            "vl_test_province_offset",
            "documented_province_offset",
            "kp_prior_coef",
            "age_prior_coef",
            "sex_prior_coef",
            "kp_probs_dynamic",
            "age_probs_dynamic",
            "sex_probs_dynamic",
            "cd4_age_low_offset",
            "cd4_kp_sex_high_offset",
            "cd4_low_coef",
            "cd4_high_coef",
            "latent_obs_diag",
            "latent_obs_art",
            "latent_obs_sup",
            "latent_obs_test",
        ],
    )
    samples = predictive(
        jax.random.PRNGKey(18),
        covariates=covariates,
        region_index=region_index_jax,
        region_count=region_count,
        observed_diag=observed_diag,
        observed_art=observed_art,
        observed_sup=observed_sup,
        observed_test=observed_test,
        init_base=init_base,
        init_province_base=init_province_base,
        init_prior_strength=init_prior_strength,
        hook_mask=hook_mask,
        mobility_operator=mobility_operator,
        service_operator=service_operator,
        information_operator=information_operator,
        mobility_pressure=mobility_pressure,
        service_pressure=service_pressure,
        information_pressure=information_pressure,
        national_anchor_diag=national_anchor_diag,
        national_anchor_art=national_anchor_art,
        national_anchor_sup=national_anchor_sup,
        national_anchor_third=national_anchor_third,
        national_anchor_weight=national_anchor_weight,
        vl_test_prior_logit=vl_test_prior_logit,
        harp_diag=harp_diag,
        harp_art=harp_art,
        harp_tested=harp_tested,
        harp_sup=harp_sup,
        harp_tested_among_art=harp_tested_among_art,
        harp_suppressed_among_art=harp_suppressed_among_art,
        harp_weight=harp_weight,
        linkage_second95_target=linkage_second95_target,
        linkage_d_to_a_target=linkage_d_to_a_target,
        linkage_weight=linkage_weight,
        suppression_overall_target=suppression_overall_target,
        suppression_among_art_target=suppression_among_art_target,
        suppression_tested_among_art_target=suppression_tested_among_art_target,
        suppression_transition_target=suppression_transition_target,
        suppression_weight=suppression_weight,
        suppressed_given_test_prior_logit=suppressed_given_test_prior_logit,
        province_transition_prior=archetype_transition_prior,
        province_reporting_weight=archetype_reporting_weight,
        province_vl_prior=archetype_vl_prior,
        province_documentation_prior=archetype_documentation_prior,
        subgroup_feature_matrix=subgroup_feature_matrix,
        base_kp_log=base_kp_log,
        base_age_log=base_age_log,
        base_sex_log=base_sex_log,
        cd4_low_base=cd4_low_base,
        cd4_high_base=cd4_high_base,
        observed_diag_support=observed_diag_support,
        observed_art_support=observed_art_support,
        observed_sup_support=observed_sup_support,
        observed_test_support=observed_test_support,
        national_mask=national_mask,
    )
    state_path = to_numpy(samples["state_path"]).mean(axis=0).astype(np.float32)
    transition_probs = to_numpy(samples["transition_probs"]).mean(axis=0).astype(np.float32)
    archetype_transition_scale = np.exp(to_numpy(samples["archetype_transition_scale_log"]).mean(axis=0)).astype(np.float32)
    archetype_process_scale = np.exp(to_numpy(samples["archetype_process_scale_log"]).mean(axis=0)).astype(np.float32)
    archetype_observation_scale = np.exp(to_numpy(samples["archetype_observation_scale_log"]).mean(axis=0)).astype(np.float32)
    vl_test_logit_mean = to_numpy(samples["vl_test_logit"]).mean(axis=0).astype(np.float32)
    suppressed_given_test_logit_mean = to_numpy(samples["suppressed_given_test_logit"]).mean(axis=0).astype(np.float32)
    province_vl_offset_raw = to_numpy(samples["vl_test_province_offset"]).mean(axis=0).astype(np.float32)
    province_doc_offset_raw = to_numpy(samples["documented_province_offset"]).mean(axis=0).astype(np.float32)
    kp_prior_coef_mean = to_numpy(samples["kp_prior_coef"]).mean(axis=0).astype(np.float32)
    age_prior_coef_mean = to_numpy(samples["age_prior_coef"]).mean(axis=0).astype(np.float32)
    sex_prior_coef_mean = to_numpy(samples["sex_prior_coef"]).mean(axis=0).astype(np.float32)
    kp_probs_dynamic_mean = to_numpy(samples["kp_probs_dynamic"]).mean(axis=0).astype(np.float32)
    age_probs_dynamic_mean = to_numpy(samples["age_probs_dynamic"]).mean(axis=0).astype(np.float32)
    sex_probs_dynamic_mean = to_numpy(samples["sex_probs_dynamic"]).mean(axis=0).astype(np.float32)
    cd4_age_low_offset_mean = to_numpy(samples["cd4_age_low_offset"]).mean(axis=0).astype(np.float32)
    cd4_kp_sex_high_offset_mean = to_numpy(samples["cd4_kp_sex_high_offset"]).mean(axis=0).astype(np.float32)
    cd4_low_coef_mean = to_numpy(samples["cd4_low_coef"]).mean(axis=0).astype(np.float32)
    cd4_high_coef_mean = to_numpy(samples["cd4_high_coef"]).mean(axis=0).astype(np.float32)
    latent_obs_diag = to_numpy(samples["latent_obs_diag"]).mean(axis=0).astype(np.float32)
    latent_obs_art = to_numpy(samples["latent_obs_art"]).mean(axis=0).astype(np.float32)
    latent_obs_sup = to_numpy(samples["latent_obs_sup"]).mean(axis=0).astype(np.float32)
    latent_obs_test = to_numpy(samples["latent_obs_test"]).mean(axis=0).astype(np.float32)
    vl_test_process = 1.0 / (1.0 + np.exp(-vl_test_logit_mean))
    suppressed_given_test = 1.0 / (1.0 + np.exp(-suppressed_given_test_logit_mean))
    testing_pred = np.clip(
        (state_path[..., 2] + state_path[..., 3])
        * (1.0 / (1.0 + np.exp(-(vl_test_logit_mean.reshape(1, len(month_axis)) + province_vl_offset_raw.reshape(len(province_axis), 1))))),
        0.0,
        1.0,
    ).astype(np.float32)
    documented_suppression = np.clip(
        np.minimum(
            state_path[..., 3],
            testing_pred
            * (1.0 / (1.0 + np.exp(-(suppressed_given_test_logit_mean.reshape(1, len(month_axis)) + province_doc_offset_raw.reshape(len(province_axis), 1))))),
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "state_estimates": state_path,
        "transition_probs": transition_probs[:, None, None, None, None, :, :].transpose(0, 1, 2, 3, 4, 6, 5),
        "prediction_stack": {
            "diagnosed_stock": np.clip(1.0 - state_path[..., 0], 0.0, 1.0).astype(np.float32),
            "art_stock": np.clip(state_path[..., 2] + state_path[..., 3], 0.0, 1.0).astype(np.float32),
            "documented_suppression": documented_suppression,
            "testing_coverage": testing_pred,
            "deaths": np.clip(state_path[..., 4] * 0.18, 0.0, 0.25).astype(np.float32),
        },
        "loss_breakdown": {
            "observation_fit_loss": float(svi_result.losses[-1]) if len(svi_result.losses) else 0.0,
            "lower_bound_suppression_penalty": 0.0,
            "diagnosed_optimism_penalty": 0.0,
            "hierarchy_reconciliation_penalty": 0.0,
            "stock_conservation_penalty": 0.0,
            "transition_plausibility_penalty": 0.0,
            "harp_program_penalty": 0.0,
            "linkage_penalty": 0.0,
            "suppression_penalty": 0.0,
            "regularization_penalty": 0.0,
            "total_loss": float(svi_result.losses[-1]) if len(svi_result.losses) else 0.0,
        },
        "loss_trace": [float(value) for value in svi_result.losses[-12:]],
        "parameters_summary": {
            "national_transition": to_numpy(jax.nn.sigmoid(samples["national_logit"]).mean(axis=0)).round(6).tolist(),
            "region_transition_sd": round(float(np.std(to_numpy(samples["region_offset"]))), 6),
            "province_transition_sd": round(float(np.std(to_numpy(samples["province_offset"]))), 6),
            "vl_test_process_share": np.round(vl_test_process, 6).tolist(),
            "suppressed_given_test_share": np.round(suppressed_given_test, 6).tolist(),
            "province_vl_test_offsets": np.round(province_vl_offset_raw, 6).tolist(),
            "province_documentation_offsets": np.round(province_doc_offset_raw, 6).tolist(),
            "observation_latent_summary": {
                "diagnosed_mean": round(float(np.mean(latent_obs_diag)) if latent_obs_diag.size else 0.0, 6),
                "art_mean": round(float(np.mean(latent_obs_art)) if latent_obs_art.size else 0.0, 6),
                "suppression_mean": round(float(np.mean(latent_obs_sup)) if latent_obs_sup.size else 0.0, 6),
                "testing_mean": round(float(np.mean(latent_obs_test)) if latent_obs_test.size else 0.0, 6),
            },
            "archetype_hyper_scales": {
                "transition": np.round(archetype_transition_scale, 6).tolist(),
                "process": np.round(archetype_process_scale, 6).tolist(),
                "observation": np.round(archetype_observation_scale, 6).tolist(),
            },
            "linkage_anchor_curves": {key: value.round(6).tolist() if isinstance(value, np.ndarray) else value for key, value in linkage_curves.items() if key != "month_axis"} | {"month_axis": month_axis},
            "suppression_anchor_curves": {key: value.round(6).tolist() if isinstance(value, np.ndarray) else value for key, value in suppression_curves.items() if key != "month_axis"} | {"month_axis": month_axis},
            "kp_coupling_matrix": np.eye(len(kp_axis), dtype=np.float32).round(6).tolist(),
            "determinant_covariates": covariate_meta,
            "metapopulation_engine": {
                "enabled": bool(operator_names),
                "operator_names": operator_names,
                "scale": np.round(1.0 / (1.0 + np.exp(-to_numpy(samples["metapopulation_scale_logit"]).mean(axis=0))), 6).tolist(),
            },
            "province_archetypes": {
                "rows": archetype_bundle.get("rows", []),
                "summary": archetype_bundle.get("summary", {}),
                "region_archetype_mixture": archetype_bundle.get("region_archetype_mixture", {}),
            },
            "subgroup_prior_learning_summary": {
                "feature_names": subgroup_feature_names,
                "kp_feature_coefficients": np.round(kp_prior_coef_mean, 6).tolist(),
                "age_feature_coefficients": np.round(age_prior_coef_mean, 6).tolist(),
                "sex_feature_coefficients": np.round(sex_prior_coef_mean, 6).tolist(),
                "mean_kp_distribution": np.round(kp_probs_dynamic_mean.mean(axis=0), 6).tolist(),
                "mean_age_distribution": np.round(age_probs_dynamic_mean.mean(axis=0), 6).tolist(),
                "mean_sex_distribution": np.round(sex_probs_dynamic_mean.mean(axis=0), 6).tolist(),
            },
            "cd4_prior_learning_summary": {
                "age_low_offset": np.round(cd4_age_low_offset_mean, 6).tolist(),
                "kp_sex_high_offset": np.round(cd4_kp_sex_high_offset_mean, 6).tolist(),
                "low_transition_coefficients": np.round(cd4_low_coef_mean, 6).tolist(),
                "high_transition_coefficients": np.round(cd4_high_coef_mean, 6).tolist(),
            },
            "synthetic_pretraining": {
                "mean_observation_weight": round(float(np.mean(archetype_bundle.get("observation_weight", []))) if len(archetype_bundle.get("observation_weight", [])) else 0.0, 6),
                "mean_reporting_noise": round(float(np.mean(archetype_bundle.get("reporting_noise", []))) if len(archetype_bundle.get("reporting_noise", [])) else 0.0, 6),
                "mean_pretraining_weight": round(float(np.mean(archetype_bundle.get("synthetic_pretraining_weight", []))) if len(archetype_bundle.get("synthetic_pretraining_weight", [])) else 0.0, 6),
                "library": archetype_bundle.get("synthetic_library", {}),
            },
        },
        "device": choose_jax_device(prefer_gpu=True),
        "inference_family": "jax_svi",
    }


def _fit_rescue_core_torch(
    *,
    run_dir: Any,
    profile_id: str,
    observation_targets: dict[str, np.ndarray],
    observation_support_bundle: dict[str, Any],
    observation_ladder: list[dict[str, Any]],
    standardized_tensor: np.ndarray,
    canonical_axis: list[str],
    candidate_profiles: list[dict[str, Any]],
    subgroup_weights: np.ndarray,
    subgroup_summary: dict[str, Any],
    cd4_overlay: np.ndarray,
    province_axis: list[str],
    month_axis: list[str],
    region_index: np.ndarray,
    kp_axis: list[str],
    age_axis: list[str],
    sex_axis: list[str],
    duration_catalog: list[str],
    requested_inference_family: str,
    archetype_bundle: dict[str, Any],
    calibration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if torch is None:  # pragma: no cover
        raise RuntimeError("Torch backend is unavailable for hiv_rescue_v1")

    province_count, month_count = observation_targets["diagnosed_stock"].shape
    duration_count = len(duration_catalog)
    device = _choose_rescue_device(province_count * month_count * len(kp_axis) * len(age_axis) * len(sex_axis) * len(STATE_NAMES) * duration_count)
    calibration_overrides = dict(calibration_overrides or {})
    loss_scale_cfg = dict(_phase3_prior("torch_map_loss_scales", {}) or {})
    reg_cfg = dict(loss_scale_cfg.get("regularization", {}) or {})
    transition_prior_override = np.asarray(calibration_overrides.get("transition_prior_override", TRANSITION_PRIOR), dtype=np.float32)
    if transition_prior_override.shape != TRANSITION_PRIOR.shape:
        transition_prior_override = np.asarray(TRANSITION_PRIOR, dtype=np.float32)
    transition_prior_override = np.clip(transition_prior_override, 0.01, 0.60).astype(np.float32)
    diagnosed_penalty_scale = float(calibration_overrides.get("diagnosed_penalty_scale", loss_scale_cfg.get("diagnosed_penalty_scale", 4.0)))
    official_reference_penalty_scale = float(calibration_overrides.get("official_reference_penalty_scale", loss_scale_cfg.get("official_reference_penalty_scale", 18.0)))
    national_anchor_penalty_scale = float(calibration_overrides.get("national_anchor_penalty_scale", loss_scale_cfg.get("national_anchor_penalty_scale", 12.0)))
    harp_program_penalty_scale = float(calibration_overrides.get("harp_program_penalty_scale", loss_scale_cfg.get("harp_program_penalty_scale", 24.0)))
    linkage_penalty_scale = float(calibration_overrides.get("linkage_penalty_scale", loss_scale_cfg.get("linkage_penalty_scale", 22.0)))
    suppression_penalty_scale = float(calibration_overrides.get("suppression_penalty_scale", loss_scale_cfg.get("suppression_penalty_scale", 0.0)))
    fit_steps = max(20, int(calibration_overrides.get("fit_steps", loss_scale_cfg.get("fit_steps", 160))))
    latent_cfg = dict(_phase3_prior("latent_observation", {}) or {})
    subgroup_cfg = dict(_phase3_prior("subgroup_hyperpriors", {}) or {})
    archetype_cfg = dict(_phase3_prior("archetype_hyperpriors", {}) or {})
    cd4_hyper_cfg = dict(_phase3_prior("cd4_hyperpriors", {}) or {})
    covariates, covariate_meta = (
        _build_mesoscopic_modifier_covariates(run_dir=run_dir, observation_targets=observation_targets)
        if profile_id == RESCUE_V2_PROFILE_ID
        else _build_modifier_covariates(
            observation_targets=observation_targets,
            standardized_tensor=standardized_tensor,
            canonical_axis=canonical_axis,
            candidate_profiles=candidate_profiles,
        )
    )
    network_family_indices = covariate_meta.get("network_family_indices", {})
    mobility_pressure_np = _network_family_pressure_np(covariates, list(network_family_indices.get("reaction_diffusion", [])))
    service_pressure_np = _network_family_pressure_np(covariates, list(network_family_indices.get("percolation_fragility", [])))
    information_pressure_np = _network_family_pressure_np(covariates, list(network_family_indices.get("information_propagation", [])))
    operator_tensor_np, operator_names = _load_network_operator_bundle(run_dir, province_count)
    if operator_tensor_np.shape[0] >= 3:
        mobility_operator_np, service_operator_np, information_operator_np = operator_tensor_np[:3]
    else:
        identity = np.eye(province_count, dtype=np.float32)
        mobility_operator_np = service_operator_np = information_operator_np = identity
    init_prior = _state_initialization_prior(observation_targets, support_bundle=observation_support_bundle)
    observed_states = np.asarray(init_prior["mean"], dtype=np.float32)
    initial_logit_guess = _logit_np(observed_states[:, 0, :])
    init_prior_strength_np = np.asarray(init_prior["concentration"][:, 0], dtype=np.float32)
    support_targets = dict(observation_support_bundle.get("targets", {}) or {})
    observed_diag_support_np = np.asarray(
        support_targets.get("diagnosed_stock", {}).get("support_strength", np.ones_like(observation_targets["diagnosed_stock"], dtype=np.float32)),
        dtype=np.float32,
    )
    observed_art_support_np = np.asarray(
        support_targets.get("art_stock", {}).get("support_strength", np.ones_like(observation_targets["art_stock"], dtype=np.float32)),
        dtype=np.float32,
    )
    observed_sup_support_np = np.asarray(
        support_targets.get("documented_suppression", {}).get("support_strength", np.ones_like(observation_targets["documented_suppression"], dtype=np.float32)),
        dtype=np.float32,
    )
    observed_test_support_np = np.asarray(
        support_targets.get("testing_coverage", {}).get("support_strength", np.ones_like(observation_targets["testing_coverage"], dtype=np.float32)),
        dtype=np.float32,
    )
    archetype_transition_prior_np = np.asarray(
        archetype_bundle.get("transition_prior_shift", np.zeros((province_count, len(TRANSITION_NAMES)), dtype=np.float32)),
        dtype=np.float32,
    )
    archetype_observation_weight_np = np.asarray(
        archetype_bundle.get("observation_weight", np.ones((province_count,), dtype=np.float32)),
        dtype=np.float32,
    )
    archetype_reporting_noise_np = np.asarray(
        archetype_bundle.get("reporting_noise", np.zeros((province_count,), dtype=np.float32)),
        dtype=np.float32,
    )
    archetype_vl_prior_np = np.asarray(
        archetype_bundle.get("vl_testing_prior_shift", np.zeros((province_count,), dtype=np.float32)),
        dtype=np.float32,
    )
    archetype_doc_prior_np = np.asarray(
        archetype_bundle.get("documentation_prior_shift", np.zeros((province_count,), dtype=np.float32)),
        dtype=np.float32,
    )
    synthetic_pretraining_weight_np = np.asarray(
        archetype_bundle.get("synthetic_pretraining_weight", np.zeros((province_count,), dtype=np.float32)),
        dtype=np.float32,
    )
    subgroup_feature_matrix_np, subgroup_feature_names = _subgroup_prior_feature_matrix(subgroup_summary, archetype_bundle, province_axis)
    base_kp_np = np.asarray(subgroup_weights.sum(axis=(2, 3)), dtype=np.float32)
    base_age_np = np.asarray(subgroup_weights.sum(axis=(1, 3)), dtype=np.float32)
    base_sex_np = np.asarray(subgroup_weights.sum(axis=(1, 2)), dtype=np.float32)
    cd4_low_score = cd4_overlay[:, :, :, :, 0, :] + 0.5 * cd4_overlay[:, :, :, :, 1, :]
    cd4_high_score = cd4_overlay[:, :, :, :, 3, :]

    subgroup_weights_t = to_torch_tensor(subgroup_weights, device=device, dtype=torch.float32)
    cd4_low_t = to_torch_tensor(cd4_low_score, device=device, dtype=torch.float32)
    cd4_high_t = to_torch_tensor(cd4_high_score, device=device, dtype=torch.float32)
    subgroup_feature_t = to_torch_tensor(subgroup_feature_matrix_np, device=device, dtype=torch.float32)
    base_kp_log_t = to_torch_tensor(np.log(np.clip(base_kp_np, 1e-5, None)), device=device, dtype=torch.float32)
    base_age_log_t = to_torch_tensor(np.log(np.clip(base_age_np, 1e-5, None)), device=device, dtype=torch.float32)
    base_sex_log_t = to_torch_tensor(np.log(np.clip(base_sex_np, 1e-5, None)), device=device, dtype=torch.float32)
    covariates_t = to_torch_tensor(covariates, device=device, dtype=torch.float32)
    hook_mask_t = to_torch_tensor(np.asarray(covariate_meta["transition_hook_masks"], dtype=np.float32), device=device, dtype=torch.float32)
    mobility_pressure_t = to_torch_tensor(mobility_pressure_np, device=device, dtype=torch.float32)
    service_pressure_t = to_torch_tensor(service_pressure_np, device=device, dtype=torch.float32)
    information_pressure_t = to_torch_tensor(information_pressure_np, device=device, dtype=torch.float32)
    mobility_operator_t = to_torch_tensor(mobility_operator_np, device=device, dtype=torch.float32)
    service_operator_t = to_torch_tensor(service_operator_np, device=device, dtype=torch.float32)
    information_operator_t = to_torch_tensor(information_operator_np, device=device, dtype=torch.float32)
    region_index_t = to_torch_tensor(region_index, device=device, dtype=torch.long)
    age_progress_t = to_torch_tensor(AGE_PROGRESS_RATES, device=device, dtype=torch.float32)
    duration_template_t = to_torch_tensor(DEFAULT_DURATION_TEMPLATE, device=device, dtype=torch.float32)
    archetype_transition_prior_t = to_torch_tensor(archetype_transition_prior_np, device=device, dtype=torch.float32)
    archetype_observation_weight_t = to_torch_tensor(archetype_observation_weight_np, device=device, dtype=torch.float32)
    archetype_reporting_noise_t = to_torch_tensor(archetype_reporting_noise_np, device=device, dtype=torch.float32)
    archetype_vl_prior_t = to_torch_tensor(archetype_vl_prior_np, device=device, dtype=torch.float32)
    archetype_doc_prior_t = to_torch_tensor(archetype_doc_prior_np, device=device, dtype=torch.float32)
    synthetic_pretraining_weight_t = to_torch_tensor(synthetic_pretraining_weight_np, device=device, dtype=torch.float32)
    observed_diag_support_t = to_torch_tensor(observed_diag_support_np, device=device, dtype=torch.float32)
    observed_art_support_t = to_torch_tensor(observed_art_support_np, device=device, dtype=torch.float32)
    observed_sup_support_t = to_torch_tensor(observed_sup_support_np, device=device, dtype=torch.float32)
    observed_test_support_t = to_torch_tensor(observed_test_support_np, device=device, dtype=torch.float32)
    init_prior_strength_t = to_torch_tensor(init_prior_strength_np, device=device, dtype=torch.float32)
    mobility_state_weights_t = to_torch_tensor(np.asarray([0.45, 0.35, 0.10, 0.00, 0.10], dtype=np.float32), device=device, dtype=torch.float32)
    service_state_weights_t = to_torch_tensor(np.asarray([0.05, 0.18, 0.36, 0.21, 0.20], dtype=np.float32), device=device, dtype=torch.float32)
    information_state_weights_t = to_torch_tensor(np.asarray([0.40, 0.34, 0.12, 0.04, 0.10], dtype=np.float32), device=device, dtype=torch.float32)

    region_count = int(region_index.max()) + 1 if region_index.size else 1
    transition_prior_logit = _logit_np(transition_prior_override)
    observation_weight_vec = to_torch_tensor(
        np.asarray(
            [
                OBSERVATION_CLASS_WEIGHT[next(item["observation_class"] for item in observation_ladder if item["target_name"] == target)]
                for target in OBSERVATION_ORDER
            ],
            dtype=np.float32,
        ),
        device=device,
        dtype=torch.float32,
    )

    national_transition = torch.nn.Parameter(to_torch_tensor(transition_prior_logit, device=device, dtype=torch.float32))
    region_transition = torch.nn.Parameter(torch.zeros((region_count, len(TRANSITION_NAMES)), device=device, dtype=torch.float32))
    province_transition = torch.nn.Parameter(archetype_transition_prior_t.clone())
    kp_transition = torch.nn.Parameter(torch.zeros((len(kp_axis), len(TRANSITION_NAMES)), device=device, dtype=torch.float32))
    age_transition = torch.nn.Parameter(torch.zeros((len(age_axis), len(TRANSITION_NAMES)), device=device, dtype=torch.float32))
    sex_transition = torch.nn.Parameter(torch.zeros((len(sex_axis), len(TRANSITION_NAMES)), device=device, dtype=torch.float32))
    kp_prior_coef = torch.nn.Parameter(torch.zeros((subgroup_feature_matrix_np.shape[-1], len(kp_axis)), device=device, dtype=torch.float32))
    age_prior_coef = torch.nn.Parameter(torch.zeros((subgroup_feature_matrix_np.shape[-1], len(age_axis)), device=device, dtype=torch.float32))
    sex_prior_coef = torch.nn.Parameter(torch.zeros((subgroup_feature_matrix_np.shape[-1], len(sex_axis)), device=device, dtype=torch.float32))
    duration_transition = torch.nn.Parameter(torch.linspace(-0.18, 0.18, steps=duration_count, device=device).unsqueeze(-1).repeat(1, len(TRANSITION_NAMES)) * 0.5)
    kp_coupling_logits = torch.nn.Parameter(torch.zeros((len(kp_axis), len(kp_axis)), device=device, dtype=torch.float32))
    metapopulation_logit = torch.nn.Parameter(torch.tensor([-2.2, -2.35, -2.3], device=device, dtype=torch.float32))
    cd4_low_coef = torch.nn.Parameter(torch.tensor([0.16, 0.12, -0.08, 0.10, -0.05], device=device, dtype=torch.float32))
    cd4_high_coef = torch.nn.Parameter(torch.tensor([-0.12, -0.06, 0.14, -0.10, 0.05], device=device, dtype=torch.float32))
    cd4_age_low_offset = torch.nn.Parameter(torch.zeros((len(age_axis),), device=device, dtype=torch.float32))
    cd4_kp_sex_high_offset = torch.nn.Parameter(torch.zeros((len(kp_axis), len(sex_axis)), device=device, dtype=torch.float32))
    covariate_coef = torch.nn.Parameter(torch.zeros((covariates.shape[-1], len(TRANSITION_NAMES)), device=device, dtype=torch.float32))
    vl_test_process_logit = torch.nn.Parameter(
        to_torch_tensor(_logit_np(np.clip(_harp_program_curves(month_axis)["viral_load_tested_among_art"], 0.10, 0.95)), device=device, dtype=torch.float32)
    )
    suppressed_given_test_logit = torch.nn.Parameter(
        to_torch_tensor(
            _logit_np(
                np.clip(
                    _harp_program_curves(month_axis)["suppressed_among_art"]
                    / np.clip(_harp_program_curves(month_axis)["viral_load_tested_among_art"], 1e-4, None),
                    0.50,
                    0.995,
                )
            ),
            device=device,
            dtype=torch.float32,
        )
    )
    province_vl_test_offset = torch.nn.Parameter(archetype_vl_prior_t.clone())
    province_documented_offset = torch.nn.Parameter(archetype_doc_prior_t.clone())
    archetype_transition_scale = torch.nn.Parameter(torch.zeros((len(TRANSITION_NAMES),), device=device, dtype=torch.float32))
    archetype_process_scale = torch.nn.Parameter(torch.zeros((2,), device=device, dtype=torch.float32))
    archetype_observation_scale = torch.nn.Parameter(torch.zeros((3,), device=device, dtype=torch.float32))
    obs_province_offset = torch.nn.Parameter(torch.zeros((province_count, 4), device=device, dtype=torch.float32))
    obs_month_offset = torch.nn.Parameter(torch.zeros((month_count, 4), device=device, dtype=torch.float32))

    national_initial = torch.nn.Parameter(to_torch_tensor(initial_logit_guess.mean(axis=0), device=device, dtype=torch.float32))
    region_initial = torch.nn.Parameter(torch.zeros((region_count, len(STATE_NAMES)), device=device, dtype=torch.float32))
    province_initial = torch.nn.Parameter(to_torch_tensor(initial_logit_guess - initial_logit_guess.mean(axis=0, keepdims=True), device=device, dtype=torch.float32))
    kp_initial = torch.nn.Parameter(torch.zeros((len(kp_axis), len(STATE_NAMES)), device=device, dtype=torch.float32))
    age_initial = torch.nn.Parameter(torch.zeros((len(age_axis), len(STATE_NAMES)), device=device, dtype=torch.float32))
    sex_initial = torch.nn.Parameter(torch.zeros((len(sex_axis), len(STATE_NAMES)), device=device, dtype=torch.float32))

    parameters = [
        national_transition,
        region_transition,
            province_transition,
            kp_transition,
            age_transition,
            sex_transition,
            kp_prior_coef,
            age_prior_coef,
            sex_prior_coef,
            duration_transition,
            kp_coupling_logits,
            metapopulation_logit,
            cd4_low_coef,
            cd4_high_coef,
            cd4_age_low_offset,
            cd4_kp_sex_high_offset,
            covariate_coef,
            vl_test_process_logit,
            suppressed_given_test_logit,
        province_vl_test_offset,
        province_documented_offset,
        archetype_transition_scale,
        archetype_process_scale,
        archetype_observation_scale,
        obs_province_offset,
        obs_month_offset,
        national_initial,
        region_initial,
        province_initial,
        kp_initial,
        age_initial,
        sex_initial,
    ]
    optimizer = torch.optim.Adam(parameters, lr=float(loss_scale_cfg.get("optimizer_lr", 0.045)))
    loss_trace: list[float] = []

    observed_diag = to_torch_tensor(observation_targets["diagnosed_stock"], device=device, dtype=torch.float32)
    observed_art = to_torch_tensor(observation_targets["art_stock"], device=device, dtype=torch.float32)
    observed_sup = to_torch_tensor(observation_targets["documented_suppression"], device=device, dtype=torch.float32)
    observed_test = to_torch_tensor(observation_targets["testing_coverage"], device=device, dtype=torch.float32)
    observed_deaths = to_torch_tensor(observation_targets["deaths"], device=device, dtype=torch.float32)

    def _forward() -> tuple[Any, Any, Any, dict[str, float], dict[str, Any], dict[str, Any], dict[str, Any]]:
        transition_scale = torch.exp(0.35 * torch.tanh(archetype_transition_scale))
        process_scale = torch.exp(0.35 * torch.tanh(archetype_process_scale))
        observation_scale = torch.exp(0.25 * torch.tanh(archetype_observation_scale))
        scaled_archetype_transition_prior = archetype_transition_prior_t * transition_scale.view(1, len(TRANSITION_NAMES))
        scaled_archetype_vl_prior = archetype_vl_prior_t * process_scale[0]
        scaled_archetype_doc_prior = archetype_doc_prior_t * process_scale[1]
        scaled_observation_weight = torch.clamp(archetype_observation_weight_t * observation_scale[0], min=0.15, max=1.25)
        kp_probs_dynamic = torch.softmax(base_kp_log_t + subgroup_feature_t @ kp_prior_coef, dim=-1)
        age_probs_dynamic = torch.softmax(base_age_log_t + subgroup_feature_t @ age_prior_coef, dim=-1)
        sex_probs_dynamic = torch.softmax(base_sex_log_t + subgroup_feature_t @ sex_prior_coef, dim=-1)
        subgroup_weights_dynamic = kp_probs_dynamic[:, :, None, None] * age_probs_dynamic[:, None, :, None] * sex_probs_dynamic[:, None, None, :]
        subgroup_weights_dynamic = subgroup_weights_dynamic / torch.clamp(subgroup_weights_dynamic.sum(dim=(1, 2, 3), keepdim=True), min=1e-6)
        cd4_low_dynamic = torch.clamp(cd4_low_t + cd4_age_low_offset.view(1, 1, len(age_axis), 1, 1), min=0.0, max=1.5)
        cd4_high_dynamic = torch.clamp(cd4_high_t + cd4_kp_sex_high_offset.view(1, len(kp_axis), 1, len(sex_axis), 1), min=0.0, max=1.5)
        init_logits = (
            national_initial.view(1, 1, 1, 1, len(STATE_NAMES))
            + region_initial[region_index_t].view(province_count, 1, 1, 1, len(STATE_NAMES))
            + province_initial.view(province_count, 1, 1, 1, len(STATE_NAMES))
            + kp_initial.view(1, len(kp_axis), 1, 1, len(STATE_NAMES))
            + age_initial.view(1, 1, len(age_axis), 1, len(STATE_NAMES))
            + sex_initial.view(1, 1, 1, len(sex_axis), len(STATE_NAMES))
        )
        init_state = torch.softmax(init_logits, dim=-1)
        initial_latent = init_state.unsqueeze(-1) * duration_template_t.transpose(0, 1).reshape(1, 1, 1, 1, len(STATE_NAMES), duration_count)

        transition_logits = (
            national_transition.view(1, 1, 1, 1, 1, 1, len(TRANSITION_NAMES))
            + region_transition[region_index_t].view(province_count, 1, 1, 1, 1, 1, len(TRANSITION_NAMES))
            + province_transition.view(province_count, 1, 1, 1, 1, 1, len(TRANSITION_NAMES))
            + kp_transition.view(1, len(kp_axis), 1, 1, 1, 1, len(TRANSITION_NAMES))
            + age_transition.view(1, 1, len(age_axis), 1, 1, 1, len(TRANSITION_NAMES))
            + sex_transition.view(1, 1, 1, len(sex_axis), 1, 1, len(TRANSITION_NAMES))
            + duration_transition.view(1, 1, 1, 1, duration_count, 1, len(TRANSITION_NAMES))
        )
        transition_logits = transition_logits + cd4_low_dynamic.unsqueeze(-2).unsqueeze(-1) * cd4_low_coef.view(1, 1, 1, 1, 1, 1, len(TRANSITION_NAMES))
        transition_logits = transition_logits + cd4_high_dynamic.unsqueeze(-2).unsqueeze(-1) * cd4_high_coef.view(1, 1, 1, 1, 1, 1, len(TRANSITION_NAMES))
        covariate_effect = torch.einsum("ptc,cr->ptr", covariates_t, covariate_coef * hook_mask_t)
        transition_logits = transition_logits + covariate_effect.view(province_count, 1, 1, 1, 1, month_count, len(TRANSITION_NAMES))
        transition_probs = torch.sigmoid(transition_logits).permute(0, 1, 2, 3, 4, 6, 5)
        coupling_bias = torch.eye(len(kp_axis), device=device, dtype=torch.float32) * 3.5
        kp_mixing_matrix = torch.softmax(kp_coupling_logits + coupling_bias, dim=-1)
        coupling_indices = [
            idx
            for idx, name in enumerate(covariate_meta.get("covariate_names", []))
            if name in set(covariate_meta.get("coupling_covariate_names", []))
        ]
        if coupling_indices:
            coupling_pressure = 0.18 * torch.sigmoid(torch.mean(covariates_t[..., coupling_indices], dim=-1))
        else:
            coupling_pressure = torch.zeros((province_count, month_count), device=device, dtype=torch.float32)

        latent_steps = [initial_latent]
        for month_idx in range(month_count - 1):
            stepped = _torch_transition_step(latent_steps[-1], transition_probs[..., month_idx], age_progress_t)
            coupled = _apply_kp_coupling_torch(stepped, kp_mixing_matrix)
            alpha = coupling_pressure[:, month_idx].view(province_count, 1, 1, 1, 1, 1)
            blended = (1.0 - alpha) * stepped + alpha * coupled
            mobility_alpha = torch.sigmoid(metapopulation_logit[0]) * mobility_pressure_t[:, month_idx].view(province_count, 1, 1, 1, 1, 1)
            service_alpha = torch.sigmoid(metapopulation_logit[1]) * service_pressure_t[:, month_idx].view(province_count, 1, 1, 1, 1, 1)
            information_alpha = torch.sigmoid(metapopulation_logit[2]) * information_pressure_t[:, month_idx].view(province_count, 1, 1, 1, 1, 1)
            blended = _blend_statewise_torch(blended, _province_operator_torch(blended, mobility_operator_t), mobility_alpha, mobility_state_weights_t)
            blended = _blend_statewise_torch(blended, _province_operator_torch(blended, service_operator_t), service_alpha, service_state_weights_t)
            blended = _blend_statewise_torch(blended, _province_operator_torch(blended, information_operator_t), information_alpha, information_state_weights_t)
            latent_steps.append(blended)
        latent = torch.stack(latent_steps, dim=-1)

        aggregate = (latent * subgroup_weights_dynamic.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=(1, 2, 3, 5)).permute(0, 2, 1)
        aggregate = aggregate / torch.clamp(aggregate.sum(dim=-1, keepdim=True), min=1e-6)
        diagnosed_pred = torch.clamp(1.0 - aggregate[..., 0], 0.0, 1.0)
        art_pred = torch.clamp(aggregate[..., 2] + aggregate[..., 3], 0.0, 1.0)
        vl_test_share = torch.sigmoid(vl_test_process_logit.view(1, month_count) + province_vl_test_offset.view(province_count, 1))
        testing_pred = torch.clamp(art_pred * vl_test_share, 0.0, 1.0)
        documented_given_test_share = torch.sigmoid(
            suppressed_given_test_logit.view(1, month_count) + province_documented_offset.view(province_count, 1)
        )
        suppression_pred = torch.clamp(torch.minimum(aggregate[..., 3], testing_pred * documented_given_test_share), 0.0, 1.0)
        deaths_pred = torch.clamp(aggregate[..., 4] * 0.18, 0.0, 0.25)
        prediction_stack = torch.stack([diagnosed_pred, art_pred, suppression_pred, testing_pred, deaths_pred], dim=-1)
        observation_stack = torch.stack([observed_diag, observed_art, observed_sup, observed_test, observed_deaths], dim=-1)
        observed_support_stack = torch.stack(
            [
                observed_diag_support_t,
                observed_art_support_t,
                observed_sup_support_t,
                observed_test_support_t,
                torch.ones_like(observed_deaths),
            ],
            dim=-1,
        )
        obs_offset_stack = obs_province_offset.unsqueeze(1) + obs_month_offset.unsqueeze(0)
        latent_diag = torch.logit(torch.clamp(observed_diag, min=1e-5, max=1.0 - 1e-5))
        latent_art = torch.logit(torch.clamp(observed_art, min=1e-5, max=1.0 - 1e-5))
        latent_sup = torch.logit(torch.clamp(observed_sup, min=1e-5, max=1.0 - 1e-5))
        latent_test = torch.logit(torch.clamp(observed_test, min=1e-5, max=1.0 - 1e-5))
        latent_diag = torch.sigmoid(latent_diag + (1.0 - observed_diag_support_t) * obs_offset_stack[..., 0])
        latent_art = torch.sigmoid(latent_art + (1.0 - observed_art_support_t) * obs_offset_stack[..., 1])
        latent_sup = torch.sigmoid(latent_sup + (1.0 - observed_sup_support_t) * obs_offset_stack[..., 2])
        latent_test = torch.sigmoid(latent_test + (1.0 - observed_test_support_t) * obs_offset_stack[..., 3])
        latent_art = torch.minimum(latent_art, latent_diag)
        latent_test = torch.minimum(latent_test, latent_art)
        latent_sup = torch.minimum(latent_sup, latent_test)
        latent_observation_stack = torch.stack([latent_diag, latent_art, latent_sup, latent_test, observed_deaths], dim=-1)

        obs_loss = torch.mean(
            ((prediction_stack - latent_observation_stack) ** 2)
            * observation_weight_vec.view(1, 1, -1)
            * scaled_observation_weight.view(province_count, 1, 1)
        )
        observation_anchor_penalty = float(latent_cfg.get("anchor_reversion_scale", 9.0)) * torch.mean(
            ((latent_observation_stack - observation_stack) ** 2)
            * observed_support_stack
            * observation_weight_vec.view(1, 1, -1)
        )
        lower_bound_penalty = torch.mean(torch.relu(suppression_pred - torch.maximum(latent_sup + 0.03, latent_test * 0.95)) ** 2)
        diagnosed_penalty = torch.mean(torch.relu(diagnosed_pred - latent_diag - 0.02) ** 2)
        official_reference_penalty, official_reference_detail = _official_reference_penalty_torch(
            diagnosed_pred=diagnosed_pred,
            art_pred=art_pred,
            suppression_pred=suppression_pred,
            province_axis=province_axis,
            month_axis=month_axis,
            device=device,
        )
        national_anchor_penalty, national_anchor_detail = _national_anchor_penalty_torch(
            diagnosed_pred=diagnosed_pred,
            art_pred=art_pred,
            suppression_pred=suppression_pred,
            province_axis=province_axis,
            month_axis=month_axis,
            device=device,
        )
        harp_program_penalty, harp_program_detail = _harp_program_penalty_torch(
            diagnosed_pred=diagnosed_pred,
            art_pred=art_pred,
            suppression_pred=suppression_pred,
            testing_pred=testing_pred,
            province_axis=province_axis,
            month_axis=month_axis,
            device=device,
        )
        linkage_penalty, linkage_detail, linkage_curves = _linkage_penalty_torch(
            latent=latent,
            transition_probs=transition_probs,
            subgroup_weights=subgroup_weights_dynamic,
            observation_targets=observation_targets,
            province_axis=province_axis,
            month_axis=month_axis,
            device=device,
        )
        suppression_penalty, suppression_detail, suppression_curves = _suppression_penalty_torch(
            latent=latent,
            transition_probs=transition_probs,
            subgroup_weights=subgroup_weights_dynamic,
            testing_pred=testing_pred,
            observation_targets=observation_targets,
            province_axis=province_axis,
            month_axis=month_axis,
            device=device,
        )

        region_pred = torch.zeros((region_count, month_count, len(STATE_NAMES)), device=device, dtype=torch.float32)
        national_pred = aggregate.mean(dim=0)
        for region_id in range(region_count):
            members = aggregate[region_index_t == region_id]
            if int(members.shape[0]) > 0:
                region_pred[region_id] = members.mean(dim=0)
        hierarchy_penalty = torch.mean((region_pred.mean(dim=0) - national_pred) ** 2)
        mass = latent.sum(dim=(-2, -1))
        stock_penalty = torch.mean((mass - 1.0) ** 2)
        duration_smooth = torch.mean((duration_transition[1:] - duration_transition[:-1]) ** 2)
        transition_reg = float(reg_cfg.get("region_transition", 0.15)) * torch.mean(region_transition**2) + float(reg_cfg.get("province_transition_prior", 0.16)) * torch.mean((province_transition - scaled_archetype_transition_prior) ** 2)
        init_prior_target = to_torch_tensor(initial_logit_guess - initial_logit_guess.mean(axis=0, keepdims=True), device=device, dtype=torch.float32)
        init_reg = (
            0.10 * torch.mean(region_initial**2)
            + 0.15 * torch.mean(((province_initial - init_prior_target) ** 2) * init_prior_strength_t.view(province_count, 1))
        )
        subgroup_feature_sigma = max(float(subgroup_cfg.get("feature_coef_sigma", 0.18)), 1e-3)
        subgroup_reg = (
            0.08 * (torch.mean(kp_transition**2) + torch.mean(age_transition**2) + torch.mean(sex_transition**2))
            + 0.5
            * (
                torch.mean((kp_prior_coef / subgroup_feature_sigma) ** 2)
                + torch.mean((age_prior_coef / subgroup_feature_sigma) ** 2)
                + torch.mean((sex_prior_coef / subgroup_feature_sigma) ** 2)
            )
        )
        coupling_reg = 0.25 * torch.mean((kp_mixing_matrix - torch.eye(len(kp_axis), device=device, dtype=torch.float32)) ** 2)
        metapop_reg = 0.08 * torch.mean(torch.sigmoid(metapopulation_logit) ** 2)
        cd4_age_sigma = max(float(cd4_hyper_cfg.get("age_low_sigma", 0.08)), 1e-3)
        cd4_high_sigma = max(float(cd4_hyper_cfg.get("kp_sex_high_sigma", 0.06)), 1e-3)
        cd4_reg = (
            0.04 * (torch.mean(cd4_low_coef**2) + torch.mean(cd4_high_coef**2))
            + 0.5
            * (
                torch.mean((cd4_age_low_offset / cd4_age_sigma) ** 2)
                + torch.mean((cd4_kp_sex_high_offset / cd4_high_sigma) ** 2)
            )
        )
        covariate_reg = 0.10 * torch.mean(covariate_coef**2)
        vl_test_reg = 0.08 * torch.mean((torch.sigmoid(vl_test_process_logit[1:]) - torch.sigmoid(vl_test_process_logit[:-1])) ** 2) if month_count > 1 else torch.tensor(0.0, device=device, dtype=torch.float32)
        documented_process_reg = 0.06 * torch.mean((torch.sigmoid(suppressed_given_test_logit[1:]) - torch.sigmoid(suppressed_given_test_logit[:-1])) ** 2) if month_count > 1 else torch.tensor(0.0, device=device, dtype=torch.float32)
        province_process_reg = float(reg_cfg.get("province_process_prior", 0.08)) * torch.mean((province_vl_test_offset - scaled_archetype_vl_prior) ** 2) + float(reg_cfg.get("province_process_prior", 0.08)) * torch.mean((province_documented_offset - scaled_archetype_doc_prior) ** 2)
        reporting_reg = (
            0.04 * torch.mean((scaled_observation_weight - (1.0 - 0.55 * archetype_reporting_noise_t * observation_scale[1])) ** 2)
            + 0.03 * torch.mean(synthetic_pretraining_weight_t**2)
            + float(latent_cfg.get("offset_regularization", 0.05)) * (torch.mean(obs_province_offset**2) + torch.mean(obs_month_offset**2))
            + 0.04 * torch.mean((transition_scale - 1.0) ** 2)
            + 0.04 * torch.mean((process_scale - 1.0) ** 2)
            + 0.03 * torch.mean((observation_scale - 1.0) ** 2)
        )
        plausibility_penalty = float(reg_cfg.get("plausibility", 0.05)) * torch.mean((torch.sigmoid(national_transition) - to_torch_tensor(transition_prior_override, device=device, dtype=torch.float32)) ** 2)
        regularization = transition_reg + init_reg + subgroup_reg + coupling_reg + metapop_reg + cd4_reg + covariate_reg + float(loss_scale_cfg.get("duration_smooth_multiplier", 0.25)) * duration_smooth + plausibility_penalty + vl_test_reg + documented_process_reg + province_process_reg + reporting_reg
        total_loss = (
            obs_loss
            + observation_anchor_penalty
            + float(loss_scale_cfg.get("lower_bound_penalty", 5.0)) * lower_bound_penalty
            + diagnosed_penalty_scale * diagnosed_penalty
            + float(loss_scale_cfg.get("hierarchy_penalty", 2.0)) * hierarchy_penalty
            + float(loss_scale_cfg.get("stock_penalty", 8.0)) * stock_penalty
            + official_reference_penalty_scale * official_reference_penalty
            + national_anchor_penalty_scale * national_anchor_penalty
            + harp_program_penalty_scale * harp_program_penalty
            + linkage_penalty_scale * linkage_penalty
            + suppression_penalty_scale * suppression_penalty
            + regularization
        )
        breakdown = {
            "observation_fit_loss": float(obs_loss.detach().cpu()),
            "observation_anchor_penalty": float(observation_anchor_penalty.detach().cpu()),
            "lower_bound_suppression_penalty": float(lower_bound_penalty.detach().cpu()),
            "diagnosed_optimism_penalty": float(diagnosed_penalty.detach().cpu()),
            "official_reference_penalty": float(official_reference_penalty.detach().cpu()),
            "national_anchor_penalty": float(national_anchor_penalty.detach().cpu()),
            "harp_program_penalty": float(harp_program_penalty.detach().cpu()),
            "linkage_penalty": float(linkage_penalty.detach().cpu()),
            "suppression_penalty": float(suppression_penalty.detach().cpu()),
            "hierarchy_reconciliation_penalty": float(hierarchy_penalty.detach().cpu()),
            "stock_conservation_penalty": float(stock_penalty.detach().cpu()),
            "transition_plausibility_penalty": float(plausibility_penalty.detach().cpu()),
            "regularization_penalty": float(regularization.detach().cpu()),
            "subgroup_prior_penalty": float(subgroup_reg.detach().cpu()),
            "coupling_penalty": float(coupling_reg.detach().cpu()),
            "metapopulation_penalty": float(metapop_reg.detach().cpu()),
            "cd4_prior_penalty": float(cd4_reg.detach().cpu()),
            "vl_test_process_penalty": float(vl_test_reg.detach().cpu()),
            "documented_suppression_process_penalty": float(documented_process_reg.detach().cpu()),
            "province_process_prior_penalty": float(province_process_reg.detach().cpu()),
            "synthetic_pretraining_penalty": float(reporting_reg.detach().cpu()),
        }
        breakdown.update(official_reference_detail)
        breakdown.update(national_anchor_detail)
        breakdown.update(harp_program_detail)
        breakdown.update(linkage_detail)
        breakdown.update(suppression_detail)
        aux_summary = {
            "observation_latent_summary": {
                "diagnosed_mean": round(float(latent_diag.detach().mean().cpu()) if latent_diag.numel() else 0.0, 6),
                "art_mean": round(float(latent_art.detach().mean().cpu()) if latent_art.numel() else 0.0, 6),
                "suppression_mean": round(float(latent_sup.detach().mean().cpu()) if latent_sup.numel() else 0.0, 6),
                "testing_mean": round(float(latent_test.detach().mean().cpu()) if latent_test.numel() else 0.0, 6),
            },
            "archetype_hyper_scales": {
                "transition": to_numpy(transition_scale).round(6).tolist(),
                "process": to_numpy(process_scale).round(6).tolist(),
                "observation": to_numpy(observation_scale).round(6).tolist(),
            },
        }
        return total_loss, aggregate, transition_probs, breakdown, linkage_curves, suppression_curves, aux_summary

    for _step in range(fit_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss, _, _, _, _, _, _ = _forward()
        total_loss.backward()
        optimizer.step()
        loss_trace.append(float(total_loss.detach().cpu()))

    with torch.no_grad():
        total_loss, aggregate, transition_probs, breakdown, linkage_curves, suppression_curves, aux_summary = _forward()
        aggregate_np = to_numpy(aggregate).astype(np.float32)
        transition_probs_np = to_numpy(transition_probs).astype(np.float32)
        kp_probs_summary = torch.softmax(base_kp_log_t + subgroup_feature_t @ kp_prior_coef, dim=-1)
        age_probs_summary = torch.softmax(base_age_log_t + subgroup_feature_t @ age_prior_coef, dim=-1)
        sex_probs_summary = torch.softmax(base_sex_log_t + subgroup_feature_t @ sex_prior_coef, dim=-1)
        parameters_summary = {
            "national_transition": to_numpy(torch.sigmoid(national_transition)).round(6).tolist(),
            "region_transition_sd": round(float(region_transition.detach().std().cpu()), 6),
            "province_transition_sd": round(float(province_transition.detach().std().cpu()), 6),
            "initial_state_national": to_numpy(torch.softmax(national_initial, dim=-1)).round(6).tolist(),
            "calibration_overrides": {
                "transition_prior_override": transition_prior_override.round(6).tolist(),
                "diagnosed_penalty_scale": diagnosed_penalty_scale,
                "official_reference_penalty_scale": official_reference_penalty_scale,
                "national_anchor_penalty_scale": national_anchor_penalty_scale,
                "harp_program_penalty_scale": harp_program_penalty_scale,
                "linkage_penalty_scale": linkage_penalty_scale,
                "suppression_penalty_scale": suppression_penalty_scale,
                "fit_steps": fit_steps,
            },
            "vl_test_process_share": to_numpy(torch.sigmoid(vl_test_process_logit)).round(6).tolist(),
            "suppressed_given_test_share": to_numpy(torch.sigmoid(suppressed_given_test_logit)).round(6).tolist(),
            "province_vl_test_offsets": to_numpy(province_vl_test_offset).round(6).tolist(),
            "province_documentation_offsets": to_numpy(province_documented_offset).round(6).tolist(),
            "linkage_anchor_curves": {key: value.round(6).tolist() if isinstance(value, np.ndarray) else value for key, value in linkage_curves.items() if key != "month_axis"} | {"month_axis": month_axis},
            "suppression_anchor_curves": {key: value.round(6).tolist() if isinstance(value, np.ndarray) else value for key, value in suppression_curves.items() if key != "month_axis"} | {"month_axis": month_axis},
            "kp_coupling_matrix": to_numpy(torch.softmax(kp_coupling_logits + torch.eye(len(kp_axis), device=device, dtype=torch.float32) * 3.5, dim=-1)).round(6).tolist(),
            "determinant_covariates": covariate_meta,
            "metapopulation_engine": {
                "enabled": bool(operator_names),
                "operator_names": operator_names,
                "scale": to_numpy(torch.sigmoid(metapopulation_logit)).round(6).tolist(),
            },
            "province_archetypes": {
                "rows": archetype_bundle.get("rows", []),
                "summary": archetype_bundle.get("summary", {}),
                "region_archetype_mixture": archetype_bundle.get("region_archetype_mixture", {}),
            },
            "subgroup_prior_learning_summary": {
                "feature_names": subgroup_feature_names,
                "kp_feature_coefficients": to_numpy(kp_prior_coef).round(6).tolist(),
                "age_feature_coefficients": to_numpy(age_prior_coef).round(6).tolist(),
                "sex_feature_coefficients": to_numpy(sex_prior_coef).round(6).tolist(),
                "mean_kp_distribution": to_numpy(kp_probs_summary.mean(dim=0)).round(6).tolist(),
                "mean_age_distribution": to_numpy(age_probs_summary.mean(dim=0)).round(6).tolist(),
                "mean_sex_distribution": to_numpy(sex_probs_summary.mean(dim=0)).round(6).tolist(),
            },
            "cd4_prior_learning_summary": {
                "age_low_offset": to_numpy(cd4_age_low_offset).round(6).tolist(),
                "kp_sex_high_offset": to_numpy(cd4_kp_sex_high_offset).round(6).tolist(),
            },
            "synthetic_pretraining": {
                "mean_observation_weight": round(float(np.mean(archetype_observation_weight_np)) if archetype_observation_weight_np.size else 0.0, 6),
                "mean_reporting_noise": round(float(np.mean(archetype_reporting_noise_np)) if archetype_reporting_noise_np.size else 0.0, 6),
                "mean_pretraining_weight": round(float(np.mean(synthetic_pretraining_weight_np)) if synthetic_pretraining_weight_np.size else 0.0, 6),
                "library": archetype_bundle.get("synthetic_library", {}),
            },
        } | aux_summary
        return {
            "state_estimates": aggregate_np,
            "transition_probs": transition_probs_np,
            "prediction_stack": {
                "diagnosed_stock": np.clip(1.0 - aggregate_np[..., 0], 0.0, 1.0).astype(np.float32),
                "art_stock": np.clip(aggregate_np[..., 2] + aggregate_np[..., 3], 0.0, 1.0).astype(np.float32),
                "documented_suppression": np.clip(
                    np.minimum(
                        aggregate_np[..., 3],
                        np.clip((aggregate_np[..., 2] + aggregate_np[..., 3]) * to_numpy(torch.sigmoid(vl_test_process_logit)).reshape(1, month_count), 0.0, 1.0)
                        * to_numpy(torch.sigmoid(suppressed_given_test_logit)).reshape(1, month_count),
                    ),
                    0.0,
                    1.0,
                ).astype(np.float32),
                "testing_coverage": np.clip((aggregate_np[..., 2] + aggregate_np[..., 3]) * to_numpy(torch.sigmoid(vl_test_process_logit)).reshape(1, month_count), 0.0, 1.0).astype(np.float32),
                "deaths": np.clip(aggregate_np[..., 4] * 0.18, 0.0, 0.25).astype(np.float32),
            },
            "loss_breakdown": breakdown | {"total_loss": float(total_loss.detach().cpu())},
            "loss_trace": loss_trace,
            "parameters_summary": parameters_summary,
            "device": device,
            "inference_family": RESCUE_INFERENCE_FAMILY if requested_inference_family == "jax_svi" else requested_inference_family,
        }


def _forecast_from_transitions(
    state_estimates: np.ndarray,
    transition_probs: np.ndarray,
    subgroup_weights: np.ndarray,
    forecast_horizon: int,
) -> np.ndarray:
    del subgroup_weights
    state = state_estimates[:, -1, :].astype(np.float32)
    latest_probs = transition_probs[..., -1].mean(axis=(1, 2, 3, 4))
    forecasts = []
    for step in range(forecast_horizon):
        decay = float(np.exp(-0.15 * step))
        p_ud, p_da, p_av, p_al, p_la = [latest_probs[:, idx] * decay for idx in range(latest_probs.shape[-1])]
        U = state[:, 0]
        D = state[:, 1]
        A = state[:, 2]
        V = state[:, 3]
        L = state[:, 4]
        flow_ud = U * p_ud
        flow_da = D * p_da
        flow_av = A * p_av
        flow_al = A * p_al
        flow_la = L * p_la
        state = np.stack(
            [
                U - flow_ud,
                D + flow_ud - flow_da,
                A + flow_da + flow_la - flow_av - flow_al,
                V + flow_av,
                L + flow_al - flow_la,
            ],
            axis=-1,
        )
        state = np.clip(state, 0.0, None)
        state = state / np.clip(state.sum(axis=-1, keepdims=True), 1e-6, None)
        forecasts.append(state.astype(np.float32))
    return np.asarray(forecasts, dtype=np.float32).transpose(1, 0, 2)


def _forecast_month_axis(last_train_month: str, forecast_horizon: int) -> list[str]:
    last_ordinal = _month_ordinal(last_train_month)
    if last_ordinal is None:
        return [f"forecast_h{step + 1}" for step in range(forecast_horizon)]
    return [_month_label_from_ordinal(last_ordinal + step + 1) for step in range(forecast_horizon)]


def _forecast_prediction_stack(
    forecast_states: np.ndarray,
    parameters_summary: dict[str, Any],
) -> dict[str, np.ndarray]:
    art_stock = np.clip(forecast_states[..., 2] + forecast_states[..., 3], 0.0, 1.0).astype(np.float32)
    diagnosed_stock = np.clip(1.0 - forecast_states[..., 0], 0.0, 1.0).astype(np.float32)
    vl_share_series = np.asarray(parameters_summary.get("vl_test_process_share", []), dtype=np.float32)
    documented_given_test_series = np.asarray(parameters_summary.get("suppressed_given_test_share", []), dtype=np.float32)
    vl_share_last = float(vl_share_series[-1]) if vl_share_series.size else 0.45
    documented_given_test_last = float(documented_given_test_series[-1]) if documented_given_test_series.size else 0.80
    testing_coverage = np.clip(art_stock * vl_share_last, 0.0, 1.0).astype(np.float32)
    documented_suppression = np.clip(
        np.minimum(
            forecast_states[..., 3],
            testing_coverage * documented_given_test_last,
        ),
        0.0,
        1.0,
    ).astype(np.float32)
    return {
        "diagnosed_stock": diagnosed_stock,
        "art_stock": art_stock,
        "testing_coverage": testing_coverage,
        "documented_suppression": documented_suppression,
        "deaths": np.clip(forecast_states[..., 4] * 0.18, 0.0, 0.25).astype(np.float32),
    }


def _point_reference_from_harp(point: dict[str, Any]) -> dict[str, float]:
    estimated_plhiv = max(float(point.get("estimated_plhiv") or 0.0), 1.0)
    diagnosed = float(point.get("diagnosed") or 0.0) / estimated_plhiv
    art = float(point.get("on_art") or 0.0) / estimated_plhiv
    tested = float(point.get("viral_load_tested") or 0.0) / estimated_plhiv
    suppressed = float(point.get("suppressed") or 0.0) / estimated_plhiv
    return {
        "diagnosed_stock": diagnosed,
        "art_stock": art,
        "viral_load_tested_stock": tested,
        "documented_suppression": suppressed,
        "second95": art / max(diagnosed, 1e-6),
        "viral_load_tested_among_art": float(point.get("viral_load_tested") or 0.0) / max(float(point.get("on_art") or 0.0), 1.0),
        "suppressed_among_art": float(point.get("suppressed") or 0.0) / max(float(point.get("on_art") or 0.0), 1.0),
    }


def _carry_forward_backtest_summary(
    train_points: list[dict[str, Any]],
    holdout_points: list[dict[str, Any]],
) -> dict[str, Any]:
    if not train_points or not holdout_points:
        return {"available": False, "comparisons": [], "mean_absolute_error": 0.0}
    train_lookup = {int(_month_year(str(point.get("month") or "")) or 0): point for point in train_points}
    holdout_lookup = {int(_month_year(str(point.get("month") or "")) or 0): point for point in holdout_points}
    latest_train_year = max(train_lookup) if train_lookup else 0
    latest_train = train_lookup.get(latest_train_year)
    comparisons: list[dict[str, Any]] = []
    for holdout_year, holdout_point in sorted(holdout_lookup.items()):
        if latest_train is None:
            continue
        baseline = _point_reference_from_harp(latest_train)
        reference = _point_reference_from_harp(holdout_point)
        errors = {f"{metric}_abs_error": round(abs(float(baseline[metric]) - float(reference[metric])), 6) for metric in reference}
        comparisons.append(
            {
                "train_year": latest_train_year,
                "holdout_year": holdout_year,
                "baseline_reference": {key: round(float(value), 6) for key, value in baseline.items()},
                "holdout_reference": {key: round(float(value), 6) for key, value in reference.items()},
                "errors": errors,
            }
        )
    mean_error = float(np.mean([np.mean(list(item["errors"].values())) for item in comparisons if item["errors"]])) if comparisons else 0.0
    return {
        "available": bool(comparisons),
        "comparisons": comparisons,
        "mean_absolute_error": round(mean_error, 6),
    }


def _compute_observation_residuals(
    predictions: dict[str, np.ndarray],
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
    month_axis: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for target_name in OBSERVATION_ORDER:
        pred = predictions[target_name]
        obs = observation_targets[target_name]
        residual = pred - obs
        for province_idx, province in enumerate(province_axis):
            for month_idx, month in enumerate(month_axis):
                rows.append(
                    {
                        "target_name": target_name,
                        "province": province,
                        "time": month,
                        "predicted": round(float(pred[province_idx, month_idx]), 6),
                        "observed": round(float(obs[province_idx, month_idx]), 6),
                        "residual": round(float(residual[province_idx, month_idx]), 6),
                    }
                )
    return rows


def _region_aggregate(surface: np.ndarray, region_index: np.ndarray, region_count: int) -> np.ndarray:
    region_surface = np.zeros((region_count, surface.shape[1]), dtype=np.float32)
    for region_id in range(region_count):
        members = surface[region_index == region_id]
        if members.size:
            region_surface[region_id] = members.mean(axis=0)
    return region_surface


def _compute_reconciliation_report(
    state_estimates: np.ndarray,
    observation_targets: dict[str, np.ndarray],
    region_axis: list[str],
    region_index: np.ndarray,
) -> dict[str, Any]:
    region_count = len(region_axis)
    diagnosed_pred = 1.0 - state_estimates[..., 0]
    art_pred = state_estimates[..., 2] + state_estimates[..., 3]
    suppression_pred = state_estimates[..., 3]
    region_diagnosed = _region_aggregate(diagnosed_pred, region_index, region_count)
    region_art = _region_aggregate(art_pred, region_index, region_count)
    region_suppression = _region_aggregate(suppression_pred, region_index, region_count)
    region_diagnosed_obs = _region_aggregate(observation_targets["diagnosed_stock"], region_index, region_count)
    region_art_obs = _region_aggregate(observation_targets["art_stock"], region_index, region_count)
    region_supp_obs = _region_aggregate(observation_targets["documented_suppression"], region_index, region_count)
    region_rows = []
    for region_id, region in enumerate(region_axis):
        region_rows.append(
            {
                "region": region,
                "diagnosed_mae": round(float(np.mean(np.abs(region_diagnosed[region_id] - region_diagnosed_obs[region_id]))), 6),
                "art_mae": round(float(np.mean(np.abs(region_art[region_id] - region_art_obs[region_id]))), 6),
                "suppression_mae": round(float(np.mean(np.abs(region_suppression[region_id] - region_supp_obs[region_id]))), 6),
            }
        )
    return {
        "region_rows": region_rows,
        "national_predicted": {
            "diagnosed_stock": diagnosed_pred.mean(axis=0).round(6).tolist(),
            "art_stock": art_pred.mean(axis=0).round(6).tolist(),
            "documented_suppression": suppression_pred.mean(axis=0).round(6).tolist(),
        },
        "national_observed": {
            "diagnosed_stock": observation_targets["diagnosed_stock"].mean(axis=0).round(6).tolist(),
            "art_stock": observation_targets["art_stock"].mean(axis=0).round(6).tolist(),
            "documented_suppression": observation_targets["documented_suppression"].mean(axis=0).round(6).tolist(),
        },
    }


def _official_reference_penalty_torch(
    *,
    diagnosed_pred: Any,
    art_pred: Any,
    suppression_pred: Any,
    province_axis: list[str],
    month_axis: list[str],
    device: Any,
) -> tuple[Any, dict[str, float]]:
    mask_np = _national_reference_mask(province_axis)
    if not np.any(mask_np):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"official_reference_penalty": 0.0}
    mask = to_torch_tensor(mask_np, device=device, dtype=torch.float32)
    penalty = torch.tensor(0.0, device=device, dtype=torch.float32)
    details: dict[str, float] = {}
    for point in OFFICIAL_REFERENCE_POINTS:
        if point["month"] not in month_axis:
            continue
        month_idx = month_axis.index(point["month"])
        diag_val = torch.sum(diagnosed_pred[:, month_idx] * mask)
        art_val = torch.sum(art_pred[:, month_idx] * mask)
        sup_val = torch.sum(suppression_pred[:, month_idx] * mask)
        second_val = art_val / torch.clamp(diag_val, min=1e-6)
        third_val = sup_val / torch.clamp(art_val, min=1e-6)
        if "first95" in point["reference"]:
            penalty = penalty + (diag_val - float(point["reference"]["first95"])) ** 2
            details[f"{point['month']}_first95_error"] = float(torch.abs(diag_val - float(point["reference"]["first95"])).detach().cpu())
        if "second95" in point["reference"]:
            penalty = penalty + (second_val - float(point["reference"]["second95"])) ** 2
            details[f"{point['month']}_second95_error"] = float(torch.abs(second_val - float(point["reference"]["second95"])).detach().cpu())
        if "overall_suppressed" in point["reference"]:
            penalty = penalty + (sup_val - float(point["reference"]["overall_suppressed"])) ** 2
            details[f"{point['month']}_overall_error"] = float(torch.abs(sup_val - float(point["reference"]["overall_suppressed"])).detach().cpu())
        if "documented_suppression_among_art" in point["reference"]:
            penalty = penalty + (third_val - float(point["reference"]["documented_suppression_among_art"])) ** 2
            details[f"{point['month']}_third95_error"] = float(torch.abs(third_val - float(point["reference"]["documented_suppression_among_art"])).detach().cpu())
    return penalty, details | {"official_reference_penalty": float(penalty.detach().cpu())}


def _national_anchor_penalty_torch(
    *,
    diagnosed_pred: Any,
    art_pred: Any,
    suppression_pred: Any,
    province_axis: list[str],
    month_axis: list[str],
    device: Any,
) -> tuple[Any, dict[str, float]]:
    mask_np = _national_reference_mask(province_axis)
    if not np.any(mask_np):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"national_anchor_penalty": 0.0}
    curves = _official_anchor_curves(month_axis)
    if not np.any(curves["weight"] > 0):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"national_anchor_penalty": 0.0}
    mask = to_torch_tensor(mask_np, device=device, dtype=torch.float32)
    weights = to_torch_tensor(curves["weight"], device=device, dtype=torch.float32)
    diag_curve = to_torch_tensor(curves["diagnosed_stock"], device=device, dtype=torch.float32)
    art_curve = to_torch_tensor(curves["art_stock"], device=device, dtype=torch.float32)
    sup_curve = to_torch_tensor(curves["documented_suppression"], device=device, dtype=torch.float32)
    third_curve = to_torch_tensor(curves["third95"], device=device, dtype=torch.float32)

    national_diag = torch.sum(diagnosed_pred * mask.view(-1, 1), dim=0)
    national_art = torch.sum(art_pred * mask.view(-1, 1), dim=0)
    national_sup = torch.sum(suppression_pred * mask.view(-1, 1), dim=0)
    national_third = national_sup / torch.clamp(national_art, min=1e-6)

    penalty = torch.mean(weights * ((national_diag - diag_curve) ** 2 + (national_art - art_curve) ** 2 + (national_sup - sup_curve) ** 2))
    active_third = (weights > 0) & (third_curve > 0)
    if torch.any(active_third):
        penalty = penalty + torch.mean(weights[active_third] * ((national_third[active_third] - third_curve[active_third]) ** 2))
    details = {
        "national_anchor_penalty": float(penalty.detach().cpu()),
        "national_anchor_diag_mae": float(torch.mean(torch.abs(national_diag - diag_curve) * weights).detach().cpu()),
        "national_anchor_art_mae": float(torch.mean(torch.abs(national_art - art_curve) * weights).detach().cpu()),
        "national_anchor_suppression_mae": float(torch.mean(torch.abs(national_sup - sup_curve) * weights).detach().cpu()),
    }
    return penalty, details


def _harp_program_penalty_torch(
    *,
    diagnosed_pred: Any,
    art_pred: Any,
    suppression_pred: Any,
    testing_pred: Any,
    province_axis: list[str],
    month_axis: list[str],
    device: Any,
) -> tuple[Any, dict[str, float]]:
    mask_np = _national_reference_mask(province_axis)
    if not np.any(mask_np):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"harp_program_penalty": 0.0}
    curves = _harp_program_curves(month_axis)
    if not np.any(curves["weight"] > 0):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"harp_program_penalty": 0.0}

    mask = to_torch_tensor(mask_np, device=device, dtype=torch.float32)
    weights = to_torch_tensor(curves["weight"], device=device, dtype=torch.float32)
    diag_curve = to_torch_tensor(curves["diagnosed_stock"], device=device, dtype=torch.float32)
    art_curve = to_torch_tensor(curves["art_stock"], device=device, dtype=torch.float32)
    tested_curve = to_torch_tensor(curves["viral_load_tested_stock"], device=device, dtype=torch.float32)
    sup_curve = to_torch_tensor(curves["documented_suppression"], device=device, dtype=torch.float32)
    tested_among_art_curve = to_torch_tensor(curves["viral_load_tested_among_art"], device=device, dtype=torch.float32)
    suppressed_among_art_curve = to_torch_tensor(curves["suppressed_among_art"], device=device, dtype=torch.float32)

    national_diag = torch.sum(diagnosed_pred * mask.view(-1, 1), dim=0)
    national_art = torch.sum(art_pred * mask.view(-1, 1), dim=0)
    national_tested = torch.sum(testing_pred * mask.view(-1, 1), dim=0)
    national_sup = torch.sum(suppression_pred * mask.view(-1, 1), dim=0)
    national_second = national_art / torch.clamp(national_diag, min=1e-6)
    national_tested_among_art = national_tested / torch.clamp(national_art, min=1e-6)
    national_suppressed_among_art = national_sup / torch.clamp(national_art, min=1e-6)

    penalty = torch.mean(
        weights
        * (
            torch.square(national_diag - diag_curve)
            + torch.square(national_art - art_curve)
            + torch.square(national_tested - tested_curve)
            + torch.square(national_sup - sup_curve)
        )
    )
    active_ratio = (weights > 0) & (tested_among_art_curve > 0)
    if torch.any(active_ratio):
        penalty = penalty + torch.mean(weights[active_ratio] * torch.square(national_tested_among_art[active_ratio] - tested_among_art_curve[active_ratio]))
    active_supp = (weights > 0) & (suppressed_among_art_curve > 0)
    if torch.any(active_supp):
        penalty = penalty + torch.mean(weights[active_supp] * torch.square(national_suppressed_among_art[active_supp] - suppressed_among_art_curve[active_supp]))

    details = {
        "harp_program_penalty": float(penalty.detach().cpu()),
        "harp_diagnosed_mae": float(torch.mean(torch.abs(national_diag - diag_curve) * weights).detach().cpu()),
        "harp_art_mae": float(torch.mean(torch.abs(national_art - art_curve) * weights).detach().cpu()),
        "harp_vl_tested_mae": float(torch.mean(torch.abs(national_tested - tested_curve) * weights).detach().cpu()),
        "harp_documented_suppression_mae": float(torch.mean(torch.abs(national_sup - sup_curve) * weights).detach().cpu()),
        "harp_second95_mae": float(torch.mean(torch.abs(national_second - torch.clamp(art_curve / torch.clamp(diag_curve, min=1e-6), 0.0, 1.0)) * weights).detach().cpu()),
        "harp_tested_among_art_mae": float(torch.mean(torch.abs(national_tested_among_art - tested_among_art_curve) * weights).detach().cpu()),
        "harp_suppressed_among_art_mae": float(torch.mean(torch.abs(national_suppressed_among_art - suppressed_among_art_curve) * weights).detach().cpu()),
    }
    return penalty, details


def _linkage_penalty_torch(
    *,
    latent: Any,
    transition_probs: Any,
    subgroup_weights: Any,
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
    month_axis: list[str],
    device: Any,
) -> tuple[Any, dict[str, float], dict[str, np.ndarray]]:
    mask_np = _national_reference_mask(province_axis)
    if not np.any(mask_np):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        curves = _linkage_anchor_curves(month_axis=month_axis, observation_targets=observation_targets, province_axis=province_axis)
        return zero, {"linkage_penalty": 0.0}, curves

    curves = _linkage_anchor_curves(month_axis=month_axis, observation_targets=observation_targets, province_axis=province_axis)
    weights_np = curves["weight"].astype(np.float32)
    if not np.any(weights_np > 0):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"linkage_penalty": 0.0}, curves

    mask = to_torch_tensor(mask_np, device=device, dtype=torch.float32)
    weights = to_torch_tensor(weights_np, device=device, dtype=torch.float32)
    second95_target = to_torch_tensor(curves["second95_target"], device=device, dtype=torch.float32)
    d_to_a_target = to_torch_tensor(curves["d_to_a_transition_target"], device=device, dtype=torch.float32)

    d_mass = latent[..., 1, :, :]
    subgroup_weighted_mass = d_mass * subgroup_weights.unsqueeze(-1).unsqueeze(-1)
    d_to_a_prob = transition_probs[..., 1, :]
    weighted_numerator = torch.sum(subgroup_weighted_mass * d_to_a_prob, dim=(1, 2, 3, 4))
    weighted_denom = torch.sum(subgroup_weighted_mass, dim=(1, 2, 3, 4))
    province_linkage = weighted_numerator / torch.clamp(weighted_denom, min=1e-6)
    national_linkage = torch.sum(province_linkage * mask.view(-1, 1), dim=0)

    aggregate = (latent * subgroup_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=(1, 2, 3, 5)).permute(0, 2, 1)
    aggregate = aggregate / torch.clamp(aggregate.sum(dim=-1, keepdim=True), min=1e-6)
    national_diag = torch.sum((1.0 - aggregate[..., 0]) * mask.view(-1, 1), dim=0)
    national_art = torch.sum((aggregate[..., 2] + aggregate[..., 3]) * mask.view(-1, 1), dim=0)
    national_second95 = national_art / torch.clamp(national_diag, min=1e-6)

    penalty = torch.mean(weights * (torch.square(national_linkage - d_to_a_target) + 0.85 * torch.square(national_second95 - second95_target)))
    details = {
        "linkage_penalty": float(penalty.detach().cpu()),
        "linkage_d_to_a_mae": float(torch.mean(torch.abs(national_linkage - d_to_a_target) * weights).detach().cpu()),
        "linkage_second95_mae": float(torch.mean(torch.abs(national_second95 - second95_target) * weights).detach().cpu()),
    }
    return penalty, details, curves


def _suppression_penalty_torch(
    *,
    latent: Any,
    transition_probs: Any,
    subgroup_weights: Any,
    testing_pred: Any,
    observation_targets: dict[str, np.ndarray],
    province_axis: list[str],
    month_axis: list[str],
    device: Any,
) -> tuple[Any, dict[str, float], dict[str, np.ndarray]]:
    mask_np = _national_reference_mask(province_axis)
    curves = _suppression_anchor_curves(month_axis=month_axis, observation_targets=observation_targets, province_axis=province_axis)
    if not np.any(mask_np) or not np.any(curves["weight"] > 0):
        zero = torch.tensor(0.0, device=device, dtype=torch.float32)
        return zero, {"suppression_penalty": 0.0}, curves

    mask = to_torch_tensor(mask_np, device=device, dtype=torch.float32)
    weights = to_torch_tensor(curves["weight"].astype(np.float32), device=device, dtype=torch.float32)
    overall_target = to_torch_tensor(curves["overall_suppression_target"].astype(np.float32), device=device, dtype=torch.float32)
    supp_art_target = to_torch_tensor(curves["suppressed_among_art_target"].astype(np.float32), device=device, dtype=torch.float32)
    tested_art_target = to_torch_tensor(curves["tested_among_art_target"].astype(np.float32), device=device, dtype=torch.float32)
    a_to_v_target = to_torch_tensor(curves["a_to_v_transition_target"].astype(np.float32), device=device, dtype=torch.float32)

    a_mass = latent[..., 2, :, :]
    subgroup_weighted_mass = a_mass * subgroup_weights.unsqueeze(-1).unsqueeze(-1)
    a_to_v_prob = transition_probs[..., 2, :]
    weighted_numerator = torch.sum(subgroup_weighted_mass * a_to_v_prob, dim=(1, 2, 3, 4))
    weighted_denom = torch.sum(subgroup_weighted_mass, dim=(1, 2, 3, 4))
    province_supp_transition = weighted_numerator / torch.clamp(weighted_denom, min=1e-6)
    national_supp_transition = torch.sum(province_supp_transition * mask.view(-1, 1), dim=0)

    aggregate = (latent * subgroup_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(dim=(1, 2, 3, 5)).permute(0, 2, 1)
    aggregate = aggregate / torch.clamp(aggregate.sum(dim=-1, keepdim=True), min=1e-6)
    national_art = torch.sum((aggregate[..., 2] + aggregate[..., 3]) * mask.view(-1, 1), dim=0)
    national_sup = torch.sum(aggregate[..., 3] * mask.view(-1, 1), dim=0)
    national_test_t = torch.sum(testing_pred * mask.view(-1, 1), dim=0)
    national_supp_art = national_sup / torch.clamp(national_art, min=1e-6)
    national_test_art = national_test_t / torch.clamp(national_art, min=1e-6)

    penalty = torch.mean(
        weights
        * (
            torch.square(national_supp_transition - a_to_v_target)
            + 0.90 * torch.square(national_supp_art - supp_art_target)
            + 0.65 * torch.square(national_sup - overall_target)
            + 0.35 * torch.square(national_test_art - tested_art_target)
        )
    )
    details = {
        "suppression_penalty": float(penalty.detach().cpu()),
        "suppression_a_to_v_mae": float(torch.mean(torch.abs(national_supp_transition - a_to_v_target) * weights).detach().cpu()),
        "suppression_overall_mae": float(torch.mean(torch.abs(national_sup - overall_target) * weights).detach().cpu()),
        "suppression_among_art_mae": float(torch.mean(torch.abs(national_supp_art - supp_art_target) * weights).detach().cpu()),
        "tested_among_art_alignment_mae": float(torch.mean(torch.abs(national_test_art - tested_art_target) * weights).detach().cpu()),
    }
    return penalty, details, curves


def _official_reference_check(
    *,
    state_estimates: np.ndarray,
    province_axis: list[str],
    month_axis: list[str],
    run_id: str,
    profile_id: str,
    inference_family: str,
) -> dict[str, Any]:
    mask = _national_reference_mask(province_axis)
    comparisons = []
    for point in OFFICIAL_REFERENCE_POINTS:
        if point["month"] not in month_axis:
            continue
        month_idx = month_axis.index(point["month"])
        state = np.tensordot(mask, state_estimates[:, month_idx], axes=(0, 0))
        first95 = float(1.0 - state[0])
        art = float(state[2] + state[3])
        overall = float(state[3])
        second95 = art / max(first95, 1e-6)
        third95 = overall / max(art, 1e-6)
        errors = {}
        if "first95" in point["reference"]:
            errors["first95_abs_error"] = round(abs(first95 - float(point["reference"]["first95"])), 6)
        if "second95" in point["reference"]:
            errors["second95_abs_error"] = round(abs(second95 - float(point["reference"]["second95"])), 6)
        if "overall_suppressed" in point["reference"]:
            errors["overall_suppressed_abs_error"] = round(abs(overall - float(point["reference"]["overall_suppressed"])), 6)
        if "documented_suppression_among_art" in point["reference"]:
            errors["documented_suppression_among_art_abs_error"] = round(abs(third95 - float(point["reference"]["documented_suppression_among_art"])), 6)
        comparisons.append(
            {
                "label": point["label"],
                "date": point["month"],
                "source_url": point["source_url"],
                "reference": point["reference"],
                "model": {
                    "first95": round(first95, 6),
                    "second95": round(second95, 6),
                    "third95": round(third95, 6),
                    "overall_suppressed": round(overall, 6),
                },
                "errors": errors,
            }
        )
    verdict = "official reference ladder unavailable"
    if comparisons:
        mean_error = np.mean([np.mean(list(item["errors"].values())) for item in comparisons if item["errors"]])
        verdict = "closer to official references" if mean_error <= 0.15 else "not calibrated to official Philippines cascade references yet"
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "inference_family": inference_family,
        "verdict": verdict,
        "comparisons": comparisons,
    }


def _harp_program_check(
    *,
    state_estimates: np.ndarray,
    prediction_stack: dict[str, np.ndarray],
    province_axis: list[str],
    month_axis: list[str],
    run_id: str,
    profile_id: str,
    inference_family: str,
) -> dict[str, Any]:
    mask = _national_reference_mask(province_axis)
    curves = _harp_program_curves(month_axis)
    comparisons = []
    if not np.any(curves["weight"] > 0):
        return {
            "run_id": run_id,
            "profile_id": profile_id,
            "inference_family": inference_family,
            "verdict": "HARP program ladder unavailable",
            "comparisons": comparisons,
        }

    national_diag = np.tensordot(mask, prediction_stack["diagnosed_stock"], axes=(0, 0))
    national_art = np.tensordot(mask, prediction_stack["art_stock"], axes=(0, 0))
    national_tested = np.tensordot(mask, prediction_stack["testing_coverage"], axes=(0, 0))
    national_sup = np.tensordot(mask, prediction_stack["documented_suppression"], axes=(0, 0))
    month_ordinals = [_month_ordinal(month) for month in month_axis]
    valid_month_indices = [idx for idx, ordinal in enumerate(month_ordinals) if ordinal is not None]
    for point in HARP_PROGRAM_POINTS:
        point_ordinal = _month_ordinal(point["month"])
        if not valid_month_indices or point_ordinal is None:
            continue
        month_idx = min(valid_month_indices, key=lambda idx: abs(month_ordinals[idx] - point_ordinal))
        state = np.tensordot(mask, state_estimates[:, month_idx], axes=(0, 0))
        diag_val = float(national_diag[month_idx])
        art_val = float(national_art[month_idx])
        tested_val = float(national_tested[month_idx])
        sup_val = float(national_sup[month_idx])
        second95 = art_val / max(diag_val, 1e-6)
        tested_among_art = tested_val / max(art_val, 1e-6)
        suppressed_among_art = sup_val / max(art_val, 1e-6)
        reference = {
            "diagnosed_stock": round(float(curves["diagnosed_stock"][month_idx]), 6),
            "art_stock": round(float(curves["art_stock"][month_idx]), 6),
            "viral_load_tested_stock": round(float(curves["viral_load_tested_stock"][month_idx]), 6),
            "documented_suppression": round(float(curves["documented_suppression"][month_idx]), 6),
            "second95": round(float(curves["art_stock"][month_idx] / max(curves["diagnosed_stock"][month_idx], 1e-6)), 6),
            "viral_load_tested_among_art": round(float(curves["viral_load_tested_among_art"][month_idx]), 6),
            "suppressed_among_art": round(float(curves["suppressed_among_art"][month_idx]), 6),
        }
        model = {
            "first95": round(float(1.0 - state[0]), 6),
            "diagnosed_stock": round(diag_val, 6),
            "art_stock": round(art_val, 6),
            "viral_load_tested_stock": round(tested_val, 6),
            "documented_suppression": round(sup_val, 6),
            "second95": round(second95, 6),
            "viral_load_tested_among_art": round(tested_among_art, 6),
            "suppressed_among_art": round(suppressed_among_art, 6),
        }
        errors = {
            "diagnosed_stock_abs_error": round(abs(model["diagnosed_stock"] - reference["diagnosed_stock"]), 6),
            "art_stock_abs_error": round(abs(model["art_stock"] - reference["art_stock"]), 6),
            "viral_load_tested_stock_abs_error": round(abs(model["viral_load_tested_stock"] - reference["viral_load_tested_stock"]), 6),
            "documented_suppression_abs_error": round(abs(model["documented_suppression"] - reference["documented_suppression"]), 6),
            "second95_abs_error": round(abs(model["second95"] - reference["second95"]), 6),
            "viral_load_tested_among_art_abs_error": round(abs(model["viral_load_tested_among_art"] - reference["viral_load_tested_among_art"]), 6),
            "suppressed_among_art_abs_error": round(abs(model["suppressed_among_art"] - reference["suppressed_among_art"]), 6),
        }
        comparisons.append(
            {
                "label": point["label"],
                "date": point["month"],
                "model_month": month_axis[month_idx],
                "source_label": point["source_label"],
                "reference": reference,
                "model": model,
                "errors": errors,
            }
        )
    verdict = "HARP program ladder unavailable"
    if comparisons:
        mean_error = np.mean([np.mean(list(item["errors"].values())) for item in comparisons if item["errors"]])
        verdict = "close to HARP program counts" if mean_error <= 0.08 else "not calibrated to HARP program counts yet"
    return {
        "run_id": run_id,
        "profile_id": profile_id,
        "inference_family": inference_family,
        "verdict": verdict,
        "comparisons": comparisons,
    }


def _frozen_history_backtest_evaluation(
    *,
    run_id: str,
    profile_id: str,
    inference_family: str,
    province_axis: list[str],
    forecast_states: np.ndarray,
    parameters_summary: dict[str, Any],
    backtest_config: dict[str, Any],
) -> dict[str, Any]:
    forecast_horizon = int(backtest_config.get("forecast_horizon") or forecast_states.shape[1] or 0)
    last_train_month = str(backtest_config.get("train_last_model_month") or "")
    forecast_month_axis = _forecast_month_axis(last_train_month, forecast_horizon)
    forecast_prediction_stack = _forecast_prediction_stack(forecast_states[:, :forecast_horizon], parameters_summary)
    holdout_points = list(backtest_config.get("holdout_harp_points") or [])
    with _temporary_reference_points(harp_program_points=holdout_points):
        holdout_check = _harp_program_check(
            state_estimates=forecast_states[:, :forecast_horizon],
            prediction_stack=forecast_prediction_stack,
            province_axis=province_axis,
            month_axis=forecast_month_axis,
            run_id=run_id,
            profile_id=profile_id,
            inference_family=inference_family,
        )
    carry_forward = _carry_forward_backtest_summary(
        list(backtest_config.get("train_harp_points") or []),
        holdout_points,
    )
    holdout_errors = [
        float(np.mean(list(item.get("errors", {}).values())))
        for item in holdout_check.get("comparisons", [])
        if item.get("errors")
    ]
    model_mean_error = round(float(np.mean(holdout_errors)) if holdout_errors else 0.0, 6)
    summary = {
        "train_years": list(backtest_config.get("train_years") or []),
        "holdout_years": list(backtest_config.get("holdout_years") or []),
        "train_last_model_month": last_train_month,
        "forecast_horizon": forecast_horizon,
        "forecast_month_axis": forecast_month_axis,
        "comparison_count": len(holdout_check.get("comparisons", [])),
        "model_mean_absolute_error": model_mean_error,
        "carry_forward_mean_absolute_error": carry_forward.get("mean_absolute_error", 0.0),
        "model_beats_carry_forward": bool(
            carry_forward.get("available") and model_mean_error < float(carry_forward.get("mean_absolute_error") or 0.0)
        ),
    }
    return {
        "mode": "frozen_history",
        "summary": summary,
        "holdout_reference_check": holdout_check,
        "carry_forward_baseline": carry_forward,
    }


def _compute_benchmark_metrics(
    state_estimates: np.ndarray,
    observation_targets: dict[str, np.ndarray],
    region_axis: list[str],
    region_index: np.ndarray,
    province_axis: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    diagnosed_pred = 1.0 - state_estimates[..., 0]
    art_pred = state_estimates[..., 2] + state_estimates[..., 3]
    suppression_pred = state_estimates[..., 3]
    naive_diag = np.roll(observation_targets["diagnosed_stock"], shift=1, axis=1)
    naive_art = np.roll(observation_targets["art_stock"], shift=1, axis=1)
    naive_sup = np.roll(observation_targets["documented_suppression"], shift=1, axis=1)
    naive_diag[:, 0] = observation_targets["diagnosed_stock"].mean(axis=0)[0]
    naive_art[:, 0] = observation_targets["art_stock"].mean(axis=0)[0]
    naive_sup[:, 0] = observation_targets["documented_suppression"].mean(axis=0)[0]
    model_mae = np.mean(
        [
            np.mean(np.abs(diagnosed_pred - observation_targets["diagnosed_stock"])),
            np.mean(np.abs(art_pred - observation_targets["art_stock"])),
            np.mean(np.abs(suppression_pred - observation_targets["documented_suppression"])),
        ]
    )
    naive_mae = np.mean(
        [
            np.mean(np.abs(naive_diag - observation_targets["diagnosed_stock"])),
            np.mean(np.abs(naive_art - observation_targets["art_stock"])),
            np.mean(np.abs(naive_sup - observation_targets["documented_suppression"])),
        ]
    )
    province_instability = np.mean(np.abs(np.diff(state_estimates, axis=1))) if state_estimates.shape[1] > 1 else 0.0
    lower_bound_miss = np.mean(np.maximum(0.0, suppression_pred - np.maximum(observation_targets["documented_suppression"] + 0.03, observation_targets["testing_coverage"] * 0.95)))
    diagnosed_optimism = np.mean(np.maximum(0.0, diagnosed_pred - observation_targets["diagnosed_stock"] - 0.02))
    second95 = art_pred / np.clip(diagnosed_pred, 1e-6, None)
    third95 = suppression_pred / np.clip(art_pred, 1e-6, None)
    boundary_hit_rate = float(np.mean((state_estimates <= 1e-4) | (state_estimates >= 0.999)))
    implausible_mass = float(np.mean(np.maximum(0.0, state_estimates.sum(axis=-1) - 1.0001)))
    validation = {
        "national_metrics": {
            "diagnosed_mae": round(float(np.mean(np.abs(diagnosed_pred.mean(axis=0) - observation_targets["diagnosed_stock"].mean(axis=0)))), 6),
            "art_mae": round(float(np.mean(np.abs(art_pred.mean(axis=0) - observation_targets["art_stock"].mean(axis=0)))), 6),
            "suppression_mae": round(float(np.mean(np.abs(suppression_pred.mean(axis=0) - observation_targets["documented_suppression"].mean(axis=0)))), 6),
        },
        "regional_metrics": _compute_reconciliation_report(state_estimates, observation_targets, region_axis, region_index)["region_rows"],
        "subnational_metrics": {
            "province_metrics": [
                {
                    "province": province_axis[idx],
                    "diagnosed_mae": round(float(np.mean(np.abs(diagnosed_pred[idx] - observation_targets["diagnosed_stock"][idx]))), 6),
                    "art_mae": round(float(np.mean(np.abs(art_pred[idx] - observation_targets["art_stock"][idx]))), 6),
                    "suppression_mae": round(float(np.mean(np.abs(suppression_pred[idx] - observation_targets["documented_suppression"][idx]))), 6),
                }
                for idx in range(len(province_axis))
            ],
            "mean_instability": round(float(province_instability), 6),
        },
        "lower_bound_suppression_checks": {"mean_exceedance": round(float(lower_bound_miss), 6)},
        "diagnosed_optimism_checks": {"mean_exceedance": round(float(diagnosed_optimism), 6)},
        "hierarchy_reconciliation_checks": {
            "national_reconciliation_mae": round(float(np.mean(np.abs(diagnosed_pred.mean(axis=0) - observation_targets["diagnosed_stock"].mean(axis=0)))), 6)
        },
        "boundary_hit_summary": {"rate": round(boundary_hit_rate, 6)},
        "stress_lab_placeholders": {"status": "not_run"},
    }
    gate_report = {
        "primary_gates": [
            {"gate": "subnational_mae_beats_naive", "passed": bool(model_mae <= naive_mae), "value": {"model_mae": round(float(model_mae), 6), "naive_mae": round(float(naive_mae), 6)}},
            {"gate": "province_instability_below_tolerance", "passed": bool(province_instability <= 0.085), "value": round(float(province_instability), 6)},
            {"gate": "lower_bound_suppression_miss_below_tolerance", "passed": bool(lower_bound_miss <= 0.065), "value": round(float(lower_bound_miss), 6)},
            {"gate": "diagnosed_optimism_below_tolerance", "passed": bool(diagnosed_optimism <= 0.055), "value": round(float(diagnosed_optimism), 6)},
            {"gate": "hierarchy_reconciliation_nearly_exact", "passed": bool(validation["hierarchy_reconciliation_checks"]["national_reconciliation_mae"] <= 0.055), "value": validation["hierarchy_reconciliation_checks"]["national_reconciliation_mae"]},
        ],
        "protected_metrics": [
            {"metric": "second95_absolute_error", "value": round(float(np.mean(np.abs(second95 - np.clip(observation_targets["art_stock"] / np.clip(observation_targets["diagnosed_stock"], 1e-6, None), 0.0, 1.0)))), 6)},
            {"metric": "third95_documented_lower_bound_consistency", "value": round(float(lower_bound_miss), 6)},
            {"metric": "province_boundary_hit_rate", "value": round(boundary_hit_rate, 6)},
            {"metric": "implausible_mass", "value": round(implausible_mass, 8)},
            {"metric": "third95_absolute_error", "value": round(float(np.mean(np.abs(third95 - np.clip(observation_targets["documented_suppression"] / np.clip(observation_targets["art_stock"], 1e-6, None), 0.0, 1.0)))), 6)},
        ],
    }
    guardrails = {
        "boundary_hit_rate": round(boundary_hit_rate, 6),
        "province_clipping_rate": round(boundary_hit_rate, 6),
        "province_instability": round(float(province_instability), 6),
        "region_inconsistency": validation["hierarchy_reconciliation_checks"]["national_reconciliation_mae"],
        "phase4_ready": False,
    }
    return validation, gate_report, guardrails


def run_phase3_rescue_core(
    *,
    run_id: str,
    plugin_id: str,
    profile_id: str = RESCUE_PROFILE_ID,
    requested_inference_family: str = RESCUE_INFERENCE_FAMILY,
    phase_dir_name: str = "phase3",
    axis_catalogs_override: dict[str, Any] | None = None,
    normalized_rows_override: list[dict[str, Any]] | None = None,
    parameter_catalog_override: list[dict[str, Any]] | None = None,
    standardized_tensor_override: np.ndarray | None = None,
    reference_overrides: dict[str, Any] | None = None,
    backtest_config: dict[str, Any] | None = None,
    calibration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    phase3_dir = ensure_dir(ctx.run_dir / phase_dir_name)
    set_global_seed(17)

    axis_catalogs = axis_catalogs_override if axis_catalogs_override is not None else read_json(ctx.run_dir / "phase1" / "axis_catalogs.json", default={})
    normalized_rows = normalized_rows_override if normalized_rows_override is not None else read_json(ctx.run_dir / "phase1" / "normalized_subparameters.json", default=[])
    parameter_catalog = parameter_catalog_override if parameter_catalog_override is not None else read_json(ctx.run_dir / "phase1" / "parameter_catalog.json", default=[])
    candidate_profiles = read_json(ctx.run_dir / "phase2" / "candidate_profiles.json", default=[])
    curated_candidate_blocks = read_json(ctx.run_dir / "phase2" / "curated_candidate_blocks.json", default=[])
    standardized_tensor = standardized_tensor_override if standardized_tensor_override is not None else load_tensor_artifact(ctx.run_dir / "phase1" / "standardized_tensor.npz")

    province_axis = list(axis_catalogs.get("province", [])) or ["national"]
    month_axis = list(axis_catalogs.get("month", [])) or ["unknown"]
    canonical_axis = list(axis_catalogs.get("canonical_name", [])) or []
    kp_axis = _default_axis(axis_catalogs, "kp_group", DEFAULT_KP_CATALOG)
    age_axis = _default_axis(axis_catalogs, "age_band", DEFAULT_AGE_CATALOG)
    sex_axis = _default_axis(axis_catalogs, "sex", DEFAULT_SEX_CATALOG)
    region_axis, region_index = _region_assignments(province_axis, normalized_rows)

    reference_overrides = reference_overrides or {}
    with _temporary_reference_points(
        official_reference_points=reference_overrides.get("official_points"),
        harp_program_points=reference_overrides.get("harp_points"),
    ):
        observation_ladder, observation_targets, observation_target_rows = build_observation_ladder(
            standardized_tensor=standardized_tensor,
            normalized_rows=normalized_rows,
            parameter_catalog=parameter_catalog,
            canonical_axis=canonical_axis,
            province_axis=province_axis,
            month_axis=month_axis,
        )
        observation_support_bundle = _build_observation_support_bundle(
            standardized_tensor=standardized_tensor,
            normalized_rows=normalized_rows,
            parameter_catalog=parameter_catalog,
            canonical_axis=canonical_axis,
            province_axis=province_axis,
            month_axis=month_axis,
            observation_ladder=observation_ladder,
        )
        observation_ladder, observation_targets, observation_target_rows, harp_program_surfaces = _inject_harp_program_targets(
            observation_ladder=observation_ladder,
            observation_targets=observation_targets,
            province_axis=province_axis,
            month_axis=month_axis,
        )
        observation_support_bundle = _apply_harp_support_to_bundle(
            observation_support_bundle,
            province_axis=province_axis,
            month_axis=month_axis,
        )
        subgroup_weights, subgroup_summary = _build_subgroup_weights(ctx.run_dir, normalized_rows, province_axis, kp_axis, age_axis, sex_axis)
        archetype_bundle = _build_province_archetype_bundle(
            province_axis=province_axis,
            month_axis=month_axis,
            subgroup_summary=subgroup_summary,
            observation_targets=observation_targets,
        )
        cd4_overlay, cd4_summary = _build_cd4_overlay(observation_targets, province_axis, kp_axis, age_axis, sex_axis)

        if requested_inference_family == "jax_svi":
            fit_result = _fit_rescue_core_jax_svi(
                run_dir=ctx.run_dir,
                profile_id=profile_id,
                observation_targets=observation_targets,
                observation_support_bundle=observation_support_bundle,
                standardized_tensor=standardized_tensor,
                canonical_axis=canonical_axis,
                candidate_profiles=candidate_profiles,
                subgroup_weights=subgroup_weights,
                subgroup_summary=subgroup_summary,
                cd4_overlay=cd4_overlay,
                region_index=region_index,
                province_axis=province_axis,
                month_axis=month_axis,
                kp_axis=kp_axis,
                age_axis=age_axis,
                sex_axis=sex_axis,
                archetype_bundle=archetype_bundle,
                calibration_overrides=calibration_overrides,
            )
        elif torch is not None:
            fit_result = _fit_rescue_core_torch(
                run_dir=ctx.run_dir,
                profile_id=profile_id,
                observation_targets=observation_targets,
                observation_support_bundle=observation_support_bundle,
                observation_ladder=observation_ladder,
                standardized_tensor=standardized_tensor,
                canonical_axis=canonical_axis,
                candidate_profiles=candidate_profiles,
                subgroup_weights=subgroup_weights,
                subgroup_summary=subgroup_summary,
                cd4_overlay=cd4_overlay,
                province_axis=province_axis,
                month_axis=month_axis,
                region_index=region_index,
                kp_axis=kp_axis,
                age_axis=age_axis,
                sex_axis=sex_axis,
                duration_catalog=DURATION_CATALOG,
                requested_inference_family=requested_inference_family,
                archetype_bundle=archetype_bundle,
                calibration_overrides=calibration_overrides,
            )
        else:  # pragma: no cover
            fit_result = _fit_rescue_core_numpy(
                run_dir=ctx.run_dir,
                profile_id=profile_id,
                observation_targets=observation_targets,
                observation_support_bundle=observation_support_bundle,
                standardized_tensor=standardized_tensor,
                canonical_axis=canonical_axis,
                candidate_profiles=candidate_profiles,
                subgroup_weights=subgroup_weights,
                cd4_overlay=cd4_overlay,
                province_axis=province_axis,
                kp_axis=kp_axis,
                age_axis=age_axis,
                sex_axis=sex_axis,
                duration_catalog=DURATION_CATALOG,
                calibration_overrides=calibration_overrides,
            )

    state_estimates = fit_result["state_estimates"]
    forecast_horizon = int(backtest_config.get("forecast_horizon") or 0) if backtest_config else 0
    if forecast_horizon <= 0:
        forecast_horizon = min(6, max(2, len(month_axis)))
    forecast_states = _forecast_from_transitions(state_estimates, fit_result["transition_probs"], subgroup_weights, forecast_horizon=forecast_horizon)
    observation_residuals = _compute_observation_residuals(fit_result["prediction_stack"], observation_targets, province_axis, month_axis)
    reconciliation = _compute_reconciliation_report(state_estimates, observation_targets, region_axis, region_index)
    validation_artifact, benchmark_gate_report, mechanistic_guardrails = _compute_benchmark_metrics(state_estimates, observation_targets, region_axis, region_index, province_axis)
    reference_check_official = _official_reference_check(
        state_estimates=state_estimates,
        province_axis=province_axis,
        month_axis=month_axis,
        run_id=run_id,
        profile_id=profile_id,
        inference_family=fit_result["inference_family"],
    )
    reference_check_harp = _harp_program_check(
        state_estimates=state_estimates,
        prediction_stack=fit_result["prediction_stack"],
        province_axis=province_axis,
        month_axis=month_axis,
        run_id=run_id,
        profile_id=profile_id,
        inference_family=fit_result["inference_family"],
    )
    frozen_history_backtest = (
        _frozen_history_backtest_evaluation(
            run_id=run_id,
            profile_id=profile_id,
            inference_family=fit_result["inference_family"],
            province_axis=province_axis,
            forecast_states=forecast_states,
            parameters_summary=fit_result["parameters_summary"],
            backtest_config=backtest_config,
        )
        if backtest_config
        else None
    )
    national_anchor_surfaces = {
        "month_axis": month_axis,
        **{key: value.round(6).tolist() for key, value in _official_anchor_curves(month_axis).items()},
    }
    linkage_anchor_surfaces = fit_result["parameters_summary"].get("linkage_anchor_curves", {"month_axis": month_axis})
    suppression_anchor_surfaces = fit_result["parameters_summary"].get("suppression_anchor_curves", {"month_axis": month_axis})
    observation_support_summary = {
        "rows": list(observation_support_bundle.get("rows", []) or []),
        "target_order": list(OBSERVATION_ORDER),
        "harp_supported_targets": ["diagnosed_stock", "art_stock", "documented_suppression", "testing_coverage"],
    }

    aggregate_cascade_summary = {
        "latest_first95": round(float(np.mean(1.0 - state_estimates[:, -1, 0])), 6),
        "latest_second95": round(float(np.mean((state_estimates[:, -1, 2] + state_estimates[:, -1, 3]) / np.clip(1.0 - state_estimates[:, -1, 0], 1e-6, None))), 6),
        "latest_trueThird95": round(float(np.mean(state_estimates[:, -1, 3] / np.clip(state_estimates[:, -1, 2] + state_estimates[:, -1, 3], 1e-6, None))), 6),
    }
    transition_parameters = {
        "transition_names": TRANSITION_NAMES,
        "state_catalog": STATE_NAMES,
        "region_axis": region_axis,
        "province_axis": province_axis,
        "mean_transition_probabilities": fit_result["transition_probs"].mean(axis=(0, 1, 2, 3, 4)).round(6).tolist(),
        "parameter_summary": fit_result["parameters_summary"],
    }
    fit_artifact = {
        "profile_id": profile_id,
        "inference_family": fit_result["inference_family"],
        "state_catalog": STATE_NAMES,
        "axis_catalogs": {
            "province": province_axis,
            "region": region_axis,
            "month": month_axis,
            "kp_catalog": kp_axis,
            "age_catalog": age_axis,
            "sex_catalog": sex_axis,
            "cd4_catalog": CD4_CATALOG,
            "duration_catalog": DURATION_CATALOG,
        },
        "transition_names": TRANSITION_NAMES,
        "loss_breakdown": fit_result["loss_breakdown"],
        "aggregate_cascade_summary": aggregate_cascade_summary,
        "subnational_summary": validation_artifact["subnational_metrics"],
        "cd4_summary": cd4_summary,
        "harp_calibration_summary": {
            "verdict": reference_check_harp.get("verdict"),
            "comparison_count": len(reference_check_harp.get("comparisons", [])),
            "vl_test_process_share": fit_result["parameters_summary"].get("vl_test_process_share", []),
        },
        "linkage_calibration_summary": {
            "linkage_penalty": fit_result["loss_breakdown"].get("linkage_penalty", 0.0),
            "linkage_d_to_a_mae": fit_result["loss_breakdown"].get("linkage_d_to_a_mae", 0.0),
            "linkage_second95_mae": fit_result["loss_breakdown"].get("linkage_second95_mae", 0.0),
        },
        "suppression_calibration_summary": {
            "suppression_penalty": fit_result["loss_breakdown"].get("suppression_penalty", 0.0),
            "suppression_a_to_v_mae": fit_result["loss_breakdown"].get("suppression_a_to_v_mae", 0.0),
            "suppression_overall_mae": fit_result["loss_breakdown"].get("suppression_overall_mae", 0.0),
            "suppression_among_art_mae": fit_result["loss_breakdown"].get("suppression_among_art_mae", 0.0),
        },
        "phase4_ready": False,
        "fit_rows": _state_rows(state_estimates, province_axis, month_axis),
        "observation_weight_summary": {row["target_name"]: row["weight"] for row in observation_ladder},
        "observation_support_summary": observation_support_summary,
        "determinant_modifier_summary": fit_result["parameters_summary"].get("determinant_covariates", {}),
        "subgroup_coupling_summary": {
            "kp_coupling_matrix": fit_result["parameters_summary"].get("kp_coupling_matrix"),
            "metapopulation_engine": fit_result["parameters_summary"].get("metapopulation_engine", {}),
        },
        "province_archetype_summary": fit_result["parameters_summary"].get("province_archetypes", {}).get("summary", {}),
        "synthetic_pretraining_summary": fit_result["parameters_summary"].get("synthetic_pretraining", {}),
        "subgroup_prior_learning_summary": fit_result["parameters_summary"].get("subgroup_prior_learning_summary", {}),
        "cd4_prior_learning_summary": fit_result["parameters_summary"].get("cd4_prior_learning_summary", {}),
        "loss_trace_tail": [round(float(value), 6) for value in fit_result["loss_trace"][-12:]],
    }
    if frozen_history_backtest is not None:
        fit_artifact["frozen_history_backtest"] = frozen_history_backtest["summary"]
    validation_artifact["profile_id"] = profile_id
    validation_artifact["phase4_ready"] = False
    validation_artifact["validation_gates"] = benchmark_gate_report["primary_gates"]
    validation_artifact["claim_eligible"] = all(row["passed"] for row in benchmark_gate_report["primary_gates"])
    validation_artifact["harp_program_checks"] = reference_check_harp
    validation_artifact["linkage_program_checks"] = linkage_anchor_surfaces
    validation_artifact["suppression_program_checks"] = suppression_anchor_surfaces
    if frozen_history_backtest is not None:
        validation_artifact["frozen_history_backtest"] = frozen_history_backtest
    mechanistic_guardrails["phase4_ready"] = False

    rescue_core_spec = {
        "profile_id": profile_id,
        "inference_family": fit_result["inference_family"],
        "state_catalog": STATE_NAMES,
        "transition_names": TRANSITION_NAMES,
        "kp_catalog": kp_axis,
        "age_catalog": age_axis,
        "sex_catalog": sex_axis,
        "cd4_catalog": CD4_CATALOG,
        "duration_catalog": DURATION_CATALOG,
        "core_latent_shape": [len(province_axis), len(kp_axis), len(age_axis), len(sex_axis), len(STATE_NAMES), len(DURATION_CATALOG), len(month_axis)],
        "minimal_covariate_hook": ["diagnosed_stock", "art_stock", "documented_suppression"],
        "requested_inference_family": requested_inference_family,
        "resolved_inference_family": fit_result["inference_family"],
        "phase2_hook": {"mode": "transition_modifiers_only", "determinants_active": len(fit_result["parameters_summary"].get("determinant_covariates", {}).get("selected_determinant_modifiers", []))},
        "phase4_ready": False,
    }

    state_estimates_artifact = save_tensor_artifact(array=state_estimates, axis_names=["province", "month", "state"], artifact_dir=phase3_dir, stem="state_estimates", backend="torch" if torch is not None else "numpy", device=fit_result["device"], notes=["phase3_rescue_core_aggregate_state_estimates"], save_pt=False)
    forecast_states_artifact = save_tensor_artifact(array=forecast_states, axis_names=["province", "forecast_step", "state"], artifact_dir=phase3_dir, stem="forecast_states", backend="torch" if torch is not None else "numpy", device=fit_result["device"], notes=["phase3_rescue_core_forecast_states"], save_pt=False)
    cd4_artifact = save_tensor_artifact(array=cd4_overlay, axis_names=["province", "kp_group", "age_band", "sex", "cd4_bin", "month"], artifact_dir=phase3_dir, stem="cd4_overlay_tensor", backend="numpy", device="cpu", notes=["phase3_rescue_core_cd4_overlay"], save_pt=False)

    write_json(phase3_dir / "rescue_core_spec.json", rescue_core_spec)
    write_json(phase3_dir / "observation_ladder.json", observation_ladder)
    write_json(phase3_dir / "observation_targets.json", observation_target_rows)
    write_json(phase3_dir / "observation_residuals.json", observation_residuals)
    write_json(phase3_dir / "observation_reconciliation.json", reconciliation)
    write_json(phase3_dir / "transition_parameters.json", transition_parameters)
    write_json(phase3_dir / "fit_artifact.json", fit_artifact)
    write_json(phase3_dir / "validation_artifact.json", validation_artifact)
    write_json(phase3_dir / "mechanistic_guardrails.json", mechanistic_guardrails)
    write_json(phase3_dir / "benchmark_gate_report.json", benchmark_gate_report)
    write_json(phase3_dir / "reference_check_official.json", reference_check_official)
    write_json(phase3_dir / "reference_check_harp.json", reference_check_harp)
    write_json(phase3_dir / "national_anchor_surfaces.json", national_anchor_surfaces)
    write_json(phase3_dir / "harp_program_surfaces.json", harp_program_surfaces)
    write_json(phase3_dir / "linkage_anchor_surfaces.json", linkage_anchor_surfaces)
    write_json(phase3_dir / "suppression_anchor_surfaces.json", suppression_anchor_surfaces)
    write_json(phase3_dir / "observation_support_summary.json", observation_support_summary)
    write_json(phase3_dir / "determinant_modifiers.json", fit_result["parameters_summary"].get("determinant_covariates", {}))
    write_json(
        phase3_dir / "subgroup_coupling.json",
        {
            "kp_coupling_matrix": fit_result["parameters_summary"].get("kp_coupling_matrix"),
            "metapopulation_engine": fit_result["parameters_summary"].get("metapopulation_engine", {}),
        },
    )
    write_json(phase3_dir / "province_archetype_mixture.json", fit_result["parameters_summary"].get("province_archetypes", {}))
    write_json(phase3_dir / "synthetic_pretraining_summary.json", fit_result["parameters_summary"].get("synthetic_pretraining", {}))
    write_json(phase3_dir / "subgroup_prior_learning_summary.json", fit_result["parameters_summary"].get("subgroup_prior_learning_summary", {}))
    write_json(phase3_dir / "cd4_prior_learning_summary.json", fit_result["parameters_summary"].get("cd4_prior_learning_summary", {}))
    write_json(phase3_dir / "state_estimates_rows.json", _state_rows(state_estimates, province_axis, month_axis))
    write_json(phase3_dir / "cd4_severity_summary.json", cd4_summary)
    write_json(phase3_dir / "subgroup_weight_summary.json", subgroup_summary)
    write_json(phase3_dir / "inference_ready_candidates.json", [])
    write_json(phase3_dir / "state_rows.json", _state_rows(state_estimates, province_axis, month_axis))
    if frozen_history_backtest is not None:
        write_json(
            phase3_dir / "frozen_history_backtest_spec.json",
            {
                "mode": "frozen_history",
                "train_years": list(backtest_config.get("train_years") or []),
                "holdout_years": list(backtest_config.get("holdout_years") or []),
                "train_last_model_month": str(backtest_config.get("train_last_model_month") or ""),
                "forecast_horizon": int(backtest_config.get("forecast_horizon") or 0),
                "holdout_eval_months": list(backtest_config.get("holdout_eval_months") or []),
            },
        )
        write_json(phase3_dir / "frozen_history_backtest_evaluation.json", frozen_history_backtest)
    write_json(
        phase3_dir / "model_artifact.json",
        {
            "model_family": "hiv_rescue_core_v2" if profile_id == RESCUE_V2_PROFILE_ID else "hiv_rescue_core_v1",
            "backend": "torch_map" if torch is not None else "numpy_map",
            "state_names": STATE_NAMES,
            "transition_names": TRANSITION_NAMES,
            "province_axis": province_axis,
            "region_axis": region_axis,
            "month_axis": month_axis,
            "curated_candidate_block_count": len(curated_candidate_blocks),
            "candidate_profile_count": len(candidate_profiles),
        },
    )
    write_json(
        phase3_dir / "forecast_bundle.json",
        {
            "forecast_horizon": forecast_states.shape[1],
            "forecast_rows": _state_rows(forecast_states, province_axis, [f"forecast_h{idx + 1}" for idx in range(forecast_states.shape[1])]),
            "latest_state_by_province": {
                province_axis[idx]: {STATE_NAMES[state_idx]: round(float(state_estimates[idx, -1, state_idx]), 6) for state_idx in range(len(STATE_NAMES))}
                for idx in range(len(province_axis))
            },
        },
    )

    backend_map = detect_backends()
    manifest = Phase0ManifestArtifact(
        plugin_id=plugin_id,
        run_id=run_id,
        generated_at=utc_now_iso(),
        raw_dir=str(ctx.run_dir / "phase0" / "raw"),
        parsed_dir=str(ctx.run_dir / "phase2"),
        extracted_dir=str(phase3_dir),
        index_dir=str(ctx.run_dir / "phase0" / "index"),
        stage_status={"phase3": "completed"},
        artifact_paths={
            "rescue_core_spec": str(phase3_dir / "rescue_core_spec.json"),
            "state_estimates": state_estimates_artifact["value_path"],
            "forecast_states": forecast_states_artifact["value_path"],
            "cd4_overlay_tensor": cd4_artifact["value_path"],
            "transition_parameters": str(phase3_dir / "transition_parameters.json"),
            "fit_artifact": str(phase3_dir / "fit_artifact.json"),
            "validation_artifact": str(phase3_dir / "validation_artifact.json"),
            "observation_ladder": str(phase3_dir / "observation_ladder.json"),
            "observation_targets": str(phase3_dir / "observation_targets.json"),
            "observation_residuals": str(phase3_dir / "observation_residuals.json"),
            "observation_reconciliation": str(phase3_dir / "observation_reconciliation.json"),
            "benchmark_gate_report": str(phase3_dir / "benchmark_gate_report.json"),
            "reference_check_official": str(phase3_dir / "reference_check_official.json"),
            "reference_check_harp": str(phase3_dir / "reference_check_harp.json"),
            "national_anchor_surfaces": str(phase3_dir / "national_anchor_surfaces.json"),
            "harp_program_surfaces": str(phase3_dir / "harp_program_surfaces.json"),
            "linkage_anchor_surfaces": str(phase3_dir / "linkage_anchor_surfaces.json"),
            "suppression_anchor_surfaces": str(phase3_dir / "suppression_anchor_surfaces.json"),
            "observation_support_summary": str(phase3_dir / "observation_support_summary.json"),
            "mechanistic_guardrails": str(phase3_dir / "mechanistic_guardrails.json"),
            "determinant_modifiers": str(phase3_dir / "determinant_modifiers.json"),
            "subgroup_coupling": str(phase3_dir / "subgroup_coupling.json"),
            "province_archetype_mixture": str(phase3_dir / "province_archetype_mixture.json"),
            "synthetic_pretraining_summary": str(phase3_dir / "synthetic_pretraining_summary.json"),
            "subgroup_prior_learning_summary": str(phase3_dir / "subgroup_prior_learning_summary.json"),
            "cd4_prior_learning_summary": str(phase3_dir / "cd4_prior_learning_summary.json"),
            "state_estimates_rows": str(phase3_dir / "state_estimates_rows.json"),
            "cd4_severity_summary": str(phase3_dir / "cd4_severity_summary.json"),
            **(
                {
                    "frozen_history_backtest_spec": str(phase3_dir / "frozen_history_backtest_spec.json"),
                    "frozen_history_backtest_evaluation": str(phase3_dir / "frozen_history_backtest_evaluation.json"),
                }
                if frozen_history_backtest is not None
                else {}
            ),
        },
        backend_status={
            "torch": Phase0BackendStatus("torch", backend_map["torch"].available, torch is not None, notes=fit_result["device"]),
            "jax": Phase0BackendStatus("jax", backend_map["jax"].available, False, notes=backend_map["jax"].device),
        },
        source_count=len(candidate_profiles),
        canonical_candidate_count=len(candidate_profiles),
        numeric_observation_count=len(observation_residuals),
        notes=["phase3_rescue_core:observation_first_hierarchical_map"],
    ).to_dict()
    truth_paths = write_ground_truth_package(
        phase_dir=phase3_dir,
        phase_name="phase3",
        profile_id=profile_id,
        checks=[
            {"name": "state_estimates_finite", "passed": bool(np.isfinite(state_estimates).all())},
            {"name": "forecast_states_finite", "passed": bool(np.isfinite(forecast_states).all())},
            {"name": "cd4_overlay_simplex", "passed": bool(np.allclose(cd4_overlay.sum(axis=4), 1.0, atol=1e-4))},
            {"name": "official_reference_comparisons_present", "passed": bool(reference_check_official.get("comparisons", []))},
            {"name": "harp_program_comparisons_present", "passed": bool(reference_check_harp.get("comparisons", []))},
            {"name": "linkage_anchor_surfaces_present", "passed": bool(linkage_anchor_surfaces.get("month_axis"))},
            {"name": "suppression_anchor_surfaces_present", "passed": bool(suppression_anchor_surfaces.get("month_axis"))},
            {"name": "observation_support_summary_present", "passed": bool(observation_support_summary.get("rows", []))},
            {"name": "province_archetypes_present", "passed": bool(fit_result["parameters_summary"].get("province_archetypes", {}).get("rows", []))},
            {"name": "synthetic_pretraining_present", "passed": bool(fit_result["parameters_summary"].get("synthetic_pretraining", {}).get("library"))},
            {"name": "subgroup_prior_learning_present", "passed": bool(fit_result["parameters_summary"].get("subgroup_prior_learning_summary", {}).get("feature_names", []))},
            {"name": "cd4_prior_learning_present", "passed": bool(fit_result["parameters_summary"].get("cd4_prior_learning_summary", {}).get("age_low_offset", []))},
            {
                "name": "no_duplicate_national_labels",
                "passed": not (("Philippines" in province_axis) and any(is_national_geo(name) for name in province_axis if name != "Philippines")),
            },
            *(
                [
                    {
                        "name": "frozen_history_holdout_present",
                        "passed": bool(frozen_history_backtest.get("holdout_reference_check", {}).get("comparisons", [])),
                    }
                ]
                if frozen_history_backtest is not None
                else []
            ),
            {"name": "phase4_stays_blocked", "passed": fit_artifact["phase4_ready"] is False},
        ],
        truth_sources=["anchor_truth", "benchmark_truth", "prior_truth", "synthetic_truth"],
        stage_manifest_path=str(phase3_dir / "phase3_manifest.json"),
        summary={
            "province_count": len(province_axis),
            "month_count": len(month_axis),
            "comparison_count": len(reference_check_official.get("comparisons", [])),
            "harp_comparison_count": len(reference_check_harp.get("comparisons", [])),
            "primary_gate_pass_count": sum(1 for row in benchmark_gate_report.get("primary_gates", []) if row.get("passed")),
            "primary_gate_count": len(benchmark_gate_report.get("primary_gates", [])),
            "frozen_history_holdout_count": len(frozen_history_backtest.get("holdout_reference_check", {}).get("comparisons", [])) if frozen_history_backtest is not None else 0,
        },
    )
    manifest["artifact_paths"].update(truth_paths)
    write_json(phase3_dir / "phase3_manifest.json", manifest)
    ctx.record_stage_outputs(
        "phase3_build",
        [
            phase3_dir / "rescue_core_spec.json",
            phase3_dir / "state_estimates.npz",
            phase3_dir / "forecast_states.npz",
            phase3_dir / "cd4_overlay_tensor.npz",
            phase3_dir / "transition_parameters.json",
            phase3_dir / "fit_artifact.json",
            phase3_dir / "validation_artifact.json",
            phase3_dir / "observation_ladder.json",
            phase3_dir / "observation_targets.json",
            phase3_dir / "observation_residuals.json",
            phase3_dir / "observation_reconciliation.json",
            phase3_dir / "mechanistic_guardrails.json",
            phase3_dir / "benchmark_gate_report.json",
            phase3_dir / "reference_check_official.json",
            phase3_dir / "reference_check_harp.json",
            phase3_dir / "national_anchor_surfaces.json",
            phase3_dir / "harp_program_surfaces.json",
            phase3_dir / "linkage_anchor_surfaces.json",
            phase3_dir / "suppression_anchor_surfaces.json",
            phase3_dir / "observation_support_summary.json",
            phase3_dir / "determinant_modifiers.json",
            phase3_dir / "subgroup_coupling.json",
            phase3_dir / "province_archetype_mixture.json",
            phase3_dir / "synthetic_pretraining_summary.json",
            phase3_dir / "state_estimates_rows.json",
            *( [phase3_dir / "frozen_history_backtest_spec.json", phase3_dir / "frozen_history_backtest_evaluation.json"] if frozen_history_backtest is not None else [] ),
            phase3_dir / "phase3_manifest.json",
        ],
    )
    return manifest
