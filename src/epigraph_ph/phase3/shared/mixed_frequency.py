from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from epigraph_ph.phase3.shared.numerics import safe_floor


def month_ordinal(month_label: str) -> int | None:
    value = str(month_label or "")
    if len(value) >= 7 and value[:4].isdigit() and value[5:7].isdigit():
        return int(value[:4]) * 12 + int(value[5:7]) - 1
    return None


def month_year(month_label: str) -> int | None:
    ordinal = month_ordinal(month_label)
    if ordinal is None:
        return None
    return ordinal // 12


def latest_month_for_year(month_axis: list[str], year: int) -> str | None:
    matches = [label for label in month_axis if month_year(label) == year]
    if not matches:
        return None
    return max(matches, key=lambda item: month_ordinal(item) or -1)


def point_effective_month(point: dict[str, Any], month_axis: list[str]) -> tuple[str | None, str]:
    precision = str(point.get("temporal_precision") or "").strip().lower()
    effective = str(point.get("effective_month") or "").strip()
    explicit = str(point.get("month") or "").strip()
    year = int(point.get("year") or 0)
    annual_snapshot = precision in {"annual_snapshot", "year_end_snapshot", "annual", "year_end"}
    if effective:
        effective_year = month_year(effective)
        if annual_snapshot and effective_year is not None:
            latest = latest_month_for_year(month_axis, int(effective_year))
            if latest:
                return latest, "year_end_snapshot"
        if effective in month_axis:
            return effective, "source_month"
    if explicit:
        explicit_year = month_year(explicit)
        if annual_snapshot and explicit_year is not None:
            latest = latest_month_for_year(month_axis, int(explicit_year))
            if latest:
                return latest, "year_end_snapshot"
        if explicit in month_axis:
            return explicit, "source_month"
        if explicit_year is not None:
            latest = latest_month_for_year(month_axis, int(explicit_year))
            if latest:
                return latest, "year_end_snapshot"
    if year > 0:
        latest = latest_month_for_year(month_axis, year)
        if latest:
            return latest, "year_end_snapshot"
    return None, "missing"


def resolve_point_month(point: dict[str, Any], month_axis: list[str]) -> tuple[str | None, str]:
    return point_effective_month(point, month_axis)


def _metric_vectors(month_axis: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((len(month_axis),), dtype=np.float32), np.zeros((len(month_axis),), dtype=np.float32)


def _append_anchor_value(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    month_label: str | None,
    month_lookup: dict[str, int],
    value: float | None,
    weight: float,
) -> None:
    if month_label is None or month_label not in month_lookup or value is None:
        return
    idx = month_lookup[month_label]
    values[idx] = float(value)
    weights[idx] = max(weights[idx], float(weight))


def build_mixed_frequency_observation_bundle(
    *,
    month_axis: list[str],
    province_axis: list[str],
    support_bundle: dict[str, Any],
    official_points: list[dict[str, Any]],
    harp_points: list[dict[str, Any]],
    region_axis: list[str] | None = None,
) -> dict[str, Any]:
    del province_axis
    division_floor = safe_floor(None)
    count_floor = 1.0
    month_lookup = {label: idx for idx, label in enumerate(month_axis)}
    region_axis = list(region_axis or [])

    official_diag, official_diag_w = _metric_vectors(month_axis)
    official_art, official_art_w = _metric_vectors(month_axis)
    official_sup, official_sup_w = _metric_vectors(month_axis)
    official_third, official_third_w = _metric_vectors(month_axis)
    harp_diag, harp_diag_w = _metric_vectors(month_axis)
    harp_art, harp_art_w = _metric_vectors(month_axis)
    harp_test, harp_test_w = _metric_vectors(month_axis)
    harp_sup, harp_sup_w = _metric_vectors(month_axis)
    harp_tested_art, harp_tested_art_w = _metric_vectors(month_axis)
    harp_supp_art, harp_supp_art_w = _metric_vectors(month_axis)
    aggregation_modes: dict[str, set[str]] = defaultdict(set)
    annual_anchor_counts: dict[str, int] = defaultdict(int)

    for point in official_points:
        month_label, mode = resolve_point_month(point, month_axis)
        if month_label is None:
            continue
        aggregation_modes["official"].add(mode)
        ref = dict(point.get("reference") or {})
        first95 = ref.get("first95")
        second95 = ref.get("second95")
        overall = ref.get("overall_suppressed")
        third95 = ref.get("documented_suppression_among_art")
        diag_val = float(first95) if first95 is not None else None
        art_val = float(first95) * float(second95) if first95 is not None and second95 is not None else None
        sup_val = float(overall) if overall is not None else (float(first95) * float(second95) * float(third95) if first95 is not None and second95 is not None and third95 is not None else None)
        third_val = (
            float(third95)
            if third95 is not None
            else (
                float(overall) / max(float(first95) * float(second95), division_floor)
                if overall is not None and first95 is not None and second95 is not None
                else None
            )
        )
        _append_anchor_value(official_diag, official_diag_w, month_label=month_label, month_lookup=month_lookup, value=diag_val, weight=1.0)
        _append_anchor_value(official_art, official_art_w, month_label=month_label, month_lookup=month_lookup, value=art_val, weight=1.0)
        _append_anchor_value(official_sup, official_sup_w, month_label=month_label, month_lookup=month_lookup, value=sup_val, weight=1.0)
        _append_anchor_value(official_third, official_third_w, month_label=month_label, month_lookup=month_lookup, value=third_val, weight=1.0)
        for target in ("diagnosed_stock", "art_stock", "documented_suppression", "third95"):
            annual_anchor_counts[target] += 1

    for point in harp_points:
        month_label, mode = resolve_point_month(point, month_axis)
        if month_label is None:
            continue
        aggregation_modes["harp"].add(mode)
        estimated = max(float(point.get("estimated_plhiv") or 0.0), count_floor)
        diagnosed = float(point.get("diagnosed") or 0.0) / estimated
        art = float(point.get("on_art") or 0.0) / estimated
        tested = float(point.get("viral_load_tested") or 0.0) / estimated
        suppressed = float(point.get("suppressed") or 0.0) / estimated
        tested_among_art = float(point.get("viral_load_tested") or 0.0) / max(float(point.get("on_art") or 0.0), count_floor)
        suppressed_among_art = float(point.get("suppressed") or 0.0) / max(float(point.get("on_art") or 0.0), count_floor)
        _append_anchor_value(harp_diag, harp_diag_w, month_label=month_label, month_lookup=month_lookup, value=diagnosed, weight=1.0)
        _append_anchor_value(harp_art, harp_art_w, month_label=month_label, month_lookup=month_lookup, value=art, weight=1.0)
        _append_anchor_value(harp_test, harp_test_w, month_label=month_label, month_lookup=month_lookup, value=tested, weight=1.0)
        _append_anchor_value(harp_sup, harp_sup_w, month_label=month_label, month_lookup=month_lookup, value=suppressed, weight=1.0)
        _append_anchor_value(harp_tested_art, harp_tested_art_w, month_label=month_label, month_lookup=month_lookup, value=tested_among_art, weight=1.0)
        _append_anchor_value(harp_supp_art, harp_supp_art_w, month_label=month_label, month_lookup=month_lookup, value=suppressed_among_art, weight=1.0)
        for target in ("diagnosed_stock", "art_stock", "testing_coverage", "documented_suppression"):
            annual_anchor_counts[target] += 1

    targets = dict(support_bundle.get("targets", {}) or {})
    summary_rows: list[dict[str, Any]] = []
    combined_modes = aggregation_modes.get("official", set()) | aggregation_modes.get("harp", set())
    if {"source_month", "year_end_snapshot"}.issubset(combined_modes):
        aggregation_mode = "mixed"
    elif "source_month" in combined_modes:
        aggregation_mode = "source_month"
    else:
        aggregation_mode = "year_end_snapshot"
    for target_name in ("diagnosed_stock", "art_stock", "documented_suppression", "testing_coverage", "deaths"):
        payload = dict(targets.get(target_name, {}) or {})
        observed_mask = np.asarray(payload.get("observed_mask", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        latent_weight = np.asarray(payload.get("latent_weight", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32)
        summary_rows.append(
            {
                "target_name": target_name,
                "direct_monthly_support_fraction": round(float(np.mean(observed_mask)) if observed_mask.size else 0.0, 6),
                "annual_anchor_count": int(annual_anchor_counts.get(target_name, 0)),
                "anchor_aggregation_mode_used": aggregation_mode,
                "effective_monthly_latent_weight": round(float(np.mean(latent_weight)) if latent_weight.size else 0.0, 6),
            }
        )

    regional_payload = {
        "available": bool(region_axis),
        "region_axis": region_axis,
        "diagnosed_stock": np.zeros((len(region_axis), len(month_axis)), dtype=np.float32).tolist(),
        "art_stock": np.zeros((len(region_axis), len(month_axis)), dtype=np.float32).tolist(),
        "documented_suppression": np.zeros((len(region_axis), len(month_axis)), dtype=np.float32).tolist(),
        "weight": np.zeros((len(region_axis), len(month_axis)), dtype=np.float32).tolist(),
    }
    return {
        "official_anchor_arrays": {
            "diagnosed_stock": official_diag.tolist(),
            "art_stock": official_art.tolist(),
            "documented_suppression": official_sup.tolist(),
            "third95": official_third.tolist(),
            "weight": np.maximum.reduce([official_diag_w, official_art_w, official_sup_w, official_third_w]).tolist() if len(month_axis) else [],
        },
        "harp_anchor_arrays": {
            "diagnosed_stock": harp_diag.tolist(),
            "art_stock": harp_art.tolist(),
            "testing_coverage": harp_test.tolist(),
            "documented_suppression": harp_sup.tolist(),
            "viral_load_tested_among_art": harp_tested_art.tolist(),
            "suppressed_among_art": harp_supp_art.tolist(),
            "weight": np.maximum.reduce([harp_diag_w, harp_art_w, harp_test_w, harp_sup_w, harp_tested_art_w, harp_supp_art_w]).tolist() if len(month_axis) else [],
        },
        "regional_anchor_arrays": regional_payload,
        "summary_rows": summary_rows,
        "aggregation_modes": {key: sorted(value) for key, value in aggregation_modes.items()},
    }
