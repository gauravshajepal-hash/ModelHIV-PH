from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from epigraph_ph.phase3.shared.numerics import safe_floor


TRANSITION_ORDER = ("U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A")


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


def logit_clip(values: np.ndarray, eps: float) -> np.ndarray:
    used_eps = safe_floor(eps)
    clipped = np.clip(np.asarray(values, dtype=np.float32), used_eps, 1.0 - used_eps)
    return np.log(clipped / np.clip(1.0 - clipped, used_eps, None)).astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-arr))).astype(np.float32)


def point_reference_from_harp(point: dict[str, Any], *, floor: float) -> dict[str, float]:
    used_floor = safe_floor(floor)
    estimated = max(float(point.get("estimated_plhiv") or 0.0), used_floor)
    diagnosed = float(point.get("diagnosed") or 0.0) / estimated
    art = float(point.get("on_art") or 0.0) / estimated
    tested = float(point.get("viral_load_tested") or 0.0) / estimated
    suppressed = float(point.get("suppressed") or 0.0) / estimated
    return {
        "diagnosed_stock": diagnosed,
        "art_stock": art,
        "viral_load_tested_stock": tested,
        "documented_suppression": suppressed,
        "second95": art / max(diagnosed, used_floor),
        "viral_load_tested_among_art": float(point.get("viral_load_tested") or 0.0) / max(float(point.get("on_art") or 0.0), used_floor),
        "suppressed_among_art": float(point.get("suppressed") or 0.0) / max(float(point.get("on_art") or 0.0), used_floor),
    }


def _monthly_hazard_from_step(rate: float, delta_months: int, *, transition_floor: float, transition_ceiling: float) -> float:
    floor = safe_floor(transition_floor)
    ceiling = float(transition_ceiling)
    bounded = float(np.clip(rate, floor, ceiling))
    return float(np.clip(1.0 - (1.0 - bounded) ** (1.0 / max(delta_months, 1)), floor, ceiling))


def derive_compartmental_scaffold(
    *,
    train_harp_points: list[dict[str, Any]],
    month_axis: list[str],
    transition_prior: np.ndarray,
    floor: float,
    transition_floor: float,
    transition_ceiling: float,
    probability_eps: float,
) -> dict[str, Any]:
    transition_prior = np.asarray(transition_prior, dtype=np.float32).reshape(len(TRANSITION_ORDER))
    used_floor = safe_floor(floor)
    used_transition_floor = safe_floor(transition_floor)
    used_probability_eps = safe_floor(probability_eps)
    if len(train_harp_points) < 2 or not month_axis:
        monthly = np.broadcast_to(
            np.clip(transition_prior, used_transition_floor, float(transition_ceiling)).reshape(1, -1),
            (len(month_axis), len(TRANSITION_ORDER)),
        ).astype(np.float32)
        return {
            "monthly_probs": monthly,
            "monthly_logits": logit_clip(monthly, used_probability_eps),
            "annual_rate_rows": [],
            "source": "transition_prior_fallback",
        }

    annual_rows: list[tuple[int, dict[str, float]]] = []
    for point in train_harp_points:
        year = month_year(str(point.get("month") or "")) or int(point.get("year") or 0)
        if year <= 0:
            continue
        annual_rows.append((year, point_reference_from_harp(point, floor=used_floor)))
    annual_rows.sort(key=lambda item: item[0])
    annual_rows = [(year, ref) for idx, (year, ref) in enumerate(annual_rows) if idx == 0 or year != annual_rows[idx - 1][0]]
    if len(annual_rows) < 2:
        monthly = np.broadcast_to(
            np.clip(transition_prior, used_transition_floor, float(transition_ceiling)).reshape(1, -1),
            (len(month_axis), len(TRANSITION_ORDER)),
        ).astype(np.float32)
        return {
            "monthly_probs": monthly,
            "monthly_logits": logit_clip(monthly, used_probability_eps),
            "annual_rate_rows": [],
            "source": "transition_prior_fallback",
        }

    segment_rows: list[dict[str, Any]] = []
    for (prev_year, prev_ref), (next_year, next_ref) in zip(annual_rows[:-1], annual_rows[1:]):
        delta_years = max(next_year - prev_year, 1)
        delta_months = max(delta_years * 12, 1)
        undiagnosed = max(1.0 - float(prev_ref["diagnosed_stock"]), used_floor)
        diagnosed_gap = max(float(prev_ref["diagnosed_stock"]) - float(prev_ref["art_stock"]), used_floor)
        art_unsuppressed = max(float(prev_ref["art_stock"]) - float(prev_ref["documented_suppression"]), used_floor)
        u_to_d = _monthly_hazard_from_step(
            max(float(next_ref["diagnosed_stock"]) - float(prev_ref["diagnosed_stock"]), 0.0) / undiagnosed,
            delta_months,
            transition_floor=used_transition_floor,
            transition_ceiling=float(transition_ceiling),
        )
        d_to_a = _monthly_hazard_from_step(
            max(float(next_ref["art_stock"]) - float(prev_ref["art_stock"]), 0.0) / diagnosed_gap,
            delta_months,
            transition_floor=used_transition_floor,
            transition_ceiling=float(transition_ceiling),
        )
        a_to_v = _monthly_hazard_from_step(
            max(float(next_ref["documented_suppression"]) - float(prev_ref["documented_suppression"]), 0.0) / art_unsuppressed,
            delta_months,
            transition_floor=used_transition_floor,
            transition_ceiling=float(transition_ceiling),
        )
        segment_rows.append(
            {
                "start_year": prev_year,
                "end_year": next_year,
                "delta_months": delta_months,
                "u_to_d": round(u_to_d, 6),
                "d_to_a": round(d_to_a, 6),
                "a_to_v": round(a_to_v, 6),
                "a_to_l": round(float(np.clip(transition_prior[3], used_transition_floor, float(transition_ceiling))), 6),
                "l_to_a": round(float(np.clip(transition_prior[4], used_transition_floor, float(transition_ceiling))), 6),
            }
        )

    monthly = np.zeros((len(month_axis), len(TRANSITION_ORDER)), dtype=np.float32)
    for month_idx, label in enumerate(month_axis):
        year = month_year(label)
        active = None
        for row in segment_rows:
            if year is not None and int(row["start_year"]) <= year <= int(row["end_year"]):
                active = row
                break
        if active is None:
            active = segment_rows[0] if year is not None and year <= int(segment_rows[0]["start_year"]) else segment_rows[-1]
        monthly[month_idx] = np.asarray(
            [active["u_to_d"], active["d_to_a"], active["a_to_v"], active["a_to_l"], active["l_to_a"]],
            dtype=np.float32,
        )
    monthly = np.clip(monthly, used_transition_floor, float(transition_ceiling)).astype(np.float32)
    return {
        "monthly_probs": monthly,
        "monthly_logits": logit_clip(monthly, used_probability_eps),
        "annual_rate_rows": segment_rows,
        "source": "simple_compartmental_scaffold",
    }


def build_temporal_basis(
    month_axis: list[str],
    *,
    slow_knot_months: int,
    medium_block_months: int,
) -> dict[str, np.ndarray]:
    month_count = len(month_axis)
    if month_count == 0:
        return {
            "slow_basis": np.zeros((0, 0), dtype=np.float32),
            "medium_basis": np.zeros((0, 0), dtype=np.float32),
            "year_index": np.zeros((0,), dtype=np.int32),
            "year_labels": np.zeros((0,), dtype=np.int32),
        }
    positions = np.arange(month_count, dtype=np.float32)
    slow_knots = list(range(0, month_count, max(int(slow_knot_months), 1)))
    if slow_knots[-1] != month_count - 1:
        slow_knots.append(month_count - 1)
    slow_basis = np.zeros((month_count, len(slow_knots)), dtype=np.float32)
    for idx, pos in enumerate(positions):
        if len(slow_knots) == 1:
            slow_basis[idx, 0] = 1.0
            continue
        for knot_idx in range(len(slow_knots) - 1):
            left = float(slow_knots[knot_idx])
            right = float(slow_knots[knot_idx + 1])
            if left <= pos <= right:
                width = max(right - left, 1.0)
                slow_basis[idx, knot_idx] = 1.0 - ((pos - left) / width)
                slow_basis[idx, knot_idx + 1] = (pos - left) / width
                break
    medium_block = max(int(medium_block_months), 1)
    medium_blocks = int(np.ceil(month_count / float(medium_block)))
    medium_basis = np.zeros((month_count, medium_blocks), dtype=np.float32)
    for month_idx in range(month_count):
        medium_basis[month_idx, min(month_idx // medium_block, medium_blocks - 1)] = 1.0
    year_labels = np.asarray([month_year(label) or 0 for label in month_axis], dtype=np.int32)
    unique_years = sorted({int(item) for item in year_labels.tolist()})
    year_lookup = {year: idx for idx, year in enumerate(unique_years)}
    year_index = np.asarray([year_lookup[int(year)] for year in year_labels.tolist()], dtype=np.int32)
    return {
        "slow_basis": slow_basis.astype(np.float32),
        "medium_basis": medium_basis.astype(np.float32),
        "year_index": year_index,
        "year_labels": np.asarray(unique_years, dtype=np.int32),
    }


def build_shock_regime_basis(month_axis: list[str], regime_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not month_axis or not regime_rows:
        return {"basis": np.zeros((len(month_axis), 0), dtype=np.float32), "names": []}
    basis = np.zeros((len(month_axis), len(regime_rows)), dtype=np.float32)
    names: list[str] = []
    month_ordinals = [month_ordinal(label) for label in month_axis]
    for regime_idx, row in enumerate(regime_rows):
        names.append(str(row.get("name") or f"regime_{regime_idx}"))
        start = month_ordinal(str(row.get("start_month") or ""))
        end = month_ordinal(str(row.get("end_month") or ""))
        for month_idx, ordinal in enumerate(month_ordinals):
            if ordinal is None:
                continue
            if start is not None and ordinal < start:
                continue
            if end is not None and ordinal > end:
                continue
            basis[month_idx, regime_idx] = 1.0
    return {"basis": basis.astype(np.float32), "names": names}


def _least_squares_projection(series: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if basis.size == 0 or series.size == 0:
        return np.zeros((basis.shape[-1],), dtype=np.float32), np.zeros_like(series, dtype=np.float32)
    coeff, *_ = np.linalg.lstsq(basis.astype(np.float32), series.astype(np.float32), rcond=None)
    coeff = np.asarray(coeff, dtype=np.float32)
    recon = (basis.astype(np.float32) @ coeff).astype(np.float32)
    return coeff.astype(np.float32), recon.astype(np.float32)


def project_covariates_temporally(
    covariates: np.ndarray,
    *,
    slow_basis: np.ndarray,
    medium_basis: np.ndarray,
    fast_enabled: bool,
    fast_proxy_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    covariates = np.asarray(covariates, dtype=np.float32)
    province_count, month_count, cov_count = covariates.shape
    projected = np.zeros_like(covariates, dtype=np.float32)
    slow_energy: list[float] = []
    medium_energy: list[float] = []
    fast_energy: list[float] = []
    for province_idx in range(province_count):
        for cov_idx in range(cov_count):
            series = covariates[province_idx, :, cov_idx]
            _, slow_recon = _least_squares_projection(series, slow_basis)
            medium_input = series - slow_recon
            _, medium_recon = _least_squares_projection(medium_input, medium_basis)
            if fast_enabled:
                fast_recon = series - slow_recon - medium_recon
                if fast_proxy_mask is not None and fast_proxy_mask.shape[:2] == covariates.shape[:2]:
                    fast_recon = fast_recon * fast_proxy_mask[province_idx, :, cov_idx]
            else:
                fast_recon = np.zeros((month_count,), dtype=np.float32)
            projected[province_idx, :, cov_idx] = slow_recon + medium_recon + fast_recon
            slow_energy.append(float(np.mean(np.abs(slow_recon))))
            medium_energy.append(float(np.mean(np.abs(medium_recon))))
            fast_energy.append(float(np.mean(np.abs(fast_recon))))
    return {
        "projected_covariates": projected.astype(np.float32),
        "summary": {
            "available": True,
            "fast_enabled": bool(fast_enabled),
            "mean_slow_energy": round(float(np.mean(slow_energy)) if slow_energy else 0.0, 6),
            "mean_medium_energy": round(float(np.mean(medium_energy)) if medium_energy else 0.0, 6),
            "mean_fast_energy": round(float(np.mean(fast_energy)) if fast_energy else 0.0, 6),
        },
    }


def covariate_group_catalog(covariate_meta: dict[str, Any]) -> dict[str, Any]:
    names = [str(name) for name in list(covariate_meta.get("covariate_names", []) or [])]
    selected = list(covariate_meta.get("selected_determinant_modifiers", []) or [])
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, name in enumerate(names):
        block_name = ""
        if idx >= 3 and idx - 3 < len(selected):
            block_name = str(selected[idx - 3].get("block_name") or "")
        group_name = block_name or str(name.split("::", 1)[0] if "::" in name else "core")
        groups[group_name].append(idx)
    return {
        "groups": {key: value for key, value in groups.items()},
        "rows": [{"group_name": key, "covariate_indices": value, "size": len(value)} for key, value in groups.items()],
    }


def temporal_representation_bundle(
    *,
    train_harp_points: list[dict[str, Any]],
    month_axis: list[str],
    covariates: np.ndarray,
    covariate_meta: dict[str, Any],
    transition_prior: np.ndarray,
    temporal_cfg: dict[str, Any],
    shock_cfg: dict[str, Any],
) -> dict[str, Any]:
    probability_eps = safe_floor(float(temporal_cfg["probability_eps"]))
    transition_floor = safe_floor(float(temporal_cfg["transition_floor"]))
    transition_ceiling = float(temporal_cfg["transition_ceiling"])
    scaffold = derive_compartmental_scaffold(
        train_harp_points=train_harp_points,
        month_axis=month_axis,
        transition_prior=transition_prior,
        floor=float(temporal_cfg["compartment_floor"]),
        transition_floor=transition_floor,
        transition_ceiling=transition_ceiling,
        probability_eps=probability_eps,
    )
    basis = build_temporal_basis(
        month_axis,
        slow_knot_months=int(temporal_cfg["slow_knot_months"]),
        medium_block_months=int(temporal_cfg["medium_block_months"]),
    )
    fast_enabled = bool(temporal_cfg["fast_enabled"])
    projected = project_covariates_temporally(
        covariates,
        slow_basis=np.asarray(basis["slow_basis"], dtype=np.float32),
        medium_basis=np.asarray(basis["medium_basis"], dtype=np.float32),
        fast_enabled=fast_enabled,
    )
    shock_rows = [dict(row) for row in list(shock_cfg.get("regimes", []) or [])]
    shock_basis = build_shock_regime_basis(month_axis, shock_rows)
    return {
        "scaffold": scaffold,
        "basis": basis,
        "shock_basis": shock_basis,
        "projected_covariates": projected["projected_covariates"].astype(np.float32),
        "projection_summary": projected["summary"],
        "group_catalog": covariate_group_catalog(covariate_meta),
        "config_summary": {
            "slow_knot_months": int(temporal_cfg["slow_knot_months"]),
            "medium_block_months": int(temporal_cfg["medium_block_months"]),
            "fast_enabled": fast_enabled,
            "transition_floor": round(float(transition_floor), 8),
            "transition_ceiling": round(float(transition_ceiling), 8),
            "probability_eps": round(float(probability_eps), 8),
            "scaffold_source": scaffold["source"],
            "shock_regime_names": list(shock_basis.get("names", [])),
        },
    }
