from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import ensure_dir, read_json, write_json

PRIMARY_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "new_diagnosed_cases_period",
)

SNAPSHOT_METRICS: tuple[str, ...] = (
    "diagnosed_plhiv",
    "alive_on_art",
    "tested_for_viral_load",
    "virally_suppressed",
    "estimated_plhiv",
)

ANNUAL_METRICS: tuple[str, ...] = (
    "annual_new_infections",
    "annual_aids_deaths",
    "estimated_plhiv",
    "art_coverage_percent",
    "population_total",
)

STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "V", "L")
TRANSITION_NAMES: tuple[str, ...] = ("U_to_D", "D_to_A", "A_to_V", "A_to_L", "L_to_A")
OBSERVED_TIERS: frozenset[str] = frozenset({"exact_observed", "bridge_observed"})

_PRIMARY_PRIORITY: dict[str, int] = {
    "official_user_provided_slide": 0,
    "official_mirror": 1,
    "official": 2,
    "official_local_corpus": 2,
    "official_doh_archive": 2,
    "official_unaids_country_data": 2,
    "official_wdi_unaids_hiv_series": 2,
    "model_estimate": 3,
    "derived": 4,
    "other": 5,
}


@dataclass(slots=True)
class QuarterlyDataset:
    holdout_years: list[int]
    train_rows: list[dict[str, Any]]
    holdout_rows: list[dict[str, Any]]
    train_state_rows: list[dict[str, Any]]
    holdout_state_rows: list[dict[str, Any]]
    train_transition_rows: list[dict[str, Any]]
    metric_scales: dict[str, float]
    eps: float


@dataclass(slots=True)
class DynamicControlConfig:
    ridge_penalty: float = 0.1
    rho_clip: float = 0.9
    trend_scale: float = 1.0


@dataclass(slots=True)
class ObservationConfig:
    calibration_ridge: float = 0.1
    share_ridge_penalty: float = 0.1
    share_rho_clip: float = 0.9
    share_trend_scale: float = 1.0


@dataclass(slots=True)
class AnnualIncidenceConfig:
    ridge_penalty: float = 0.1
    inflow_scale_clip: float = 4.0


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_archive_run_id() -> str:
    return "harp-archive-wdi-standard-20260411-s00"


def default_frozen_04d_report() -> Path:
    return repo_root() / "Phase3(dynamic)" / "artifacts" / "runs" / "tr-v3-04d-sandbox-20260410-s00" / "analysis" / "tr_v3_04d_autoresearch_report.json"


def quarter_sort_key(value: str) -> tuple[int, int]:
    year_text, quarter_text = str(value).split("-Q", 1)
    return int(year_text), int(quarter_text)


def quarter_year(value: str) -> int:
    return int(quarter_sort_key(value)[0])


def quarter_ordinal(value: str) -> int:
    year, quarter = quarter_sort_key(value)
    return (int(year) * 4) + int(quarter) - 1


def quarter_gap(previous_quarter: str, current_quarter: str) -> int:
    return int(quarter_ordinal(current_quarter) - quarter_ordinal(previous_quarter))


def inv_logit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(value))))


def logit(value: float, *, eps: float) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / max(1.0 - clipped, eps)))


def softplus(value: float) -> float:
    return float(np.log1p(np.exp(-abs(float(value)))) + max(float(value), 0.0))


def _safe_div(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return 0.0
    denominator_value = float(denominator)
    if abs(denominator_value) <= 1e-9:
        return 0.0
    return float(numerator) / denominator_value


def _priority_key(row: dict[str, Any]) -> tuple[float, float, str]:
    quality = str(row.get("source_quality_tier") or row.get("measurement_class") or "other")
    quality_rank = _PRIMARY_PRIORITY.get(quality, _PRIMARY_PRIORITY["other"])
    confidence = -float(row.get("evidence_confidence") or 0.0)
    source_id = str(row.get("source_id") or "")
    return float(quality_rank), confidence, source_id


def _quarter_from_month(value: str) -> str:
    year_text, month_text = str(value).split("-", 1)
    month = int(month_text[:2])
    return f"{int(year_text):04d}-Q{((month - 1) // 3) + 1}"


def _quarter_end_month_label(quarter: str) -> str:
    year_text, quarter_text = str(quarter).split("-Q", 1)
    month = {1: 3, 2: 6, 3: 9, 4: 12}[int(quarter_text)]
    return f"{int(year_text):04d}-{month:02d}"


def _month_ordinal(value: str) -> int | None:
    text = str(value or "")
    if len(text) < 7 or not text[:4].isdigit() or not text[5:7].isdigit():
        return None
    return (int(text[:4]) * 12) + int(text[5:7]) - 1


def _metric_row_candidates(rows: list[dict[str, Any]], metric_name: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("region") or "").lower() != "national":
            continue
        if str(row.get("metric_name") or "") != metric_name:
            continue
        series_kind = str(row.get("series_kind") or row.get("temporal_precision") or "").lower()
        if "monthly" in series_kind:
            continue
        time_label = str(row.get("time") or row.get("period_end") or "")
        if not time_label:
            continue
        quarter = _quarter_from_month(time_label)
        grouped.setdefault(quarter, []).append(dict(row))
    return grouped


def _choose_primary_rows(candidates_by_key: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for key, rows in candidates_by_key.items():
        if rows:
            chosen[key] = min(rows, key=_priority_key)
    return chosen


def _snapshot_selection_key(quarter: str, row: dict[str, Any]) -> tuple[int, int, float, float, str]:
    time_label = str(row.get("time") or row.get("period_end") or "")
    month_ordinal = _month_ordinal(time_label) or -1
    quarter_end_month = _quarter_end_month_label(quarter)
    quarter_end_rank = 0 if time_label[:7] == quarter_end_month else 1
    quality_rank, confidence, source_id = _priority_key(row)
    return quarter_end_rank, -month_ordinal, quality_rank, confidence, source_id


def _choose_snapshot_rows(candidates_by_key: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    chosen: dict[str, dict[str, Any]] = {}
    for quarter, rows in candidates_by_key.items():
        if rows:
            chosen[quarter] = min(rows, key=lambda row: _snapshot_selection_key(quarter, row))
    return chosen


def _quarterly_diagnosis_flow(rows: list[dict[str, Any]]) -> dict[str, float]:
    period_candidates: dict[str, list[dict[str, Any]]] = {}
    monthly_rows_by_quarter: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("region") or "").lower() != "national":
            continue
        metric_name = str(row.get("metric_name") or "")
        if metric_name == "new_diagnosed_cases_period":
            period_end = str(row.get("period_end") or row.get("time") or "")
            period_start = str(row.get("period_start") or period_end)
            series_kind = str(row.get("series_kind") or row.get("temporal_precision") or "").lower()
            if "quarterly" in series_kind:
                if period_end:
                    quarter = _quarter_from_month(period_end)
                    period_candidates.setdefault(quarter, []).append(dict(row))
                continue
            start_ordinal = _month_ordinal(period_start)
            end_ordinal = _month_ordinal(period_end)
            if start_ordinal is None or end_ordinal is None or end_ordinal < start_ordinal:
                continue
            if start_ordinal == end_ordinal:
                quarter = _quarter_from_month(period_end)
                monthly_row = dict(row)
                monthly_row["time"] = period_end
                monthly_rows_by_quarter.setdefault(quarter, []).append(monthly_row)
                continue
            if _quarter_from_month(period_start) != _quarter_from_month(period_end):
                continue
            if (end_ordinal - start_ordinal + 1) < 2:
                continue
            quarter = _quarter_from_month(period_end)
            period_candidates.setdefault(quarter, []).append(dict(row))
        elif metric_name == "new_diagnosed_cases_monthly":
            time_label = str(row.get("time") or "")
            if not time_label:
                continue
            quarter = _quarter_from_month(time_label)
            monthly_rows_by_quarter.setdefault(quarter, []).append(dict(row))
    flow_map: dict[str, float] = {}
    for quarter, row in _choose_primary_rows(period_candidates).items():
        flow_map[quarter] = float(row.get("value") or 0.0)
    for quarter, qrows in monthly_rows_by_quarter.items():
        if quarter in flow_map:
            continue
        months = {str(row.get("time") or "")[:7] for row in qrows if str(row.get("time") or "")}
        if len(months) < 3:
            continue
        flow_map[quarter] = float(sum(float(row.get("value") or 0.0) for row in qrows))
    return flow_map


def _year_rows(rows: list[dict[str, Any]], metric_name: str) -> dict[int, dict[str, Any]]:
    chosen: dict[int, dict[str, Any]] = {}
    for row in rows:
        if str(row.get("region") or "").lower() != "national":
            continue
        if str(row.get("metric_name") or "") != metric_name:
            continue
        time_label = str(row.get("time") or row.get("period_end") or "")
        if not time_label:
            continue
        year = int(time_label[:4])
        candidate = dict(row)
        previous = chosen.get(year)
        if previous is None or _priority_key(candidate) < _priority_key(previous):
            chosen[year] = candidate
    return chosen


def load_archive_rows(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    root = repo or repo_root()
    archive_path = root / "artifacts" / "runs" / archive_run_id / "harp_archive" / "historical_metric_rows.json"
    return list(read_json(archive_path, default=[]) or [])


def build_quarterly_observation_rows(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    metric_rows = load_archive_rows(archive_run_id, repo=repo)
    snapshot_maps = {
        metric_name: _choose_snapshot_rows(_metric_row_candidates(metric_rows, metric_name))
        for metric_name in SNAPSHOT_METRICS
    }
    quarterly_flow_map = _quarterly_diagnosis_flow(metric_rows)
    quarter_set = sorted({quarter for snapshot in snapshot_maps.values() for quarter in snapshot.keys()}, key=quarter_sort_key)
    rows: list[dict[str, Any]] = []
    for quarter in quarter_set:
        diagnosed_row = snapshot_maps["diagnosed_plhiv"].get(quarter)
        alive_row = snapshot_maps["alive_on_art"].get(quarter)
        if diagnosed_row is None or alive_row is None:
            continue
        row: dict[str, Any] = {
            "quarter": quarter,
            "diagnosed_plhiv": float(diagnosed_row.get("value") or 0.0),
            "alive_on_art": float(alive_row.get("value") or 0.0),
            "new_diagnosed_cases_period": float(quarterly_flow_map[quarter]) if quarter in quarterly_flow_map else None,
        }
        for metric_name in ("tested_for_viral_load", "virally_suppressed", "estimated_plhiv"):
            candidate = snapshot_maps[metric_name].get(quarter)
            row[metric_name] = float(candidate.get("value") or 0.0) if candidate is not None else None
        rows.append(row)
    return rows


def build_annual_anchor_rows(archive_run_id: str, *, repo: Path | None = None) -> list[dict[str, Any]]:
    metric_rows = load_archive_rows(archive_run_id, repo=repo)
    chosen: dict[tuple[str, int], dict[str, Any]] = {}
    for metric_name in ANNUAL_METRICS:
        for year, row in _year_rows(metric_rows, metric_name).items():
            chosen[(metric_name, year)] = {
                "metric_name": metric_name,
                "year": int(year),
                "value": float(row.get("value") or 0.0),
                "source_quality_tier": str(row.get("source_quality_tier") or row.get("measurement_class") or "other"),
                "evidence_confidence": float(row.get("evidence_confidence") or 0.0),
                "source_id": str(row.get("source_id") or ""),
            }
    return sorted(chosen.values(), key=lambda row: (str(row["metric_name"]), int(row["year"])))


def _coverage_mean(rows: list[dict[str, Any]]) -> float:
    coverages = []
    for row in rows:
        estimated = row.get("estimated_plhiv")
        diagnosed = row.get("diagnosed_plhiv")
        if estimated is None or diagnosed is None:
            continue
        estimated_value = float(estimated)
        if estimated_value > 0.0:
            coverages.append(float(diagnosed) / estimated_value)
    return float(np.mean(coverages)) if coverages else 0.5


def _share_mean(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float:
    shares = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value > 0.0:
            shares.append(float(numerator) / denominator_value)
    return float(np.mean(shares)) if shares else 0.0


def _lost_gap_share(rows: list[dict[str, Any]]) -> float:
    shares = []
    for previous, current in zip(rows[:-1], rows[1:]):
        diagnosed_gap = max(float(previous.get("diagnosed_plhiv") or 0.0) - float(previous.get("alive_on_art") or 0.0), 0.0)
        art_drop = max(float(previous.get("alive_on_art") or 0.0) - float(current.get("alive_on_art") or 0.0), 0.0)
        if diagnosed_gap > 0.0 and art_drop > 0.0:
            shares.append(art_drop / diagnosed_gap)
    return float(np.mean(shares)) if shares else 0.0


def _row_metric_tiers(row: dict[str, Any]) -> dict[str, str]:
    metric_names = SNAPSHOT_METRICS + ("new_diagnosed_cases_period",)
    return {metric_name: str(row.get(f"{metric_name}_tier") or "exact_observed") for metric_name in metric_names}


def _tier_is_observed(value: str | None) -> bool:
    return str(value or "") in OBSERVED_TIERS


def _transition_support_metadata(previous: dict[str, Any], current: dict[str, Any]) -> tuple[dict[str, bool], dict[str, str]]:
    previous_tiers = dict(previous.get("metric_tiers") or {})
    current_tiers = dict(current.get("metric_tiers") or {})
    diagnosed_support = _tier_is_observed(previous_tiers.get("diagnosed_plhiv")) and _tier_is_observed(current_tiers.get("diagnosed_plhiv"))
    art_support = _tier_is_observed(previous_tiers.get("alive_on_art")) and _tier_is_observed(current_tiers.get("alive_on_art"))
    flow_support = _tier_is_observed(current_tiers.get("new_diagnosed_cases_period"))
    suppressed_support = (
        _tier_is_observed(previous_tiers.get("virally_suppressed"))
        and _tier_is_observed(current_tiers.get("virally_suppressed"))
        and art_support
    )
    net_leakage_support = art_support and flow_support
    support_flags = {
        "U_to_D": bool(flow_support),
        "D_to_A": bool(diagnosed_support and art_support),
        "A_to_V": bool(suppressed_support),
        "A_to_L": bool(net_leakage_support),
        "L_to_A": bool(net_leakage_support),
    }
    support_sources = {
        "U_to_D": "observed_diagnosis_flow" if flow_support else "unsupported_diagnosis_flow",
        "D_to_A": "observed_diagnosed_and_art_stocks" if diagnosed_support and art_support else "unsupported_stock_pair",
        "A_to_V": "observed_suppression_and_art_stocks" if suppressed_support else "unsupported_suppression_pair",
        "A_to_L": "net_art_stock_plus_diagnosis_flow" if net_leakage_support else "unsupported_leakage_signal",
        "L_to_A": "net_art_stock_plus_diagnosis_flow" if net_leakage_support else "unsupported_leakage_signal",
    }
    return support_flags, support_sources


def build_state_rows(observation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage_mean = _coverage_mean(observation_rows)
    suppression_share_mean = _share_mean(observation_rows, "virally_suppressed", "alive_on_art")
    lost_gap_share = _lost_gap_share(observation_rows)
    rows: list[dict[str, Any]] = []
    for row in observation_rows:
        diagnosed = max(float(row.get("diagnosed_plhiv") or 0.0), 0.0)
        alive_on_art = max(float(row.get("alive_on_art") or 0.0), 0.0)
        observed_suppressed = row.get("virally_suppressed")
        virally_suppressed = min(
            alive_on_art,
            max(float(observed_suppressed), 0.0) if observed_suppressed is not None else suppression_share_mean * alive_on_art,
        )
        diagnosed_gap = max(diagnosed - alive_on_art, 0.0)
        lost = min(diagnosed_gap, max(lost_gap_share * diagnosed_gap, 0.0))
        diagnosed_not_on_art = max(diagnosed_gap - lost, 0.0)
        engaged_unsuppressed = max(alive_on_art - virally_suppressed, 0.0)
        estimated_total = row.get("estimated_plhiv")
        if estimated_total is None or float(estimated_total) <= 0.0:
            estimated_total = diagnosed / max(coverage_mean, 1e-6)
        estimated_total = max(float(estimated_total), diagnosed)
        undiagnosed = max(float(estimated_total) - diagnosed, 0.0)
        rows.append(
            {
                "quarter": str(row.get("quarter") or ""),
                "diagnosed_plhiv": diagnosed,
                "alive_on_art": alive_on_art,
                "new_diagnosed_cases_period": row.get("new_diagnosed_cases_period"),
                "tested_for_viral_load": row.get("tested_for_viral_load"),
                "virally_suppressed": row.get("virally_suppressed"),
                "estimated_plhiv": estimated_total,
                "metric_tiers": _row_metric_tiers(row),
                "row_tier": str(row.get("row_tier") or "exact_observed"),
                "score_eligible": bool(row.get("score_eligible") or False),
                "score_eligible_metrics": list(row.get("score_eligible_metrics") or []),
                "state_values": {
                    "U": float(undiagnosed),
                    "D": float(diagnosed_not_on_art),
                    "A": float(engaged_unsuppressed),
                    "V": float(virally_suppressed),
                    "L": float(lost),
                },
            }
        )
    return rows


def _u_to_d_flow(current: dict[str, Any], previous_state: dict[str, float], next_state: dict[str, float]) -> float:
    flow = current.get("new_diagnosed_cases_period")
    if flow is not None:
        return max(float(flow), 0.0)
    diagnosed_delta = max(
        float(next_state["D"] + next_state["A"] + next_state["V"] + next_state["L"])
        - float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"]),
        0.0,
    )
    return float(diagnosed_delta)


def transition_rows(state_rows: list[dict[str, Any]], *, eps: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for previous, current in zip(state_rows[:-1], state_rows[1:]):
        gap_quarters = max(quarter_gap(str(previous["quarter"]), str(current["quarter"])), 1)
        previous_state = dict(previous["state_values"])
        current_state = dict(current["state_values"])
        support_flags, support_sources = _transition_support_metadata(previous, current)
        u_to_d = _u_to_d_flow(current, previous_state, current_state)
        d_to_a = max(float(previous_state["D"]) + u_to_d - float(current_state["D"]), 0.0)
        a_to_v = max(float(current_state["V"]) - float(previous_state["V"]), 0.0)
        l_delta = float(current_state["L"]) - float(previous_state["L"])
        if l_delta >= 0.0:
            a_to_l = float(l_delta)
            l_to_a = 0.0
        else:
            a_to_l = 0.0
            l_to_a = float(-l_delta)
        flows = {
            "U_to_D": float(u_to_d),
            "D_to_A": float(d_to_a),
            "A_to_V": float(a_to_v),
            "A_to_L": float(a_to_l),
            "L_to_A": float(l_to_a),
        }
        denominators = {
            "U_to_D": max(float(previous_state["U"]), eps),
            "D_to_A": max(float(previous_state["D"]), eps),
            "A_to_V": max(float(previous_state["A"]), eps),
            "A_to_L": max(float(previous_state["A"]), eps),
            "L_to_A": max(float(previous_state["L"]), eps),
        }
        rows.append(
            {
                "quarter": str(current["quarter"]),
                "previous_quarter": str(previous["quarter"]),
                "gap_quarters": int(gap_quarters),
                "flows": flows,
                "hazards": {name: float(flows[name]) / denominators[name] for name in TRANSITION_NAMES},
                "state_values_previous": {name: float(previous_state[name]) for name in STATE_NAMES},
                "state_values_current": {name: float(current_state[name]) for name in STATE_NAMES},
                "support_flags": dict(support_flags),
                "support_sources": dict(support_sources),
                "current_metric_tiers": dict(current.get("metric_tiers") or {}),
                "previous_metric_tiers": dict(previous.get("metric_tiers") or {}),
            }
        )
    return rows


def build_quarterly_dataset(observation_rows: list[dict[str, Any]], holdout_years: list[int]) -> QuarterlyDataset:
    min_holdout_year = min(holdout_years)
    train_rows = [dict(row) for row in observation_rows if quarter_year(str(row["quarter"])) < min_holdout_year]
    holdout_rows = [dict(row) for row in observation_rows if quarter_year(str(row["quarter"])) in set(holdout_years)]
    full_state_rows = build_state_rows(train_rows + holdout_rows)
    train_state_rows = [row for row in full_state_rows if quarter_year(str(row["quarter"])) < min_holdout_year]
    holdout_state_rows = [row for row in full_state_rows if quarter_year(str(row["quarter"])) in set(holdout_years)]
    eps = float(np.finfo(np.float32).eps)
    train_transition_rows = [
        dict(row)
        for row in transition_rows(full_state_rows, eps=eps)
        if quarter_year(str(row["quarter"])) < min_holdout_year and int(row.get("gap_quarters") or 1) == 1
    ]
    metric_scales = {}
    for metric_name in PRIMARY_METRICS:
        observed = [abs(float(row.get(metric_name) or 0.0)) for row in train_rows + holdout_rows if row.get(metric_name) is not None]
        metric_scales[metric_name] = max(observed) if observed else eps
    return QuarterlyDataset(
        holdout_years=list(holdout_years),
        train_rows=train_rows,
        holdout_rows=holdout_rows,
        train_state_rows=train_state_rows,
        holdout_state_rows=holdout_state_rows,
        train_transition_rows=train_transition_rows,
        metric_scales=metric_scales,
        eps=eps,
    )


def rolling_origin_year_splits(years: list[int], *, start_year: int, end_year: int, min_train_years: int, horizon_years: int) -> list[dict[str, Any]]:
    bounded_years = [year for year in sorted(set(years)) if int(start_year) <= int(year) <= int(end_year)]
    splits: list[dict[str, Any]] = []
    for train_end_year in bounded_years:
        train_years = [year for year in bounded_years if year <= train_end_year]
        holdout_years = [year for year in bounded_years if train_end_year < year <= train_end_year + int(horizon_years)]
        if len(train_years) < int(min_train_years) or not holdout_years:
            continue
        splits.append({"train_end_year": int(train_end_year), "train_years": train_years, "holdout_years": holdout_years})
    return splits

def _lag_diff(values: list[float], lag: int = 1) -> list[float]:
    out = [0.0] * len(values)
    for idx in range(lag, len(values)):
        out[idx] = float(values[idx] - values[idx - lag])
    return out


def _log1p_series(values: list[float]) -> list[float]:
    return [float(math.log1p(max(value, 0.0))) for value in values]


def _pca_first_component(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0 or matrix.shape[0] == 0:
        return np.zeros((matrix.shape[0],), dtype=np.float64)
    standardized = np.asarray(matrix, dtype=np.float64)
    means = standardized.mean(axis=0)
    stds = standardized.std(axis=0)
    stds = np.where(stds > 1e-9, stds, 1.0)
    standardized = (standardized - means) / stds
    if standardized.shape[1] == 1:
        return standardized[:, 0]
    _u, _s, vt = np.linalg.svd(standardized, full_matrices=False)
    component = vt[0]
    if component[0] < 0.0:
        component = -component
    return standardized @ component


def build_control_channels(train_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    diagnosed = [float(row.get("diagnosed_plhiv") or 0.0) for row in train_rows]
    alive = [float(row.get("alive_on_art") or 0.0) for row in train_rows]
    estimated = [float(row.get("estimated_plhiv") or 0.0) for row in train_rows]
    tested = [float(row.get("tested_for_viral_load") or 0.0) for row in train_rows]
    suppressed = [float(row.get("virally_suppressed") or 0.0) for row in train_rows]

    diagnosed_log = _log1p_series(diagnosed)
    alive_log = _log1p_series(alive)
    testing_share = [_safe_div(test, art) for test, art in zip(tested, alive)]
    suppression_share = [_safe_div(supp, art) for supp, art in zip(suppressed, alive)]
    diagnosed_coverage = [_safe_div(diag, est) for diag, est in zip(diagnosed, estimated)]
    art_coverage = [_safe_div(art, diag) for art, diag in zip(alive, diagnosed)]
    undiagnosed_share = [_safe_div(max(est - diag, 0.0), est) for est, diag in zip(estimated, diagnosed)]
    care_load = [_safe_div(art, est) for art, est in zip(alive, estimated)]

    diagnosed_growth = _lag_diff(diagnosed_log, 1)
    alive_growth = _lag_diff(alive_log, 1)
    diagnosed_accel = _lag_diff(diagnosed_growth, 1)
    alive_accel = _lag_diff(alive_growth, 1)
    testing_share_change = _lag_diff(testing_share, 1)
    diag_alive_gap_change = [dg - ag for dg, ag in zip(diagnosed_growth, alive_growth)]

    a_matrix = np.asarray(list(zip(diagnosed_coverage, diagnosed_growth, undiagnosed_share)), dtype=np.float64)
    c_matrix = np.asarray(list(zip(art_coverage, suppression_share, testing_share, care_load)), dtype=np.float64)
    r_matrix = np.asarray(list(zip(diagnosed_accel, alive_accel, testing_share_change, diag_alive_gap_change)), dtype=np.float64)
    return {
        "A": list(_pca_first_component(a_matrix)),
        "C": list(_pca_first_component(c_matrix)),
        "R": list(_pca_first_component(r_matrix)),
    }


def _fit_scalar_ar_trend(train_quarters: list[str], train_values: list[float], forecast_quarters: list[str], *, ridge_penalty: float, rho_clip: float, trend_scale: float) -> dict[str, float]:
    if not train_values:
        return {quarter: 0.0 for quarter in forecast_quarters}
    if len(train_values) < 2:
        return {quarter: float(train_values[-1]) for quarter in forecast_quarters}
    all_quarters = sorted(set(train_quarters + forecast_quarters), key=quarter_sort_key)
    base_position = quarter_ordinal(all_quarters[0])
    positions = {quarter: float(quarter_ordinal(quarter) - base_position) for quarter in all_quarters}
    y = np.asarray(train_values[1:], dtype=np.float64)
    x = np.asarray([[1.0, positions[quarter] * trend_scale, train_values[idx - 1]] for idx, quarter in enumerate(train_quarters[1:], start=1)], dtype=np.float64)
    ridge = np.diag([0.0, ridge_penalty, ridge_penalty])
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    alpha = float(beta[0])
    slope = float(beta[1])
    rho = float(np.clip(beta[2], -rho_clip, rho_clip))
    out: dict[str, float] = {}
    last_value = float(train_values[-1])
    for quarter in forecast_quarters:
        predicted = alpha + slope * positions[quarter] * trend_scale + rho * last_value
        out[quarter] = float(predicted)
        last_value = float(predicted)
    return out


def _fit_logit_transition_model(
    train_transition_rows: list[dict[str, Any]],
    forecast_quarters: list[str],
    transition_name: str,
    controls_train: dict[str, list[float]],
    *,
    cfg: DynamicControlConfig,
    eps: float,
) -> dict[str, float]:
    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    train_hazards = [float(row["hazards"][transition_name]) for row in train_transition_rows]
    if len(train_hazards) < 2:
        constant = float(train_hazards[-1]) if train_hazards else 0.0
        return {quarter: constant for quarter in forecast_quarters}
    control_names = {
        "U_to_D": ("A", "R"),
        "D_to_A": ("C", "R"),
        "A_to_V": ("C", "R"),
        "A_to_L": ("C", "R"),
        "L_to_A": ("C", "R"),
    }[transition_name]
    all_quarters = sorted(set(train_quarters + forecast_quarters), key=quarter_sort_key)
    base_position = quarter_ordinal(all_quarters[0])
    quarter_positions = {quarter: float(quarter_ordinal(quarter) - base_position) for quarter in all_quarters}
    etas = [logit(value, eps=eps) for value in train_hazards]
    y = np.asarray(etas[1:], dtype=np.float64)
    x_rows = []
    for idx in range(1, len(train_quarters)):
        quarter = train_quarters[idx]
        row = [1.0, quarter_positions[quarter] * cfg.trend_scale, etas[idx - 1]]
        for control_name in control_names:
            row.append(float(controls_train[control_name][idx]))
        x_rows.append(row)
    x = np.asarray(x_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    beta[2] = float(np.clip(beta[2], -cfg.rho_clip, cfg.rho_clip))
    control_forecasts = {
        control_name: _fit_scalar_ar_trend(train_quarters, controls_train[control_name], forecast_quarters, ridge_penalty=cfg.ridge_penalty, rho_clip=cfg.rho_clip, trend_scale=cfg.trend_scale)
        for control_name in control_names
    }
    last_eta = float(etas[-1])
    hazards: dict[str, float] = {}
    for quarter in forecast_quarters:
        row = [1.0, quarter_positions[quarter] * cfg.trend_scale, last_eta]
        row.extend(float(control_forecasts[control_name][quarter]) for control_name in control_names)
        eta = float(np.dot(np.asarray(row, dtype=np.float64), beta))
        hazards[quarter] = float(inv_logit(eta))
        last_eta = eta
    return hazards


def _quarter_positions(rows: list[dict[str, Any]]) -> dict[str, int]:
    quarters = sorted({str(row["quarter"]) for row in rows}, key=quarter_sort_key)
    if not quarters:
        return {}
    base_position = quarter_ordinal(quarters[0])
    return {quarter: int(quarter_ordinal(quarter) - base_position) for quarter in quarters}


def _fit_linear_calibration(raw_values: list[float], observed_values: list[float], positions: list[float], ridge_penalty: float) -> tuple[dict[str, float], float]:
    if len(raw_values) < 2 or len(observed_values) < 2:
        return {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0}, 0.0
    x = np.asarray([[1.0, raw, pos] for raw, pos in zip(raw_values, positions)], dtype=np.float64)
    y = np.asarray(observed_values, dtype=np.float64)
    ridge = np.diag([0.0, ridge_penalty, ridge_penalty])
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    fitted = x @ beta
    raw_mae = float(np.mean(np.abs(y - np.asarray(raw_values, dtype=np.float64))))
    calibrated_mae = float(np.mean(np.abs(y - fitted)))
    gain = max(raw_mae - calibrated_mae, 0.0) / max(raw_mae, 1e-6) if raw_mae > 0.0 else 0.0
    return {"intercept": float(beta[0]), "raw_scale": float(beta[1]), "time_slope": float(beta[2])}, float(gain)


def _apply_linear_calibration(raw_value: float, position: float, coeffs: dict[str, float]) -> float:
    return float(coeffs["intercept"] + coeffs["raw_scale"] * raw_value + coeffs["time_slope"] * position)


def fit_observation_model(dataset: QuarterlyDataset, train_hazard_map: dict[str, dict[str, float]], cfg: ObservationConfig) -> dict[str, Any]:
    train_rows = sorted(list(dataset.train_rows), key=lambda row: quarter_sort_key(str(row["quarter"])))
    if len(train_rows) < 3:
        return {"coefficients": {}, "shares": {}}
    initial_state = dict(dataset.train_state_rows[0]["state_values"])
    raw_prediction_rows = simulate_closed_flow(initial_state, train_rows[1:], train_hazard_map)["prediction_rows"]
    target_rows = train_rows[1:]
    quarter_positions = _quarter_positions(train_rows + list(dataset.holdout_rows))
    coefficients: dict[str, dict[str, float]] = {}
    for metric_name in ("diagnosed_plhiv", "alive_on_art"):
        coeffs, _gain = _fit_linear_calibration(
            [float(row.get(metric_name) or 0.0) for row in raw_prediction_rows],
            [float(row.get(metric_name) or 0.0) for row in target_rows],
            [float(quarter_positions[str(row["quarter"])]) for row in target_rows],
            cfg.calibration_ridge,
        )
        coefficients[metric_name] = coeffs
    return {"coefficients": coefficients}


def apply_observation_model(prediction_rows: list[dict[str, float]], model: dict[str, Any], *, positions: dict[str, int]) -> list[dict[str, float]]:
    coeffs = dict(model.get("coefficients") or {})
    out: list[dict[str, float]] = []
    for row in prediction_rows:
        quarter = str(row["quarter"])
        position = float(positions.get(quarter, 0))
        diagnosed_raw = float(row.get("diagnosed_plhiv") or 0.0)
        alive_raw = float(row.get("alive_on_art") or 0.0)
        diagnosed = _apply_linear_calibration(diagnosed_raw, position, coeffs.get("diagnosed_plhiv") or {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0})
        alive = _apply_linear_calibration(alive_raw, position, coeffs.get("alive_on_art") or {"intercept": 0.0, "raw_scale": 1.0, "time_slope": 0.0})
        alive = max(alive, 0.0)
        diagnosed = max(diagnosed, alive)
        out.append({**row, "diagnosed_plhiv": float(diagnosed), "alive_on_art": float(alive)})
    return out


def carry_forward_hazards(dataset: QuarterlyDataset) -> dict[str, dict[str, float]]:
    holdout_quarters = [str(row["quarter"]) for row in sorted(dataset.holdout_rows, key=lambda row: quarter_sort_key(str(row["quarter"]))) ]
    if not dataset.train_transition_rows:
        return {quarter: {transition: 0.0 for transition in TRANSITION_NAMES} for quarter in holdout_quarters}
    baseline = {transition: float(dataset.train_transition_rows[-1]["hazards"][transition]) for transition in TRANSITION_NAMES}
    return {quarter: dict(baseline) for quarter in holdout_quarters}


def simulate_closed_flow(initial_state: dict[str, float], target_rows: list[dict[str, Any]], hazard_map: dict[str, dict[str, float]]) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(hazard_map.get(quarter) or {})
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        d_to_a = float(hazards.get("D_to_A") or 0.0) * max(float(previous_state["D"]), 0.0)
        a_to_v = float(hazards.get("A_to_V") or 0.0) * max(float(previous_state["A"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)
        current_state = {
            "U": max(float(previous_state["U"]) - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        prediction_rows.append({
            "quarter": quarter,
            "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
            "alive_on_art": float(current_state["A"] + current_state["V"]),
            "new_diagnosed_cases_period": float(u_to_d),
            "tested_for_viral_load": None,
            "virally_suppressed": float(current_state["V"]),
        })
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def normalized_mae(prediction_rows: list[dict[str, float]], target_rows: list[dict[str, Any]], metric_scales: dict[str, float], *, eps: float) -> float:
    errors: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            errors.append(abs(float(prediction_value) - float(target_value)) / max(float(metric_scales.get(metric_name) or eps), eps))
    return float(np.mean(errors)) if errors else float("inf")


def smape(prediction_rows: list[dict[str, float]], target_rows: list[dict[str, Any]], *, eps: float) -> float:
    scores: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            denom = abs(float(prediction_value)) + abs(float(target_value))
            if denom > eps:
                scores.append((2.0 * abs(float(prediction_value) - float(target_value))) / denom)
    return float(np.mean(scores)) if scores else 0.0

def _annual_metric_map(annual_rows: list[dict[str, Any]], metric_name: str) -> dict[int, float]:
    return {int(row["year"]): float(row["value"]) for row in annual_rows if str(row["metric_name"]) == metric_name}


def _aggregate_yearly_quarter_rows(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(quarter_year(str(row["quarter"])), []).append(dict(row))
    return {year: sorted(grouped_rows, key=lambda row: quarter_sort_key(str(row["quarter"]))) for year, grouped_rows in grouped.items()}


def _yearly_feature_rows(observation_rows: list[dict[str, Any]], annual_rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    incidence_map = _annual_metric_map(annual_rows, "annual_new_infections")
    death_map = _annual_metric_map(annual_rows, "annual_aids_deaths")
    plhiv_map = _annual_metric_map(annual_rows, "estimated_plhiv")
    art_cov_map = _annual_metric_map(annual_rows, "art_coverage_percent")
    population_map = _annual_metric_map(annual_rows, "population_total")
    grouped = _aggregate_yearly_quarter_rows(observation_rows)
    years = sorted(set(incidence_map) & set(plhiv_map))
    rows: list[dict[str, float]] = []
    for year in years:
        qrows = grouped.get(year, [])
        if qrows:
            diagnosed_cov_mean = float(np.mean([_safe_div(row.get("diagnosed_plhiv"), row.get("estimated_plhiv")) for row in qrows]))
            care_cov_mean = float(np.mean([_safe_div(row.get("alive_on_art"), row.get("diagnosed_plhiv")) for row in qrows]))
            suppression_mean = float(np.mean([_safe_div(row.get("virally_suppressed"), row.get("alive_on_art")) for row in qrows]))
        else:
            diagnosed_cov_mean = 0.0
            care_cov_mean = 0.0
            suppression_mean = 0.0
        rows.append(
            {
                "year": float(year),
                "annual_new_infections": float(incidence_map[year]),
                "annual_aids_deaths": float(death_map.get(year, 0.0)),
                "estimated_plhiv": float(plhiv_map[year]),
                "art_coverage_percent": float(art_cov_map.get(year, 0.0)),
                "population_total": float(population_map.get(year, 0.0)),
                "uninfected_population_total": max(float(population_map.get(year, 0.0)) - float(plhiv_map[year]), 0.0),
                "diagnosed_coverage_mean": diagnosed_cov_mean,
                "care_coverage_mean": care_cov_mean,
                "suppression_share_mean": suppression_mean,
            }
        )
    rows.sort(key=lambda row: int(row["year"]))
    for idx, row in enumerate(rows):
        row["prev_annual_new_infections"] = float(rows[idx - 1]["annual_new_infections"]) if idx > 0 else float(row["annual_new_infections"])
    return rows


def _fit_annual_incidence_model(train_rows: list[dict[str, float]], holdout_years: list[int], cfg: AnnualIncidenceConfig) -> dict[int, float]:
    if not train_rows:
        return {year: 0.0 for year in holdout_years}
    feature_names = [
        "year",
        "estimated_plhiv",
        "population_total",
        "uninfected_population_total",
        "prev_annual_new_infections",
        "art_coverage_percent",
        "diagnosed_coverage_mean",
        "care_coverage_mean",
        "suppression_share_mean",
    ]
    x = np.asarray([[1.0] + [math.log1p(max(float(row[name]), 0.0)) if name != "year" else float(row[name]) for name in feature_names] for row in train_rows], dtype=np.float64)
    y = np.asarray([math.log1p(max(float(row["annual_new_infections"]), 0.0)) for row in train_rows], dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    by_year = {int(row["year"]): dict(row) for row in train_rows}
    latest = dict(train_rows[-1])
    forecasts: dict[int, float] = {}
    for year in holdout_years:
        row = dict(latest)
        row["year"] = float(year)
        if year in by_year:
            row.update(by_year[year])
        row["prev_annual_new_infections"] = float(forecasts.get(year - 1, latest["annual_new_infections"] if year - 1 <= int(latest["year"]) else latest["prev_annual_new_infections"]))
        x_row = np.asarray([1.0] + [math.log1p(max(float(row[name]), 0.0)) if name != "year" else float(row[name]) for name in feature_names], dtype=np.float64)
        value = float(np.expm1(np.dot(x_row, beta)))
        forecasts[year] = max(value, 0.0)
    return forecasts


def _provisional_quarterly_inflow(transition_rows_list: list[dict[str, Any]], annual_rows: list[dict[str, Any]]) -> dict[str, float]:
    annual_incidence = _annual_metric_map(annual_rows, "annual_new_infections")
    provisional: dict[str, float] = {}
    by_year: dict[int, list[tuple[str, float]]] = {}
    for transition_row in transition_rows_list:
        quarter = str(transition_row["quarter"])
        year = quarter_year(quarter)
        if year not in annual_incidence:
            continue
        previous_state = dict(transition_row["state_values_previous"])
        current_state = dict(transition_row["state_values_current"])
        u_to_d = float(transition_row["flows"]["U_to_D"])
        inflow = max(float(current_state["U"]) - float(previous_state["U"]) + u_to_d, 0.0)
        by_year.setdefault(year, []).append((quarter, inflow))
    for year, entries in by_year.items():
        total = sum(value for _quarter, value in entries)
        annual_total = float(annual_incidence.get(year, 0.0))
        if total <= 1e-9:
            equal_share = annual_total / max(len(entries), 1)
            for quarter, _value in entries:
                provisional[quarter] = float(equal_share)
        else:
            for quarter, value in entries:
                provisional[quarter] = float(annual_total * value / total)
    return provisional


def _fit_quarterly_inflow_share_model(train_transition_rows: list[dict[str, Any]], train_controls: dict[str, list[float]], annual_rows: list[dict[str, Any]], cfg: AnnualIncidenceConfig) -> dict[str, float]:
    if not train_transition_rows:
        return {}
    provisional = _provisional_quarterly_inflow(train_transition_rows, annual_rows)
    train_quarters = [str(row["quarter"]) for row in train_transition_rows if str(row["quarter"]) in provisional]
    if len(train_quarters) < 2:
        return {}
    quarter_to_idx = {str(row["quarter"]): idx for idx, row in enumerate(train_transition_rows)}
    x_rows = []
    y_rows = []
    for quarter in train_quarters:
        idx = quarter_to_idx[quarter]
        state_previous = dict(train_transition_rows[idx]["state_values_previous"])
        infectious_pool = float(state_previous["U"] + state_previous["D"] + state_previous["A"] + state_previous["L"] + 0.02 * state_previous["V"])
        x_rows.append([1.0, math.log1p(max(infectious_pool, 0.0)), float(train_controls["A"][idx]), float(train_controls["C"][idx]), float(train_controls["R"][idx])])
        y_rows.append(math.log1p(max(float(provisional[quarter]), 0.0)))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    ridge = np.eye(x.shape[1], dtype=np.float64) * float(cfg.ridge_penalty)
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    return {"intercept": float(beta[0]), "log_infectious": float(beta[1]), "A": float(beta[2]), "C": float(beta[3]), "R": float(beta[4])}


def _quarterly_inflow_scores(trajectory_rows: list[dict[str, Any]], quarter_controls: dict[str, dict[str, float]], coeffs: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for row in trajectory_rows:
        quarter = str(row["quarter"])
        state_values = dict(row["state_values"])
        infectious_pool = float(state_values["U"] + state_values["D"] + state_values["A"] + state_values["L"] + 0.02 * state_values["V"])
        controls = quarter_controls.get(quarter, {})
        value = (
            float(coeffs.get("intercept", 0.0))
            + float(coeffs.get("log_infectious", 0.0)) * math.log1p(max(infectious_pool, 0.0))
            + float(coeffs.get("A", 0.0)) * float(controls.get("A", 0.0))
            + float(coeffs.get("C", 0.0)) * float(controls.get("C", 0.0))
            + float(coeffs.get("R", 0.0)) * float(controls.get("R", 0.0))
        )
        scores[quarter] = softplus(value)
    return scores


def simulate_open_inflow(initial_state: dict[str, float], target_rows: list[dict[str, Any]], hazard_map: dict[str, dict[str, float]], quarter_inflows: dict[str, float]) -> dict[str, Any]:
    previous_state = {name: float(initial_state[name]) for name in STATE_NAMES}
    prediction_rows: list[dict[str, float]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for row in target_rows:
        quarter = str(row["quarter"])
        hazards = dict(hazard_map.get(quarter) or {})
        inflow = max(float(quarter_inflows.get(quarter, 0.0)), 0.0)
        u_to_d = float(hazards.get("U_to_D") or 0.0) * max(float(previous_state["U"]), 0.0)
        d_to_a = float(hazards.get("D_to_A") or 0.0) * max(float(previous_state["D"]), 0.0)
        a_to_v = float(hazards.get("A_to_V") or 0.0) * max(float(previous_state["A"]), 0.0)
        a_to_l = float(hazards.get("A_to_L") or 0.0) * max(float(previous_state["A"]), 0.0)
        l_to_a = float(hazards.get("L_to_A") or 0.0) * max(float(previous_state["L"]), 0.0)
        current_state = {
            "U": max(float(previous_state["U"]) + inflow - u_to_d, 0.0),
            "D": max(float(previous_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(previous_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(float(previous_state["V"]) + a_to_v, 0.0),
            "L": max(float(previous_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        prediction_rows.append({
            "quarter": quarter,
            "diagnosed_plhiv": float(current_state["D"] + current_state["A"] + current_state["V"] + current_state["L"]),
            "alive_on_art": float(current_state["A"] + current_state["V"]),
            "new_diagnosed_cases_period": float(u_to_d),
            "tested_for_viral_load": None,
            "virally_suppressed": float(current_state["V"]),
        })
        trajectory_rows.append({"quarter": quarter, "state_values": current_state, "hazards": hazards, "inflow": inflow})
        previous_state = current_state
    return {"prediction_rows": prediction_rows, "trajectory_rows": trajectory_rows}


def _fit_05a_candidate(dataset: QuarterlyDataset, dynamic_cfg: DynamicControlConfig, observation_cfg: ObservationConfig) -> dict[str, Any]:
    holdout_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    controls_train_full = build_control_channels(dataset.train_rows)
    controls_transition = {name: list(values[1:]) for name, values in controls_train_full.items()}
    hazard_map = {quarter: {} for quarter in holdout_quarters}
    train_hazard_map = {str(row["quarter"]): {} for row in dataset.train_transition_rows}
    for transition_name in TRANSITION_NAMES:
        forecast = _fit_logit_transition_model(dataset.train_transition_rows, holdout_quarters, transition_name, controls_transition, cfg=dynamic_cfg, eps=dataset.eps)
        for quarter, value in forecast.items():
            hazard_map[quarter][transition_name] = value
        for row in dataset.train_transition_rows:
            train_hazard_map[str(row["quarter"])][transition_name] = float(row["hazards"][transition_name])
    raw = simulate_closed_flow(dict(dataset.train_state_rows[-1]["state_values"]), dataset.holdout_rows, hazard_map)
    observation_model = fit_observation_model(dataset, train_hazard_map, observation_cfg)
    calibrated_rows = apply_observation_model(raw["prediction_rows"], observation_model, positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows))
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": raw["trajectory_rows"],
        "hazard_map": hazard_map,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
    }


def _fit_05b_candidate(dataset: QuarterlyDataset, annual_rows: list[dict[str, Any]], dynamic_cfg: DynamicControlConfig, observation_cfg: ObservationConfig, annual_cfg: AnnualIncidenceConfig) -> dict[str, Any]:
    base = _fit_05a_candidate(dataset, dynamic_cfg, observation_cfg)
    full_rows = dataset.train_rows + dataset.holdout_rows
    yearly_rows = _yearly_feature_rows(full_rows, annual_rows)
    train_years = sorted({quarter_year(str(row["quarter"])) for row in dataset.train_rows})
    holdout_years = sorted({quarter_year(str(row["quarter"])) for row in dataset.holdout_rows})
    annual_train_rows = [row for row in yearly_rows if int(row["year"]) in set(train_years)]
    annual_forecast = _fit_annual_incidence_model(annual_train_rows, holdout_years, annual_cfg)

    controls_train_full = build_control_channels(dataset.train_rows)
    controls_transition = {name: list(values[1:]) for name, values in controls_train_full.items()}
    inflow_coeffs = _fit_quarterly_inflow_share_model(dataset.train_transition_rows, controls_transition, annual_rows, annual_cfg)
    forecast_quarters = [str(row["quarter"]) for row in dataset.holdout_rows]
    forecast_controls = {
        control_name: _fit_scalar_ar_trend([str(row["quarter"]) for row in dataset.train_rows], controls_train_full[control_name], forecast_quarters, ridge_penalty=dynamic_cfg.ridge_penalty, rho_clip=dynamic_cfg.rho_clip, trend_scale=dynamic_cfg.trend_scale)
        for control_name in ("A", "C", "R")
    }

    interim = simulate_closed_flow(dict(dataset.train_state_rows[-1]["state_values"]), dataset.holdout_rows, base["hazard_map"])
    quarter_controls = {quarter: {control_name: float(forecast_controls[control_name].get(quarter, 0.0)) for control_name in ("A", "C", "R")} for quarter in forecast_quarters}
    score_map = _quarterly_inflow_scores(interim["trajectory_rows"], quarter_controls, inflow_coeffs) if inflow_coeffs else {quarter: 1.0 for quarter in forecast_quarters}
    quarter_inflows: dict[str, float] = {}
    by_year: dict[int, list[str]] = {}
    for quarter in forecast_quarters:
        by_year.setdefault(quarter_year(quarter), []).append(quarter)
    for year, quarters in by_year.items():
        total = float(annual_forecast.get(year, 0.0))
        weights = np.asarray([float(score_map.get(quarter, 1.0)) for quarter in quarters], dtype=np.float64)
        weight_sum = float(weights.sum())
        if weight_sum <= 1e-9:
            weights = np.full((len(quarters),), 1.0 / max(len(quarters), 1), dtype=np.float64)
        else:
            weights = weights / weight_sum
        for quarter, weight in zip(quarters, weights):
            quarter_inflows[quarter] = float(total * float(weight))
    open_sim = simulate_open_inflow(dict(dataset.train_state_rows[-1]["state_values"]), dataset.holdout_rows, base["hazard_map"], quarter_inflows)
    observation_model = fit_observation_model(dataset, {str(row["quarter"]): {transition: float(row["hazards"][transition]) for transition in TRANSITION_NAMES} for row in dataset.train_transition_rows}, observation_cfg)
    calibrated_rows = apply_observation_model(open_sim["prediction_rows"], observation_model, positions=_quarter_positions(dataset.train_rows + dataset.holdout_rows))
    predicted_annual_infections = {year: float(sum(quarter_inflows[quarter] for quarter in quarters)) for year, quarters in by_year.items()}
    return {
        "prediction_rows": calibrated_rows,
        "trajectory_rows": open_sim["trajectory_rows"],
        "annual_infections": predicted_annual_infections,
        "mae": normalized_mae(calibrated_rows, dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps),
        "smape": smape(calibrated_rows, dataset.holdout_rows, eps=dataset.eps),
    }

def _annual_metric_error(yearly_rows: list[dict[str, float]], holdout_years: list[int], predictions: dict[int, float], metric_name: str) -> float:
    target_map = {int(row["year"]): float(row[metric_name]) for row in yearly_rows}
    scales = [abs(float(row[metric_name])) for row in yearly_rows if float(row[metric_name]) != 0.0]
    scale = max(scales) if scales else 1.0
    errors = [abs(float(predictions[year]) - float(target_map[year])) / max(scale, 1e-6) for year in holdout_years if year in predictions and year in target_map]
    return float(np.mean(errors)) if errors else float("inf")


def _run_quarterly_splits(
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    candidate_type: str,
    dynamic_cfg: DynamicControlConfig,
    observation_cfg: ObservationConfig,
    annual_cfg: AnnualIncidenceConfig | None,
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    splits = rolling_origin_year_splits(sorted({quarter_year(str(row["quarter"])) for row in observation_rows}), start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    rows: list[dict[str, Any]] = []
    candidate_maes: list[float] = []
    carry_forward_maes: list[float] = []
    for split in splits:
        dataset = build_quarterly_dataset(observation_rows, list(split["holdout_years"]))
        carry_forward = simulate_closed_flow(dict(dataset.train_state_rows[-1]["state_values"]), dataset.holdout_rows, carry_forward_hazards(dataset))
        carry_forward_mae = normalized_mae(carry_forward["prediction_rows"], dataset.holdout_rows, dataset.metric_scales, eps=dataset.eps)
        carry_forward_smape = smape(carry_forward["prediction_rows"], dataset.holdout_rows, eps=dataset.eps)
        if candidate_type == "05a":
            candidate = _fit_05a_candidate(dataset, dynamic_cfg, observation_cfg)
        else:
            assert annual_cfg is not None
            candidate = _fit_05b_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, annual_cfg)
        rows.append({
            "train_end_year": int(split["train_end_year"]),
            "holdout_years": list(split["holdout_years"]),
            "carry_forward": {"mae": float(carry_forward_mae), "smape": float(carry_forward_smape)},
            "candidate": {"mae": float(candidate["mae"]), "smape": float(candidate["smape"])},
            "candidate_annual_infections": dict(candidate.get("annual_infections") or {}),
        })
        candidate_maes.append(float(candidate["mae"]))
        carry_forward_maes.append(float(carry_forward_mae))
    summary = {
        "candidate_mean_mae": float(np.mean(candidate_maes)) if candidate_maes else float("inf"),
        "carry_forward_mean_mae": float(np.mean(carry_forward_maes)) if carry_forward_maes else float("inf"),
        "candidate_worst_mae": float(max(candidate_maes)) if candidate_maes else float("inf"),
        "carry_forward_worst_mae": float(max(carry_forward_maes)) if carry_forward_maes else float("inf"),
    }
    return rows, summary


def _run_annual_diagnostics(
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    annual_cfg: AnnualIncidenceConfig,
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizon_years: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    yearly_rows = _yearly_feature_rows(observation_rows, annual_rows)
    years = [int(row["year"]) for row in yearly_rows]
    splits = rolling_origin_year_splits(years, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=horizon_years)
    rows: list[dict[str, Any]] = []
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    incidence_map = {int(row["year"]): float(row["annual_new_infections"]) for row in yearly_rows}
    plhiv_map = {int(row["year"]): float(row["estimated_plhiv"]) for row in yearly_rows}
    for split in splits:
        train = [row for row in yearly_rows if int(row["year"]) <= int(split["train_end_year"])]
        candidate_pred = _fit_annual_incidence_model(train, list(split["holdout_years"]), annual_cfg)
        baseline_pred = {year: float(train[-1]["annual_new_infections"]) for year in split["holdout_years"]}
        candidate_error = _annual_metric_error(yearly_rows, list(split["holdout_years"]), candidate_pred, "annual_new_infections")
        baseline_error = _annual_metric_error(yearly_rows, list(split["holdout_years"]), baseline_pred, "annual_new_infections")
        plhiv_baseline = {year: float(plhiv_map.get(year - 1, train[-1]["estimated_plhiv"])) for year in split["holdout_years"]}
        plhiv_candidate = {year: float(plhiv_map.get(year - 1, train[-1]["estimated_plhiv"]) + candidate_pred.get(year, 0.0) - baseline_pred.get(year, 0.0)) for year in split["holdout_years"]}
        rows.append({
            "train_end_year": int(split["train_end_year"]),
            "holdout_years": list(split["holdout_years"]),
            "candidate_incidence_error": float(candidate_error),
            "baseline_incidence_error": float(baseline_error),
            "candidate_incidence": candidate_pred,
            "baseline_incidence": baseline_pred,
            "candidate_plhiv_error": float(_annual_metric_error(yearly_rows, list(split["holdout_years"]), plhiv_candidate, "estimated_plhiv")),
            "baseline_plhiv_error": float(_annual_metric_error(yearly_rows, list(split["holdout_years"]), plhiv_baseline, "estimated_plhiv")),
        })
        candidate_errors.append(float(candidate_error))
        baseline_errors.append(float(baseline_error))
    summary = {
        "candidate_mean_incidence_error": float(np.mean(candidate_errors)) if candidate_errors else float("inf"),
        "baseline_mean_incidence_error": float(np.mean(baseline_errors)) if baseline_errors else float("inf"),
    }
    return rows, summary


def _load_frozen_04d_reference(path: Path) -> dict[str, Any]:
    payload = dict(read_json(path, default={}) or {})
    best = dict(payload.get("best_candidate") or {})
    score = dict(best.get("score") or {})
    rows = list(best.get("rows") or [])
    return {"path": str(path), "score": score, "rows": rows}


def _candidate_grid_05a() -> list[tuple[DynamicControlConfig, ObservationConfig]]:
    return [
        (DynamicControlConfig(ridge_penalty=dynamic_ridge, rho_clip=rho_clip, trend_scale=1.0), ObservationConfig(calibration_ridge=obs_ridge, share_ridge_penalty=0.1, share_rho_clip=0.9, share_trend_scale=1.0))
        for dynamic_ridge in (0.01, 0.1, 1.0)
        for rho_clip in (0.8, 0.95)
        for obs_ridge in (0.01, 0.1, 1.0)
    ]


def _candidate_grid_05b() -> list[tuple[DynamicControlConfig, ObservationConfig, AnnualIncidenceConfig]]:
    return [
        (dynamic_cfg, observation_cfg, AnnualIncidenceConfig(ridge_penalty=annual_ridge, inflow_scale_clip=4.0))
        for dynamic_cfg, observation_cfg in _candidate_grid_05a()
        for annual_ridge in (0.01, 0.1, 1.0)
    ]


def _score_payload(payload: dict[str, Any]) -> tuple[float, float, float]:
    modern = float(payload["quarterly_summary"]["candidate_mean_mae"])
    annual = float(payload["annual_summary"]["candidate_mean_incidence_error"])
    worst = float(payload["quarterly_summary"]["candidate_worst_mae"])
    return modern, annual, worst


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['family_name']} Autoresearch Report",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Decision: `{payload['decision']}`",
        f"- Reason: `{payload['decision_reason']}`",
    ]
    if str(payload.get("implementation_boundary") or "").strip():
        lines.append(f"- Implementation boundary: `{payload['implementation_boundary']}`")
    lines.extend([
        "",
        "## Best Candidate",
        "",
        f"- Dynamic config: `{payload['best_candidate']['dynamic_cfg']}`",
        f"- Observation config: `{payload['best_candidate']['observation_cfg']}`",
    ])
    if payload["best_candidate"].get("annual_cfg") is not None:
        lines.append(f"- Annual incidence config: `{payload['best_candidate']['annual_cfg']}`")
    lines.extend([
        f"- Quarterly mean MAE: `{payload['quarterly_summary']['candidate_mean_mae']:.6f}`",
        f"- Quarterly carry-forward mean MAE: `{payload['quarterly_summary']['carry_forward_mean_mae']:.6f}`",
        f"- Annual incidence mean error: `{payload['annual_summary']['candidate_mean_incidence_error']:.6f}`",
        f"- Annual incidence carry-forward mean error: `{payload['annual_summary']['baseline_mean_incidence_error']:.6f}`",
    ])
    frozen = payload.get("frozen_04d_reference") or {}
    frozen_score = dict(frozen.get("score") or {})
    if frozen_score:
        lines.extend([
            "",
            "## Frozen 04d Reference",
            "",
            f"- Reference quarterly mean MAE: `{float(frozen_score.get('candidate_mean_mae', float('nan'))):.6f}`",
            f"- Reference quarterly worst MAE: `{float(frozen_score.get('candidate_worst_mae', float('nan'))):.6f}`",
            "- This is a frozen incumbent from the older sandbox archive. It is a reference, not a gating truth for the new merged archive.",
        ])
    lines.extend(["", "## Quarterly Splits", "", "| Train end | Holdout | Carry-forward MAE | Candidate MAE | Delta |", "|---|---:|---:|---:|---:|"])
    for row in payload["quarterly_rows"]:
        carry = float(row["carry_forward"]["mae"])
        cand = float(row["candidate"]["mae"])
        lines.append(f"| {row['train_end_year']} | {','.join(str(value) for value in row['holdout_years'])} | {carry:.6f} | {cand:.6f} | {cand - carry:+.6f} |")
    lines.extend(["", "## Annual Diagnostics", "", "| Train end | Holdout | Baseline Incidence Error | Candidate Incidence Error | Baseline PLHIV Error | Candidate PLHIV Error |", "|---|---:|---:|---:|---:|---:|"])
    for row in payload["annual_rows"]:
        lines.append(f"| {row['train_end_year']} | {','.join(str(value) for value in row['holdout_years'])} | {float(row['baseline_incidence_error']):.6f} | {float(row['candidate_incidence_error']):.6f} | {float(row['baseline_plhiv_error']):.6f} | {float(row['candidate_plhiv_error']):.6f} |")
    return "\n".join(lines) + "\n"


def run_tr_v3_05a_loop(*, run_id: str, archive_run_id: str, frozen_04d_report: Path, quarterly_start_year: int = 2017, quarterly_end_year: int = 2025, quarterly_min_train_years: int = 5, horizon_years: int = 1, annual_start_year: int = 2010, annual_end_year: int = 2024, annual_min_train_years: int = 5) -> dict[str, Any]:
    observation_rows = build_quarterly_observation_rows(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    frozen_reference = _load_frozen_04d_reference(frozen_04d_report)
    candidates: list[dict[str, Any]] = []
    for dynamic_cfg, observation_cfg in _candidate_grid_05a():
        quarterly_rows, quarterly_summary = _run_quarterly_splits(observation_rows, annual_rows, "05a", dynamic_cfg, observation_cfg, None, start_year=quarterly_start_year, end_year=quarterly_end_year, min_train_years=quarterly_min_train_years, horizon_years=horizon_years)
        annual_rows_diag, annual_summary = _run_annual_diagnostics(observation_rows, annual_rows, AnnualIncidenceConfig(ridge_penalty=0.1), start_year=annual_start_year, end_year=annual_end_year, min_train_years=annual_min_train_years, horizon_years=horizon_years)
        candidates.append({"dynamic_cfg": asdict(dynamic_cfg), "observation_cfg": asdict(observation_cfg), "annual_cfg": None, "quarterly_rows": quarterly_rows, "quarterly_summary": quarterly_summary, "annual_rows": annual_rows_diag, "annual_summary": annual_summary})
    best_candidate = min(candidates, key=_score_payload)
    quarterly_ok = float(best_candidate["quarterly_summary"]["candidate_mean_mae"]) < float(best_candidate["quarterly_summary"]["carry_forward_mean_mae"])
    annual_ok = float(best_candidate["annual_summary"]["candidate_mean_incidence_error"]) <= float(best_candidate["annual_summary"]["baseline_mean_incidence_error"])
    decision = "keep" if quarterly_ok and annual_ok else "revert"
    reason = "Beat the live carry-forward baseline on the modern quarterly window and did not regress the longer annual incidence diagnostic." if decision == "keep" else "Did not clear both the live quarterly and longer annual gates."
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "family_name": "TR-V3-05a", "archive_run_id": archive_run_id, "loop_variant": "evidence-to-model-loop", "decision": decision, "decision_reason": reason, "frozen_04d_reference": frozen_reference, "best_candidate": best_candidate, "quarterly_rows": best_candidate["quarterly_rows"], "quarterly_summary": best_candidate["quarterly_summary"], "annual_rows": best_candidate["annual_rows"], "annual_summary": best_candidate["annual_summary"], "all_candidates": candidates}
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / "tr_v3_05a_autoresearch_report.json", payload)
    (analysis_dir / "tr_v3_05a_autoresearch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def run_tr_v3_05b_loop(*, run_id: str, archive_run_id: str, frozen_04d_report: Path, quarterly_start_year: int = 2017, quarterly_end_year: int = 2025, quarterly_min_train_years: int = 5, horizon_years: int = 1, annual_start_year: int = 2010, annual_end_year: int = 2024, annual_min_train_years: int = 5) -> dict[str, Any]:
    observation_rows = build_quarterly_observation_rows(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    frozen_reference = _load_frozen_04d_reference(frozen_04d_report)
    candidates: list[dict[str, Any]] = []
    for dynamic_cfg, observation_cfg, annual_cfg in _candidate_grid_05b():
        quarterly_rows, quarterly_summary = _run_quarterly_splits(observation_rows, annual_rows, "05b", dynamic_cfg, observation_cfg, annual_cfg, start_year=quarterly_start_year, end_year=quarterly_end_year, min_train_years=quarterly_min_train_years, horizon_years=horizon_years)
        annual_rows_diag, annual_summary = _run_annual_diagnostics(observation_rows, annual_rows, annual_cfg, start_year=annual_start_year, end_year=annual_end_year, min_train_years=annual_min_train_years, horizon_years=horizon_years)
        candidates.append({"dynamic_cfg": asdict(dynamic_cfg), "observation_cfg": asdict(observation_cfg), "annual_cfg": asdict(annual_cfg), "quarterly_rows": quarterly_rows, "quarterly_summary": quarterly_summary, "annual_rows": annual_rows_diag, "annual_summary": annual_summary})
    best_candidate = min(candidates, key=_score_payload)
    quarterly_ok = float(best_candidate["quarterly_summary"]["candidate_mean_mae"]) < float(best_candidate["quarterly_summary"]["carry_forward_mean_mae"])
    annual_ok = float(best_candidate["annual_summary"]["candidate_mean_incidence_error"]) < float(best_candidate["annual_summary"]["baseline_mean_incidence_error"])
    decision = "keep" if quarterly_ok and annual_ok else "revert"
    reason = "Open incidence-flow improved the live quarterly baseline and the longer annual incidence diagnostic." if decision == "keep" else "Open incidence-flow did not beat both the live quarterly baseline and the longer annual incidence diagnostic."
    has_annual_deaths = any(str(row.get("metric_name") or "") == "annual_aids_deaths" for row in annual_rows)
    implementation_boundary = (
        "05b currently activates explicit incidence inflow into U and retains the observed A<->L leakage structure. The rebuilt archive now includes a trustworthy annual national AIDS-deaths series, but 05b still does not activate separate D->L, V->L, or mortality blocks because the live archive does not yet provide quarterly or state-specific mortality signal that can identify those extra flows."
        if has_annual_deaths
        else "05b currently activates explicit incidence inflow into U and retains the observed A<->L leakage structure. It does not yet activate separate D->L, V->L, or mortality blocks because the live merged archive does not contain a trustworthy death series and does not uniquely identify those extra flows."
    )
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "family_name": "TR-V3-05b", "archive_run_id": archive_run_id, "loop_variant": "evidence-to-model-loop", "decision": decision, "decision_reason": reason, "implementation_boundary": implementation_boundary, "frozen_04d_reference": frozen_reference, "best_candidate": best_candidate, "quarterly_rows": best_candidate["quarterly_rows"], "quarterly_summary": best_candidate["quarterly_summary"], "annual_rows": best_candidate["annual_rows"], "annual_summary": best_candidate["annual_summary"], "all_candidates": candidates}
    analysis_dir = ensure_dir(repo_root() / "artifacts" / "runs" / run_id / "analysis")
    write_json(analysis_dir / "tr_v3_05b_autoresearch_report.json", payload)
    (analysis_dir / "tr_v3_05b_autoresearch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr-v3-05-autoresearch")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("tr-v3-05a-loop", "tr-v3-05b-loop"):
        cmd = subparsers.add_parser(name)
        cmd.add_argument("--run-id", required=True)
        cmd.add_argument("--archive-run-id", default=default_archive_run_id())
        cmd.add_argument("--frozen-04d-report", default=str(default_frozen_04d_report()))
        cmd.add_argument("--quarterly-start-year", type=int, default=2017)
        cmd.add_argument("--quarterly-end-year", type=int, default=2025)
        cmd.add_argument("--quarterly-min-train-years", type=int, default=5)
        cmd.add_argument("--annual-start-year", type=int, default=2010)
        cmd.add_argument("--annual-end-year", type=int, default=2024)
        cmd.add_argument("--annual-min-train-years", type=int, default=5)
        cmd.add_argument("--horizon-years", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    kwargs = {
        "run_id": args.run_id,
        "archive_run_id": args.archive_run_id,
        "frozen_04d_report": Path(args.frozen_04d_report),
        "quarterly_start_year": args.quarterly_start_year,
        "quarterly_end_year": args.quarterly_end_year,
        "quarterly_min_train_years": args.quarterly_min_train_years,
        "annual_start_year": args.annual_start_year,
        "annual_end_year": args.annual_end_year,
        "annual_min_train_years": args.annual_min_train_years,
        "horizon_years": args.horizon_years,
    }
    if args.command == "tr-v3-05a-loop":
        run_tr_v3_05a_loop(**kwargs)
        return 0
    if args.command == "tr-v3-05b-loop":
        run_tr_v3_05b_loop(**kwargs)
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
