from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backhalf_channels import BackHalfChannelContext
from .data import build_blocked_time_dataset, build_observation_rows, build_source_row_path, default_epigraph_root, sandbox_repo_root
from .hybrid_champion import (
    OBSERVATION_HEAD_METRICS,
    _apply_endpoint_head,
    _carry_forward_result,
    _delegate_family_for_horizon,
    _filter_rows_before_holdout,
    _forecast_reference_for_family,
    _fit_endpoint_head,
    _load_r10_baseline,
    _load_reference_config,
    _r10_reference_scores,
    _rolling_splits,
    _train_backbone_prediction_rows,
    _train_multihorizon_examples,
)
from .metrics import quarter_ordinal, quarter_sort_key, quarter_year
from .monthly_joint_observation import MonthlyJointContext
from .monthly_latent_state import MonthlyLatentContext
from .monthly_shock import (
    MonthlyShockContext,
    _chosen_month_metric_values,
    _month_index,
    _month_label,
    _quarter_month_range,
    load_monthly_signal_rows,
)
from .observation_ledger import (
    build_contract_row_hash,
    build_observation_contract_lookup,
    resolve_active_source_run_id,
    resolve_baseline_source_run_id,
)
from .runtime import ensure_dir, read_json, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID

DIAGNOSIS_INCIDENCE_REPAIR_SCHEMA_VERSION = "phase3_dynamic_diagnosis_delay_backcalc.v5_module_decomposition"

DEFAULT_SOURCE_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-reengagement-sensitivity-20260426-s01-public-stock-flow-proxy"
    / "analysis"
    / "hybrid_champion_search_report.json"
)

REPAIR_FAMILIES: tuple[str, ...] = (
    "baseline_no_repair",
    "incidence_readout_from_diag_flow",
    "monthly_delay_backcalc",
    "monthly_delay_backcalc_late_constrained",
    "monthly_backlog_late_emission",
    "monthly_backlog_late_emission_readout",
    "monthly_reporting_nowcast_flow_only",
    "monthly_reporting_nowcast",
    "monthly_reporting_nowcast_readout",
    "monthly_reporting_nowcast_backlog",
    "selective_monthly_nowcast_backlog",
    "selective_monthly_nowcast_backlog_readout",
)

DIAGNOSIS_FLOW_REPAIR_FAMILIES: tuple[str, ...] = (
    "diag_flow_incidence_only",
    "diag_flow_incidence_backbone",
    "diag_flow_incidence_backlog",
    "diag_flow_incidence_backbone_backlog",
    "diag_flow_incidence_backbone_backlog_readout",
)

INCIDENCE_READOUT_REPAIR_FAMILIES: tuple[str, ...] = (
    "incidence_readout_from_diag_flow",
    "monthly_backlog_late_emission_readout",
    "monthly_reporting_nowcast_readout",
    "selective_monthly_nowcast_backlog_readout",
    "diag_flow_incidence_backbone_backlog_readout",
)

DIAGNOSIS_DELAY_BACKCALC_FAMILIES: tuple[str, ...] = (
    "monthly_delay_backcalc",
    "monthly_delay_backcalc_late_constrained",
)

BACKLOG_LATE_EMISSION_FAMILIES: tuple[str, ...] = (
    "monthly_backlog_late_emission",
    "monthly_backlog_late_emission_readout",
    "monthly_reporting_nowcast_backlog",
)

MONTHLY_REPORTING_NOWCAST_FAMILIES: tuple[str, ...] = (
    "monthly_reporting_nowcast_flow_only",
    "monthly_reporting_nowcast",
    "monthly_reporting_nowcast_readout",
    "monthly_reporting_nowcast_backlog",
    "selective_monthly_nowcast_backlog",
    "selective_monthly_nowcast_backlog_readout",
)

SELECTIVE_REPAIR_GATE_FAMILIES: tuple[str, ...] = (
    "selective_monthly_nowcast_backlog",
    "selective_monthly_nowcast_backlog_readout",
)

POST_FLOW_INCIDENCE_READOUT_FAMILIES: tuple[str, ...] = (
    "incidence_readout_from_diag_flow",
    "monthly_backlog_late_emission_readout",
    "monthly_reporting_nowcast_readout",
    "selective_monthly_nowcast_backlog_readout",
    "diag_flow_incidence_backbone_backlog_readout",
)

NOWCAST_INTERNAL_INCIDENCE_FAMILIES: tuple[str, ...] = (
    "monthly_reporting_nowcast",
    "monthly_reporting_nowcast_backlog",
    "selective_monthly_nowcast_backlog",
)

LATE_DIAGNOSIS_EVIDENCE_METRICS: tuple[str, ...] = (
    "median_cd4_at_enrollment",
    "median_cd4_at_diagnosis",
    "advanced_hiv_cases_period",
    "advanced_hiv_disease_share",
    "late_hiv_diagnosis_percent",
)
LATE_DIAGNOSIS_ALLOWED_ROLES = {"auxiliary_likelihood", "prior_context"}


@dataclass(slots=True)
class DiagnosisFlowRepairHead:
    family: str
    feature_names: list[str]
    coefficients: list[float]
    train_row_count: int
    training_objective: str = "direct_harp_diagnosis_flow_log_readout"


@dataclass(slots=True)
class IncidenceReadoutRepairHead:
    family: str
    feature_names: list[str]
    coefficients: list[float]
    train_row_count: int
    training_objective: str = "training_window_mechanistic_incidence_from_direct_diagnosis_flow"


@dataclass(slots=True)
class DiagnosisDelayBackcalcHead:
    family: str
    delay_lags_months: list[int]
    delay_kernel: list[float]
    ascertainment_scale: float
    train_month_count: int
    observed_diagnosis_month_count: int
    backcalculated_month_count: int
    forecast_month_count: int
    monthly_ar_parameters: dict[str, float]
    late_constraint_summary: dict[str, Any] | None = None
    training_objective: str = "monthly_diagnosis_delay_backcalculation_without_incidence_validation"


@dataclass(slots=True)
class BacklogLateEmissionHead:
    family: str
    early_to_late_hazard: float
    early_diagnosis_hazard: float
    late_diagnosis_hazard: float
    initial_early_undiagnosed: float
    initial_late_undiagnosed: float
    train_month_count: int
    observed_diagnosis_month_count: int
    late_share_emission_month_count: int
    advanced_count_emission_month_count: int
    forecast_month_count: int
    train_loss: float
    diagnosis_loss: float | None
    late_share_loss: float | None
    advanced_count_loss: float | None
    monthly_ar_parameters: dict[str, float]
    emission_summary: dict[str, Any]
    training_objective: str = "joint_monthly_backlog_diagnosis_and_late_presenter_emission"


@dataclass(slots=True)
class MonthlyReportingNowcastHead:
    family: str
    completion_by_month_mask: dict[str, float]
    training_partial_by_month_mask: dict[str, list[float]]
    lead_completion_by_month_mask: dict[str, float]
    training_lead_partial_by_month_mask: dict[str, list[float]]
    fallback_completion_scale: float
    fallback_lead_completion_scale: float
    incidence_feature_names: list[str]
    incidence_coefficients: list[float]
    incidence_prediction_floor: float
    incidence_prediction_ceiling: float
    train_quarter_count: int
    direct_monthly_row_count: int
    train_partial_quarter_count: int
    holdout_nowcast_quarter_count: int
    training_objective: str = "partial_monthly_HARP_diagnosis_reporting_nowcast_with_validation_only_incidence_readout"


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _safe_scale(metric_name: str, target_value: float, metric_scales: dict[str, Any]) -> float:
    configured = _finite_float(metric_scales.get(metric_name))
    if configured is not None and configured > 0.0:
        return max(configured, 1.0)
    return max(abs(float(target_value)), 1.0)


def _default_source_report() -> Path:
    expanded_harp_candidates: list[Path] = []
    for candidate in sorted(
        (sandbox_repo_root() / "artifacts" / "runs").glob(
            "p3d-hybrid-champion-search-*/analysis/hybrid_champion_search_report.json"
        )
    ):
        payload = dict(read_json(candidate, default={}) or {})
        if str(payload.get("source_run_id") or "") == DEFAULT_ACTIVE_SOURCE_RUN_ID:
            expanded_harp_candidates.append(candidate)
    if expanded_harp_candidates:
        return max(expanded_harp_candidates, key=lambda path: path.stat().st_mtime)
    if DEFAULT_SOURCE_REPORT.exists():
        return DEFAULT_SOURCE_REPORT
    candidates = sorted(
        (sandbox_repo_root() / "artifacts" / "runs").glob(
            "p3d-reengagement-sensitivity-*-public-stock-flow-proxy/analysis/hybrid_champion_search_report.json"
        )
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No public-stock-flow hybrid report found for diagnosis/incidence repair.")


def _selected_family(report: dict[str, Any], requested_family: str | None) -> str:
    if requested_family:
        return requested_family
    champion = dict(report.get("champion_by_claim_aware_promotion") or {})
    if champion.get("family"):
        return str(champion["family"])
    best = dict(report.get("best_family") or {})
    if best.get("family"):
        return str(best["family"])
    raise ValueError("Hybrid report does not contain a selectable family.")


def _feature_names(family: str) -> list[str]:
    if family == "diag_flow_incidence_only":
        return ["intercept", "log_incidence_inflow"]
    if family == "diag_flow_incidence_backbone":
        return ["intercept", "log_incidence_inflow", "log_backbone_diagnosis_flow"]
    if family == "diag_flow_incidence_backlog":
        return ["intercept", "log_incidence_inflow", "log_undiagnosed_state"]
    if family in {"diag_flow_incidence_backbone_backlog", "diag_flow_incidence_backbone_backlog_readout"}:
        return [
            "intercept",
            "log_incidence_inflow",
            "log_backbone_diagnosis_flow",
            "log_undiagnosed_state",
            "horizon_fraction",
        ]
    raise ValueError(f"Unsupported diagnosis-flow repair family: {family}")


def _incidence_readout_feature_names(family: str) -> list[str]:
    if family in INCIDENCE_READOUT_REPAIR_FAMILIES:
        return ["intercept", "log_diagnosis_flow", "horizon_fraction"]
    raise ValueError(f"Unsupported incidence-readout repair family: {family}")


def _trajectory_incidence(trajectory_row: dict[str, Any] | None) -> float:
    stock_balance = dict((trajectory_row or {}).get("stock_balance") or {})
    return max(float(stock_balance.get("incidence_inflow") or 0.0), 0.0)


def _trajectory_u_state(trajectory_row: dict[str, Any] | None) -> float:
    state_values = dict((trajectory_row or {}).get("state_values") or {})
    return max(float(state_values.get("U") or 0.0), 0.0)


def _month_to_quarter(month_index: int) -> str:
    year = int(month_index) // 12
    month = int(month_index) % 12 + 1
    quarter = ((month - 1) // 3) + 1
    return f"{year:04d}-Q{quarter}"


def _quarter_month_indices(quarter: str) -> list[int]:
    start, end = _quarter_month_range(quarter)
    return list(range(int(start), int(end) + 1))


def _quarterly_incidence_to_monthly(rows: list[dict[str, Any]], *, value_key: str) -> dict[int, float]:
    monthly: dict[int, float] = {}
    for row in rows:
        quarter = str(row.get("quarter") or "")
        value = _finite_float(row.get(value_key))
        if value is None:
            value = _finite_float(((row.get("stock_balance") or {}).get("incidence_inflow")))
        if not quarter or value is None:
            continue
        months = _quarter_month_indices(quarter)
        share = max(float(value), 0.0) / float(len(months))
        for month in months:
            monthly[int(month)] = float(share)
    return monthly


def _direct_monthly_diagnosis_values(context: MonthlyShockContext, *, train_end_month: int) -> dict[int, float]:
    rows = [
        row
        for row in load_monthly_signal_rows(context)
        if int(row["_month_index"]) <= int(train_end_month)
        and str(row.get("metric_name") or "") in {"new_diagnosed_cases_period", "new_diagnosed_cases_monthly"}
        and str((row.get("_contract") or {}).get("observation_role") or "") == "direct_target"
    ]
    chosen = _chosen_month_metric_values(rows)
    values: dict[int, float] = {}
    for (month, _metric_name), row in chosen.items():
        value = _finite_float(row.get("value"))
        if value is not None:
            values[int(month)] = max(float(value), 0.0)
    return values


def _direct_monthly_diagnosis_rows(
    context: MonthlyShockContext,
    *,
    max_month: int | None = None,
    monthly_only: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in load_monthly_signal_rows(context):
        if str(row.get("metric_name") or "") not in {"new_diagnosed_cases_period", "new_diagnosed_cases_monthly"}:
            continue
        if str((row.get("_contract") or {}).get("observation_role") or "") != "direct_target":
            continue
        if monthly_only and not str(row.get("series_kind") or "").startswith("monthly"):
            continue
        month_index = int(row["_month_index"])
        if max_month is not None and month_index > int(max_month):
            continue
        value = _finite_float(row.get("value"))
        if value is None:
            continue
        enriched = dict(row)
        enriched["_month_index"] = month_index
        enriched["_value"] = max(float(value), 0.0)
        rows.append(enriched)
    return sorted(rows, key=lambda row: (int(row["_month_index"]), str(row.get("metric_name") or "")))


def _diagnosis_monthly_values_by_month(
    rows: list[dict[str, Any]],
) -> dict[int, float]:
    chosen = _chosen_month_metric_values(rows)
    values: dict[int, float] = {}
    for (month, _metric_name), row in chosen.items():
        value = _finite_float(row.get("value"))
        if value is not None:
            values[int(month)] = max(float(value), 0.0)
    return values


def _quarter_month_mask_and_sum(
    monthly_values: dict[int, float],
    quarter: str,
) -> tuple[str, float, int]:
    start, end = _quarter_month_range(quarter)
    offsets: list[str] = []
    total = 0.0
    for month in range(int(start), int(end) + 1):
        if month in monthly_values:
            offsets.append(str(int(month) - int(start) + 1))
            total += max(float(monthly_values.get(month) or 0.0), 0.0)
    return "".join(offsets), float(total), len(offsets)


def _monthly_nowcast_feature_names() -> list[str]:
    return ["intercept", "log_diag_nowcast", "partial_month_support", "reporting_shock_score", "lead_signal"]


def _quarter_from_ordinal(index: int) -> str:
    year = int(index) // 4
    quarter = int(index) % 4 + 1
    return f"{year:04d}-Q{quarter}"


def _previous_quarter(quarter: str) -> str:
    return _quarter_from_ordinal(quarter_ordinal(quarter) - 1)


def _completion_scale_summary(values: list[float]) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if np.isfinite(float(value)) and float(value) > 0.0]
    if not finite:
        return None, None
    array = np.asarray(finite, dtype=np.float64)
    return float(np.median(array)), float(np.median(np.abs(array - float(np.median(array)))))


def _reporting_shock_score(
    *,
    partial_sum: float,
    month_mask: str,
    training_partial_by_mask: dict[str, list[float]],
) -> float:
    training_values = training_partial_by_mask.get(month_mask) or []
    if len(training_values) < 2:
        training_values = [value for values in training_partial_by_mask.values() for value in values]
    if not training_values:
        return 0.0
    transformed = np.asarray([float(np.log1p(max(float(value), 0.0))) for value in training_values], dtype=np.float64)
    center = float(np.median(transformed))
    scale = float(np.median(np.abs(transformed - center)))
    if scale <= 0.0:
        scale = float(np.std(transformed))
    if scale <= 0.0:
        scale = max(float(np.max(np.abs(transformed - center))), 1.0)
    raw = (float(np.log1p(max(float(partial_sum), 0.0))) - center) / scale
    return float(np.tanh(raw))


def _fit_monthly_reporting_nowcast_head(
    *,
    family: str,
    dataset: Any,
    monthly_context: MonthlyShockContext,
) -> MonthlyReportingNowcastHead | None:
    if family not in MONTHLY_REPORTING_NOWCAST_FAMILIES:
        return None
    train_rows = sorted(list(getattr(dataset, "train_rows", []) or []), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_transition_rows = {
        str(row.get("quarter") or ""): dict(row)
        for row in list(getattr(dataset, "train_transition_rows", []) or [])
    }
    if not train_rows:
        return None
    train_quarters = [str(row.get("quarter") or "") for row in train_rows]
    train_end_quarter = train_quarters[-1]
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    monthly_rows = _direct_monthly_diagnosis_rows(monthly_context, max_month=train_end_month, monthly_only=True)
    monthly_values = _diagnosis_monthly_values_by_month(monthly_rows)
    completion_by_mask_values: dict[str, list[float]] = {}
    partial_by_mask: dict[str, list[float]] = {}
    lead_completion_by_mask_values: dict[str, list[float]] = {}
    lead_partial_by_mask: dict[str, list[float]] = {}
    incidence_x_rows: list[list[float]] = []
    incidence_y_rows: list[float] = []
    feature_names = _monthly_nowcast_feature_names()
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        observed_quarter_diagnosis = _finite_float(row.get("new_diagnosed_cases_period"))
        if observed_quarter_diagnosis is None or observed_quarter_diagnosis <= 0.0:
            continue
        month_mask, partial_sum, support_count = _quarter_month_mask_and_sum(monthly_values, quarter)
        if not month_mask or partial_sum <= 0.0:
            continue
        completion = max(float(observed_quarter_diagnosis), 0.0) / max(float(partial_sum), float(np.finfo(np.float64).eps))
        completion_by_mask_values.setdefault(month_mask, []).append(float(completion))
        partial_by_mask.setdefault(month_mask, []).append(float(partial_sum))
        lead_mask, lead_partial_sum, _lead_support_count = _quarter_month_mask_and_sum(monthly_values, _previous_quarter(quarter))
        if lead_mask and lead_partial_sum > 0.0:
            lead_completion = max(float(observed_quarter_diagnosis), 0.0) / max(float(lead_partial_sum), float(np.finfo(np.float64).eps))
            lead_completion_by_mask_values.setdefault(lead_mask, []).append(float(lead_completion))
            lead_partial_by_mask.setdefault(lead_mask, []).append(float(lead_partial_sum))
    fallback_scale, _fallback_mad = _completion_scale_summary(
        [value for values in completion_by_mask_values.values() for value in values]
    )
    if fallback_scale is None:
        return None
    fallback_lead_scale, _fallback_lead_mad = _completion_scale_summary(
        [value for values in lead_completion_by_mask_values.values() for value in values]
    )
    if fallback_lead_scale is None:
        fallback_lead_scale = float(fallback_scale)
    completion_by_mask: dict[str, float] = {}
    for month_mask, values in completion_by_mask_values.items():
        scale, _mad = _completion_scale_summary(values)
        if scale is not None:
            completion_by_mask[month_mask] = float(scale)
    lead_completion_by_mask: dict[str, float] = {}
    for month_mask, values in lead_completion_by_mask_values.items():
        scale, _mad = _completion_scale_summary(values)
        if scale is not None:
            lead_completion_by_mask[month_mask] = float(scale)
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        transition_row = train_transition_rows.get(quarter) or {}
        incidence = _finite_float((transition_row.get("stock_balance") or {}).get("incidence_inflow"))
        if incidence is None:
            incidence = _finite_float(row.get("incident_infections_period"))
        if incidence is None or incidence < 0.0:
            continue
        observed_quarter_diagnosis = _finite_float(row.get("new_diagnosed_cases_period"))
        if observed_quarter_diagnosis is None or observed_quarter_diagnosis <= 0.0:
            continue
        month_mask, partial_sum, support_count = _quarter_month_mask_and_sum(monthly_values, quarter)
        lead_signal = 0.0
        if month_mask and partial_sum > 0.0:
            completion = completion_by_mask.get(month_mask, float(fallback_scale))
            diagnosis_nowcast = max(float(partial_sum) * float(completion), 0.0)
            shock_score = _reporting_shock_score(
                partial_sum=partial_sum,
                month_mask=month_mask,
                training_partial_by_mask=partial_by_mask,
            )
            support_fraction = float(support_count) / 3.0
        else:
            lead_mask, lead_partial_sum, lead_support_count = _quarter_month_mask_and_sum(monthly_values, _previous_quarter(quarter))
            if lead_mask and lead_partial_sum > 0.0:
                lead_completion = lead_completion_by_mask.get(lead_mask, float(fallback_lead_scale))
                diagnosis_nowcast = max(float(lead_partial_sum) * float(lead_completion), 0.0)
                shock_score = _reporting_shock_score(
                    partial_sum=lead_partial_sum,
                    month_mask=lead_mask,
                    training_partial_by_mask=lead_partial_by_mask,
                )
                support_fraction = float(lead_support_count) / 3.0
                lead_signal = 1.0
            else:
                diagnosis_nowcast = max(float(observed_quarter_diagnosis), 0.0)
                shock_score = 0.0
                support_fraction = 0.0
        incidence_x_rows.append(
            [
                1.0,
                float(np.log1p(max(diagnosis_nowcast, 0.0))),
                float(support_fraction),
                float(shock_score),
                float(lead_signal),
            ]
        )
        incidence_y_rows.append(float(np.log1p(max(float(incidence), 0.0))))
    if not incidence_x_rows:
        return None
    x = np.asarray(incidence_x_rows, dtype=np.float64)
    y = np.asarray(incidence_y_rows, dtype=np.float64)
    beta = np.linalg.pinv(x) @ y
    incidence_train_values = [max(float(np.expm1(value)), 0.0) for value in y]
    floor = min(incidence_train_values) if incidence_train_values else 0.0
    ceiling = max(incidence_train_values) if incidence_train_values else 0.0
    return MonthlyReportingNowcastHead(
        family=family,
        completion_by_month_mask={key: float(value) for key, value in sorted(completion_by_mask.items())},
        training_partial_by_month_mask={
            key: [float(value) for value in values]
            for key, values in sorted(partial_by_mask.items())
        },
        lead_completion_by_month_mask={key: float(value) for key, value in sorted(lead_completion_by_mask.items())},
        training_lead_partial_by_month_mask={
            key: [float(value) for value in values]
            for key, values in sorted(lead_partial_by_mask.items())
        },
        fallback_completion_scale=float(fallback_scale),
        fallback_lead_completion_scale=float(fallback_lead_scale),
        incidence_feature_names=feature_names,
        incidence_coefficients=[float(value) for value in beta],
        incidence_prediction_floor=float(floor),
        incidence_prediction_ceiling=float(ceiling),
        train_quarter_count=len(train_rows),
        direct_monthly_row_count=len(monthly_rows),
        train_partial_quarter_count=sum(len(values) for values in completion_by_mask_values.values()),
        holdout_nowcast_quarter_count=0,
    )


def _apply_monthly_reporting_nowcast_head(
    *,
    head: MonthlyReportingNowcastHead | None,
    monthly_context: MonthlyShockContext,
    base_rows: list[dict[str, Any]],
    update_incidence: bool = True,
) -> tuple[list[dict[str, Any]], MonthlyReportingNowcastHead | None, list[dict[str, Any]]]:
    if head is None:
        return [dict(row) for row in base_rows], None, []
    if not base_rows:
        return [], head, []
    ordered = sorted(list(base_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    last_quarter = str(ordered[-1].get("quarter") or "")
    _start, max_month = _quarter_month_range(last_quarter)
    monthly_rows = _direct_monthly_diagnosis_rows(monthly_context, max_month=max_month, monthly_only=True)
    monthly_values = _diagnosis_monthly_values_by_month(monthly_rows)
    training_partial_by_mask = {
        str(key): [float(value) for value in list(values)]
        for key, values in dict(head.training_partial_by_month_mask).items()
    }
    training_lead_partial_by_mask = {
        str(key): [float(value) for value in list(values)]
        for key, values in dict(head.training_lead_partial_by_month_mask).items()
    }
    beta = np.asarray(head.incidence_coefficients, dtype=np.float64)
    feature_names = list(head.incidence_feature_names)
    output: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    nowcast_count = 0
    for row in ordered:
        quarter = str(row.get("quarter") or "")
        mask, partial_sum, support_count = _quarter_month_mask_and_sum(monthly_values, quarter)
        next_row = dict(row)
        nowcast_source = ""
        lead_signal = 0.0
        if mask and partial_sum > 0.0:
            completion = float(head.completion_by_month_mask.get(mask, head.fallback_completion_scale))
            diagnosis_nowcast = max(float(partial_sum) * completion, 0.0)
            support_fraction = float(support_count) / 3.0
            shock_score = _reporting_shock_score(
                partial_sum=partial_sum,
                month_mask=mask,
                training_partial_by_mask=training_partial_by_mask,
            )
            nowcast_source = "same_quarter_partial_months"
        else:
            lead_mask, lead_partial_sum, lead_support_count = _quarter_month_mask_and_sum(monthly_values, _previous_quarter(quarter))
            if lead_mask and lead_partial_sum > 0.0:
                mask = lead_mask
                partial_sum = lead_partial_sum
                support_count = lead_support_count
                completion = float(head.lead_completion_by_month_mask.get(mask, head.fallback_lead_completion_scale))
                diagnosis_nowcast = max(float(partial_sum) * completion, 0.0)
                support_fraction = float(support_count) / 3.0
                shock_score = _reporting_shock_score(
                    partial_sum=partial_sum,
                    month_mask=mask,
                    training_partial_by_mask=training_lead_partial_by_mask,
                )
                nowcast_source = "previous_quarter_leading_months"
                lead_signal = 1.0
            else:
                output.append(next_row)
                continue
        if nowcast_source:
            feature_values = {
                "intercept": 1.0,
                "log_diag_nowcast": float(np.log1p(diagnosis_nowcast)),
                "partial_month_support": float(support_fraction),
                "reporting_shock_score": float(shock_score),
                "lead_signal": float(lead_signal),
            }
            x = np.asarray([float(feature_values.get(name) or 0.0) for name in feature_names], dtype=np.float64)
            if update_incidence and beta.size == x.size:
                incidence = max(float(np.expm1(float(x @ beta))), 0.0)
                if head.incidence_prediction_ceiling > 0.0:
                    incidence = float(np.clip(incidence, head.incidence_prediction_floor, head.incidence_prediction_ceiling))
                next_row["incident_infections_period"] = incidence
            next_row["new_diagnosed_cases_period"] = diagnosis_nowcast
            next_row["monthly_reporting_nowcast_support"] = support_fraction
            next_row["monthly_reporting_shock_score"] = float(shock_score)
            nowcast_count += 1
            diagnostics.append(
                {
                    "quarter": quarter,
                    "month_mask": mask,
                    "partial_month_count": int(support_count),
                    "partial_monthly_diagnosis_sum": float(partial_sum),
                    "completion_scale": float(completion),
                    "diagnosis_nowcast": float(diagnosis_nowcast),
                    "incidence_nowcast": _finite_float(next_row.get("incident_infections_period")),
                    "incidence_updated": bool(update_incidence and beta.size == x.size),
                    "reporting_shock_score": float(shock_score),
                    "nowcast_source": nowcast_source,
                    "contract": "uses direct monthly HARP diagnosis rows inside or immediately before the target quarter; no quarterly target row is used",
                }
            )
        output.append(next_row)
    updated_head = MonthlyReportingNowcastHead(
        family=head.family,
        completion_by_month_mask=dict(head.completion_by_month_mask),
        training_partial_by_month_mask={
            str(key): [float(value) for value in list(values)]
            for key, values in dict(head.training_partial_by_month_mask).items()
        },
        lead_completion_by_month_mask=dict(head.lead_completion_by_month_mask),
        training_lead_partial_by_month_mask={
            str(key): [float(value) for value in list(values)]
            for key, values in dict(head.training_lead_partial_by_month_mask).items()
        },
        fallback_completion_scale=float(head.fallback_completion_scale),
        fallback_lead_completion_scale=float(head.fallback_lead_completion_scale),
        incidence_feature_names=list(head.incidence_feature_names),
        incidence_coefficients=list(head.incidence_coefficients),
        incidence_prediction_floor=float(head.incidence_prediction_floor),
        incidence_prediction_ceiling=float(head.incidence_prediction_ceiling),
        train_quarter_count=int(head.train_quarter_count),
        direct_monthly_row_count=int(head.direct_monthly_row_count),
        train_partial_quarter_count=int(head.train_partial_quarter_count),
        holdout_nowcast_quarter_count=int(nowcast_count),
        training_objective=head.training_objective,
    )
    return output, updated_head, diagnostics


def _row_time_label(row: dict[str, Any]) -> str:
    return str(row.get("time") or row.get("period_end") or row.get("period_start") or "")


def _empirical_metric_severity(values: list[float], *, high_is_late: bool) -> list[float]:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return []
    if len(finite) == 1:
        return [0.5]
    ordered = np.asarray(sorted(finite), dtype=np.float64)
    denominator = float(len(ordered) - 1)
    severities: list[float] = []
    for value in finite:
        rank = float(np.searchsorted(ordered, float(value), side="right") - 1) / denominator
        severities.append(rank if high_is_late else 1.0 - rank)
    return severities


def _late_diagnosis_severity_by_month(
    context: MonthlyShockContext,
    *,
    diagnosis_by_month: dict[int, float],
    train_end_month: int,
) -> dict[int, float]:
    archive_path = (
        Path(context.epigraph_root)
        / "artifacts"
        / "runs"
        / context.source_run_id
        / "harp_archive"
        / "historical_metric_rows.json"
    )
    contract_lookup, _ledger = build_observation_contract_lookup(
        Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )
    metric_values: dict[str, list[tuple[int, float]]] = {}
    for row_index, row in enumerate(list(read_json(archive_path, default=[]) or [])):
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in LATE_DIAGNOSIS_EVIDENCE_METRICS:
            continue
        time_label = _row_time_label(row)
        if not time_label:
            continue
        month = _month_index(time_label)
        if int(month) > int(train_end_month):
            continue
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = dict(contract_lookup.get(row_hash) or {})
        if str(contract.get("observation_role") or "") not in LATE_DIAGNOSIS_ALLOWED_ROLES:
            continue
        value = _finite_float(row.get("value"))
        if value is None:
            continue
        if metric_name == "advanced_hiv_cases_period":
            diagnosis_denominator = sum(
                float(diagnosis_by_month.get(q_month) or 0.0)
                for q_month in _quarter_month_indices(_month_to_quarter(month))
            )
            if diagnosis_denominator > 0.0:
                value = float(value) / diagnosis_denominator
        metric_values.setdefault(metric_name, []).append((int(month), max(float(value), 0.0)))
    severities_by_month: dict[int, list[float]] = {}
    for metric_name, pairs in metric_values.items():
        high_is_late = metric_name not in {"median_cd4_at_enrollment", "median_cd4_at_diagnosis"}
        severities = _empirical_metric_severity([value for _month, value in pairs], high_is_late=high_is_late)
        for (month, _value), severity in zip(pairs, severities):
            severities_by_month.setdefault(int(month), []).append(float(np.clip(severity, 0.0, 1.0)))
    return {
        int(month): float(np.mean(np.asarray(values, dtype=np.float64)))
        for month, values in severities_by_month.items()
        if values
    }


def _late_diagnosis_emission_observations(
    context: MonthlyShockContext,
    *,
    diagnosis_by_month: dict[int, float],
    train_end_month: int,
) -> dict[str, Any]:
    archive_path = (
        Path(context.epigraph_root)
        / "artifacts"
        / "runs"
        / context.source_run_id
        / "harp_archive"
        / "historical_metric_rows.json"
    )
    contract_lookup, _ledger = build_observation_contract_lookup(
        Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )
    cd4_pairs: list[tuple[int, float]] = []
    late_share_by_month: dict[int, list[float]] = {}
    advanced_count_by_month: dict[int, float] = {}
    metric_counts: dict[str, int] = {metric_name: 0 for metric_name in LATE_DIAGNOSIS_EVIDENCE_METRICS}
    for row_index, row in enumerate(list(read_json(archive_path, default=[]) or [])):
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in LATE_DIAGNOSIS_EVIDENCE_METRICS:
            continue
        time_label = _row_time_label(row)
        if not time_label:
            continue
        month = _month_index(time_label)
        if int(month) > int(train_end_month):
            continue
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = dict(contract_lookup.get(row_hash) or {})
        if str(contract.get("observation_role") or "") not in LATE_DIAGNOSIS_ALLOWED_ROLES:
            continue
        value = _finite_float(row.get("value"))
        if value is None:
            continue
        metric_counts[metric_name] = int(metric_counts.get(metric_name, 0)) + 1
        if metric_name in {"median_cd4_at_enrollment", "median_cd4_at_diagnosis"}:
            cd4_pairs.append((int(month), max(float(value), 0.0)))
            continue
        if metric_name == "late_hiv_diagnosis_percent":
            late_share_by_month.setdefault(int(month), []).append(float(np.clip(float(value) / 100.0, 0.0, 1.0)))
            continue
        if metric_name == "advanced_hiv_cases_period":
            advanced_count_by_month[int(month)] = max(float(value), 0.0)
            diagnosis_denominator = sum(
                float(diagnosis_by_month.get(q_month) or 0.0)
                for q_month in _quarter_month_indices(_month_to_quarter(month))
            )
            if diagnosis_denominator > 0.0:
                late_share_by_month.setdefault(int(month), []).append(float(np.clip(float(value) / diagnosis_denominator, 0.0, 1.0)))
            continue
        if metric_name == "advanced_hiv_disease_share":
            late_share_by_month.setdefault(int(month), []).append(float(np.clip(float(value), 0.0, 1.0)))
    if cd4_pairs:
        severities = _empirical_metric_severity([value for _month, value in cd4_pairs], high_is_late=False)
        for (month, _value), severity in zip(cd4_pairs, severities):
            late_share_by_month.setdefault(int(month), []).append(float(np.clip(severity, 0.0, 1.0)))
    combined_late_share = {
        int(month): float(np.mean(np.asarray(values, dtype=np.float64)))
        for month, values in late_share_by_month.items()
        if values
    }
    return {
        "late_share_by_month": combined_late_share,
        "advanced_count_by_month": advanced_count_by_month,
        "metric_counts": metric_counts,
        "late_share_month_count": len(combined_late_share),
        "advanced_count_month_count": len(advanced_count_by_month),
    }


def _late_diagnosis_evidence_months(
    context: MonthlyShockContext,
    *,
    max_month: int,
) -> dict[int, dict[str, Any]]:
    archive_path = (
        Path(context.epigraph_root)
        / "artifacts"
        / "runs"
        / context.source_run_id
        / "harp_archive"
        / "historical_metric_rows.json"
    )
    contract_lookup, _ledger = build_observation_contract_lookup(
        Path(context.epigraph_root),
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
    )
    months: dict[int, dict[str, Any]] = {}
    for row_index, row in enumerate(list(read_json(archive_path, default=[]) or [])):
        metric_name = str(row.get("metric_name") or "")
        if metric_name not in LATE_DIAGNOSIS_EVIDENCE_METRICS:
            continue
        time_label = _row_time_label(row)
        if not time_label:
            continue
        month = _month_index(time_label)
        if int(month) > int(max_month):
            continue
        source_path = build_source_row_path(archive_path, row_index)
        row_hash = build_contract_row_hash(row=row, source_path=source_path)
        contract = dict(contract_lookup.get(row_hash) or {})
        if str(contract.get("observation_role") or "") not in LATE_DIAGNOSIS_ALLOWED_ROLES:
            continue
        if _finite_float(row.get("value")) is None:
            continue
        entry = months.setdefault(
            int(month),
            {
                "metric_names": set(),
                "row_hashes": set(),
                "observation_roles": set(),
            },
        )
        entry["metric_names"].add(metric_name)
        entry["row_hashes"].add(row_hash)
        entry["observation_roles"].add(str(contract.get("observation_role") or ""))
    return {
        month: {
            "metric_names": sorted(value["metric_names"]),
            "row_hashes": sorted(value["row_hashes"]),
            "observation_roles": sorted(value["observation_roles"]),
        }
        for month, value in sorted(months.items())
    }


def _late_evidence_supported_quarters(
    context: MonthlyShockContext,
    *,
    quarters: list[str],
) -> dict[str, dict[str, Any]]:
    if not quarters:
        return {}
    last_quarter = max(quarters, key=quarter_sort_key)
    _start, max_month = _quarter_month_range(last_quarter)
    evidence_by_month = _late_diagnosis_evidence_months(context, max_month=max_month)
    supported: dict[str, dict[str, Any]] = {}
    for quarter in quarters:
        quarter_months = set(_quarter_month_indices(quarter))
        matched_months = [month for month in sorted(evidence_by_month) if month in quarter_months]
        if not matched_months:
            continue
        metric_names = sorted(
            {
                metric_name
                for month in matched_months
                for metric_name in list((evidence_by_month.get(month) or {}).get("metric_names") or [])
            }
        )
        supported[quarter] = {
            "support_months": [_month_label(month) for month in matched_months],
            "metric_names": metric_names,
            "support_rule": "same_quarter_ledger_allowed_late_diagnosis_evidence",
        }
    return supported


def _late_evidence_agreement_by_quarter(
    *,
    context: MonthlyShockContext,
    dataset: Any,
    base_rows: list[dict[str, Any]],
    backlog_head: BacklogLateEmissionHead,
) -> dict[str, dict[str, Any]]:
    holdout_quarters = [str(row.get("quarter") or "") for row in sorted(base_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    if not holdout_quarters:
        return {}
    train_transition_rows = sorted(
        list(getattr(dataset, "train_transition_rows", []) or []),
        key=lambda row: quarter_sort_key(str(row.get("quarter") or "")),
    )
    if not train_transition_rows:
        return {}
    train_end_quarter = str(train_transition_rows[-1].get("quarter") or "")
    _train_start, train_end_month = _quarter_month_range(train_end_quarter)
    train_diagnosis_by_month = _direct_monthly_diagnosis_values(context, train_end_month=train_end_month)
    train_emissions = _late_diagnosis_emission_observations(
        context,
        diagnosis_by_month=train_diagnosis_by_month,
        train_end_month=train_end_month,
    )
    train_late_mean = _mean([float(value) for value in dict(train_emissions.get("late_share_by_month") or {}).values()])
    if train_late_mean is None:
        train_late_mean = 0.0
    train_advanced_counts = [float(value) for value in dict(train_emissions.get("advanced_count_by_month") or {}).values()]
    train_advanced_reference = float(np.median(np.asarray(train_advanced_counts, dtype=np.float64))) if train_advanced_counts else 0.0
    last_quarter = max(holdout_quarters, key=quarter_sort_key)
    _holdout_start, max_month = _quarter_month_range(last_quarter)
    diagnosis_by_month = _direct_monthly_diagnosis_values(context, train_end_month=max_month)
    all_emissions = _late_diagnosis_emission_observations(
        context,
        diagnosis_by_month=diagnosis_by_month,
        train_end_month=max_month,
    )
    late_share_by_month = dict(all_emissions.get("late_share_by_month") or {})
    advanced_count_by_month = dict(all_emissions.get("advanced_count_by_month") or {})
    train_prior = _quarterly_incidence_to_monthly(train_transition_rows, value_key="incident_infections_period")
    holdout_incidence = _quarterly_incidence_to_monthly(base_rows, value_key="incident_infections_period")
    forecast_months = [month for quarter in holdout_quarters for month in _quarter_month_indices(quarter)]
    combined_incidence = {
        **{int(month): float(value) for month, value in train_prior.items()},
        **{int(month): float(value) for month, value in holdout_incidence.items() if int(month) in set(forecast_months)},
    }
    simulated = _simulate_backlog_late_emission(
        incidence_by_month=combined_incidence,
        months=sorted(set(combined_incidence).union(forecast_months)),
        early_to_late_hazard=float(backlog_head.early_to_late_hazard),
        early_diagnosis_hazard=float(backlog_head.early_diagnosis_hazard),
        late_diagnosis_hazard=float(backlog_head.late_diagnosis_hazard),
        initial_early_undiagnosed=float(backlog_head.initial_early_undiagnosed),
        initial_late_undiagnosed=float(backlog_head.initial_late_undiagnosed),
    )
    agreement: dict[str, dict[str, Any]] = {}
    for quarter in holdout_quarters:
        quarter_months = _quarter_month_indices(quarter)
        candidate_losses: list[float] = []
        neutral_losses: list[float] = []
        metric_names: set[str] = set()
        support_months: set[int] = set()
        for month in quarter_months:
            if month in late_share_by_month:
                target = float(np.clip(float(late_share_by_month[month]), 0.0, 1.0))
                prediction = float((simulated.get(month) or {}).get("late_diagnosis_share") or 0.0)
                candidate_losses.append(abs(prediction - target))
                neutral_losses.append(abs(float(train_late_mean) - target))
                metric_names.add("late_share_proxy")
                support_months.add(month)
            if month in advanced_count_by_month:
                target = max(float(advanced_count_by_month[month]), 0.0)
                predicted_late_count = sum(
                    float((simulated.get(candidate_month) or {}).get("late_diagnoses") or 0.0)
                    for candidate_month in quarter_months
                )
                scale = max(float(target), 1.0)
                candidate_losses.append(abs(predicted_late_count - target) / scale)
                neutral_losses.append(abs(float(train_advanced_reference) - target) / scale)
                metric_names.add("advanced_hiv_cases_period")
                support_months.add(month)
        if not candidate_losses or not neutral_losses:
            continue
        candidate_loss = float(np.mean(np.asarray(candidate_losses, dtype=np.float64)))
        neutral_loss = float(np.mean(np.asarray(neutral_losses, dtype=np.float64)))
        agreement[quarter] = {
            "agreement_supported": bool(candidate_loss <= neutral_loss),
            "candidate_late_evidence_loss": candidate_loss,
            "neutral_late_evidence_loss": neutral_loss,
            "support_months": [_month_label(month) for month in sorted(support_months)],
            "metric_names": sorted(metric_names),
            "support_rule": "same_quarter_late_diagnosis_evidence_and_candidate_loss_not_above_neutral_loss",
        }
    return agreement


def _latest_late_severity(month: int, severity_by_month: dict[int, float]) -> float | None:
    candidates = [candidate for candidate in severity_by_month if int(candidate) <= int(month)]
    if not candidates:
        return None
    return float(severity_by_month[max(candidates)])


def _late_delay_prior_by_lag(
    *,
    lags: list[int],
    diagnosis_by_month: dict[int, float],
    incidence_prior_by_month: dict[int, float],
    late_severity_by_month: dict[int, float] | None,
) -> dict[int, float] | None:
    if not lags or not late_severity_by_month:
        return None
    min_lag = min(lags)
    max_lag = max(lags)
    denominator = float(max_lag - min_lag) if max_lag > min_lag else 1.0
    scores = {int(lag): 0.0 for lag in lags}
    support_count = 0
    for month in sorted(diagnosis_by_month):
        severity = _latest_late_severity(month, late_severity_by_month)
        if severity is None:
            continue
        support_count += 1
        for lag in lags:
            lag_rank = (float(lag) - float(min_lag)) / denominator
            monotone_late_shape = float(severity) * lag_rank + (1.0 - float(severity)) * (1.0 - lag_rank)
            scores[int(lag)] += max(float(incidence_prior_by_month.get(int(month) - int(lag)) or 0.0), 0.0) * monotone_late_shape
    total = float(sum(scores.values()))
    if support_count == 0 or total <= float(np.finfo(np.float64).eps):
        return None
    return {lag: float(value) / total for lag, value in scores.items()}


def _fit_delay_kernel_from_prior(
    *,
    diagnosis_by_month: dict[int, float],
    incidence_prior_by_month: dict[int, float],
    late_severity_by_month: dict[int, float] | None = None,
) -> dict[str, Any] | None:
    observed_months = sorted(month for month, value in diagnosis_by_month.items() if _finite_float(value) is not None)
    prior_months = sorted(month for month, value in incidence_prior_by_month.items() if _finite_float(value) is not None)
    if len(observed_months) < 2 or not prior_months:
        return None
    max_lag = max(max(observed_months) - min(prior_months), 0)
    candidate_lags = [
        lag
        for lag in range(max_lag + 1)
        if any((month - lag) in incidence_prior_by_month for month in observed_months)
    ]
    if not candidate_lags:
        return None
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for month in observed_months:
        x_rows.append([max(float(incidence_prior_by_month.get(month - lag) or 0.0), 0.0) for lag in candidate_lags])
        y_rows.append(max(float(diagnosis_by_month.get(month) or 0.0), 0.0))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    theta = np.linalg.pinv(x) @ y
    theta = np.maximum(theta, 0.0)
    unconstrained_scale = float(np.sum(theta))
    if unconstrained_scale <= float(np.finfo(np.float64).eps):
        return None
    kernel = theta / unconstrained_scale
    late_prior = _late_delay_prior_by_lag(
        lags=[int(lag) for lag in candidate_lags],
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=incidence_prior_by_month,
        late_severity_by_month=late_severity_by_month,
    )
    if late_prior:
        prior_vector = np.asarray([float(late_prior.get(int(lag)) or 0.0) for lag in candidate_lags], dtype=np.float64)
        constrained_kernel = np.maximum(kernel, 0.0) * np.maximum(prior_vector, 0.0)
        constrained_sum = float(np.sum(constrained_kernel))
        if constrained_sum > float(np.finfo(np.float64).eps):
            kernel = constrained_kernel / constrained_sum
    scale = min(unconstrained_scale, 1.0)
    fitted = x @ (kernel * scale)
    residual = y - fitted
    return {
        "lags": [int(lag) for lag in candidate_lags],
        "kernel": [float(value) for value in kernel],
        "ascertainment_scale": scale,
        "unconstrained_ascertainment_scale": unconstrained_scale,
        "train_fit_mae": float(np.mean(np.abs(residual))) if residual.size else None,
        "observed_month_count": len(observed_months),
        "late_constraint": None if not late_prior else {
            "status": "applied",
            "metric_ids": list(LATE_DIAGNOSIS_EVIDENCE_METRICS),
            "late_severity_month_count": len(late_severity_by_month or {}),
            "prior_support_lag_count": len(late_prior),
            "prior_top_lag_months": max(late_prior, key=lambda lag: float(late_prior[lag])),
            "prior_top_lag_weight": max(float(value) for value in late_prior.values()),
        },
    }


def _compressed_probability_grid(values: list[float]) -> list[float]:
    finite = sorted(
        {
            float(np.clip(value, 0.0, 1.0))
            for value in values
            if np.isfinite(float(value))
            and float(value) > float(np.finfo(np.float64).eps)
            and float(value) < 1.0 - float(np.finfo(np.float64).eps)
        }
    )
    if len(finite) <= 1:
        return finite
    target_count = int(np.ceil(np.sqrt(float(len(finite)))))
    quantiles = np.linspace(0.0, 1.0, target_count)
    return sorted(
        {
            float(np.clip(np.quantile(np.asarray(finite, dtype=np.float64), float(q)), 0.0, 1.0))
            for q in quantiles
        }
    )


def _backlog_candidate_hazard_grid(
    *,
    diagnosis_by_month: dict[int, float],
    incidence_prior_by_month: dict[int, float],
    late_share_by_month: dict[int, float],
) -> list[float]:
    values: list[float] = []
    for month, diagnosis_value in diagnosis_by_month.items():
        incidence_value = max(float(incidence_prior_by_month.get(int(month)) or 0.0), 0.0)
        diagnosis = max(float(diagnosis_value), 0.0)
        denominator = diagnosis + incidence_value
        if denominator > 0.0:
            values.append(diagnosis / denominator)
    for value in late_share_by_month.values():
        share = float(np.clip(value, 0.0, 1.0))
        values.extend([share, 1.0 - share])
    return _compressed_probability_grid(values)


def _simulate_backlog_late_emission(
    *,
    incidence_by_month: dict[int, float],
    months: list[int],
    early_to_late_hazard: float,
    early_diagnosis_hazard: float,
    late_diagnosis_hazard: float,
    initial_early_undiagnosed: float,
    initial_late_undiagnosed: float,
) -> dict[int, dict[str, float]]:
    early_state = max(float(initial_early_undiagnosed), 0.0)
    late_state = max(float(initial_late_undiagnosed), 0.0)
    gamma = float(np.clip(early_to_late_hazard, 0.0, 1.0))
    h_early = float(np.clip(early_diagnosis_hazard, 0.0, 1.0))
    h_late = float(np.clip(late_diagnosis_hazard, 0.0, 1.0))
    output: dict[int, dict[str, float]] = {}
    for month in sorted(int(value) for value in months):
        infections = max(float(incidence_by_month.get(month) or 0.0), 0.0)
        early_available = early_state + infections
        early_diagnoses = h_early * early_available
        progressed = gamma * max(early_available - early_diagnoses, 0.0)
        late_available = late_state + progressed
        late_diagnoses = h_late * late_available
        total_diagnoses = early_diagnoses + late_diagnoses
        next_early = max(early_available - early_diagnoses - progressed, 0.0)
        next_late = max(late_available - late_diagnoses, 0.0)
        output[month] = {
            "incident_infections": float(infections),
            "early_undiagnosed_start": float(early_state),
            "late_undiagnosed_start": float(late_state),
            "early_diagnoses": float(early_diagnoses),
            "late_diagnoses": float(late_diagnoses),
            "new_diagnosed_cases_period": float(total_diagnoses),
            "late_diagnosis_share": 0.0 if total_diagnoses <= 0.0 else float(late_diagnoses / total_diagnoses),
            "early_to_late_progression": float(progressed),
            "early_undiagnosed_end": float(next_early),
            "late_undiagnosed_end": float(next_late),
        }
        early_state = next_early
        late_state = next_late
    return output


def _advanced_count_loss(
    *,
    simulated: dict[int, dict[str, float]],
    advanced_count_by_month: dict[int, float],
) -> float | None:
    losses: list[float] = []
    for month, target in advanced_count_by_month.items():
        quarter_months = _quarter_month_indices(_month_to_quarter(int(month)))
        predicted = sum(float((simulated.get(q_month) or {}).get("late_diagnoses") or 0.0) for q_month in quarter_months)
        target_value = max(float(target), 0.0)
        losses.append(abs(predicted - target_value) / max(target_value, 1.0))
    return _mean(losses)


def _backlog_emission_losses(
    *,
    simulated: dict[int, dict[str, float]],
    diagnosis_by_month: dict[int, float],
    late_share_by_month: dict[int, float],
    advanced_count_by_month: dict[int, float],
) -> dict[str, float | None]:
    diagnosis_losses: list[float] = []
    for month, observed in diagnosis_by_month.items():
        predicted = max(float((simulated.get(int(month)) or {}).get("new_diagnosed_cases_period") or 0.0), 0.0)
        observed_value = max(float(observed), 0.0)
        diagnosis_losses.append(abs(float(np.log1p(predicted)) - float(np.log1p(observed_value))))
    late_losses = [
        abs(float((simulated.get(int(month)) or {}).get("late_diagnosis_share") or 0.0) - float(np.clip(target, 0.0, 1.0)))
        for month, target in late_share_by_month.items()
        if int(month) in simulated
    ]
    advanced_loss = _advanced_count_loss(
        simulated=simulated,
        advanced_count_by_month={month: value for month, value in advanced_count_by_month.items() if int(month) in simulated},
    )
    component_losses = [
        value
        for value in (
            _mean(diagnosis_losses),
            _mean(late_losses),
            advanced_loss,
        )
        if value is not None
    ]
    return {
        "train_loss": _mean(component_losses),
        "diagnosis_loss": _mean(diagnosis_losses),
        "late_share_loss": _mean(late_losses),
        "advanced_count_loss": advanced_loss,
    }


def _fit_backlog_late_emission_model(
    *,
    diagnosis_by_month: dict[int, float],
    incidence_prior_by_month: dict[int, float],
    emission_observations: dict[str, Any],
) -> dict[str, Any] | None:
    observed_months = sorted(month for month, value in diagnosis_by_month.items() if _finite_float(value) is not None)
    prior_months = sorted(month for month, value in incidence_prior_by_month.items() if _finite_float(value) is not None)
    if len(observed_months) < 2 or not prior_months:
        return None
    late_share_by_month = dict(emission_observations.get("late_share_by_month") or {})
    advanced_count_by_month = dict(emission_observations.get("advanced_count_by_month") or {})
    hazard_grid = _backlog_candidate_hazard_grid(
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=incidence_prior_by_month,
        late_share_by_month=late_share_by_month,
    )
    if not hazard_grid:
        return None
    first_month = observed_months[0]
    first_diagnosis = max(float(diagnosis_by_month.get(first_month) or 0.0), 0.0)
    first_late_share = _latest_late_severity(first_month, late_share_by_month)
    if first_late_share is None:
        first_late_share = _mean([float(value) for value in late_share_by_month.values()])
    if first_late_share is None:
        first_late_share = 0.0
    best: dict[str, Any] | None = None
    train_months = sorted(set(prior_months).union(observed_months))
    for early_to_late_hazard in hazard_grid:
        for early_diagnosis_hazard in hazard_grid:
            for late_diagnosis_hazard in hazard_grid:
                initial_early = (1.0 - float(first_late_share)) * first_diagnosis / max(float(early_diagnosis_hazard), float(np.finfo(np.float64).eps))
                initial_late = float(first_late_share) * first_diagnosis / max(float(late_diagnosis_hazard), float(np.finfo(np.float64).eps))
                simulated = _simulate_backlog_late_emission(
                    incidence_by_month=incidence_prior_by_month,
                    months=train_months,
                    early_to_late_hazard=float(early_to_late_hazard),
                    early_diagnosis_hazard=float(early_diagnosis_hazard),
                    late_diagnosis_hazard=float(late_diagnosis_hazard),
                    initial_early_undiagnosed=float(initial_early),
                    initial_late_undiagnosed=float(initial_late),
                )
                losses = _backlog_emission_losses(
                    simulated=simulated,
                    diagnosis_by_month=diagnosis_by_month,
                    late_share_by_month=late_share_by_month,
                    advanced_count_by_month=advanced_count_by_month,
                )
                train_loss = _finite_float(losses.get("train_loss"))
                if train_loss is None:
                    continue
                candidate = {
                    "early_to_late_hazard": float(early_to_late_hazard),
                    "early_diagnosis_hazard": float(early_diagnosis_hazard),
                    "late_diagnosis_hazard": float(late_diagnosis_hazard),
                    "initial_early_undiagnosed": float(initial_early),
                    "initial_late_undiagnosed": float(initial_late),
                    "train_loss": float(train_loss),
                    "diagnosis_loss": losses.get("diagnosis_loss"),
                    "late_share_loss": losses.get("late_share_loss"),
                    "advanced_count_loss": losses.get("advanced_count_loss"),
                    "train_end_early_undiagnosed": float((simulated.get(train_months[-1]) or {}).get("early_undiagnosed_end") or 0.0),
                    "train_end_late_undiagnosed": float((simulated.get(train_months[-1]) or {}).get("late_undiagnosed_end") or 0.0),
                }
                if best is None or (
                    float(candidate["train_loss"]),
                    float(candidate["diagnosis_loss"] or float("inf")),
                    float(candidate["late_share_loss"] or float("inf")),
                ) < (
                    float(best["train_loss"]),
                    float(best["diagnosis_loss"] or float("inf")),
                    float(best["late_share_loss"] or float("inf")),
                ):
                    best = candidate
    if best is None:
        return None
    best["hazard_grid_size"] = len(hazard_grid)
    best["observed_month_count"] = len(observed_months)
    best["train_month_count"] = len(prior_months)
    best["late_share_emission_month_count"] = len(late_share_by_month)
    best["advanced_count_emission_month_count"] = len(advanced_count_by_month)
    best["emission_summary"] = {
        "metric_counts": dict(emission_observations.get("metric_counts") or {}),
        "late_share_month_count": int(emission_observations.get("late_share_month_count") or 0),
        "advanced_count_month_count": int(emission_observations.get("advanced_count_month_count") or 0),
        "hazard_grid_size": len(hazard_grid),
    }
    return best


def _backcalculate_monthly_incidence(
    *,
    diagnosis_by_month: dict[int, float],
    incidence_prior_by_month: dict[int, float],
    lags: list[int],
    kernel: list[float],
    ascertainment_scale: float,
) -> dict[int, float]:
    if not lags or not kernel or float(ascertainment_scale) <= 0.0:
        return dict(incidence_prior_by_month)
    lag_kernel = {int(lag): max(float(value), 0.0) for lag, value in zip(lags, kernel)}
    zero_lag = max(float(lag_kernel.get(0) or 0.0), 0.0)
    if zero_lag <= float(np.finfo(np.float64).eps):
        return dict(incidence_prior_by_month)
    all_months = sorted(set(incidence_prior_by_month).union(diagnosis_by_month))
    prior_cap = max([max(float(value), 0.0) for value in incidence_prior_by_month.values()] or [0.0])
    backcalc: dict[int, float] = {}
    for month in all_months:
        prior = max(float(incidence_prior_by_month.get(month) or 0.0), 0.0)
        observed = _finite_float(diagnosis_by_month.get(month))
        if observed is None:
            backcalc[month] = prior
            continue
        lagged_sum = 0.0
        for lag, weight in lag_kernel.items():
            if lag == 0:
                continue
            lagged_sum += float(weight) * max(float(backcalc.get(month - lag, incidence_prior_by_month.get(month - lag, 0.0)) or 0.0), 0.0)
        implied = (max(float(observed), 0.0) / float(ascertainment_scale) - lagged_sum) / zero_lag
        backcalc[month] = min(max(float(implied), 0.0), prior_cap) if prior_cap > 0.0 else max(float(implied), 0.0)
    return backcalc


def _fit_monthly_log_ar_forecast(train_values_by_month: dict[int, float], forecast_months: list[int]) -> dict[str, Any]:
    train_months = sorted(train_values_by_month)
    if not train_months:
        return {"forecast": {month: 0.0 for month in forecast_months}, "parameters": {"alpha": 0.0, "slope": 0.0, "rho": 0.0}}
    transformed = [float(np.log1p(max(float(train_values_by_month[month]), 0.0))) for month in train_months]
    train_cap = max([max(float(value), 0.0) for value in train_values_by_month.values()] or [0.0])
    if len(transformed) < 2:
        eta = float(transformed[-1])
        return {
            "forecast": {month: float(min(max(np.expm1(eta), 0.0), train_cap)) for month in forecast_months},
            "parameters": {"alpha": eta, "slope": 0.0, "rho": 0.0, "train_monthly_cap": train_cap},
        }
    y = np.asarray(transformed[1:], dtype=np.float64)
    x = np.asarray(
        [[1.0, float(train_months[idx]), float(transformed[idx - 1])] for idx in range(1, len(transformed))],
        dtype=np.float64,
    )
    beta = np.linalg.pinv(x) @ y
    alpha = float(beta[0])
    slope = float(beta[1])
    rho = float(np.clip(beta[2], -1.0, 1.0))
    last_eta = float(transformed[-1])
    forecast: dict[int, float] = {}
    for month in sorted(forecast_months):
        eta = alpha + slope * float(month) + rho * last_eta
        forecast[int(month)] = float(min(max(np.expm1(eta), 0.0), train_cap)) if train_cap > 0.0 else float(max(np.expm1(eta), 0.0))
        last_eta = float(eta)
    return {"forecast": forecast, "parameters": {"alpha": alpha, "slope": slope, "rho": rho, "train_monthly_cap": train_cap}}


def _convolve_monthly_diagnoses(
    *,
    incidence_by_month: dict[int, float],
    lags: list[int],
    kernel: list[float],
    ascertainment_scale: float,
    months: list[int],
) -> dict[int, float]:
    lag_kernel = {int(lag): max(float(value), 0.0) for lag, value in zip(lags, kernel)}
    output: dict[int, float] = {}
    for month in sorted(months):
        value = 0.0
        for lag, weight in lag_kernel.items():
            value += float(weight) * max(float(incidence_by_month.get(month - lag) or 0.0), 0.0)
        output[int(month)] = float(max(float(ascertainment_scale) * value, 0.0))
    return output


def _aggregate_months_to_quarters(values_by_month: dict[int, float], quarters: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for quarter in quarters:
        result[str(quarter)] = float(sum(max(float(values_by_month.get(month) or 0.0), 0.0) for month in _quarter_month_indices(str(quarter))))
    return result


def _feature_row(
    *,
    family: str,
    backbone_row: dict[str, Any],
    trajectory_row: dict[str, Any] | None,
    step_index: int,
    horizon_quarters: int,
) -> list[float]:
    values = {
        "intercept": 1.0,
        "log_incidence_inflow": float(np.log1p(_trajectory_incidence(trajectory_row))),
        "log_backbone_diagnosis_flow": float(np.log1p(max(float(backbone_row.get("new_diagnosed_cases_period") or 0.0), 0.0))),
        "log_undiagnosed_state": float(np.log1p(_trajectory_u_state(trajectory_row))),
        "horizon_fraction": float(int(step_index) + 1) / max(float(horizon_quarters), 1.0),
    }
    return [float(values[name]) for name in _feature_names(family)]


def _incidence_readout_feature_row(
    *,
    family: str,
    diagnosis_row: dict[str, Any],
    step_index: int,
    horizon_quarters: int,
) -> list[float]:
    values = {
        "intercept": 1.0,
        "log_diagnosis_flow": float(np.log1p(max(float(diagnosis_row.get("new_diagnosed_cases_period") or 0.0), 0.0))),
        "horizon_fraction": float(int(step_index) + 1) / max(float(horizon_quarters), 1.0),
    }
    return [float(values[name]) for name in _incidence_readout_feature_names(family)]


def fit_diagnosis_flow_repair_head(
    *,
    repair_family: str,
    base_family: str,
    train_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext | None,
    monthly_latent_context: MonthlyLatentContext | None,
    monthly_joint_context: MonthlyJointContext | None,
    backhalf_context: BackHalfChannelContext | None,
    min_train_years: int,
) -> DiagnosisFlowRepairHead | None:
    if repair_family not in DIAGNOSIS_FLOW_REPAIR_FAMILIES:
        return None
    examples = _train_multihorizon_examples(
        family=base_family,
        train_rows=train_rows,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        min_train_years=min_train_years,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for example in examples:
        target = _finite_float((example.get("target_row") or {}).get("new_diagnosed_cases_period"))
        if target is None:
            continue
        x_rows.append(
            _feature_row(
                family=repair_family,
                backbone_row=dict(example.get("backbone_row") or {}),
                trajectory_row=dict(example.get("trajectory_row") or {}),
                step_index=int(example.get("step_index") or 0),
                horizon_quarters=int(example.get("horizon_quarters") or 1),
            )
        )
        y_rows.append(float(np.log1p(max(float(target), 0.0))))
    if not x_rows:
        return None
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    beta = np.linalg.pinv(x) @ y
    return DiagnosisFlowRepairHead(
        family=repair_family,
        feature_names=_feature_names(repair_family),
        coefficients=[float(value) for value in beta],
        train_row_count=len(x_rows),
    )


def fit_incidence_readout_repair_head(
    *,
    repair_family: str,
    base_family: str,
    train_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    constraint_rows: list[dict[str, Any]],
    monthly_context: MonthlyShockContext | None,
    monthly_latent_context: MonthlyLatentContext | None,
    monthly_joint_context: MonthlyJointContext | None,
    backhalf_context: BackHalfChannelContext | None,
    min_train_years: int,
) -> IncidenceReadoutRepairHead | None:
    if repair_family not in INCIDENCE_READOUT_REPAIR_FAMILIES:
        return None
    examples = _train_multihorizon_examples(
        family=base_family,
        train_rows=train_rows,
        reference_config=reference_config,
        constraint_rows=constraint_rows,
        min_train_years=min_train_years,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    x_rows: list[list[float]] = []
    y_rows: list[float] = []
    for example in examples:
        trajectory_row = dict(example.get("trajectory_row") or {})
        target = _trajectory_incidence(trajectory_row)
        diagnosis_row = dict(example.get("target_row") or {})
        if _finite_float(diagnosis_row.get("new_diagnosed_cases_period")) is None:
            continue
        x_rows.append(
            _incidence_readout_feature_row(
                family=repair_family,
                diagnosis_row=diagnosis_row,
                step_index=int(example.get("step_index") or 0),
                horizon_quarters=int(example.get("horizon_quarters") or 1),
            )
        )
        y_rows.append(float(np.log1p(max(float(target), 0.0))))
    if not x_rows:
        return None
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.float64)
    beta = np.linalg.pinv(x) @ y
    return IncidenceReadoutRepairHead(
        family=repair_family,
        feature_names=_incidence_readout_feature_names(repair_family),
        coefficients=[float(value) for value in beta],
        train_row_count=len(x_rows),
    )


def _inject_incidence_from_trajectory(
    rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trajectory_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in trajectory_rows
    }
    output: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        quarter = str(row.get("quarter") or "")
        if quarter in trajectory_by_quarter:
            next_row["incident_infections_period"] = _trajectory_incidence(trajectory_by_quarter[quarter])
        output.append(next_row)
    return output


def apply_diagnosis_flow_repair_head(
    *,
    head: DiagnosisFlowRepairHead | None,
    base_rows: list[dict[str, Any]],
    backbone_rows: list[dict[str, Any]],
    trajectory_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if head is None:
        return [dict(row) for row in base_rows]
    backbone_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in backbone_rows
    }
    trajectory_by_quarter = {
        str(row.get("quarter") or ""): dict(row)
        for row in trajectory_rows
    }
    beta = np.asarray(head.coefficients, dtype=np.float64)
    ordered = sorted(list(base_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    horizon_quarters = len(ordered)
    output: list[dict[str, Any]] = []
    for step_index, row in enumerate(ordered):
        quarter = str(row.get("quarter") or "")
        backbone_row = backbone_by_quarter.get(quarter, row)
        trajectory_row = trajectory_by_quarter.get(quarter)
        features = np.asarray(
            _feature_row(
                family=head.family,
                backbone_row=backbone_row,
                trajectory_row=trajectory_row,
                step_index=step_index,
                horizon_quarters=horizon_quarters,
            ),
            dtype=np.float64,
        )
        next_row = dict(row)
        if beta.size == features.size:
            eta = float(features @ beta)
            next_row["new_diagnosed_cases_period"] = float(max(np.expm1(eta), 0.0))
        output.append(next_row)
    return output


def apply_incidence_readout_repair_head(
    *,
    head: IncidenceReadoutRepairHead | None,
    base_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if head is None:
        return [dict(row) for row in base_rows]
    beta = np.asarray(head.coefficients, dtype=np.float64)
    ordered = sorted(list(base_rows), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    horizon_quarters = len(ordered)
    output: list[dict[str, Any]] = []
    for step_index, row in enumerate(ordered):
        features = np.asarray(
            _incidence_readout_feature_row(
                family=head.family,
                diagnosis_row=dict(row),
                step_index=step_index,
                horizon_quarters=horizon_quarters,
            ),
            dtype=np.float64,
        )
        next_row = dict(row)
        if beta.size == features.size:
            eta = float(features @ beta)
            next_row["incident_infections_period"] = float(max(np.expm1(eta), 0.0))
        output.append(next_row)
    return output


def apply_diagnosis_delay_backcalc_branch(
    *,
    repair_family: str,
    dataset: Any,
    monthly_context: MonthlyShockContext,
    base_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], DiagnosisDelayBackcalcHead | None]:
    if repair_family not in DIAGNOSIS_DELAY_BACKCALC_FAMILIES:
        return [dict(row) for row in base_rows], None
    train_transition_rows = sorted(
        list(getattr(dataset, "train_transition_rows", []) or []),
        key=lambda row: quarter_sort_key(str(row.get("quarter") or "")),
    )
    if not train_transition_rows or not base_rows:
        return [dict(row) for row in base_rows], None
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in sorted(base_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    train_end_quarter = train_quarters[-1]
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    diagnosis_by_month = _direct_monthly_diagnosis_values(monthly_context, train_end_month=train_end_month)
    train_prior = _quarterly_incidence_to_monthly(train_transition_rows, value_key="incident_infections_period")
    late_severity_by_month = None
    if repair_family == "monthly_delay_backcalc_late_constrained":
        late_severity_by_month = _late_diagnosis_severity_by_month(
            monthly_context,
            diagnosis_by_month=diagnosis_by_month,
            train_end_month=train_end_month,
        )
    kernel_fit = _fit_delay_kernel_from_prior(
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=train_prior,
        late_severity_by_month=late_severity_by_month,
    )
    if kernel_fit is None:
        return [dict(row) for row in base_rows], None
    lags = [int(value) for value in list(kernel_fit["lags"])]
    kernel = [float(value) for value in list(kernel_fit["kernel"])]
    scale = float(kernel_fit["ascertainment_scale"])
    backcalc_train = _backcalculate_monthly_incidence(
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=train_prior,
        lags=lags,
        kernel=kernel,
        ascertainment_scale=scale,
    )
    forecast_months = [month for quarter in holdout_quarters for month in _quarter_month_indices(quarter)]
    monthly_forecast = _fit_monthly_log_ar_forecast(backcalc_train, forecast_months)
    holdout_incidence = {int(month): float(value) for month, value in dict(monthly_forecast["forecast"]).items()}
    combined_incidence = {**backcalc_train, **holdout_incidence}
    holdout_diagnoses = _convolve_monthly_diagnoses(
        incidence_by_month=combined_incidence,
        lags=lags,
        kernel=kernel,
        ascertainment_scale=scale,
        months=forecast_months,
    )
    incidence_by_quarter = _aggregate_months_to_quarters(holdout_incidence, holdout_quarters)
    diagnosis_by_quarter = _aggregate_months_to_quarters(holdout_diagnoses, holdout_quarters)
    candidate_rows: list[dict[str, Any]] = []
    for row in base_rows:
        quarter = str(row.get("quarter") or "")
        next_row = dict(row)
        if quarter in incidence_by_quarter:
            next_row["incident_infections_period"] = float(incidence_by_quarter[quarter])
        if quarter in diagnosis_by_quarter:
            next_row["new_diagnosed_cases_period"] = float(diagnosis_by_quarter[quarter])
        candidate_rows.append(next_row)
    head = DiagnosisDelayBackcalcHead(
        family=repair_family,
        delay_lags_months=lags,
        delay_kernel=kernel,
        ascertainment_scale=scale,
        train_month_count=len(train_prior),
        observed_diagnosis_month_count=int(kernel_fit["observed_month_count"]),
        backcalculated_month_count=len(backcalc_train),
        forecast_month_count=len(holdout_incidence),
        monthly_ar_parameters=dict(monthly_forecast["parameters"]),
        late_constraint_summary=kernel_fit.get("late_constraint"),
    )
    return candidate_rows, head


def apply_backlog_late_emission_branch(
    *,
    repair_family: str,
    dataset: Any,
    monthly_context: MonthlyShockContext,
    base_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], BacklogLateEmissionHead | None]:
    if repair_family not in BACKLOG_LATE_EMISSION_FAMILIES:
        return [dict(row) for row in base_rows], None
    train_transition_rows = sorted(
        list(getattr(dataset, "train_transition_rows", []) or []),
        key=lambda row: quarter_sort_key(str(row.get("quarter") or "")),
    )
    if not train_transition_rows or not base_rows:
        return [dict(row) for row in base_rows], None
    train_quarters = [str(row.get("quarter") or "") for row in train_transition_rows]
    holdout_quarters = [str(row.get("quarter") or "") for row in sorted(base_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    train_end_quarter = train_quarters[-1]
    _start_month, train_end_month = _quarter_month_range(train_end_quarter)
    diagnosis_by_month = _direct_monthly_diagnosis_values(monthly_context, train_end_month=train_end_month)
    train_prior = _quarterly_incidence_to_monthly(train_transition_rows, value_key="incident_infections_period")
    emission_observations = _late_diagnosis_emission_observations(
        monthly_context,
        diagnosis_by_month=diagnosis_by_month,
        train_end_month=train_end_month,
    )
    fit = _fit_backlog_late_emission_model(
        diagnosis_by_month=diagnosis_by_month,
        incidence_prior_by_month=train_prior,
        emission_observations=emission_observations,
    )
    if fit is None:
        return [dict(row) for row in base_rows], None
    forecast_months = [month for quarter in holdout_quarters for month in _quarter_month_indices(quarter)]
    holdout_incidence = _quarterly_incidence_to_monthly(base_rows, value_key="incident_infections_period")
    combined_incidence = {
        **{int(month): float(value) for month, value in train_prior.items()},
        **{int(month): float(value) for month, value in holdout_incidence.items() if int(month) in set(forecast_months)},
    }
    all_months = sorted(set(combined_incidence).union(forecast_months))
    simulated = _simulate_backlog_late_emission(
        incidence_by_month=combined_incidence,
        months=all_months,
        early_to_late_hazard=float(fit["early_to_late_hazard"]),
        early_diagnosis_hazard=float(fit["early_diagnosis_hazard"]),
        late_diagnosis_hazard=float(fit["late_diagnosis_hazard"]),
        initial_early_undiagnosed=float(fit["initial_early_undiagnosed"]),
        initial_late_undiagnosed=float(fit["initial_late_undiagnosed"]),
    )
    holdout_incidence = {int(month): float(combined_incidence.get(month) or 0.0) for month in forecast_months}
    holdout_diagnoses = {
        int(month): float((simulated.get(int(month)) or {}).get("new_diagnosed_cases_period") or 0.0)
        for month in forecast_months
    }
    incidence_by_quarter = _aggregate_months_to_quarters(holdout_incidence, holdout_quarters)
    diagnosis_by_quarter = _aggregate_months_to_quarters(holdout_diagnoses, holdout_quarters)
    candidate_rows: list[dict[str, Any]] = []
    for row in base_rows:
        quarter = str(row.get("quarter") or "")
        next_row = dict(row)
        if quarter in incidence_by_quarter:
            next_row["incident_infections_period"] = float(incidence_by_quarter[quarter])
        if quarter in diagnosis_by_quarter:
            next_row["new_diagnosed_cases_period"] = float(diagnosis_by_quarter[quarter])
        candidate_rows.append(next_row)
    head = BacklogLateEmissionHead(
        family=repair_family,
        early_to_late_hazard=float(fit["early_to_late_hazard"]),
        early_diagnosis_hazard=float(fit["early_diagnosis_hazard"]),
        late_diagnosis_hazard=float(fit["late_diagnosis_hazard"]),
        initial_early_undiagnosed=float(fit["initial_early_undiagnosed"]),
        initial_late_undiagnosed=float(fit["initial_late_undiagnosed"]),
        train_month_count=int(fit["train_month_count"]),
        observed_diagnosis_month_count=int(fit["observed_month_count"]),
        late_share_emission_month_count=int(fit["late_share_emission_month_count"]),
        advanced_count_emission_month_count=int(fit["advanced_count_emission_month_count"]),
        forecast_month_count=len(forecast_months),
        train_loss=float(fit["train_loss"]),
        diagnosis_loss=None if fit.get("diagnosis_loss") is None else float(fit["diagnosis_loss"]),
        late_share_loss=None if fit.get("late_share_loss") is None else float(fit["late_share_loss"]),
        advanced_count_loss=None if fit.get("advanced_count_loss") is None else float(fit["advanced_count_loss"]),
        monthly_ar_parameters={
            "train_monthly_cap": max([max(float(value), 0.0) for value in train_prior.values()] or [0.0]),
            "holdout_month_count": float(len(forecast_months)),
        },
        emission_summary={
            **dict(fit["emission_summary"]),
            "holdout_incidence_source": "frozen_base_incidence_readout",
        },
    )
    return candidate_rows, head


def apply_selective_repair_gate(
    *,
    repair_family: str,
    dataset: Any,
    monthly_context: MonthlyShockContext,
    base_rows: list[dict[str, Any]],
    monthly_nowcast_head: MonthlyReportingNowcastHead | None,
    update_nowcast_incidence: bool = True,
) -> tuple[list[dict[str, Any]], BacklogLateEmissionHead | None, MonthlyReportingNowcastHead | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if repair_family not in SELECTIVE_REPAIR_GATE_FAMILIES:
        return [dict(row) for row in base_rows], None, monthly_nowcast_head, [], []
    if monthly_nowcast_head is None:
        return [dict(row) for row in base_rows], None, None, [], []
    backlog_rows, backlog_head = apply_backlog_late_emission_branch(
        repair_family="monthly_backlog_late_emission",
        dataset=dataset,
        monthly_context=monthly_context,
        base_rows=base_rows,
    )
    if backlog_head is None:
        return [dict(row) for row in base_rows], None, monthly_nowcast_head, [], []
    nowcast_rows, updated_nowcast_head, nowcast_diagnostics = _apply_monthly_reporting_nowcast_head(
        head=monthly_nowcast_head,
        monthly_context=monthly_context,
        base_rows=base_rows,
        update_incidence=update_nowcast_incidence,
    )
    holdout_quarters = [str(row.get("quarter") or "") for row in sorted(base_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))]
    late_available = _late_evidence_supported_quarters(monthly_context, quarters=holdout_quarters)
    late_agreement = _late_evidence_agreement_by_quarter(
        context=monthly_context,
        dataset=dataset,
        base_rows=base_rows,
        backlog_head=backlog_head,
    )
    base_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_rows}
    backlog_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in backlog_rows}
    nowcast_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in nowcast_rows}
    nowcast_diag_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in nowcast_diagnostics}
    selected_rows: list[dict[str, Any]] = []
    gate_diagnostics: list[dict[str, Any]] = []
    for quarter in holdout_quarters:
        source = "strict_base"
        reason = "no_monthly_nowcast_or_late_diagnosis_support"
        selected = dict(base_by_quarter.get(quarter) or {})
        if quarter in nowcast_diag_by_quarter:
            selected = dict(nowcast_by_quarter.get(quarter) or selected)
            source = "monthly_reporting_nowcast"
            reason = str((nowcast_diag_by_quarter.get(quarter) or {}).get("nowcast_source") or "monthly_support")
        elif bool((late_agreement.get(quarter) or {}).get("agreement_supported")):
            selected = dict(backlog_by_quarter.get(quarter) or selected)
            source = "backlog_late_emission"
            reason = "late_diagnosis_evidence_agreement"
        selected["selective_repair_source"] = source
        selected["selective_repair_reason"] = reason
        selected_rows.append(selected)
        gate_diagnostics.append(
            {
                "quarter": quarter,
                "selected_source": source,
                "selection_reason": reason,
                "monthly_nowcast_available": quarter in nowcast_diag_by_quarter,
                "late_diagnosis_evidence_available": quarter in late_available,
                "late_diagnosis_evidence_agreement": bool((late_agreement.get(quarter) or {}).get("agreement_supported")),
                "late_diagnosis_support": late_available.get(quarter),
                "late_diagnosis_agreement": late_agreement.get(quarter),
                "nowcast_support": nowcast_diag_by_quarter.get(quarter),
                "contract": (
                    "monthly nowcast wins when direct monthly diagnosis support exists; backlog emission is allowed only when "
                    "same-quarter ledger-allowed late-diagnosis evidence exists and the backlog late-evidence loss is not above "
                    "the neutral late-evidence loss; otherwise the strict base row is retained"
                ),
            }
        )
    return selected_rows, backlog_head, updated_nowcast_head, nowcast_diagnostics, gate_diagnostics


def _metric_entries(
    *,
    candidate_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    carry_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    metric_scales: dict[str, Any],
    train_end_year: int,
    horizon_years: int,
    repair_family: str,
) -> list[dict[str, Any]]:
    candidate_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in candidate_rows}
    base_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_rows}
    carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_rows}
    entries: list[dict[str, Any]] = []
    for target in sorted(holdout_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or ""))):
        quarter = str(target.get("quarter") or "")
        candidate = candidate_by_quarter.get(quarter)
        base = base_by_quarter.get(quarter)
        carry = carry_by_quarter.get(quarter)
        if candidate is None or base is None or carry is None:
            continue
        for metric_name in OBSERVATION_HEAD_METRICS:
            target_value = _finite_float(target.get(metric_name))
            candidate_value = _finite_float(candidate.get(metric_name))
            base_value = _finite_float(base.get(metric_name))
            carry_value = _finite_float(carry.get(metric_name))
            if target_value is None or candidate_value is None or base_value is None or carry_value is None:
                continue
            scale = _safe_scale(metric_name, target_value, metric_scales)
            entries.append(
                {
                    "repair_family": repair_family,
                    "train_end_year": int(train_end_year),
                    "horizon_years": int(horizon_years),
                    "quarter": quarter,
                    "year": quarter_year(quarter),
                    "metric_name": metric_name,
                    "observed_value": float(target_value),
                    "candidate_value": float(candidate_value),
                    "base_value": float(base_value),
                    "carry_forward_value": float(carry_value),
                    "candidate_norm_error": abs(float(candidate_value) - float(target_value)) / scale,
                    "base_norm_error": abs(float(base_value) - float(target_value)) / scale,
                    "carry_forward_norm_error": abs(float(carry_value) - float(target_value)) / scale,
                    "candidate_minus_base_norm_error": (
                        abs(float(candidate_value) - float(target_value))
                        - abs(float(base_value) - float(target_value))
                    )
                    / scale,
                    "candidate_minus_carry_forward_norm_error": (
                        abs(float(candidate_value) - float(target_value))
                        - abs(float(carry_value) - float(target_value))
                    )
                    / scale,
                }
            )
    return entries


def _annual_incidence_targets(validation_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    targets: dict[int, dict[str, Any]] = {}
    for row in validation_rows:
        quarter = str(row.get("quarter") or "")
        value = _finite_float(row.get("annual_new_infections"))
        if not quarter.endswith("-Q4") or value is None:
            continue
        targets[quarter_year(quarter)] = dict(row)
    return targets


def _annual_incidence_entries(
    *,
    candidate_rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    carry_rows: list[dict[str, Any]],
    validation_targets: dict[int, dict[str, Any]],
    train_end_year: int,
    horizon_years: int,
    repair_family: str,
) -> list[dict[str, Any]]:
    def annual_sum(rows: list[dict[str, Any]], year: int) -> tuple[float | None, int]:
        values = [
            _finite_float(row.get("incident_infections_period"))
            for row in rows
            if str(row.get("quarter") or "").startswith(f"{int(year):04d}-Q")
        ]
        finite = [float(value) for value in values if value is not None]
        quarters = {
            str(row.get("quarter") or "")
            for row in rows
            if str(row.get("quarter") or "").startswith(f"{int(year):04d}-Q")
        }
        if len(quarters) < 4 or len(finite) < 4:
            return None, len(quarters)
        return float(sum(finite)), len(quarters)

    entries: list[dict[str, Any]] = []
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in candidate_rows})
    for year in years:
        target_row = validation_targets.get(year)
        if target_row is None:
            continue
        target_value = _finite_float(target_row.get("annual_new_infections"))
        if target_value is None or target_value <= 0.0:
            continue
        candidate_sum, candidate_quarters = annual_sum(candidate_rows, year)
        base_sum, base_quarters = annual_sum(base_rows, year)
        carry_sum, carry_quarters = annual_sum(carry_rows, year)
        if candidate_sum is None or base_sum is None or carry_sum is None:
            continue
        scale = max(float(target_value), 1.0)
        entries.append(
            {
                "repair_family": repair_family,
                "train_end_year": int(train_end_year),
                "horizon_years": int(horizon_years),
                "year": int(year),
                "target_annual_new_infections": float(target_value),
                "candidate_annual_incidence": float(candidate_sum),
                "base_annual_incidence": float(base_sum),
                "carry_forward_annual_incidence": float(carry_sum),
                "candidate_norm_error": abs(float(candidate_sum) - float(target_value)) / scale,
                "base_norm_error": abs(float(base_sum) - float(target_value)) / scale,
                "carry_forward_norm_error": abs(float(carry_sum) - float(target_value)) / scale,
                "candidate_minus_base_norm_error": (
                    abs(float(candidate_sum) - float(target_value))
                    - abs(float(base_sum) - float(target_value))
                )
                / scale,
                "candidate_minus_carry_forward_norm_error": (
                    abs(float(candidate_sum) - float(target_value))
                    - abs(float(carry_sum) - float(target_value))
                )
                / scale,
                "candidate_quarter_count": int(candidate_quarters),
                "base_quarter_count": int(base_quarters),
                "carry_forward_quarter_count": int(carry_quarters),
                "target_allowed_use": str(((target_row.get("metric_provenance") or {}).get("annual_new_infections") or {}).get("allowed_use") or ""),
            }
        )
    return entries


def _score_entries(entries: list[dict[str, Any]], *, metric_filter: str | None = None) -> dict[str, Any]:
    subset = [entry for entry in entries if metric_filter is None or str(entry.get("metric_name") or "") == metric_filter]
    return {
        "entry_count": len(subset),
        "candidate_norm_mae": _mean([float(entry["candidate_norm_error"]) for entry in subset]),
        "base_norm_mae": _mean([float(entry["base_norm_error"]) for entry in subset]),
        "carry_forward_norm_mae": _mean([float(entry["carry_forward_norm_error"]) for entry in subset]),
        "candidate_minus_base_norm_mae": _mean([float(entry["candidate_minus_base_norm_error"]) for entry in subset]),
        "candidate_minus_carry_forward_norm_mae": _mean([float(entry["candidate_minus_carry_forward_norm_error"]) for entry in subset]),
    }


def _r10_metric_proxy(r10_baseline: dict[str, Any], metric_name: str) -> float | None:
    reference = _r10_reference_scores(r10_baseline)
    source_path = Path(str(r10_baseline.get("source_report") or ""))
    payload = read_json(source_path, default={}) if source_path.exists() else {}
    for contract in list((payload or {}).get("contracts") or []):
        champion = dict((contract or {}).get("merged_current_champion") or {})
        if str(champion.get("contract") or (contract or {}).get("contract") or "") != str(reference.get("reference_contract") or ""):
            continue
        for row in list(champion.get("residual_rows") or []):
            if str(row.get("metric") or "") == metric_name and str(row.get("tier") or "") == "overall":
                return _finite_float(row.get("abs_residual_mean"))
    return None


def _r10_annual_incidence_error(r10_baseline: dict[str, Any], r10_reference: dict[str, Any]) -> float | None:
    source_path = Path(str(r10_baseline.get("source_report") or ""))
    payload = read_json(source_path, default={}) if source_path.exists() else {}
    reference_contract = str(r10_reference.get("reference_contract") or "")
    reference_experiment = str(r10_reference.get("reference_experiment_id") or "")
    for contract in list((payload or {}).get("contracts") or []):
        for key in ("merged_current_champion", "baseline_current_champion"):
            champion = dict((contract or {}).get(key) or {})
            if not champion:
                continue
            if str(champion.get("contract") or (contract or {}).get("contract") or "") != reference_contract:
                continue
            if reference_experiment and str(champion.get("experiment_id") or "") != reference_experiment:
                continue
            if str(champion.get("archive_variant") or "") == "merged":
                return _finite_float(champion.get("annual_mean_incidence_error"))
    for contract in list((payload or {}).get("contracts") or []):
        champion = dict((contract or {}).get("baseline_current_champion") or {})
        if str(champion.get("contract") or (contract or {}).get("contract") or "") == reference_contract:
            return _finite_float(champion.get("annual_mean_incidence_error"))
    return None


def _evaluate_repair_split(
    *,
    repair_family: str,
    base_family: str,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    split: dict[str, Any],
    reference_config: dict[str, Any],
    validation_targets: dict[int, dict[str, Any]],
    monthly_context: MonthlyShockContext,
    monthly_latent_context: MonthlyLatentContext,
    monthly_joint_context: MonthlyJointContext,
    backhalf_context: BackHalfChannelContext,
    min_train_years: int,
    horizon_years: int,
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    model_family = _delegate_family_for_horizon(base_family, holdout_year_count=len(list(split["holdout_years"])))
    train_constraint_rows = _filter_rows_before_holdout(constraint_rows, list(split["holdout_years"]))
    train_rows = sorted(dataset.train_rows, key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
    train_backbone_rows = _train_backbone_prediction_rows(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    endpoint_head = _fit_endpoint_head(
        family=model_family,
        train_rows=train_rows,
        train_backbone_rows=train_backbone_rows,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    if endpoint_head is None:
        return None
    repair_head = fit_diagnosis_flow_repair_head(
        repair_family=repair_family,
        base_family=model_family,
        train_rows=train_rows,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    incidence_head = fit_incidence_readout_repair_head(
        repair_family=repair_family,
        base_family=model_family,
        train_rows=train_rows,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
        min_train_years=min_train_years,
    )
    if repair_family in DIAGNOSIS_FLOW_REPAIR_FAMILIES and repair_head is None:
        return None
    if repair_family in INCIDENCE_READOUT_REPAIR_FAMILIES and incidence_head is None:
        return None
    monthly_nowcast_head = _fit_monthly_reporting_nowcast_head(
        family=repair_family,
        dataset=dataset,
        monthly_context=monthly_context,
    )
    if repair_family in MONTHLY_REPORTING_NOWCAST_FAMILIES and monthly_nowcast_head is None:
        return None
    backbone = _forecast_reference_for_family(
        family=model_family,
        dataset=dataset,
        reference_config=reference_config,
        constraint_rows=train_constraint_rows,
        monthly_context=monthly_context,
        monthly_latent_context=monthly_latent_context,
        monthly_joint_context=monthly_joint_context,
        backhalf_context=backhalf_context,
    )
    backbone_rows = list(backbone.get("prediction_rows") or [])
    trajectory_rows = list(backbone.get("trajectory_rows") or [])
    base_rows = _inject_incidence_from_trajectory(
        _apply_endpoint_head(
            head=endpoint_head,
            train_rows=train_rows,
            backbone_rows=backbone_rows,
            trajectory_rows=trajectory_rows,
            horizon_quarters=len(backbone_rows),
        ),
        trajectory_rows,
    )
    candidate_rows = apply_diagnosis_flow_repair_head(
        head=repair_head,
        base_rows=base_rows,
        backbone_rows=backbone_rows,
        trajectory_rows=trajectory_rows,
    )
    candidate_rows, delay_head = apply_diagnosis_delay_backcalc_branch(
        repair_family=repair_family,
        dataset=dataset,
        monthly_context=monthly_context,
        base_rows=candidate_rows,
    )
    if repair_family in DIAGNOSIS_DELAY_BACKCALC_FAMILIES and delay_head is None:
        return None
    selective_gate_diagnostics: list[dict[str, Any]] = []
    if repair_family in SELECTIVE_REPAIR_GATE_FAMILIES:
        candidate_rows, backlog_head, monthly_nowcast_head, monthly_nowcast_diagnostics, selective_gate_diagnostics = apply_selective_repair_gate(
            repair_family=repair_family,
            dataset=dataset,
            monthly_context=monthly_context,
            base_rows=candidate_rows,
            monthly_nowcast_head=monthly_nowcast_head,
            update_nowcast_incidence=repair_family in NOWCAST_INTERNAL_INCIDENCE_FAMILIES,
        )
        if backlog_head is None or monthly_nowcast_head is None:
            return None
    else:
        candidate_rows, backlog_head = apply_backlog_late_emission_branch(
            repair_family=repair_family,
            dataset=dataset,
            monthly_context=monthly_context,
            base_rows=candidate_rows,
        )
        if repair_family in BACKLOG_LATE_EMISSION_FAMILIES and backlog_head is None:
            return None
        candidate_rows, monthly_nowcast_head, monthly_nowcast_diagnostics = _apply_monthly_reporting_nowcast_head(
            head=monthly_nowcast_head,
            monthly_context=monthly_context,
            base_rows=candidate_rows,
            update_incidence=repair_family in NOWCAST_INTERNAL_INCIDENCE_FAMILIES,
        )
        if repair_family in MONTHLY_REPORTING_NOWCAST_FAMILIES and monthly_nowcast_head is None:
            return None
    candidate_rows = apply_incidence_readout_repair_head(
        head=incidence_head,
        base_rows=candidate_rows,
    )
    carry = _carry_forward_result(dataset)
    carry_rows = _inject_incidence_from_trajectory(
        list(carry.get("prediction_rows") or []),
        list(carry.get("trajectory_rows") or []),
    )
    metric_entries = _metric_entries(
        candidate_rows=candidate_rows,
        base_rows=base_rows,
        carry_rows=carry_rows,
        holdout_rows=list(dataset.holdout_rows),
        metric_scales=dict(dataset.metric_scales),
        train_end_year=int(split["train_end_year"]),
        horizon_years=int(horizon_years),
        repair_family=repair_family,
    )
    incidence_entries = _annual_incidence_entries(
        candidate_rows=candidate_rows,
        base_rows=base_rows,
        carry_rows=carry_rows,
        validation_targets=validation_targets,
        train_end_year=int(split["train_end_year"]),
        horizon_years=int(horizon_years),
        repair_family=repair_family,
    )
    return {
        "repair_family": repair_family,
        "base_family": base_family,
        "delegate_family": model_family,
        "train_end_year": int(split["train_end_year"]),
        "holdout_years": list(split["holdout_years"]),
        "horizon_years": int(horizon_years),
        "repair_head": None if repair_head is None else {
            "family": repair_head.family,
            "feature_names": list(repair_head.feature_names),
            "coefficients": list(repair_head.coefficients),
            "train_row_count": int(repair_head.train_row_count),
            "training_objective": repair_head.training_objective,
        },
        "incidence_readout_head": None if incidence_head is None else {
            "family": incidence_head.family,
            "feature_names": list(incidence_head.feature_names),
            "coefficients": list(incidence_head.coefficients),
            "train_row_count": int(incidence_head.train_row_count),
            "training_objective": incidence_head.training_objective,
        },
        "diagnosis_delay_backcalc_head": None if delay_head is None else {
            "family": delay_head.family,
            "delay_lags_months": list(delay_head.delay_lags_months),
            "delay_kernel": list(delay_head.delay_kernel),
            "ascertainment_scale": float(delay_head.ascertainment_scale),
            "train_month_count": int(delay_head.train_month_count),
            "observed_diagnosis_month_count": int(delay_head.observed_diagnosis_month_count),
            "backcalculated_month_count": int(delay_head.backcalculated_month_count),
            "forecast_month_count": int(delay_head.forecast_month_count),
            "monthly_ar_parameters": dict(delay_head.monthly_ar_parameters),
            "late_constraint_summary": None if delay_head.late_constraint_summary is None else dict(delay_head.late_constraint_summary),
            "training_objective": delay_head.training_objective,
        },
        "backlog_late_emission_head": None if backlog_head is None else {
            "family": backlog_head.family,
            "early_to_late_hazard": float(backlog_head.early_to_late_hazard),
            "early_diagnosis_hazard": float(backlog_head.early_diagnosis_hazard),
            "late_diagnosis_hazard": float(backlog_head.late_diagnosis_hazard),
            "initial_early_undiagnosed": float(backlog_head.initial_early_undiagnosed),
            "initial_late_undiagnosed": float(backlog_head.initial_late_undiagnosed),
            "train_month_count": int(backlog_head.train_month_count),
            "observed_diagnosis_month_count": int(backlog_head.observed_diagnosis_month_count),
            "late_share_emission_month_count": int(backlog_head.late_share_emission_month_count),
            "advanced_count_emission_month_count": int(backlog_head.advanced_count_emission_month_count),
            "forecast_month_count": int(backlog_head.forecast_month_count),
            "train_loss": float(backlog_head.train_loss),
            "diagnosis_loss": backlog_head.diagnosis_loss,
            "late_share_loss": backlog_head.late_share_loss,
            "advanced_count_loss": backlog_head.advanced_count_loss,
            "monthly_ar_parameters": dict(backlog_head.monthly_ar_parameters),
            "emission_summary": dict(backlog_head.emission_summary),
            "training_objective": backlog_head.training_objective,
        },
        "monthly_reporting_nowcast_head": None if monthly_nowcast_head is None else {
            "family": monthly_nowcast_head.family,
            "completion_by_month_mask": dict(monthly_nowcast_head.completion_by_month_mask),
            "lead_completion_by_month_mask": dict(monthly_nowcast_head.lead_completion_by_month_mask),
            "fallback_completion_scale": float(monthly_nowcast_head.fallback_completion_scale),
            "fallback_lead_completion_scale": float(monthly_nowcast_head.fallback_lead_completion_scale),
            "incidence_feature_names": list(monthly_nowcast_head.incidence_feature_names),
            "incidence_coefficients": list(monthly_nowcast_head.incidence_coefficients),
            "incidence_prediction_floor": float(monthly_nowcast_head.incidence_prediction_floor),
            "incidence_prediction_ceiling": float(monthly_nowcast_head.incidence_prediction_ceiling),
            "train_quarter_count": int(monthly_nowcast_head.train_quarter_count),
            "direct_monthly_row_count": int(monthly_nowcast_head.direct_monthly_row_count),
            "train_partial_quarter_count": int(monthly_nowcast_head.train_partial_quarter_count),
            "holdout_nowcast_quarter_count": int(monthly_nowcast_head.holdout_nowcast_quarter_count),
            "training_objective": monthly_nowcast_head.training_objective,
        },
        "monthly_reporting_nowcast_diagnostics": monthly_nowcast_diagnostics,
        "selective_repair_gate_diagnostics": selective_gate_diagnostics,
        "metric_entries": metric_entries,
        "incidence_validation_entries": incidence_entries,
    }


def _family_summary(
    *,
    repair_family: str,
    rows: list[dict[str, Any]],
    r10_reference: dict[str, Any],
    r10_diagnosis_flow_proxy: float | None,
) -> dict[str, Any]:
    metric_entries = [entry for row in rows for entry in list(row.get("metric_entries") or [])]
    incidence_entries = [entry for row in rows for entry in list(row.get("incidence_validation_entries") or [])]
    delay_heads = [
        dict(row.get("diagnosis_delay_backcalc_head") or {})
        for row in rows
        if row.get("diagnosis_delay_backcalc_head")
    ]
    backlog_heads = [
        dict(row.get("backlog_late_emission_head") or {})
        for row in rows
        if row.get("backlog_late_emission_head")
    ]
    nowcast_heads = [
        dict(row.get("monthly_reporting_nowcast_head") or {})
        for row in rows
        if row.get("monthly_reporting_nowcast_head")
    ]
    selective_gate_rows = [
        dict(item)
        for row in rows
        for item in list(row.get("selective_repair_gate_diagnostics") or [])
    ]
    flow_score = _score_entries(metric_entries, metric_filter="new_diagnosed_cases_period")
    full_score = _score_entries(metric_entries)
    incidence_score = _score_entries(incidence_entries)
    r10_scalar = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))
    r10_annual = _finite_float(r10_reference.get("reference_annual_incidence_error"))
    blockers: list[str] = []
    if flow_score.get("candidate_norm_mae") is None:
        blockers.append("no_diagnosis_flow_entries")
    elif flow_score.get("base_norm_mae") is not None and float(flow_score["candidate_norm_mae"]) > float(flow_score["base_norm_mae"]):
        blockers.append("diagnosis_flow_regresses_vs_base")
    if full_score.get("candidate_norm_mae") is None:
        blockers.append("no_full_path_entries")
    elif full_score.get("base_norm_mae") is not None and float(full_score["candidate_norm_mae"]) > float(full_score["base_norm_mae"]):
        blockers.append("full_lifted_path_regresses_vs_base")
    if incidence_score.get("candidate_norm_mae") is None:
        blockers.append("no_complete_annual_incidence_validation_entries")
    elif incidence_score.get("base_norm_mae") is not None and float(incidence_score["candidate_norm_mae"]) > float(incidence_score["base_norm_mae"]):
        blockers.append("incidence_validation_regresses_vs_base")
    if r10_scalar is not None and full_score.get("candidate_norm_mae") is not None and float(full_score["candidate_norm_mae"]) >= r10_scalar:
        blockers.append("full_lifted_path_not_better_than_r10_scalar")
    if r10_diagnosis_flow_proxy is not None and flow_score.get("candidate_norm_mae") is not None:
        # The proxy is raw in the archived R10 report, so this gate is reported but not promoted on its own.
        flow_score["r10_diagnosis_flow_raw_proxy"] = r10_diagnosis_flow_proxy
    if r10_annual is not None and incidence_score.get("candidate_norm_mae") is not None and float(incidence_score["candidate_norm_mae"]) >= r10_annual:
        blockers.append("annual_incidence_validation_not_better_than_r10")
    delay_summary = {
        "split_count": len(delay_heads),
        "mean_ascertainment_scale": _mean([float(head.get("ascertainment_scale") or 0.0) for head in delay_heads]),
        "mean_observed_diagnosis_month_count": _mean([float(head.get("observed_diagnosis_month_count") or 0.0) for head in delay_heads]),
        "mean_backcalculated_month_count": _mean([float(head.get("backcalculated_month_count") or 0.0) for head in delay_heads]),
        "mean_forecast_month_count": _mean([float(head.get("forecast_month_count") or 0.0) for head in delay_heads]),
        "late_constrained_split_count": sum(1 for head in delay_heads if (head.get("late_constraint_summary") or {}).get("status") == "applied"),
        "mean_late_severity_month_count": _mean([
            float((head.get("late_constraint_summary") or {}).get("late_severity_month_count") or 0.0)
            for head in delay_heads
            if head.get("late_constraint_summary")
        ]),
        "median_top_delay_lag_months": None if not delay_heads else float(np.median(np.asarray([
            int((head.get("delay_lags_months") or [0])[int(np.argmax(np.asarray(head.get("delay_kernel") or [1.0], dtype=np.float64)))])
            for head in delay_heads
            if head.get("delay_lags_months") and head.get("delay_kernel")
        ], dtype=np.float64))),
    }
    backlog_summary = {
        "split_count": len(backlog_heads),
        "mean_early_to_late_hazard": _mean([float(head.get("early_to_late_hazard") or 0.0) for head in backlog_heads]),
        "mean_early_diagnosis_hazard": _mean([float(head.get("early_diagnosis_hazard") or 0.0) for head in backlog_heads]),
        "mean_late_diagnosis_hazard": _mean([float(head.get("late_diagnosis_hazard") or 0.0) for head in backlog_heads]),
        "mean_train_loss": _mean([float(head.get("train_loss") or 0.0) for head in backlog_heads]),
        "mean_late_share_emission_month_count": _mean([float(head.get("late_share_emission_month_count") or 0.0) for head in backlog_heads]),
        "mean_advanced_count_emission_month_count": _mean([float(head.get("advanced_count_emission_month_count") or 0.0) for head in backlog_heads]),
    }
    nowcast_summary = {
        "split_count": len(nowcast_heads),
        "mean_direct_monthly_row_count": _mean([float(head.get("direct_monthly_row_count") or 0.0) for head in nowcast_heads]),
        "mean_train_partial_quarter_count": _mean([float(head.get("train_partial_quarter_count") or 0.0) for head in nowcast_heads]),
        "mean_holdout_nowcast_quarter_count": _mean([float(head.get("holdout_nowcast_quarter_count") or 0.0) for head in nowcast_heads]),
        "mean_completion_scale": _mean([float(head.get("fallback_completion_scale") or 0.0) for head in nowcast_heads]),
    }
    selected_source_counts: dict[str, int] = {}
    for row in selective_gate_rows:
        source = str(row.get("selected_source") or "unknown")
        selected_source_counts[source] = selected_source_counts.get(source, 0) + 1
    selective_summary = {
        "gate_row_count": len(selective_gate_rows),
        "selected_source_counts": selected_source_counts,
        "monthly_nowcast_available_count": sum(1 for row in selective_gate_rows if bool(row.get("monthly_nowcast_available"))),
        "late_diagnosis_evidence_available_count": sum(1 for row in selective_gate_rows if bool(row.get("late_diagnosis_evidence_available"))),
    }
    return {
        "repair_family": repair_family,
        "split_count": len(rows),
        "metric_entry_count": len(metric_entries),
        "incidence_validation_entry_count": len(incidence_entries),
        "diagnosis_flow_score": flow_score,
        "full_lifted_path_score": full_score,
        "annual_incidence_validation_score": incidence_score,
        "diagnosis_delay_backcalc_summary": delay_summary,
        "backlog_late_emission_summary": backlog_summary,
        "monthly_reporting_nowcast_summary": nowcast_summary,
        "selective_repair_gate_summary": selective_summary,
        "r10_reference": {
            "quarterly_scalar_mae": r10_scalar,
            "annual_incidence_error": r10_annual,
            "diagnosis_flow_raw_proxy": r10_diagnosis_flow_proxy,
        },
        "promotion_eligible": not blockers and repair_family != "baseline_no_repair",
        "blockers": blockers,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    summaries = list(payload.get("family_summaries") or [])
    labels = [str(row.get("repair_family") or "") for row in summaries]
    x = np.arange(len(labels), dtype=np.float64)
    flow = [float(((row.get("diagnosis_flow_score") or {}).get("candidate_norm_mae")) or 0.0) for row in summaries]
    base_flow = [float(((row.get("diagnosis_flow_score") or {}).get("base_norm_mae")) or 0.0) for row in summaries]
    incidence = [float(((row.get("annual_incidence_validation_score") or {}).get("candidate_norm_mae")) or 0.0) for row in summaries]
    full = [float(((row.get("full_lifted_path_score") or {}).get("candidate_norm_mae")) or 0.0) for row in summaries]
    r10_scalar = _finite_float((payload.get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    r10_annual = _finite_float((payload.get("r10_reference") or {}).get("reference_annual_incidence_error"))
    metric_entries = list(payload.get("best_metric_entries") or [])
    delay_entries = list(payload.get("delay_head_diagnostics") or [])
    backlog_entries = list(payload.get("backlog_emission_diagnostics") or [])
    worst_flow = sorted(
        [entry for entry in metric_entries if str(entry.get("metric_name") or "") == "new_diagnosed_cases_period"],
        key=lambda entry: float(entry.get("candidate_minus_base_norm_error") or 0.0),
        reverse=True,
    )[:8]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.0), constrained_layout=True)
    flat = axes.ravel()
    width = 0.34
    flat[0].bar(x - width / 2, flow, width=width, color="#1f77b4", label="repair")
    flat[0].bar(x + width / 2, base_flow, width=width, color="#8c8c8c", label="base")
    flat[0].set_xticks(x)
    flat[0].set_xticklabels(labels, rotation=25, ha="right")
    flat[0].set_ylabel("normalized MAE")
    flat[0].set_title("A. Delayed diagnosis-flow fit")
    flat[0].legend(frameon=False, fontsize=8)

    flat[1].bar(x, incidence, color="#2ca02c")
    if r10_annual is not None:
        flat[1].axhline(r10_annual, color="#111827", linestyle="--", linewidth=1.4, label="R10 annual incidence")
        flat[1].legend(frameon=False, fontsize=8)
    flat[1].set_xticks(x)
    flat[1].set_xticklabels(labels, rotation=25, ha="right")
    flat[1].set_ylabel("annual incidence validation error")
    flat[1].set_title("B. Validation-only incidence readout")

    flat[2].bar(x, full, color="#e45756")
    if r10_scalar is not None:
        flat[2].axhline(r10_scalar, color="#111827", linestyle="--", linewidth=1.4, label="R10 scalar")
        flat[2].legend(frameon=False, fontsize=8)
    flat[2].set_xticks(x)
    flat[2].set_xticklabels(labels, rotation=25, ha="right")
    flat[2].set_ylabel("full lifted-path normalized MAE")
    flat[2].set_title("C. Full path side-effect gate")

    if backlog_entries:
        ordered_backlog = sorted(backlog_entries, key=lambda row: (int(row.get("horizon_years") or 0), int(row.get("train_end_year") or 0)))
        y = np.arange(len(ordered_backlog), dtype=np.float64)
        offset = 0.24
        flat[3].barh(y - offset, [float(row.get("early_to_late_hazard") or 0.0) for row in ordered_backlog], height=0.22, color="#7f3c8d", label="U_early -> U_late")
        flat[3].barh(y, [float(row.get("early_diagnosis_hazard") or 0.0) for row in ordered_backlog], height=0.22, color="#11a579", label="early diagnosis")
        flat[3].barh(y + offset, [float(row.get("late_diagnosis_hazard") or 0.0) for row in ordered_backlog], height=0.22, color="#f2b701", label="late diagnosis")
        flat[3].set_yticks(y)
        flat[3].set_yticklabels([f"{row.get('train_end_year')} h{row.get('horizon_years')}" for row in ordered_backlog], fontsize=8)
        flat[3].invert_yaxis()
        flat[3].set_xlabel("monthly hazard")
        flat[3].set_title("D. Backlog-emission transition hazards")
        flat[3].legend(frameon=False, fontsize=7)
    elif delay_entries:
        ordered_delay = sorted(delay_entries, key=lambda row: (int(row.get("horizon_years") or 0), int(row.get("train_end_year") or 0)))
        y = np.arange(len(ordered_delay), dtype=np.float64)
        flat[3].barh(y, [float(row.get("top_delay_lag_months") or 0.0) for row in ordered_delay], color="#7f3c8d")
        flat[3].set_yticks(y)
        flat[3].set_yticklabels([f"{row.get('train_end_year')} h{row.get('horizon_years')}" for row in ordered_delay], fontsize=8)
        flat[3].invert_yaxis()
        flat[3].set_xlabel("kernel mode lag, months")
        flat[3].set_title("D. Learned delay-kernel modes")
    else:
        y = np.arange(len(worst_flow), dtype=np.float64)
        flat[3].barh(y, [float(row.get("candidate_minus_base_norm_error") or 0.0) for row in worst_flow], color="#d62728")
        flat[3].set_yticks(y)
        flat[3].set_yticklabels([f"{row.get('quarter')} h{row.get('horizon_years')}" for row in worst_flow], fontsize=8)
        flat[3].invert_yaxis()
        flat[3].axvline(0.0, color="#111827", linewidth=0.8)
        flat[3].set_xlabel("repair minus base normalized error")
        flat[3].set_title("D. Worst diagnosis-flow side effects")

    fig.suptitle("Monthly Diagnosis-Delay / Incidence Back-Calculation", fontsize=15, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Monthly Diagnosis-Delay / Incidence Back-Calculation",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Base family: `{payload['base_family']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Baseline run: `{payload['baseline_source_run_id']}`",
        f"- R10 reference: `{(payload.get('r10_reference') or {}).get('reference_experiment_id')}` / `{(payload.get('r10_reference') or {}).get('reference_quarterly_mean_mae')}`",
        "",
        "## Contract",
        "",
        "- Monthly diagnosis-delay branches fit `D_m = rho * sum_l k_l I_{m-l}` with nonnegative learned delay kernels.",
        "- Diagnosis-flow targets used for delay fitting are direct HARP monthly diagnosis-flow rows only.",
        "- Late-constrained branches use only ledger-allowed auxiliary/prior CD4, AHD, and late-diagnosis emissions to shape delay-kernel mass.",
        "- Backlog-emission branches fit `U_early -> U_late -> diagnosed` and score CD4/AHD/late-presenter rows as auxiliary emissions.",
        "- Monthly reporting-nowcast branches complete partial in-quarter direct HARP monthly diagnosis rows using training-estimated month-mask completion scales; they are surveillance nowcasts, not pure forecasts.",
        "- Selective repair branches choose per quarter: monthly nowcast when monthly diagnosis support exists, backlog emission only when same-quarter late-diagnosis evidence exists, otherwise strict base.",
        "- Annual new infections are validation-only and are never used to fit repair coefficients.",
        "- Annual incidence validation is scored only when all four quarters have simulated incident-infection readouts.",
        "",
        "## Family Summary",
        "",
        "| family | promoted | diagnosis-flow MAE | base flow MAE | full path MAE | annual incidence validation | blockers |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in list(payload.get("family_summaries") or []):
        flow = dict(row.get("diagnosis_flow_score") or {})
        full = dict(row.get("full_lifted_path_score") or {})
        incidence = dict(row.get("annual_incidence_validation_score") or {})
        lines.append(
            "| {family} | {promoted} | {flow:.6f} | {base:.6f} | {full:.6f} | {incidence:.6f} | {blockers} |".format(
                family=row.get("repair_family"),
                promoted=bool(row.get("promotion_eligible")),
                flow=float(flow.get("candidate_norm_mae") or 0.0),
                base=float(flow.get("base_norm_mae") or 0.0),
                full=float(full.get("candidate_norm_mae") or 0.0),
                incidence=float(incidence.get("candidate_norm_mae") or 0.0),
                blockers=", ".join(str(value) for value in list(row.get("blockers") or [])) or "none",
            )
        )
    best = dict(payload.get("best_family") or {})
    delay_rows = list(payload.get("delay_head_diagnostics") or [])
    backlog_rows = list(payload.get("backlog_emission_diagnostics") or [])
    nowcast_rows = list(payload.get("monthly_reporting_nowcast_diagnostics") or [])
    selective_rows = list(payload.get("selective_repair_gate_diagnostics") or [])
    lines.extend(
        [
            "",
            "## Best Family",
            "",
            f"- Best family: `{best.get('repair_family')}`",
            f"- Promotion eligible: `{best.get('promotion_eligible')}`",
            f"- Blockers: `{', '.join(str(value) for value in list(best.get('blockers') or [])) or 'none'}`",
        ]
    )
    if delay_rows:
        top_lags = [float(row.get("top_delay_lag_months") or 0.0) for row in delay_rows]
        lines.extend(
            [
                "",
                "## Delay-Kernel Diagnostics",
                "",
                f"- Split-level delay kernels: `{len(delay_rows)}`",
                f"- Median kernel-mode lag months: `{float(np.median(np.asarray(top_lags, dtype=np.float64))):.3f}`",
                f"- Late-constrained kernels: `{sum(1 for row in delay_rows if str(row.get('late_constraint_status') or '') == 'applied')}`",
                "- Interpretation: long kernel modes mean diagnosis-flow evidence is trying to explain current diagnoses using old latent infections, so the incidence/diagnosis split is weakly identified under the current public evidence.",
            ]
        )
    if backlog_rows:
        losses = [float(row.get("train_loss") or 0.0) for row in backlog_rows]
        late_months = [float(row.get("late_share_emission_month_count") or 0.0) for row in backlog_rows]
        lines.extend(
            [
                "",
                "## Backlog-Emission Diagnostics",
                "",
                f"- Split-level backlog models: `{len(backlog_rows)}`",
                f"- Median training emission loss: `{float(np.median(np.asarray(losses, dtype=np.float64))):.6f}`",
                f"- Median late-emission months per split: `{float(np.median(np.asarray(late_months, dtype=np.float64))):.3f}`",
                "- Interpretation: this branch turns late-presenter evidence into an auxiliary emission target; promotion still depends on blocked diagnosis-flow, full-path, incidence-validation, and R10 gates.",
            ]
        )
    nowcast_head_rows = [row for row in nowcast_rows if row.get("fallback_completion_scale") is not None]
    nowcast_quarter_rows = [row for row in nowcast_rows if row.get("quarter")]
    if nowcast_head_rows:
        nowcast_counts = [float(row.get("holdout_nowcast_quarter_count") or 0.0) for row in nowcast_head_rows]
        lines.extend(
            [
                "",
                "## Monthly Reporting-Nowcast Diagnostics",
                "",
                f"- Split-level nowcast heads: `{len(nowcast_head_rows)}`",
                f"- Median holdout quarters with partial monthly nowcast support: `{float(np.median(np.asarray(nowcast_counts, dtype=np.float64))):.3f}`",
                f"- Nowcasted quarter diagnostics: `{len(nowcast_quarter_rows)}`",
                "- Interpretation: these rows quantify whether the 2020 disruption and 2023 rebound are observable from monthly surveillance support before quarterly endpoint scoring.",
            ]
        )
    if selective_rows:
        source_counts: dict[str, int] = {}
        for row in selective_rows:
            source = str(row.get("selected_source") or "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        lines.extend(
            [
                "",
                "## Selective Repair Gate Diagnostics",
                "",
                f"- Gate rows: `{len(selective_rows)}`",
                f"- Selected source counts: `{source_counts}`",
                "- Interpretation: this gate prevents backlog emission from changing unsupported quarters, which is the specific failure mode behind the 2023-Q4 side effect.",
            ]
        )
    return "\n".join(lines) + "\n"


def run_diagnosis_incidence_repair(
    *,
    run_id: str = "p3d-diagnosis-incidence-repair-20260426-s00",
    source_report_path: str | Path | None = None,
    family: str | None = None,
    repair_families: tuple[str, ...] = REPAIR_FAMILIES,
    start_year: int | None = None,
    end_year: int | None = None,
    min_train_years: int | None = None,
) -> dict[str, Any]:
    source_report = Path(source_report_path) if source_report_path is not None else _default_source_report()
    report = dict(read_json(source_report, default={}) or {})
    if not report:
        raise FileNotFoundError(f"Hybrid champion report not found or empty: {source_report}")
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(report.get("source_run_id") or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=str(report.get("baseline_source_run_id") or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    base_family = _selected_family(report, family)
    benchmark_contract = dict(report.get("benchmark_contract") or {})
    start = int(start_year if start_year is not None else benchmark_contract.get("start_year") or 2010)
    end = int(end_year if end_year is not None else benchmark_contract.get("end_year") or 2025)
    minimum_train = int(min_train_years if min_train_years is not None else benchmark_contract.get("min_train_years") or 5)
    reference_config = _load_reference_config(Path(str(report.get("reference_report_path") or "")) if report.get("reference_report_path") else None)
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    constraint_rows = validation_rows
    validation_targets = _annual_incidence_targets(validation_rows)
    monthly_context = MonthlyShockContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    monthly_latent_context = MonthlyLatentContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    monthly_joint_context = MonthlyJointContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    backhalf_context = BackHalfChannelContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        reengagement_sensitivity_mode=str(report.get("reengagement_sensitivity_mode") or "public_stock_flow_proxy"),
    )
    r10_baseline = _load_r10_baseline(epigraph_root)
    r10_reference = _r10_reference_scores(r10_baseline)
    r10_annual_error = _r10_annual_incidence_error(r10_baseline, r10_reference)
    r10_reference_with_annual = dict(r10_reference)
    r10_reference_with_annual["reference_annual_incidence_error"] = r10_annual_error
    r10_diagnosis_proxy = _r10_metric_proxy(r10_baseline, "new_diagnosed_cases_period")
    family_rows: dict[str, list[dict[str, Any]]] = {family_name: [] for family_name in repair_families}
    for horizon_years in (1, 5):
        splits = _rolling_splits(
            observation_rows,
            start_year=start,
            end_year=end,
            min_train_years=minimum_train,
            horizon_years=horizon_years,
        )
        for split in splits:
            for repair_family in repair_families:
                row = _evaluate_repair_split(
                    repair_family=repair_family,
                    base_family=base_family,
                    observation_rows=observation_rows,
                    constraint_rows=constraint_rows,
                    split=split,
                    reference_config=reference_config,
                    validation_targets=validation_targets,
                    monthly_context=monthly_context,
                    monthly_latent_context=monthly_latent_context,
                    monthly_joint_context=monthly_joint_context,
                    backhalf_context=backhalf_context,
                    min_train_years=minimum_train,
                    horizon_years=horizon_years,
                )
                if row is not None:
                    family_rows[repair_family].append(row)
    summaries = [
        _family_summary(
            repair_family=family_name,
            rows=rows,
            r10_reference=r10_reference_with_annual,
            r10_diagnosis_flow_proxy=r10_diagnosis_proxy,
        )
        for family_name, rows in family_rows.items()
    ]
    promoted_summaries = [row for row in summaries if bool(row.get("promotion_eligible"))]
    if promoted_summaries:
        best = min(
            promoted_summaries,
            key=lambda row: (
                float(((row.get("diagnosis_flow_score") or {}).get("candidate_norm_mae")) or float("inf")),
                float(((row.get("annual_incidence_validation_score") or {}).get("candidate_norm_mae")) or float("inf")),
                float(((row.get("full_lifted_path_score") or {}).get("candidate_norm_mae")) or float("inf")),
                str(row.get("repair_family") or ""),
            ),
        )
    else:
        best = next(
            (row for row in summaries if str(row.get("repair_family") or "") == "baseline_no_repair"),
            summaries[0],
        )
    best_rejected_candidate = min(
        [row for row in summaries if str(row.get("repair_family") or "") != "baseline_no_repair"],
        key=lambda row: (
            len(list(row.get("blockers") or [])),
            float(((row.get("diagnosis_flow_score") or {}).get("candidate_norm_mae")) or float("inf")),
            float(((row.get("annual_incidence_validation_score") or {}).get("candidate_norm_mae")) or float("inf")),
            str(row.get("repair_family") or ""),
        ),
        default=None,
    )
    best_rows = family_rows.get(str(best.get("repair_family") or ""), [])
    best_metric_entries = [entry for row in best_rows for entry in list(row.get("metric_entries") or [])]
    best_incidence_entries = [entry for row in best_rows for entry in list(row.get("incidence_validation_entries") or [])]
    delay_head_diagnostics: list[dict[str, Any]] = []
    backlog_emission_diagnostics: list[dict[str, Any]] = []
    monthly_reporting_nowcast_diagnostics: list[dict[str, Any]] = []
    selective_repair_gate_diagnostics: list[dict[str, Any]] = []
    for family_name, rows in family_rows.items():
        for row in rows:
            head = dict(row.get("diagnosis_delay_backcalc_head") or {})
            if head:
                kernel = np.asarray(head.get("delay_kernel") or [], dtype=np.float64)
                lags = list(head.get("delay_lags_months") or [])
                top_index = int(np.argmax(kernel)) if kernel.size else 0
                late_constraint = dict(head.get("late_constraint_summary") or {})
                delay_head_diagnostics.append(
                    {
                        "repair_family": family_name,
                        "train_end_year": int(row.get("train_end_year") or 0),
                        "horizon_years": int(row.get("horizon_years") or 0),
                        "delegate_family": str(row.get("delegate_family") or ""),
                        "ascertainment_scale": float(head.get("ascertainment_scale") or 0.0),
                        "observed_diagnosis_month_count": int(head.get("observed_diagnosis_month_count") or 0),
                        "backcalculated_month_count": int(head.get("backcalculated_month_count") or 0),
                        "forecast_month_count": int(head.get("forecast_month_count") or 0),
                        "top_delay_lag_months": int(lags[top_index]) if lags and top_index < len(lags) else 0,
                        "top_delay_weight": float(kernel[top_index]) if kernel.size else 0.0,
                        "train_monthly_cap": _finite_float((head.get("monthly_ar_parameters") or {}).get("train_monthly_cap")),
                        "late_constraint_status": str(late_constraint.get("status") or "not_applied"),
                        "late_severity_month_count": int(late_constraint.get("late_severity_month_count") or 0),
                        "late_prior_top_lag_months": late_constraint.get("prior_top_lag_months"),
                        "late_prior_top_lag_weight": late_constraint.get("prior_top_lag_weight"),
                    }
                )
            backlog_head = dict(row.get("backlog_late_emission_head") or {})
            if backlog_head:
                backlog_emission_diagnostics.append(
                    {
                        "repair_family": family_name,
                        "train_end_year": int(row.get("train_end_year") or 0),
                        "horizon_years": int(row.get("horizon_years") or 0),
                        "delegate_family": str(row.get("delegate_family") or ""),
                        "early_to_late_hazard": float(backlog_head.get("early_to_late_hazard") or 0.0),
                        "early_diagnosis_hazard": float(backlog_head.get("early_diagnosis_hazard") or 0.0),
                        "late_diagnosis_hazard": float(backlog_head.get("late_diagnosis_hazard") or 0.0),
                        "initial_early_undiagnosed": float(backlog_head.get("initial_early_undiagnosed") or 0.0),
                        "initial_late_undiagnosed": float(backlog_head.get("initial_late_undiagnosed") or 0.0),
                        "observed_diagnosis_month_count": int(backlog_head.get("observed_diagnosis_month_count") or 0),
                        "late_share_emission_month_count": int(backlog_head.get("late_share_emission_month_count") or 0),
                        "advanced_count_emission_month_count": int(backlog_head.get("advanced_count_emission_month_count") or 0),
                        "forecast_month_count": int(backlog_head.get("forecast_month_count") or 0),
                        "train_loss": float(backlog_head.get("train_loss") or 0.0),
                        "diagnosis_loss": _finite_float(backlog_head.get("diagnosis_loss")),
                        "late_share_loss": _finite_float(backlog_head.get("late_share_loss")),
                        "advanced_count_loss": _finite_float(backlog_head.get("advanced_count_loss")),
                    }
                )
            nowcast_head = dict(row.get("monthly_reporting_nowcast_head") or {})
            if nowcast_head:
                monthly_reporting_nowcast_diagnostics.append(
                    {
                        "repair_family": family_name,
                        "train_end_year": int(row.get("train_end_year") or 0),
                        "horizon_years": int(row.get("horizon_years") or 0),
                        "delegate_family": str(row.get("delegate_family") or ""),
                        "direct_monthly_row_count": int(nowcast_head.get("direct_monthly_row_count") or 0),
                        "train_partial_quarter_count": int(nowcast_head.get("train_partial_quarter_count") or 0),
                        "holdout_nowcast_quarter_count": int(nowcast_head.get("holdout_nowcast_quarter_count") or 0),
                        "fallback_completion_scale": float(nowcast_head.get("fallback_completion_scale") or 0.0),
                        "incidence_prediction_floor": float(nowcast_head.get("incidence_prediction_floor") or 0.0),
                        "incidence_prediction_ceiling": float(nowcast_head.get("incidence_prediction_ceiling") or 0.0),
                        "completion_month_masks": sorted(dict(nowcast_head.get("completion_by_month_mask") or {}).keys()),
                        "lead_completion_month_masks": sorted(dict(nowcast_head.get("lead_completion_by_month_mask") or {}).keys()),
                    }
                )
                for diagnostic in list(row.get("monthly_reporting_nowcast_diagnostics") or []):
                    monthly_reporting_nowcast_diagnostics.append(
                        {
                            "repair_family": family_name,
                            "train_end_year": int(row.get("train_end_year") or 0),
                            "horizon_years": int(row.get("horizon_years") or 0),
                            **dict(diagnostic),
                        }
                    )
            for diagnostic in list(row.get("selective_repair_gate_diagnostics") or []):
                selective_repair_gate_diagnostics.append(
                    {
                        "repair_family": family_name,
                        "train_end_year": int(row.get("train_end_year") or 0),
                        "horizon_years": int(row.get("horizon_years") or 0),
                        **dict(diagnostic),
                    }
                )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    payload = {
        "schema_version": DIAGNOSIS_INCIDENCE_REPAIR_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_report_path": source_report.as_posix(),
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "base_family": base_family,
        "repair_families": list(repair_families),
        "contract": {
            "purpose": "Focused monthly diagnosis-delay/back-calculation repair for the R10-comparable diagnosis-flow/incidence alignment gap identified by lifted residual anatomy.",
            "training_targets": "direct_target monthly HARP diagnosis-flow rows only",
            "diagnosis_delay_equation": "D_m = rho * sum_l k_l I_{m-l}, k_l >= 0, sum_l k_l = 1",
            "late_diagnosis_constraint": "CD4/AHD/late-diagnosis auxiliary-prior emissions produce an empirical monotone delay prior; annual incidence is excluded.",
            "backlog_emission_equation": "I_m -> U_early; U_early -> U_late; diagnoses are emitted from both states; CD4/AHD/late-presenter rows score the late-diagnosis emission directly.",
            "monthly_reporting_nowcast_equation": "observed partial monthly HARP diagnosis flow inside quarter q is completed by a training-estimated month-mask completion scale; incidence readout is a bounded training-estimated function of the diagnosis nowcast and reporting-shock score",
            "module_decomposition_gate": "explicit decomposed families separate reporting-nowcast-only, backlog-only, incidence-readout-only, nowcast+readout, backlog+readout, and selective-nowcast/backlog+readout so diagnosis-flow gains cannot be attributed to an overloaded joint branch",
            "selective_repair_gate": "selective_monthly_nowcast_backlog starts from the strict base path, uses monthly nowcast only for quarters with monthly diagnosis support, uses backlog emission only when same-quarter ledger-allowed late-diagnosis evidence exists and the backlog late-evidence loss is not above a neutral late-evidence reference, and otherwise leaves the quarter unchanged",
            "backcalculation_contract": "training-window latent monthly infections are inferred from direct monthly diagnoses through the learned delay kernel, then forecast into blocked holdouts",
            "nowcast_contract": "nowcast branches may use direct monthly diagnosis rows inside the target quarter, never quarterly target rows; they support surveillance nowcasting, not pure long-horizon forecasting",
            "validation_only_targets": "annual_new_infections used only for complete-year incidence readout validation",
            "back_half_tuning": "frozen",
            "promotion_rule": "diagnosis flow must improve over base, full lifted path must not regress, annual incidence validation must not regress, and full path must beat R10 scalar.",
        },
        "benchmark_contract": {
            "start_year": start,
            "end_year": end,
            "min_train_years": minimum_train,
            "horizons": [1, 5],
        },
        "validation_target_years": sorted(validation_targets.keys()),
        "r10_reference": r10_reference_with_annual,
        "family_summaries": summaries,
        "best_family": best,
        "best_rejected_candidate": best_rejected_candidate,
        "best_metric_entries": best_metric_entries,
        "best_incidence_validation_entries": best_incidence_entries,
        "delay_head_diagnostics": delay_head_diagnostics,
        "backlog_emission_diagnostics": backlog_emission_diagnostics,
        "monthly_reporting_nowcast_diagnostics": monthly_reporting_nowcast_diagnostics,
        "selective_repair_gate_diagnostics": selective_repair_gate_diagnostics,
        "artifact_paths": {
            "json": (analysis_dir / "diagnosis_incidence_repair.json").as_posix(),
            "markdown": (analysis_dir / "diagnosis_incidence_repair.md").as_posix(),
            "dashboard_png": (analysis_dir / "diagnosis_incidence_repair_dashboard.png").as_posix(),
            "best_metric_entries_csv": (analysis_dir / "diagnosis_incidence_best_metric_entries.csv").as_posix(),
            "best_incidence_validation_csv": (analysis_dir / "diagnosis_incidence_best_incidence_validation.csv").as_posix(),
            "delay_head_diagnostics_csv": (analysis_dir / "diagnosis_delay_head_diagnostics.csv").as_posix(),
            "backlog_emission_diagnostics_csv": (analysis_dir / "backlog_late_emission_diagnostics.csv").as_posix(),
            "monthly_reporting_nowcast_diagnostics_csv": (analysis_dir / "monthly_reporting_nowcast_diagnostics.csv").as_posix(),
            "selective_repair_gate_diagnostics_csv": (analysis_dir / "selective_repair_gate_diagnostics.csv").as_posix(),
        },
    }
    write_json(Path(payload["artifact_paths"]["json"]), payload)
    Path(payload["artifact_paths"]["markdown"]).write_text(_markdown_report(payload), encoding="utf-8")
    _write_csv(Path(payload["artifact_paths"]["best_metric_entries_csv"]), best_metric_entries)
    _write_csv(Path(payload["artifact_paths"]["best_incidence_validation_csv"]), best_incidence_entries)
    _write_csv(Path(payload["artifact_paths"]["delay_head_diagnostics_csv"]), delay_head_diagnostics)
    _write_csv(Path(payload["artifact_paths"]["backlog_emission_diagnostics_csv"]), backlog_emission_diagnostics)
    _write_csv(Path(payload["artifact_paths"]["monthly_reporting_nowcast_diagnostics_csv"]), monthly_reporting_nowcast_diagnostics)
    _write_csv(Path(payload["artifact_paths"]["selective_repair_gate_diagnostics_csv"]), selective_repair_gate_diagnostics)
    _write_dashboard(payload, Path(payload["artifact_paths"]["dashboard_png"]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused monthly diagnosis-delay / incidence back-calculation.")
    parser.add_argument("--run-id", default="p3d-diagnosis-incidence-repair-20260426-s00")
    parser.add_argument("--source-report-path", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-train-years", type=int, default=None)
    args = parser.parse_args()
    payload = run_diagnosis_incidence_repair(
        run_id=args.run_id,
        source_report_path=args.source_report_path,
        family=args.family,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"]["json"])


if __name__ == "__main__":
    main()
