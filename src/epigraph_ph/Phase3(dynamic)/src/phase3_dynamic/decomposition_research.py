from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import (
    STATE_NAMES,
    build_blocked_time_dataset,
    build_observation_rows,
    default_epigraph_root,
    rolling_origin_splits,
    sandbox_repo_root,
)
from .decomposition import DecompositionControlConfig
from .incidence import carry_forward_incidence_flow_paths
from .metrics import (
    SUPPORT_AWARE_BACK_HALF_METRICS,
    SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
    support_aware_metric_scales,
    support_aware_normalized_mae,
    support_aware_rate_mae,
)
from .model import carry_forward_hazards, simulate_holdout
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .runtime import ensure_dir, write_json
from .scenario_lab import (
    DEFAULT_ACTIVE_SOURCE_RUN_ID,
    DEFAULT_BASELINE_SOURCE_RUN_ID,
    _filter_rows_before_holdout,
    _forecast_constrained_reference,
    _future_quarters,
    _load_reference_config,
    _projection_dataset,
)


DECOMPOSITION_RESEARCH_SCHEMA_VERSION = "phase3_dynamic_decomposition_research.v1"
SCORE_TOLERANCE = 1e-9


def _reference_with_decomposition(reference_config: dict[str, Any], cfg: DecompositionControlConfig | None) -> dict[str, Any]:
    updated = dict(reference_config)
    updated["decomposition_cfg"] = cfg
    return updated


def _carry_forward_result(dataset: Any) -> dict[str, Any]:
    incidence = carry_forward_incidence_flow_paths(dataset, mode="last_train")
    return simulate_holdout(
        dataset,
        carry_forward_hazards(dataset, mode="last_train"),
        incidence_inflow_map=dict(incidence.get("incidence_inflow_map") or {}),
        incidence_hazard_map=dict(incidence.get("incidence_hazard_map") or {}),
        population_denominator_map=dict(incidence.get("population_denominator_map") or {}),
        attrition_outflow_map=dict(incidence.get("attrition_outflow_map") or {}),
        state_attrition_outflow_map=dict(incidence.get("state_attrition_outflow_map") or {}),
        exit_channel_state_outflow_map=dict(incidence.get("exit_channel_state_outflow_map") or {}),
    )


def _evaluate_split(
    *,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    split: dict[str, Any],
    reference_config: dict[str, Any],
    decomposition_cfg: DecompositionControlConfig,
) -> dict[str, Any] | None:
    dataset = build_blocked_time_dataset(observation_rows, list(split["holdout_years"]))
    if not dataset.train_transition_rows or not dataset.holdout_rows:
        return None
    train_constraint_rows = _filter_rows_before_holdout(constraint_rows, list(split["holdout_years"]))
    baseline = _forecast_constrained_reference(
        dataset=dataset,
        reference_config=_reference_with_decomposition(reference_config, None),
        constraint_rows=train_constraint_rows,
    )
    candidate = _forecast_constrained_reference(
        dataset=dataset,
        reference_config=_reference_with_decomposition(reference_config, decomposition_cfg),
        constraint_rows=train_constraint_rows,
    )
    carry_forward = _carry_forward_result(dataset)
    back_half_scales = support_aware_metric_scales(
        dataset.train_rows,
        SUPPORT_AWARE_BACK_HALF_METRICS,
        eps=dataset.eps,
    )
    return {
        "train_end_year": int(split["train_end_year"]),
        "holdout_years": list(split["holdout_years"]),
        "baseline_mae": float(baseline.get("mae") or float("inf")),
        "candidate_mae": float(candidate.get("mae") or float("inf")),
        "carry_forward_mae": float(carry_forward.get("mae") or float("inf")),
        "candidate_minus_baseline_mae": float(candidate.get("mae") or float("inf")) - float(baseline.get("mae") or float("inf")),
        "candidate_minus_carry_forward_mae": float(candidate.get("mae") or float("inf")) - float(carry_forward.get("mae") or float("inf")),
        "support_aware_back_half": {
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_METRICS),
            "metric_scales": {key: float(value) for key, value in back_half_scales.items()},
            "baseline": support_aware_normalized_mae(
                list(baseline.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                back_half_scales,
                metric_names=SUPPORT_AWARE_BACK_HALF_METRICS,
                eps=dataset.eps,
            ),
            "candidate": support_aware_normalized_mae(
                list(candidate.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                back_half_scales,
                metric_names=SUPPORT_AWARE_BACK_HALF_METRICS,
                eps=dataset.eps,
            ),
            "carry_forward": support_aware_normalized_mae(
                list(carry_forward.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                back_half_scales,
                metric_names=SUPPORT_AWARE_BACK_HALF_METRICS,
                eps=dataset.eps,
            ),
        },
        "support_aware_back_half_rates": {
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_RATE_METRICS),
            "baseline": support_aware_rate_mae(
                list(baseline.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                list(dataset.train_rows),
                eps=dataset.eps,
            ),
            "candidate": support_aware_rate_mae(
                list(candidate.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                list(dataset.train_rows),
                eps=dataset.eps,
            ),
            "carry_forward": support_aware_rate_mae(
                list(carry_forward.get("prediction_rows") or []),
                list(dataset.holdout_rows),
                list(dataset.train_rows),
                eps=dataset.eps,
            ),
        },
    }


def _score_gate(rows: list[dict[str, Any]], *, gate_name: str) -> dict[str, Any]:
    if not rows:
        return {
            "gate": gate_name,
            "status": "fail",
            "blockers": ["no_valid_splits"],
        }
    candidate = np.asarray([float(row["candidate_mae"]) for row in rows], dtype=np.float64)
    baseline = np.asarray([float(row["baseline_mae"]) for row in rows], dtype=np.float64)
    carry = np.asarray([float(row["carry_forward_mae"]) for row in rows], dtype=np.float64)
    blockers: list[str] = []
    if float(np.mean(candidate)) >= float(np.mean(baseline)):
        blockers.append("candidate_mean_not_better_than_current_reference")
    if float(np.mean(candidate)) >= float(np.mean(carry)):
        blockers.append("candidate_mean_not_better_than_carry_forward")
    if float(np.max(candidate)) > float(np.max(baseline)):
        blockers.append("candidate_worst_regresses_against_current_reference")
    if float(np.max(candidate)) > float(np.max(carry)):
        blockers.append("candidate_worst_regresses_against_carry_forward")
    return {
        "gate": gate_name,
        "status": "pass" if not blockers else "fail",
        "blockers": blockers,
        "split_count": len(rows),
        "candidate_mean_mae": float(np.mean(candidate)),
        "baseline_mean_mae": float(np.mean(baseline)),
        "carry_forward_mean_mae": float(np.mean(carry)),
        "candidate_worst_mae": float(np.max(candidate)),
        "baseline_worst_mae": float(np.max(baseline)),
        "carry_forward_worst_mae": float(np.max(carry)),
        "rows": rows,
    }


def _finite_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return score


def _support_score_issue_counts(score: dict[str, Any]) -> dict[str, int]:
    missing_predictions = 0
    missing_train_scales = 0
    train_unsupported = 0
    for row in list(score.get("metric_rows") or []):
        missing_predictions += int(row.get("missing_prediction_count") or 0)
        missing_train_scales += int(row.get("missing_train_scale_count") or 0)
        train_supported_count = row.get("train_supported_count")
        if train_supported_count is not None and int(train_supported_count) <= 0:
            train_unsupported += int(row.get("supported_target_count") or 0)
    return {
        "missing_prediction_count": missing_predictions,
        "missing_train_scale_count": missing_train_scales,
        "train_unsupported_target_count": train_unsupported,
    }


def _score_supported_back_half_gate(rows: list[dict[str, Any]], *, gate_name: str) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    for row in rows:
        support = dict(row.get("support_aware_back_half") or {})
        candidate = dict(support.get("candidate") or {})
        baseline = dict(support.get("baseline") or {})
        carry = dict(support.get("carry_forward") or {})
        candidate_issues = _support_score_issue_counts(candidate)
        baseline_issues = _support_score_issue_counts(baseline)
        carry_issues = _support_score_issue_counts(carry)
        candidate_score = _finite_score(candidate.get("mean_normalized_mae"))
        baseline_score = _finite_score(baseline.get("mean_normalized_mae"))
        carry_score = _finite_score(carry.get("mean_normalized_mae"))
        supported_count = int(candidate.get("supported_target_count") or 0)
        if supported_count <= 0:
            continue
        gate_rows.append(
            {
                "train_end_year": int(row.get("train_end_year") or 0),
                "holdout_years": list(row.get("holdout_years") or []),
                "candidate_back_half_mae": candidate_score,
                "baseline_back_half_mae": baseline_score,
                "carry_forward_back_half_mae": carry_score,
                "candidate_status": candidate.get("status"),
                "baseline_status": baseline.get("status"),
                "carry_forward_status": carry.get("status"),
                "supported_target_count": supported_count,
                "candidate_missing_prediction_count": candidate_issues["missing_prediction_count"],
                "candidate_missing_train_scale_count": candidate_issues["missing_train_scale_count"],
                "baseline_missing_prediction_count": baseline_issues["missing_prediction_count"],
                "baseline_missing_train_scale_count": baseline_issues["missing_train_scale_count"],
                "carry_forward_missing_prediction_count": carry_issues["missing_prediction_count"],
                "carry_forward_missing_train_scale_count": carry_issues["missing_train_scale_count"],
                "candidate_metric_rows": list(candidate.get("metric_rows") or []),
                "baseline_metric_rows": list(baseline.get("metric_rows") or []),
                "carry_forward_metric_rows": list(carry.get("metric_rows") or []),
            }
        )
    if not gate_rows:
        return {
            "gate": gate_name,
            "status": "not_observed",
            "blockers": ["no_supported_vl_or_suppression_holdout_targets"],
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_METRICS),
            "rows": [],
        }
    comparable_rows = [
        row for row in gate_rows
        if row["candidate_back_half_mae"] is not None and row["baseline_back_half_mae"] is not None
    ]
    candidate_scores = [float(row["candidate_back_half_mae"]) for row in comparable_rows]
    baseline_scores = [float(row["baseline_back_half_mae"]) for row in comparable_rows]
    carry_scores = [float(row["carry_forward_back_half_mae"]) for row in comparable_rows if row["carry_forward_back_half_mae"] is not None]
    unscalable_rows = [
        row for row in gate_rows
        if row["candidate_back_half_mae"] is None
        and int(row.get("candidate_missing_prediction_count") or 0) == 0
        and int(row.get("candidate_missing_train_scale_count") or 0) > 0
    ]
    candidate_missing_prediction_rows = [
        row for row in gate_rows if int(row.get("candidate_missing_prediction_count") or 0) > 0
    ]
    baseline_missing_prediction_rows = [
        row for row in gate_rows if int(row.get("baseline_missing_prediction_count") or 0) > 0
    ]
    blockers: list[str] = []
    if candidate_missing_prediction_rows:
        blockers.append("candidate_missing_supported_back_half_predictions")
    if baseline_missing_prediction_rows:
        blockers.append("baseline_missing_supported_back_half_predictions")
    if not comparable_rows and unscalable_rows and not blockers:
        return {
            "gate": gate_name,
            "status": "not_train_scalable",
            "claim_status": "not_claimable",
            "blockers": ["no_train_origin_scale_for_supported_vl_or_suppression_targets"],
            "split_count": len(gate_rows),
            "comparable_split_count": 0,
            "excluded_train_unscalable_split_count": len(unscalable_rows),
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_METRICS),
            "rows": gate_rows,
        }
    if not comparable_rows:
        blockers.append("no_comparable_supported_back_half_splits")
    if candidate_scores and baseline_scores:
        if float(np.mean(candidate_scores)) > float(np.mean(baseline_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_back_half_mean_regresses_against_current_reference")
        if float(np.max(candidate_scores)) > float(np.max(baseline_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_back_half_worst_regresses_against_current_reference")
    comparable_to_carry = bool(carry_scores) and len(carry_scores) == len(comparable_rows)
    if comparable_to_carry and candidate_scores:
        if float(np.mean(candidate_scores)) > float(np.mean(carry_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_back_half_mean_regresses_against_carry_forward")
        if float(np.max(candidate_scores)) > float(np.max(carry_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_back_half_worst_regresses_against_carry_forward")
    mean_delta = (
        float(np.mean(candidate_scores) - np.mean(baseline_scores))
        if candidate_scores and baseline_scores
        else None
    )
    claim_status = "not_claimable"
    if mean_delta is not None:
        if mean_delta < -SCORE_TOLERANCE:
            claim_status = "supports_back_half_improvement_claim"
        elif abs(mean_delta) <= SCORE_TOLERANCE:
            claim_status = "back_half_non_regression_only"
        else:
            claim_status = "rejects_back_half_improvement_claim"
    return {
        "gate": gate_name,
        "status": "pass" if not blockers else "fail",
        "claim_status": claim_status,
        "blockers": blockers,
        "split_count": len(gate_rows),
        "comparable_split_count": len(comparable_rows),
        "excluded_train_unscalable_split_count": len(unscalable_rows),
        "candidate_mean_mae": float(np.mean(candidate_scores)) if candidate_scores else float("inf"),
        "baseline_mean_mae": float(np.mean(baseline_scores)) if baseline_scores else float("inf"),
        "carry_forward_mean_mae": float(np.mean(carry_scores)) if carry_scores else None,
        "candidate_worst_mae": float(np.max(candidate_scores)) if candidate_scores else float("inf"),
        "baseline_worst_mae": float(np.max(baseline_scores)) if baseline_scores else float("inf"),
        "carry_forward_worst_mae": float(np.max(carry_scores)) if carry_scores else None,
        "candidate_minus_baseline_mean_mae": mean_delta,
        "carry_forward_comparable": comparable_to_carry,
        "metric_names": list(SUPPORT_AWARE_BACK_HALF_METRICS),
        "contract": "VL/suppression gate scores only direct-target exact/bridge observed holdout rows, scaled only from supported train rows; carry-forward comparison is reported only when carry-forward emits the same supported metrics",
        "rows": gate_rows,
    }


def _score_supported_rate_gate(rows: list[dict[str, Any]], *, gate_name: str) -> dict[str, Any]:
    gate_rows: list[dict[str, Any]] = []
    for row in rows:
        support = dict(row.get("support_aware_back_half_rates") or {})
        candidate = dict(support.get("candidate") or {})
        baseline = dict(support.get("baseline") or {})
        carry = dict(support.get("carry_forward") or {})
        candidate_issues = _support_score_issue_counts(candidate)
        baseline_issues = _support_score_issue_counts(baseline)
        carry_issues = _support_score_issue_counts(carry)
        candidate_score = _finite_score(candidate.get("mean_normalized_mae"))
        baseline_score = _finite_score(baseline.get("mean_normalized_mae"))
        carry_score = _finite_score(carry.get("mean_normalized_mae"))
        supported_count = int(candidate.get("supported_target_count") or 0)
        if supported_count <= 0:
            continue
        gate_rows.append(
            {
                "train_end_year": int(row.get("train_end_year") or 0),
                "holdout_years": list(row.get("holdout_years") or []),
                "candidate_rate_mae": candidate_score,
                "baseline_rate_mae": baseline_score,
                "carry_forward_rate_mae": carry_score,
                "candidate_status": candidate.get("status"),
                "baseline_status": baseline.get("status"),
                "carry_forward_status": carry.get("status"),
                "supported_target_count": supported_count,
                "candidate_missing_prediction_count": candidate_issues["missing_prediction_count"],
                "candidate_train_unsupported_target_count": candidate_issues["train_unsupported_target_count"],
                "baseline_missing_prediction_count": baseline_issues["missing_prediction_count"],
                "baseline_train_unsupported_target_count": baseline_issues["train_unsupported_target_count"],
                "carry_forward_missing_prediction_count": carry_issues["missing_prediction_count"],
                "carry_forward_train_unsupported_target_count": carry_issues["train_unsupported_target_count"],
                "candidate_metric_rows": list(candidate.get("metric_rows") or []),
                "baseline_metric_rows": list(baseline.get("metric_rows") or []),
                "carry_forward_metric_rows": list(carry.get("metric_rows") or []),
            }
        )
    if not gate_rows:
        return {
            "gate": gate_name,
            "status": "not_observed",
            "blockers": ["no_supported_conditional_rate_holdout_targets"],
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_RATE_METRICS),
            "rows": [],
        }
    comparable_rows = [
        row for row in gate_rows
        if row["candidate_rate_mae"] is not None and row["baseline_rate_mae"] is not None
    ]
    candidate_scores = [float(row["candidate_rate_mae"]) for row in comparable_rows]
    baseline_scores = [float(row["baseline_rate_mae"]) for row in comparable_rows]
    carry_scores = [float(row["carry_forward_rate_mae"]) for row in comparable_rows if row["carry_forward_rate_mae"] is not None]
    untrained_rows = [
        row for row in gate_rows
        if row["candidate_rate_mae"] is None
        and int(row.get("candidate_missing_prediction_count") or 0) == 0
        and int(row.get("candidate_train_unsupported_target_count") or 0) > 0
    ]
    blockers: list[str] = []
    if any(int(row.get("candidate_missing_prediction_count") or 0) > 0 for row in gate_rows):
        blockers.append("candidate_missing_supported_conditional_rate_predictions")
    if any(int(row.get("baseline_missing_prediction_count") or 0) > 0 for row in gate_rows):
        blockers.append("baseline_missing_supported_conditional_rate_predictions")
    if not comparable_rows and untrained_rows and not blockers:
        return {
            "gate": gate_name,
            "status": "not_train_supported",
            "claim_status": "not_claimable",
            "blockers": ["no_train_origin_support_for_conditional_rate_targets"],
            "split_count": len(gate_rows),
            "comparable_split_count": 0,
            "excluded_train_unsupported_split_count": len(untrained_rows),
            "metric_names": list(SUPPORT_AWARE_BACK_HALF_RATE_METRICS),
            "rows": gate_rows,
        }
    if not comparable_rows:
        blockers.append("no_comparable_supported_conditional_rate_splits")
    if candidate_scores and baseline_scores:
        if float(np.mean(candidate_scores)) > float(np.mean(baseline_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_conditional_rate_mean_regresses_against_current_reference")
        if float(np.max(candidate_scores)) > float(np.max(baseline_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_conditional_rate_worst_regresses_against_current_reference")
    comparable_to_carry = bool(carry_scores) and len(carry_scores) == len(comparable_rows)
    if comparable_to_carry and candidate_scores:
        if float(np.mean(candidate_scores)) > float(np.mean(carry_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_conditional_rate_mean_regresses_against_carry_forward")
        if float(np.max(candidate_scores)) > float(np.max(carry_scores)) + SCORE_TOLERANCE:
            blockers.append("candidate_conditional_rate_worst_regresses_against_carry_forward")
    mean_delta = (
        float(np.mean(candidate_scores) - np.mean(baseline_scores))
        if candidate_scores and baseline_scores
        else None
    )
    claim_status = "not_claimable"
    if mean_delta is not None:
        if mean_delta < -SCORE_TOLERANCE:
            claim_status = "supports_conditional_back_half_improvement_claim"
        elif abs(mean_delta) <= SCORE_TOLERANCE:
            claim_status = "conditional_back_half_non_regression_only"
        else:
            claim_status = "rejects_conditional_back_half_improvement_claim"
    return {
        "gate": gate_name,
        "status": "pass" if not blockers else "fail",
        "claim_status": claim_status,
        "blockers": blockers,
        "split_count": len(gate_rows),
        "comparable_split_count": len(comparable_rows),
        "excluded_train_unsupported_split_count": len(untrained_rows),
        "candidate_mean_mae": float(np.mean(candidate_scores)) if candidate_scores else float("inf"),
        "baseline_mean_mae": float(np.mean(baseline_scores)) if baseline_scores else float("inf"),
        "carry_forward_mean_mae": float(np.mean(carry_scores)) if carry_scores else None,
        "candidate_worst_mae": float(np.max(candidate_scores)) if candidate_scores else float("inf"),
        "baseline_worst_mae": float(np.max(baseline_scores)) if baseline_scores else float("inf"),
        "carry_forward_worst_mae": float(np.max(carry_scores)) if carry_scores else None,
        "candidate_minus_baseline_mean_mae": mean_delta,
        "carry_forward_comparable": comparable_to_carry,
        "metric_names": list(SUPPORT_AWARE_BACK_HALF_RATE_METRICS),
        "contract": "conditional back-half gate scores VL tested / ART and suppressed / VL-tested rates only when numerator and denominator are direct-target exact/bridge observed in holdout and have train-origin support; rate errors use the natural unit interval scale",
        "rows": gate_rows,
    }


def _long_horizon_status(
    *,
    observation_rows: list[dict[str, Any]],
    constraint_rows: list[dict[str, Any]],
    reference_config: dict[str, Any],
    decomposition_cfg: DecompositionControlConfig,
    scenario_start_year: int,
    scenario_end_year: int,
) -> dict[str, Any]:
    future_quarters = _future_quarters(scenario_start_year, scenario_end_year)
    dataset = _projection_dataset(observation_rows, future_quarters)
    candidate = _forecast_constrained_reference(
        dataset=dataset,
        reference_config=_reference_with_decomposition(reference_config, decomposition_cfg),
        constraint_rows=constraint_rows,
    )
    trajectory_rows = list(candidate.get("trajectory_rows") or [])
    first_zero_stock_quarter = None
    bound_violations = 0
    constraints = dict(candidate.get("constraints") or {})
    lower = dict(constraints.get("lower_bound_by_quarter") or {})
    upper = dict(constraints.get("upper_bound_by_quarter") or {})
    for row in trajectory_rows:
        quarter = str(row.get("quarter") or "")
        state_values = dict(row.get("state_values") or {})
        total = float(sum(float(state_values.get(name) or 0.0) for name in STATE_NAMES))
        if total <= 0.0 and first_zero_stock_quarter is None:
            first_zero_stock_quarter = quarter
        if quarter in lower and total < float(lower[quarter]):
            bound_violations += 1
        if quarter in upper and total > float(upper[quarter]):
            bound_violations += 1
    return {
        "status": "pass" if first_zero_stock_quarter is None and bound_violations == 0 else "fail",
        "first_zero_stock_quarter": first_zero_stock_quarter,
        "demographic_bound_violation_count": int(bound_violations),
        "final_prediction": dict((candidate.get("prediction_rows") or [{}])[-1]) if candidate.get("prediction_rows") else {},
    }


def _promotion_gate(
    one_year_gate: dict[str, Any],
    five_year_gate: dict[str, Any],
    long_horizon_status: dict[str, Any],
    back_half_gate: dict[str, Any] | None = None,
    conditional_rate_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blockers = []
    if one_year_gate.get("status") != "pass":
        blockers.append("one_year_blocked_time_gate_failed")
    if five_year_gate.get("status") != "pass":
        blockers.append("five_year_bounded_gate_failed")
    if long_horizon_status.get("status") != "pass":
        blockers.append("long_horizon_bounded_projection_failed")
    if back_half_gate is not None and back_half_gate.get("status") == "fail":
        blockers.append("support_aware_back_half_gate_failed")
    if conditional_rate_gate is not None and conditional_rate_gate.get("status") == "fail":
        blockers.append("support_aware_conditional_rate_gate_failed")
    return {
        "status": "promote" if not blockers else "diagnostic_only",
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "contract": "decomposition controls promote only if they beat current reference and carry-forward on one-year blocked time, five-year bounded analog, bounded long-horizon stability, and do not regress on supported VL/suppression counts or conditional rates when that evidence is observed",
    }


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    one = dict(payload.get("one_year_gate") or {})
    five = dict(payload.get("five_year_gate") or {})
    labels = ["1y candidate", "1y reference", "1y carry", "5y candidate", "5y reference", "5y carry"]
    values = [
        float(one.get("candidate_mean_mae") or 0.0),
        float(one.get("baseline_mean_mae") or 0.0),
        float(one.get("carry_forward_mean_mae") or 0.0),
        float(five.get("candidate_mean_mae") or 0.0),
        float(five.get("baseline_mean_mae") or 0.0),
        float(five.get("carry_forward_mean_mae") or 0.0),
    ]
    colors = ["#b23a48", "#6b7280", "#2f6f73", "#b23a48", "#6b7280", "#2f6f73"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, values, color=colors, width=0.72)
    axes[0].set_xticks(x, labels, rotation=30, ha="right")
    axes[0].set_ylabel("Mean normalized MAE")
    axes[0].set_title("A. Promotion Gates", loc="left", fontweight="bold")
    axes[0].grid(axis="y", color="#e5e7eb", linewidth=0.7)
    promotion = dict(payload.get("promotion_gate") or {})
    long_status = dict(payload.get("long_horizon_status") or {})
    text = [
        "HIV decomposition controls",
        f"Promotion: {promotion.get('status')}",
        f"1y gate: {one.get('status')}",
        f"5y gate: {five.get('status')}",
        f"Long horizon: {long_status.get('status')}",
        "",
        "Blockers:",
    ]
    text.extend([f"- {value}" for value in list(promotion.get("blockers") or [])] or ["- none"])
    axes[1].axis("off")
    axes[1].text(0.0, 1.0, "\n".join(text), va="top", ha="left", family="monospace", fontsize=10)
    for ax in axes[:1]:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 HIV Stream Decomposition Branch", fontsize=14, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def run_decomposition_research(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    reference_report_path: str | Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
    scenario_start_year: int = 2026,
    scenario_end_year: int = 2035,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_run = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    observation_rows = build_observation_rows(epigraph_root, source_run_id=source_run, baseline_source_run_id=baseline_run)
    constraint_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run,
        baseline_source_run_id=baseline_run,
        include_validation_only=True,
    )
    reference_config = _load_reference_config(None if reference_report_path is None else Path(reference_report_path))
    decomposition_cfg = DecompositionControlConfig()
    one_year_splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=1,
    )
    five_year_splits = rolling_origin_splits(
        observation_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizon_years=5,
    )
    one_year_rows = [
        row
        for split in one_year_splits
        if (row := _evaluate_split(
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            split=split,
            reference_config=reference_config,
            decomposition_cfg=decomposition_cfg,
        )) is not None
    ]
    five_year_rows = [
        row
        for split in five_year_splits
        if (row := _evaluate_split(
            observation_rows=observation_rows,
            constraint_rows=constraint_rows,
            split=split,
            reference_config=reference_config,
            decomposition_cfg=decomposition_cfg,
        )) is not None
    ]
    one_year_gate = _score_gate(one_year_rows, gate_name="one_year_blocked_time")
    five_year_gate = _score_gate(five_year_rows, gate_name="five_year_bounded_analog")
    back_half_gate = _score_supported_back_half_gate(one_year_rows, gate_name="one_year_support_aware_vl_suppression")
    conditional_rate_gate = _score_supported_rate_gate(one_year_rows, gate_name="one_year_support_aware_back_half_rates")
    long_horizon = _long_horizon_status(
        observation_rows=observation_rows,
        constraint_rows=constraint_rows,
        reference_config=reference_config,
        decomposition_cfg=decomposition_cfg,
        scenario_start_year=scenario_start_year,
        scenario_end_year=scenario_end_year,
    )
    promotion = _promotion_gate(
        one_year_gate,
        five_year_gate,
        long_horizon,
        back_half_gate=back_half_gate,
        conditional_rate_gate=conditional_rate_gate,
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "decomposition_research_report.json"
    dashboard_path = analysis_dir / "decomposition_research_dashboard.png"
    payload = {
        "schema_version": DECOMPOSITION_RESEARCH_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run,
        "baseline_source_run_id": baseline_run,
        "reference_config": {
            "source": reference_config.get("source"),
            "path": reference_config.get("path"),
        },
        "paper_inspiration": {
            "source": "https://arxiv.org/html/2602.06323v1",
            "adapted_idea": "decomposition controls for explicit non-stationarity under partial observability",
            "hiv_adaptation": "trend + reporting/support shift + residual shock for incidence, diagnosis, ART, VL, and suppression streams",
        },
        "one_year_gate": one_year_gate,
        "five_year_gate": five_year_gate,
        "support_aware_back_half_gate": back_half_gate,
        "support_aware_conditional_rate_gate": conditional_rate_gate,
        "long_horizon_status": long_horizon,
        "promotion_gate": promotion,
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_dashboard(payload, dashboard_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="phase3-dynamic-decomposition-research")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--reference-report-path")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--scenario-start-year", type=int, default=2026)
    parser.add_argument("--scenario-end-year", type=int, default=2035)
    args = parser.parse_args()
    payload = run_decomposition_research(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        reference_report_path=args.reference_report_path,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
        scenario_start_year=args.scenario_start_year,
        scenario_end_year=args.scenario_end_year,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
