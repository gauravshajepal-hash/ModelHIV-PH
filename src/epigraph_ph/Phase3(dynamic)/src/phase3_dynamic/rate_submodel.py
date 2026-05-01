from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .data import build_observation_rows, default_epigraph_root, rolling_origin_splits, sandbox_repo_root
from .metrics import (
    OBSERVED_SUPPORT_TIERS,
    RATE_METRIC_DEFINITIONS,
    SUPPORTED_SCORING_ROLES,
    SUPPORT_AWARE_BACK_HALF_RATE_METRICS,
    _rate_support_provenance,
    _rate_support_value,
    inv_logit,
    logit,
    quarter_ordinal,
    quarter_sort_key,
    quarter_year,
)
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .runtime import ensure_dir, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID


RATE_SUBMODEL_SCHEMA_VERSION = "phase3_dynamic_vl_suppression_rate_submodel.v1"
SCORE_TOLERANCE = 1e-9
BASELINE_FAMILY = "last_observed_carry_forward"
RATE_MODEL_FAMILIES: tuple[str, ...] = (
    BASELINE_FAMILY,
    "train_mean",
    "logit_linear_trend",
    "logit_last_slope",
)
CANDIDATE_RATE_MODEL_FAMILIES: tuple[str, ...] = tuple(
    family for family in RATE_MODEL_FAMILIES if family != BASELINE_FAMILY
)
CLAIM_GRADE_REQUIRED_HOLDOUT_ERAS: tuple[str, ...] = (
    "post_covid_rebound",
    "recent",
)
MIN_CLAIM_GRADE_COMPARABLE_SPLITS = len(CLAIM_GRADE_REQUIRED_HOLDOUT_ERAS)


@dataclass(frozen=True, slots=True)
class RateTarget:
    quarter: str
    rate_name: str
    value: float
    ordinal: int
    provenance: dict[str, object]


def _supported_rate_targets(rows: list[dict[str, Any]], *, eps: float) -> list[RateTarget]:
    targets: list[RateTarget] = []
    for row in sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        for rate_name in SUPPORT_AWARE_BACK_HALF_RATE_METRICS:
            value = _rate_support_value(
                row,
                rate_name,
                eps=eps,
                require_supported_target=True,
                allowed_roles=SUPPORTED_SCORING_ROLES,
                observed_tiers=OBSERVED_SUPPORT_TIERS,
            )
            if value is None:
                continue
            targets.append(
                RateTarget(
                    quarter=quarter,
                    rate_name=rate_name,
                    value=float(value),
                    ordinal=quarter_ordinal(quarter),
                    provenance=_rate_support_provenance(row, rate_name),
                )
            )
    return targets


def _rows_before_holdout(observation_rows: list[dict[str, Any]], holdout_years: list[int]) -> list[dict[str, Any]]:
    holdout_start = min(int(year) for year in holdout_years)
    return [
        dict(row)
        for row in observation_rows
        if quarter_year(str(row.get("quarter") or "")) < holdout_start
    ]


def _rows_in_holdout(observation_rows: list[dict[str, Any]], holdout_years: list[int]) -> list[dict[str, Any]]:
    years = {int(year) for year in holdout_years}
    return [
        dict(row)
        for row in observation_rows
        if quarter_year(str(row.get("quarter") or "")) in years
    ]


def _forecast_one_rate(
    train_targets: list[RateTarget],
    holdout_quarters: list[str],
    *,
    family: str,
    eps: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    train_targets = sorted(train_targets, key=lambda target: (target.ordinal, target.quarter))
    if not train_targets:
        return {}, {"status": "no_train_support", "train_supported_count": 0}
    last = train_targets[-1]
    if family == BASELINE_FAMILY:
        return {quarter: float(last.value) for quarter in holdout_quarters}, {
            "status": "fit",
            "train_supported_count": len(train_targets),
            "last_observed_quarter": last.quarter,
        }
    values = np.asarray([target.value for target in train_targets], dtype=np.float64)
    if family == "train_mean":
        mean_value = float(np.clip(np.mean(values), 0.0, 1.0))
        return {quarter: mean_value for quarter in holdout_quarters}, {
            "status": "fit",
            "train_supported_count": len(train_targets),
            "mean_value": mean_value,
        }
    if len(train_targets) < 2:
        return {}, {"status": "insufficient_train_support", "train_supported_count": len(train_targets)}
    x = np.asarray([float(target.ordinal) for target in train_targets], dtype=np.float64)
    y = np.asarray([logit(float(target.value), eps=eps) for target in train_targets], dtype=np.float64)
    if family == "logit_linear_trend":
        centered_x = x - float(np.mean(x))
        design = np.column_stack([np.ones_like(centered_x), centered_x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        holdout = {}
        for quarter in holdout_quarters:
            position = float(quarter_ordinal(quarter)) - float(np.mean(x))
            holdout[quarter] = float(inv_logit(float(beta[0]) + float(beta[1]) * position))
        return holdout, {
            "status": "fit",
            "train_supported_count": len(train_targets),
            "intercept": float(beta[0]),
            "slope_per_quarter": float(beta[1]),
        }
    if family == "logit_last_slope":
        previous = train_targets[-2]
        gap = max(float(last.ordinal - previous.ordinal), 1.0)
        slope = float((logit(last.value, eps=eps) - logit(previous.value, eps=eps)) / gap)
        last_eta = logit(last.value, eps=eps)
        holdout = {
            quarter: float(inv_logit(last_eta + slope * max(float(quarter_ordinal(quarter) - last.ordinal), 0.0)))
            for quarter in holdout_quarters
        }
        return holdout, {
            "status": "fit",
            "train_supported_count": len(train_targets),
            "last_observed_quarter": last.quarter,
            "slope_per_quarter": slope,
        }
    raise ValueError(f"Unsupported rate model family: {family}")


def forecast_rate_family(
    train_targets: list[RateTarget],
    holdout_quarters: list[str],
    *,
    family: str,
    eps: float,
) -> dict[str, Any]:
    predictions: dict[tuple[str, str], float] = {}
    diagnostics: dict[str, Any] = {}
    for rate_name in SUPPORT_AWARE_BACK_HALF_RATE_METRICS:
        rate_train = [target for target in train_targets if target.rate_name == rate_name]
        rate_prediction, rate_diagnostics = _forecast_one_rate(
            rate_train,
            holdout_quarters,
            family=family,
            eps=eps,
        )
        for quarter, value in rate_prediction.items():
            predictions[(quarter, rate_name)] = float(value)
        diagnostics[rate_name] = rate_diagnostics
    return {
        "family": family,
        "predictions": predictions,
        "diagnostics": diagnostics,
    }


def _score_rate_forecast(
    prediction_map: dict[tuple[str, str], float],
    targets: list[RateTarget],
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for rate_name in SUPPORT_AWARE_BACK_HALF_RATE_METRICS:
        rate_targets = [target for target in targets if target.rate_name == rate_name]
        rate_errors: list[float] = []
        missing_prediction_count = 0
        support_partition_counts: dict[str, int] = {}
        for target in rate_targets:
            partition = str(target.provenance.get("support_partition") or "unknown")
            support_partition_counts[partition] = support_partition_counts.get(partition, 0) + 1
            prediction = prediction_map.get((target.quarter, rate_name))
            if prediction is None:
                missing_prediction_count += 1
                continue
            error = abs(float(prediction) - float(target.value))
            rate_errors.append(error)
        errors.extend(rate_errors)
        if missing_prediction_count:
            status = "not_evaluable"
            mean_mae = float("inf")
            worst_mae = float("inf")
        elif rate_errors:
            status = "scored"
            mean_mae = float(np.mean(rate_errors))
            worst_mae = float(np.max(rate_errors))
        elif rate_targets:
            status = "not_evaluable"
            mean_mae = float("inf")
            worst_mae = float("inf")
        else:
            status = "not_observed"
            mean_mae = float("inf")
            worst_mae = float("inf")
        metric_rows.append(
            {
                "metric_name": rate_name,
                "status": status,
                "target_count": len(rate_targets),
                "scored_entry_count": len(rate_errors),
                "missing_prediction_count": missing_prediction_count,
                "mean_mae": mean_mae,
                "worst_mae": worst_mae,
                "support_partition_counts": support_partition_counts,
                "definition": dict(RATE_METRIC_DEFINITIONS[rate_name]),
            }
        )
    missing_predictions = int(sum(int(row["missing_prediction_count"]) for row in metric_rows))
    scored_count = int(sum(int(row["scored_entry_count"]) for row in metric_rows))
    target_count = int(sum(int(row["target_count"]) for row in metric_rows))
    if missing_predictions:
        status = "not_evaluable"
        mean_mae = float("inf")
        worst_mae = float("inf")
    elif scored_count:
        status = "scored"
        mean_mae = float(np.mean(errors))
        worst_mae = float(np.max(errors))
    elif target_count:
        status = "not_evaluable"
        mean_mae = float("inf")
        worst_mae = float("inf")
    else:
        status = "not_observed"
        mean_mae = float("inf")
        worst_mae = float("inf")
    return {
        "status": status,
        "mean_mae": mean_mae,
        "worst_mae": worst_mae,
        "target_count": target_count,
        "scored_entry_count": scored_count,
        "metric_rows": metric_rows,
    }


def _evaluate_split(
    observation_rows: list[dict[str, Any]],
    split: dict[str, Any],
    *,
    family: str,
    eps: float,
) -> dict[str, Any] | None:
    train_rows = _rows_before_holdout(observation_rows, list(split["holdout_years"]))
    holdout_rows = _rows_in_holdout(observation_rows, list(split["holdout_years"]))
    train_targets = _supported_rate_targets(train_rows, eps=eps)
    holdout_targets = _supported_rate_targets(holdout_rows, eps=eps)
    if not holdout_targets:
        return None
    holdout_quarters = sorted({target.quarter for target in holdout_targets}, key=quarter_sort_key)
    baseline = forecast_rate_family(
        train_targets,
        holdout_quarters,
        family=BASELINE_FAMILY,
        eps=eps,
    )
    candidate = forecast_rate_family(
        train_targets,
        holdout_quarters,
        family=family,
        eps=eps,
    )
    baseline_score = _score_rate_forecast(dict(baseline["predictions"]), holdout_targets)
    candidate_score = _score_rate_forecast(dict(candidate["predictions"]), holdout_targets)
    return {
        "train_end_year": int(split["train_end_year"]),
        "holdout_years": list(split["holdout_years"]),
        "holdout_quarters": holdout_quarters,
        "train_rate_target_count": len(train_targets),
        "holdout_rate_target_count": len(holdout_targets),
        "baseline": {
            "family": BASELINE_FAMILY,
            "score": baseline_score,
            "diagnostics": baseline["diagnostics"],
        },
        "candidate": {
            "family": family,
            "score": candidate_score,
            "diagnostics": candidate["diagnostics"],
        },
    }


def _finite(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return score


def _score_family_gate(rows: list[dict[str, Any]], *, family: str, gate_name: str) -> dict[str, Any]:
    if family == BASELINE_FAMILY:
        baseline_rows = []
        for row in rows:
            score = dict((row.get("baseline") or {}).get("score") or {})
            mean_mae = _finite(score.get("mean_mae"))
            worst_mae = _finite(score.get("worst_mae"))
            if mean_mae is None or worst_mae is None:
                continue
            baseline_rows.append(
                {
                    "train_end_year": row.get("train_end_year"),
                    "holdout_years": row.get("holdout_years"),
                    "mean_mae": mean_mae,
                    "worst_mae": worst_mae,
                    "target_count": int(score.get("target_count") or 0),
                }
            )
        means = [float(row["mean_mae"]) for row in baseline_rows]
        worst = [float(row["worst_mae"]) for row in baseline_rows]
        return {
            "gate": gate_name,
            "family": family,
            "status": "baseline",
            "claim_status": "reference_only",
            "blockers": [],
            "split_count": len(rows),
            "comparable_split_count": len(baseline_rows),
            "candidate_mean_mae": float(np.mean(means)) if means else float("inf"),
            "baseline_mean_mae": float(np.mean(means)) if means else float("inf"),
            "candidate_worst_mae": float(np.max(worst)) if worst else float("inf"),
            "baseline_worst_mae": float(np.max(worst)) if worst else float("inf"),
            "candidate_minus_baseline_mean_mae": 0.0,
            "rows": baseline_rows,
        }
    gate_rows: list[dict[str, Any]] = []
    for row in rows:
        baseline_score = dict((row.get("baseline") or {}).get("score") or {})
        candidate_score = dict((row.get("candidate") or {}).get("score") or {})
        baseline_mean = _finite(baseline_score.get("mean_mae"))
        candidate_mean = _finite(candidate_score.get("mean_mae"))
        baseline_worst = _finite(baseline_score.get("worst_mae"))
        candidate_worst = _finite(candidate_score.get("worst_mae"))
        if baseline_mean is None or candidate_mean is None or baseline_worst is None or candidate_worst is None:
            gate_rows.append(
                {
                    "train_end_year": row.get("train_end_year"),
                    "holdout_years": row.get("holdout_years"),
                    "status": "not_comparable",
                    "baseline_status": baseline_score.get("status"),
                    "candidate_status": candidate_score.get("status"),
                    "target_count": int(candidate_score.get("target_count") or baseline_score.get("target_count") or 0),
                }
            )
            continue
        gate_rows.append(
            {
                "train_end_year": row.get("train_end_year"),
                "holdout_years": row.get("holdout_years"),
                "status": "comparable",
                "baseline_mean_mae": baseline_mean,
                "candidate_mean_mae": candidate_mean,
                "baseline_worst_mae": baseline_worst,
                "candidate_worst_mae": candidate_worst,
                "mean_delta_vs_last_observed": float(candidate_mean - baseline_mean),
                "worst_delta_vs_last_observed": float(candidate_worst - baseline_worst),
                "target_count": int(candidate_score.get("target_count") or 0),
            }
        )
    comparable_rows = [row for row in gate_rows if row.get("status") == "comparable"]
    if not comparable_rows:
        return {
            "gate": gate_name,
            "family": family,
            "status": "fail",
            "claim_status": "not_claimable",
            "blockers": ["no_comparable_conditional_rate_splits"],
            "split_count": len(rows),
            "comparable_split_count": 0,
            "rows": gate_rows,
        }
    candidate_mean_values = np.asarray([float(row["candidate_mean_mae"]) for row in comparable_rows], dtype=np.float64)
    baseline_mean_values = np.asarray([float(row["baseline_mean_mae"]) for row in comparable_rows], dtype=np.float64)
    candidate_worst_values = np.asarray([float(row["candidate_worst_mae"]) for row in comparable_rows], dtype=np.float64)
    baseline_worst_values = np.asarray([float(row["baseline_worst_mae"]) for row in comparable_rows], dtype=np.float64)
    mean_delta = float(np.mean(candidate_mean_values) - np.mean(baseline_mean_values))
    worst_delta = float(np.max(candidate_worst_values) - np.max(baseline_worst_values))
    blockers: list[str] = []
    if mean_delta >= -SCORE_TOLERANCE:
        blockers.append("candidate_mean_not_better_than_last_observed_rate_carry_forward")
    if worst_delta > SCORE_TOLERANCE:
        blockers.append("candidate_worst_regresses_against_last_observed_rate_carry_forward")
    if mean_delta < -SCORE_TOLERANCE:
        claim_status = "supports_conditional_rate_improvement_claim"
    elif abs(mean_delta) <= SCORE_TOLERANCE and worst_delta <= SCORE_TOLERANCE:
        claim_status = "conditional_rate_non_regression_only"
    else:
        claim_status = "rejects_conditional_rate_improvement_claim"
    return {
        "gate": gate_name,
        "family": family,
        "status": "pass" if not blockers else "fail",
        "claim_status": claim_status,
        "blockers": blockers,
        "split_count": len(rows),
        "comparable_split_count": len(comparable_rows),
        "candidate_mean_mae": float(np.mean(candidate_mean_values)),
        "baseline_mean_mae": float(np.mean(baseline_mean_values)),
        "candidate_worst_mae": float(np.max(candidate_worst_values)),
        "baseline_worst_mae": float(np.max(baseline_worst_values)),
        "candidate_minus_baseline_mean_mae": mean_delta,
        "candidate_minus_baseline_worst_mae": worst_delta,
        "rows": gate_rows,
    }


def _holdout_era_labels(holdout_years: list[int]) -> list[str]:
    labels: list[str] = []
    years = {int(year) for year in holdout_years}
    if any(year <= 2019 for year in years):
        labels.append("pre_covid")
    if any(2020 <= year <= 2021 for year in years):
        labels.append("covid_disruption")
    if any(2022 <= year <= 2023 for year in years):
        labels.append("post_covid_rebound")
    if any(year >= 2024 for year in years):
        labels.append("recent")
    return labels


def _covered_comparable_eras(gate: dict[str, Any]) -> list[str]:
    covered: set[str] = set()
    for row in list(gate.get("rows") or []):
        if row.get("status") != "comparable":
            continue
        covered.update(_holdout_era_labels([int(year) for year in row.get("holdout_years") or []]))
    return sorted(covered)


def _metric_row_map(score: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("metric_name")): dict(row) for row in list(score.get("metric_rows") or [])}


def _per_rate_consistency(split_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for rate_name in SUPPORT_AWARE_BACK_HALF_RATE_METRICS:
        rate_mean_deltas: list[float] = []
        rate_worst_deltas: list[float] = []
        scored_split_count = 0
        target_count = 0
        for split in split_rows:
            baseline_score = dict((split.get("baseline") or {}).get("score") or {})
            candidate_score = dict((split.get("candidate") or {}).get("score") or {})
            baseline_metric = _metric_row_map(baseline_score).get(rate_name)
            candidate_metric = _metric_row_map(candidate_score).get(rate_name)
            if not baseline_metric or not candidate_metric:
                continue
            baseline_mean = _finite(baseline_metric.get("mean_mae"))
            candidate_mean = _finite(candidate_metric.get("mean_mae"))
            baseline_worst = _finite(baseline_metric.get("worst_mae"))
            candidate_worst = _finite(candidate_metric.get("worst_mae"))
            if baseline_mean is None or candidate_mean is None or baseline_worst is None or candidate_worst is None:
                continue
            scored_split_count += 1
            target_count += int(candidate_metric.get("target_count") or 0)
            rate_mean_deltas.append(float(candidate_mean - baseline_mean))
            rate_worst_deltas.append(float(candidate_worst - baseline_worst))
        if not rate_mean_deltas:
            rows.append(
                {
                    "metric_name": rate_name,
                    "status": "not_claimable",
                    "blockers": ["no_comparable_metric_support"],
                    "scored_split_count": 0,
                    "target_count": 0,
                    "mean_delta_vs_last_observed": None,
                    "worst_delta_vs_last_observed": None,
                }
            )
            continue
        mean_delta = float(np.mean(rate_mean_deltas))
        worst_delta = float(np.max(rate_worst_deltas))
        blockers: list[str] = []
        if mean_delta >= -SCORE_TOLERANCE:
            blockers.append("metric_mean_not_better_than_last_observed")
        if worst_delta > SCORE_TOLERANCE:
            blockers.append("metric_worst_regresses_against_last_observed")
        rows.append(
            {
                "metric_name": rate_name,
                "status": "pass" if not blockers else "not_claimable",
                "blockers": blockers,
                "scored_split_count": scored_split_count,
                "target_count": target_count,
                "mean_delta_vs_last_observed": mean_delta,
                "worst_delta_vs_last_observed": worst_delta,
            }
        )
    return {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "rows": rows,
    }


def _claim_grade_support_gate(
    one_year: dict[str, Any],
    five_year: dict[str, Any],
    *,
    family: str,
) -> dict[str, Any]:
    if family == BASELINE_FAMILY:
        return {
            "status": "baseline",
            "claim_status": "reference_only",
            "promotion_eligible": False,
            "blockers": ["baseline_reference_not_candidate"],
        }
    blockers: list[str] = []
    one_gate = dict(one_year.get("gate") or {})
    five_gate = dict(five_year.get("gate") or {})
    if one_gate.get("status") != "pass":
        blockers.append("one_year_rate_gate_failed")
    if five_gate.get("status") != "pass":
        blockers.append("five_year_rate_gate_failed")
    one_comparable = int(one_gate.get("comparable_split_count") or 0)
    five_comparable = int(five_gate.get("comparable_split_count") or 0)
    if one_comparable < MIN_CLAIM_GRADE_COMPARABLE_SPLITS:
        blockers.append("insufficient_one_year_comparable_splits_for_claim_grade")
    if five_comparable < MIN_CLAIM_GRADE_COMPARABLE_SPLITS:
        blockers.append("insufficient_five_year_comparable_splits_for_claim_grade")
    one_eras = _covered_comparable_eras(one_gate)
    five_eras = _covered_comparable_eras(five_gate)
    covered_eras = sorted(set(one_eras).union(five_eras))
    for era in CLAIM_GRADE_REQUIRED_HOLDOUT_ERAS:
        if era not in set(one_eras) or era not in set(five_eras):
            blockers.append(f"missing_{era}_coverage_in_both_blocked_gates")
    one_consistency = _per_rate_consistency(list(one_year.get("split_rows") or []))
    five_consistency = _per_rate_consistency(list(five_year.get("split_rows") or []))
    for horizon_name, consistency in (("one_year", one_consistency), ("five_year", five_consistency)):
        for row in list(consistency.get("rows") or []):
            if row.get("status") != "pass":
                blockers.append(f"{horizon_name}_{row.get('metric_name')}_not_consistently_better")
    return {
        "status": "pass" if not blockers else "fail",
        "claim_status": "publication_grade_third_95_rate_process" if not blockers else "exploratory_or_not_claimable",
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "contract": (
            "Claim-grade third-95 rate promotion requires both blocked-time rate gates to pass, "
            "coverage of post-COVID rebound and recent holdout eras in both gates, and every "
            "supported conditional-rate metric to improve mean error without worst-error regression."
        ),
        "minimum_comparable_splits_per_gate": MIN_CLAIM_GRADE_COMPARABLE_SPLITS,
        "required_holdout_eras": list(CLAIM_GRADE_REQUIRED_HOLDOUT_ERAS),
        "covered_holdout_eras": covered_eras,
        "one_year_covered_holdout_eras": one_eras,
        "five_year_covered_holdout_eras": five_eras,
        "one_year_per_rate_consistency": one_consistency,
        "five_year_per_rate_consistency": five_consistency,
        "scope_limitations": [
            "This local VL/suppression rate gate does not address whether the full Phase 3 model beats the R10 endpoint family.",
            "A global third-95 claim still requires the full state/readout model to beat carry-forward and R10-family comparators.",
        ],
    }


def _evaluate_family(
    observation_rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    *,
    family: str,
    eps: float,
    gate_name: str,
) -> dict[str, Any]:
    rows = [
        row
        for split in splits
        if (row := _evaluate_split(observation_rows, split, family=family, eps=eps)) is not None
    ]
    gate = _score_family_gate(rows, family=family, gate_name=gate_name)
    return {
        "family": family,
        "gate": gate,
        "split_rows": rows,
    }


def _promotion_gate(one_year: dict[str, Any], five_year: dict[str, Any], claim_grade: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    if (one_year.get("gate") or {}).get("status") != "pass":
        blockers.append("one_year_rate_gate_failed")
    if (five_year.get("gate") or {}).get("status") != "pass":
        blockers.append("five_year_rate_gate_failed")
    if claim_grade.get("status") != "pass":
        blockers.append("claim_grade_support_gate_failed")
        blockers.extend(str(blocker) for blocker in claim_grade.get("blockers") or [])
    blockers = list(dict.fromkeys(blockers))
    return {
        "status": "promote" if not blockers else "diagnostic_only",
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "contract": "A VL/suppression rate submodel promotes only if it beats last-observed conditional-rate carry-forward on both blocked-time gates and passes the claim-grade support gate.",
    }


def _summary_rows(family_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in family_reports:
        one = dict((report.get("one_year") or {}).get("gate") or {})
        five = dict((report.get("five_year") or {}).get("gate") or {})
        claim_grade = dict(report.get("claim_grade_gate") or {})
        promotion = dict(report.get("promotion_gate") or {})
        rows.append(
            {
                "family": report.get("family"),
                "one_year_status": one.get("status"),
                "one_year_claim_status": one.get("claim_status"),
                "one_year_candidate_mean_mae": one.get("candidate_mean_mae"),
                "one_year_baseline_mean_mae": one.get("baseline_mean_mae"),
                "one_year_delta_vs_last_observed": one.get("candidate_minus_baseline_mean_mae"),
                "five_year_status": five.get("status"),
                "five_year_claim_status": five.get("claim_status"),
                "five_year_candidate_mean_mae": five.get("candidate_mean_mae"),
                "five_year_baseline_mean_mae": five.get("baseline_mean_mae"),
                "five_year_delta_vs_last_observed": five.get("candidate_minus_baseline_mean_mae"),
                "claim_grade_status": claim_grade.get("status"),
                "claim_grade_claim_status": claim_grade.get("claim_status"),
                "claim_grade_blockers": list(claim_grade.get("blockers") or []),
                "promotion_status": promotion.get("status"),
                "promotion_blockers": list(promotion.get("blockers") or []),
            }
        )
    return rows


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    summaries = list(payload.get("summary") or [])
    candidate_summaries = [row for row in summaries if row.get("family") != BASELINE_FAMILY]
    families = [str(row.get("family")) for row in candidate_summaries]
    x = np.arange(len(families))
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.8), constrained_layout=True)
    flat = axes.ravel()
    one_values = [float(row.get("one_year_candidate_mean_mae") or 0.0) for row in candidate_summaries]
    one_baseline = [float(row.get("one_year_baseline_mean_mae") or 0.0) for row in candidate_summaries]
    five_values = [float(row.get("five_year_candidate_mean_mae") or 0.0) for row in candidate_summaries]
    five_baseline = [float(row.get("five_year_baseline_mean_mae") or 0.0) for row in candidate_summaries]
    flat[0].bar(x - 0.18, one_values, width=0.36, color="#b23a48", label="candidate")
    flat[0].bar(x + 0.18, one_baseline, width=0.36, color="#6b7280", label="last observed")
    flat[0].set_title("A. One-Year Conditional-Rate MAE", loc="left", fontweight="bold")
    flat[0].set_ylabel("Absolute rate MAE")
    flat[0].set_xticks(x, families, rotation=25, ha="right")
    flat[0].legend(frameon=False)
    flat[1].bar(x - 0.18, five_values, width=0.36, color="#2f6f73", label="candidate")
    flat[1].bar(x + 0.18, five_baseline, width=0.36, color="#6b7280", label="last observed")
    flat[1].set_title("B. Five-Year Conditional-Rate MAE", loc="left", fontweight="bold")
    flat[1].set_ylabel("Absolute rate MAE")
    flat[1].set_xticks(x, families, rotation=25, ha="right")
    flat[1].legend(frameon=False)
    one_delta = [float(row.get("one_year_delta_vs_last_observed") or 0.0) for row in candidate_summaries]
    five_delta = [float(row.get("five_year_delta_vs_last_observed") or 0.0) for row in candidate_summaries]
    flat[2].bar(x - 0.18, one_delta, width=0.36, color="#b23a48", label="1-year")
    flat[2].bar(x + 0.18, five_delta, width=0.36, color="#2f6f73", label="5-year")
    flat[2].axhline(0.0, color="#111827", linewidth=0.9)
    flat[2].set_title("C. Delta vs Last-Observed Baseline", loc="left", fontweight="bold")
    flat[2].set_ylabel("Candidate minus baseline MAE")
    flat[2].set_xticks(x, families, rotation=25, ha="right")
    flat[2].text(0.01, 0.02, "Negative = candidate improves rate process", transform=flat[2].transAxes, va="bottom", ha="left", fontsize=9)
    flat[2].legend(frameon=False)
    promoted = list(payload.get("promoted_families") or [])
    lines = [
        "Dedicated VL/suppression rate submodel",
        f"Promoted families: {', '.join(promoted) if promoted else 'none'}",
        f"Champion: {payload.get('champion_family') or 'none'}",
        "",
        "Claim rule:",
        "Improve conditional rates over last-observed",
        "carry-forward, then pass support-grade",
        "COVID/rebound and per-rate gates.",
        "",
        "If no family promotes, third-95 process",
        "claim remains non-supported.",
    ]
    flat[3].axis("off")
    flat[3].text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)
    for ax in flat[:3]:
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Phase 3 Dedicated VL/Suppression Conditional-Rate Submodel", fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=330, bbox_inches="tight")
    plt.close(fig)


def run_rate_submodel_research(
    *,
    run_id: str,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> dict[str, Any]:
    epigraph_root = default_epigraph_root()
    source_run = resolve_active_source_run_id(epigraph_root, source_run_id or DEFAULT_ACTIVE_SOURCE_RUN_ID)
    baseline_run = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run,
        preferred=baseline_source_run_id or DEFAULT_BASELINE_SOURCE_RUN_ID,
    )
    observation_rows = build_observation_rows(epigraph_root, source_run_id=source_run, baseline_source_run_id=baseline_run)
    eps = float(np.finfo(np.float32).eps)
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
    family_reports: list[dict[str, Any]] = []
    for family in RATE_MODEL_FAMILIES:
        one_year = _evaluate_family(
            observation_rows,
            one_year_splits,
            family=family,
            eps=eps,
            gate_name="one_year_conditional_rate_gate",
        )
        five_year = _evaluate_family(
            observation_rows,
            five_year_splits,
            family=family,
            eps=eps,
            gate_name="five_year_conditional_rate_gate",
        )
        claim_grade = _claim_grade_support_gate(one_year, five_year, family=family)
        promotion = (
            {
                "status": "baseline",
                "promotion_eligible": False,
                "blockers": ["baseline_reference_not_candidate"],
            }
            if family == BASELINE_FAMILY
            else _promotion_gate(one_year, five_year, claim_grade)
        )
        family_reports.append(
            {
                "family": family,
                "one_year": one_year,
                "five_year": five_year,
                "claim_grade_gate": claim_grade,
                "promotion_gate": promotion,
            }
        )
    summary = _summary_rows(family_reports)
    promoted = [str(report["family"]) for report in family_reports if (report.get("promotion_gate") or {}).get("promotion_eligible")]
    promoted_reports = [report for report in family_reports if str(report.get("family")) in set(promoted)]
    champion_family = None
    if promoted_reports:
        champion_family = min(
            promoted_reports,
            key=lambda report: (
                float(((report.get("one_year") or {}).get("gate") or {}).get("candidate_mean_mae") or float("inf")),
                float(((report.get("five_year") or {}).get("gate") or {}).get("candidate_mean_mae") or float("inf")),
            ),
        ).get("family")
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    report_path = analysis_dir / "rate_submodel_report.json"
    dashboard_path = analysis_dir / "rate_submodel_dashboard.png"
    payload = {
        "schema_version": RATE_SUBMODEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_run_id": source_run,
        "baseline_source_run_id": baseline_run,
        "rate_metrics": {
            key: dict(value)
            for key, value in RATE_METRIC_DEFINITIONS.items()
            if key in set(SUPPORT_AWARE_BACK_HALF_RATE_METRICS)
        },
        "families": list(RATE_MODEL_FAMILIES),
        "baseline_family": BASELINE_FAMILY,
        "contract": {
            "allowed_training_support": {
                "roles": list(SUPPORTED_SCORING_ROLES),
                "tiers": list(OBSERVED_SUPPORT_TIERS),
            },
            "baseline": "last observed supported conditional rate within the training window",
            "promotion_rule": "candidate must improve mean absolute conditional-rate error, not regress worst error versus last-observed carry-forward on both one-year and five-year blocked-time gates, and pass the claim-grade support gate",
            "claim_grade_support_rule": "claim-grade third-95 process language requires comparable post-COVID rebound and recent-era blocked splits plus per-rate consistency for VL testing among ART and suppression among VL-tested",
            "covid_reporting_shock_note": "2020-2021 diagnosis and service observations are treated as a disruption era; post-2022 rebound coverage is required before promoting third-95 process claims",
            "r10_comparator_note": "this local rate submodel does not establish superiority over the R10 endpoint family; the full Phase 3 champion must still beat R10 and carry-forward under blocked time",
            "claim_scope": "third-95 process claim allowed only if a candidate promotes after claim-grade support; otherwise report ART-mediated back-half count improvement only",
        },
        "promoted_families": promoted,
        "champion_family": champion_family,
        "summary": summary,
        "family_reports": family_reports,
        "artifact_paths": {
            "report_json": report_path.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(report_path, payload)
    _write_dashboard(payload, dashboard_path)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(prog="phase3-dynamic-rate-submodel")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-run-id")
    parser.add_argument("--baseline-source-run-id")
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    payload = run_rate_submodel_research(
        run_id=args.run_id,
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(json.dumps(payload.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
