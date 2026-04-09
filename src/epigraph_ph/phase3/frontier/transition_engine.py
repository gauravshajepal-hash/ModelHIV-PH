from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase3._lineage.national_reset_core import PRIMARY_METRICS, quarter_sort_key
from epigraph_ph.phase3._lineage.national_reset_pipeline import build_national_reset_observation_table_payload
from epigraph_ph.runtime import ROOT_DIR, read_json, save_tensor_artifact, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .analytics import _collapse_kp_distribution, _factor_transition_rows
from .numeric_policy import numerical_guard_entry
from .registry import KP_COLLAPSED_NAMES, TRANSITION_NAMES
from .sources import load_transition_research_inputs

STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "V", "L")
NR01_EXPERIMENT_ID = "NR-01-national-uda-baseline"
MECH01A_EXPERIMENT_ID = "MECH-01A-national-udavl-baseline"
MECH01E_EXPERIMENT_ID = "MECH-01E-anchored-downstream-residual-helpers"
DOWNSTREAM_TRANSITIONS: tuple[str, ...] = tuple(
    transition for transition in TRANSITION_NAMES if transition != "U_to_D"
)


def _quarter_year(quarter: str) -> int:
    return int(quarter_sort_key(str(quarter))[0])


def _discover_latest_nr01_experiment() -> tuple[str, Path]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "phase3_national_reset" / NR01_EXPERIMENT_ID
        evaluation_path = experiment_dir / "evaluation.json"
        if evaluation_path.exists():
            candidates.append((evaluation_path.stat().st_mtime, run_dir.name, experiment_dir))
    if not candidates:
        raise FileNotFoundError("No NR-01 national reset experiment found under artifacts/runs")
    preferred = [row for row in candidates if "pytest" not in str(row[1]).lower()]
    pool = preferred or candidates
    _, run_id, experiment_dir = max(pool, key=lambda row: row[0])
    return run_id, experiment_dir


def _discover_latest_transition_experiment(experiment_id: str) -> tuple[str, Path]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / experiment_id
        evaluation_path = experiment_dir / "evaluation.json"
        if evaluation_path.exists():
            candidates.append((evaluation_path.stat().st_mtime, run_dir.name, experiment_dir))
    if not candidates:
        raise FileNotFoundError(f"No transition research experiment found for {experiment_id}")
    preferred = [row for row in candidates if "pytest" not in str(row[1]).lower()]
    pool = preferred or candidates
    _, run_id, experiment_dir = max(pool, key=lambda row: row[0])
    return run_id, experiment_dir


def _load_front_half_reference() -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_nr01_experiment()
    evaluation = read_json(experiment_dir / "evaluation.json", default={})
    baseline_comparison = read_json(experiment_dir / "baseline_comparison.json", default={})
    diagnosis_flow_evaluation = read_json(experiment_dir / "diagnosis_flow_evaluation.json", default={})
    fit_artifact = read_json(experiment_dir / "fit_artifact.json", default={})
    observation_table = read_json(experiment_dir / "observation_table.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "evaluation": dict(evaluation or {}),
        "baseline_comparison": dict(baseline_comparison or {}),
        "diagnosis_flow_evaluation": dict(diagnosis_flow_evaluation or {}),
        "fit_artifact": dict(fit_artifact or {}),
        "observation_table": dict(observation_table or {}),
    }


def _earliest_supported_quarter(archive_dir: Path) -> str:
    metric_rows = read_json(archive_dir / "historical_metric_rows.json", default=[])
    quarter_candidates: list[str] = []
    for row in list(metric_rows or []):
        metric_name = str(row.get("metric_name") or "")
        region = str(row.get("region") or "").lower()
        if region != "national":
            continue
        if metric_name not in {"diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period", "tested_for_viral_load", "virally_suppressed"}:
            continue
        month_label = str(row.get("period_end") or row.get("time") or "")
        if not month_label:
            continue
        year_text, month_text = month_label.split("-", 1)
        quarter_candidates.append(f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}")
    if not quarter_candidates:
        raise ValueError(f"No supported national observation quarter found in {archive_dir}")
    return min(quarter_candidates, key=quarter_sort_key)


def _holdout_years_from_reference(reference: dict[str, Any]) -> list[int]:
    evaluation = dict(reference.get("evaluation") or {})
    holdout_quarters = [str(value) for value in list(evaluation.get("holdout_quarters") or []) if str(value)]
    if not holdout_quarters:
        raise ValueError("MECH-01A requires holdout_quarters in the front-half reference evaluation")
    return sorted({_quarter_year(quarter) for quarter in holdout_quarters})


def _reference_forecast_rows(reference: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = dict(reference.get("evaluation") or {})
    holdout_rows = [dict(row) for row in list(evaluation.get("holdout_rows") or [])]
    rows: list[dict[str, Any]] = []
    for row in holdout_rows:
        prediction = dict(row.get("prediction") or {})
        prediction["quarter"] = str(row.get("quarter") or "")
        rows.append(prediction)
    if not rows:
        raise ValueError("MECH-01A requires holdout_rows in the front-half reference evaluation")
    return rows


def _build_observation_payload(ctx: TransitionResearchContext) -> dict[str, Any]:
    archive_dir = ctx.source_run_dir / "harp_archive"
    start_quarter = _earliest_supported_quarter(archive_dir)
    return build_national_reset_observation_table_payload(
        archive_dir=archive_dir,
        archive_run_id=ctx.source_run_id,
        start_quarter=start_quarter,
    )


def _with_holdout_forecasts(payload_rows: list[dict[str, Any]], forecast_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    forecast_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in forecast_rows}
    combined: list[dict[str, Any]] = []
    for row in payload_rows:
        quarter = str(row.get("quarter") or "")
        candidate = dict(row)
        if quarter in forecast_by_quarter:
            forecast = forecast_by_quarter[quarter]
            for metric_name in PRIMARY_METRICS:
                if forecast.get(metric_name) is not None:
                    candidate[metric_name] = float(forecast[metric_name])
        combined.append(candidate)
    return combined


def _train_rows(rows: list[dict[str, Any]], holdout_years: list[int]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("diagnosed_plhiv") is not None
        and row.get("alive_on_art") is not None
        and _quarter_year(str(row.get("quarter") or "")) < min(holdout_years)
    ]


def _holdout_rows(rows: list[dict[str, Any]], holdout_years: list[int]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in rows
        if row.get("diagnosed_plhiv") is not None
        and row.get("alive_on_art") is not None
        and _quarter_year(str(row.get("quarter") or "")) in set(holdout_years)
    ]


def _coverage_mean(reference: dict[str, Any], estimated_plhiv_by_quarter: dict[str, float], train_rows: list[dict[str, Any]]) -> float:
    fit_artifact = dict(reference.get("fit_artifact") or {})
    coverage_summary = dict(fit_artifact.get("coverage_summary") or {})
    coverage_mean = coverage_summary.get("coverage_mean")
    if coverage_mean is not None:
        return float(coverage_mean)
    coverages = []
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        estimated = float(estimated_plhiv_by_quarter.get(quarter) or 0.0)
        diagnosed = float(row.get("diagnosed_plhiv") or 0.0)
        if estimated > 0.0 and diagnosed > 0.0:
            coverages.append(diagnosed / estimated)
    if not coverages:
        raise ValueError("MECH-01A requires an estimated diagnosed coverage mean")
    return float(np.mean(coverages))


def _share_mean(rows: list[dict[str, Any]], numerator_key: str, denominator_key: str) -> float:
    shares = []
    for row in rows:
        numerator = row.get(numerator_key)
        denominator = row.get(denominator_key)
        if numerator is None or denominator is None:
            continue
        denominator_value = float(denominator)
        if denominator_value <= 0.0:
            continue
        shares.append(float(numerator) / denominator_value)
    return float(np.mean(shares)) if shares else 0.0


def _lost_gap_share(train_rows: list[dict[str, Any]]) -> float:
    shares = []
    for previous, current in zip(train_rows[:-1], train_rows[1:]):
        diagnosed_gap = max(float(previous.get("diagnosed_plhiv") or 0.0) - float(previous.get("alive_on_art") or 0.0), 0.0)
        art_drop = max(float(previous.get("alive_on_art") or 0.0) - float(current.get("alive_on_art") or 0.0), 0.0)
        if diagnosed_gap > 0.0 and art_drop > 0.0:
            shares.append(art_drop / diagnosed_gap)
    return float(np.mean(shares)) if shares else 0.0


def _state_row(
    *,
    row: dict[str, Any],
    estimated_plhiv_by_quarter: dict[str, float],
    coverage_mean: float,
    suppression_share_mean: float,
    lost_gap_share: float,
) -> dict[str, Any]:
    quarter = str(row.get("quarter") or "")
    diagnosed = max(float(row.get("diagnosed_plhiv") or 0.0), 0.0)
    alive_on_art = max(float(row.get("alive_on_art") or 0.0), 0.0)
    observed_suppressed = row.get("virally_suppressed")
    if observed_suppressed is None:
        virally_suppressed = min(alive_on_art, max(suppression_share_mean * alive_on_art, 0.0))
    else:
        virally_suppressed = min(alive_on_art, max(float(observed_suppressed), 0.0))
    diagnosed_gap = max(diagnosed - alive_on_art, 0.0)
    lost = min(diagnosed_gap, max(lost_gap_share * diagnosed_gap, 0.0))
    diagnosed_not_on_art = max(diagnosed_gap - lost, 0.0)
    engaged_unsuppressed = max(alive_on_art - virally_suppressed, 0.0)
    estimated_total = float(estimated_plhiv_by_quarter.get(quarter) or 0.0)
    if estimated_total <= 0.0 and coverage_mean > 0.0:
        estimated_total = diagnosed / coverage_mean
    estimated_total = max(estimated_total, diagnosed)
    undiagnosed = max(estimated_total - diagnosed, 0.0)
    state_values = {
        "U": float(undiagnosed),
        "D": float(diagnosed_not_on_art),
        "A": float(engaged_unsuppressed),
        "V": float(virally_suppressed),
        "L": float(lost),
    }
    return {
        "quarter": quarter,
        "diagnosed_plhiv": diagnosed,
        "alive_on_art": alive_on_art,
        "new_diagnosed_cases_period": row.get("new_diagnosed_cases_period"),
        "tested_for_viral_load": row.get("tested_for_viral_load"),
        "virally_suppressed": row.get("virally_suppressed"),
        "estimated_plhiv": estimated_total,
        "state_values": state_values,
    }


def _transition_denominator(state_name: str, state_values: dict[str, float]) -> float:
    return float(state_values.get(state_name) or 0.0)


def _flow_from_row(row: dict[str, Any], previous_state: dict[str, float], next_state: dict[str, float], eps: float) -> float:
    flow = row.get("new_diagnosed_cases_period")
    if flow is not None:
        return max(float(flow), 0.0)
    diagnosed_delta = max(
        float(next_state["D"] + next_state["A"] + next_state["V"] + next_state["L"])
        - float(previous_state["D"] + previous_state["A"] + previous_state["V"] + previous_state["L"]),
        0.0,
    )
    return max(diagnosed_delta, eps * 0.0)


def _transition_rows(state_rows: list[dict[str, Any]], eps: float) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    residuals: list[float] = []
    for previous, current in zip(state_rows[:-1], state_rows[1:]):
        previous_state = dict(previous["state_values"])
        current_state = dict(current["state_values"])
        u_to_d = _flow_from_row(current, previous_state, current_state, eps)
        d_to_a = max(float(previous_state["D"]) + u_to_d - float(current_state["D"]), 0.0)
        a_to_v = max(float(current_state["V"]) - float(previous_state["V"]), 0.0)
        l_delta = float(current_state["L"]) - float(previous_state["L"])
        if l_delta >= 0.0:
            a_to_l = float(l_delta)
            l_to_a = 0.0
        else:
            a_to_l = 0.0
            l_to_a = float(-l_delta)
        a_balance_prediction = float(previous_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l
        a_balance_residual = float(current_state["A"]) - a_balance_prediction
        residuals.append(a_balance_residual)
        flows = {
            "U_to_D": u_to_d,
            "D_to_A": d_to_a,
            "A_to_V": a_to_v,
            "A_to_L": a_to_l,
            "L_to_A": l_to_a,
        }
        denominators = {
            "U_to_D": _transition_denominator("U", previous_state),
            "D_to_A": _transition_denominator("D", previous_state),
            "A_to_V": _transition_denominator("A", previous_state),
            "A_to_L": _transition_denominator("A", previous_state),
            "L_to_A": _transition_denominator("L", previous_state),
        }
        hazard_row = {
            "quarter": str(current["quarter"]),
            "previous_quarter": str(previous["quarter"]),
            "a_balance_residual": round(float(a_balance_residual), 6),
            "flows": {name: round(float(value), 6) for name, value in flows.items()},
            "hazards": {
                name: round(float(value) / max(float(denominators[name]), eps), 6) if float(denominators[name]) > 0.0 else 0.0
                for name, value in flows.items()
            },
            "state_values_previous": {name: round(float(previous_state[name]), 6) for name in STATE_NAMES},
            "state_values_current": {name: round(float(current_state[name]), 6) for name in STATE_NAMES},
        }
        rows.append(hazard_row)
    summary = {
        "transition_count": len(rows),
        "mean_absolute_a_balance_residual": round(float(np.mean(np.abs(residuals))) if residuals else 0.0, 6),
        "max_absolute_a_balance_residual": round(float(np.max(np.abs(residuals))) if residuals else 0.0, 6),
        "transition_mean_hazards": {
            transition: round(float(np.mean([float(row["hazards"][transition]) for row in rows])) if rows else 0.0, 6)
            for transition in TRANSITION_NAMES
        },
    }
    return rows, summary


def _state_tensor(state_rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(row["state_values"][state_name]) for state_name in STATE_NAMES] for row in state_rows],
        dtype=np.float32,
    )


def _state_trajectory_rows(state_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for row in state_rows:
        payload.append(
            {
                "quarter": str(row["quarter"]),
                "estimated_plhiv": round(float(row["estimated_plhiv"]), 6),
                "diagnosed_plhiv": round(float(row["diagnosed_plhiv"]), 6),
                "alive_on_art": round(float(row["alive_on_art"]), 6),
                "new_diagnosed_cases_period": round(float(row.get("new_diagnosed_cases_period") or 0.0), 6)
                if row.get("new_diagnosed_cases_period") is not None
                else None,
                "state_values": {state_name: round(float(row["state_values"][state_name]), 6) for state_name in STATE_NAMES},
            }
        )
    return payload


def _metric_scales(rows: list[dict[str, Any]], eps: float) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric_name in PRIMARY_METRICS:
        observed = [abs(float(row.get(metric_name) or 0.0)) for row in rows if row.get(metric_name) is not None]
        scales[metric_name] = max(observed) if observed else eps
    return scales


def _normalized_mae(prediction_rows: list[dict[str, float]], target_rows: list[dict[str, Any]], metric_scales: dict[str, float], eps: float) -> float:
    errors: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            errors.append(abs(float(prediction_value) - float(target_value)) / max(float(metric_scales.get(metric_name) or eps), eps))
    return float(np.mean(errors)) if errors else float("inf")


def _smape(prediction_rows: list[dict[str, float]], target_rows: list[dict[str, Any]], eps: float) -> float:
    scores: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            prediction_value = prediction.get(metric_name)
            target_value = target.get(metric_name)
            if prediction_value is None or target_value is None:
                continue
            denom = abs(float(prediction_value)) + abs(float(target_value))
            if denom <= eps:
                continue
            scores.append((2.0 * abs(float(prediction_value) - float(target_value))) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _quarter_from_month(month_label: str) -> str:
    year_text, month_text = str(month_label).split("-", 1)
    return f"{int(year_text):04d}-Q{((int(month_text[:2]) - 1) // 3) + 1}"


def _quarter_factor_surface(ctx: TransitionResearchContext) -> dict[str, dict[str, float]]:
    inputs = load_transition_research_inputs(ctx)
    grouped_months: dict[str, list[int]] = defaultdict(list)
    for month_idx, month_label in enumerate(inputs.month_axis):
        grouped_months[_quarter_from_month(str(month_label))].append(month_idx)
    factor_surface: dict[str, dict[str, float]] = {}
    for factor_id, factor_idx in inputs.factor_index.items():
        values = np.asarray(inputs.national_tensor[0, :, factor_idx], dtype=np.float64)
        factor_surface[factor_id] = {
            quarter: float(np.mean(values[month_indices])) if month_indices else 0.0
            for quarter, month_indices in grouped_months.items()
        }
    return factor_surface


def _train_transition_rows(state_rows: list[dict[str, Any]], holdout_years: list[int], eps: float) -> list[dict[str, Any]]:
    transition_rows, _ = _transition_rows(state_rows, eps)
    return [dict(row) for row in transition_rows if _quarter_year(str(row.get("quarter") or "")) < min(holdout_years)]


def _zscore_map(values_by_quarter: dict[str, float], train_quarters: list[str], eps: float) -> tuple[dict[str, float], float, float]:
    train_values = [float(values_by_quarter.get(quarter) or 0.0) for quarter in train_quarters]
    mean_value = float(np.mean(train_values)) if train_values else 0.0
    std_value = float(np.std(train_values)) if train_values else 0.0
    if std_value <= eps:
        return ({quarter: 0.0 for quarter in values_by_quarter}, mean_value, std_value)
    return ({quarter: (float(value) - mean_value) / std_value for quarter, value in values_by_quarter.items()}, mean_value, std_value)


def _pearson_corr(x_values: list[float], y_values: list[float], eps: float) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_array = np.asarray(x_values, dtype=np.float64)
    y_array = np.asarray(y_values, dtype=np.float64)
    if float(np.std(x_array)) <= eps or float(np.std(y_array)) <= eps:
        return 0.0
    return float(np.corrcoef(x_array, y_array)[0, 1])


def _helper_model_summary(
    *,
    ctx: TransitionResearchContext,
    train_transition_rows: list[dict[str, Any]],
    eps: float,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    factor_surface = _quarter_factor_surface(ctx)
    transition_rows = _factor_transition_rows(load_transition_research_inputs(ctx))
    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    helper_rows: list[dict[str, Any]] = []
    quarter_signals: dict[str, dict[str, float]] = {transition: {} for transition in TRANSITION_NAMES}
    model_rows: list[dict[str, Any]] = []

    for transition in TRANSITION_NAMES:
        candidate_rows = [
            dict(row)
            for row in transition_rows
            if str(row.get("transition") or "") == transition and float(row.get("relevance") or 0.0) > 0.0
        ]
        hazard_targets = {
            str(row["quarter"]): float(row["hazards"][transition])
            for row in train_transition_rows
        }
        weighted_factors: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            factor_id = str(candidate["factor_id"])
            values_by_quarter = factor_surface.get(factor_id, {})
            standardized_map, mean_value, std_value = _zscore_map(values_by_quarter, train_quarters, eps)
            corr = _pearson_corr(
                [float(standardized_map.get(quarter) or 0.0) for quarter in train_quarters],
                [float(hazard_targets.get(quarter) or 0.0) for quarter in train_quarters],
                eps,
            )
            pre_weight = float(candidate["relevance"]) * abs(float(corr))
            weighted_factors.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(candidate.get("factor_name") or factor_id),
                    "transition_relevance": float(candidate["relevance"]),
                    "empirical_correlation": float(corr),
                    "pre_weight": float(pre_weight),
                    "train_mean": float(mean_value),
                    "train_std": float(std_value),
                    "standardized_map": standardized_map,
                }
            )
        weight_total = float(sum(row["pre_weight"] for row in weighted_factors))
        for row in weighted_factors:
            normalized_weight = float(row["pre_weight"] / weight_total) if weight_total > eps else 0.0
            row["normalized_weight"] = normalized_weight
            helper_rows.append(
                {
                    "transition": transition,
                    "factor_id": row["factor_id"],
                    "factor_name": row["factor_name"],
                    "transition_relevance": round(float(row["transition_relevance"]), 6),
                    "empirical_correlation": round(float(row["empirical_correlation"]), 6),
                    "pre_weight": round(float(row["pre_weight"]), 6),
                    "normalized_weight": round(normalized_weight, 6),
                    "train_mean": round(float(row["train_mean"]), 6),
                    "train_std": round(float(row["train_std"]), 6),
                }
            )
        all_quarters = sorted(
            {
                quarter
                for row in weighted_factors
                for quarter in row["standardized_map"].keys()
            },
            key=quarter_sort_key,
        )
        for quarter in all_quarters:
            quarter_signals[transition][quarter] = float(
                sum(float(row["normalized_weight"]) * float(row["standardized_map"].get(quarter) or 0.0) for row in weighted_factors)
            )

        train_signal = [float(quarter_signals[transition].get(quarter) or 0.0) for quarter in train_quarters]
        train_hazard = [float(hazard_targets[quarter]) for quarter in train_quarters]
        if len(train_hazard) >= 2:
            design = np.asarray(
                [
                    [1.0, float(train_hazard[idx - 1]), float(train_signal[idx])]
                    for idx in range(1, len(train_hazard))
                ],
                dtype=np.float64,
            )
            target = np.asarray([float(train_hazard[idx]) for idx in range(1, len(train_hazard))], dtype=np.float64)
            coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
            intercept, persistence, helper_scale = [float(value) for value in coefficients.tolist()]
        else:
            intercept = float(np.mean(train_hazard)) if train_hazard else 0.0
            persistence = 0.0
            helper_scale = 0.0
        model_rows.append(
            {
                "transition": transition,
                "intercept": round(intercept, 6),
                "persistence": round(persistence, 6),
                "helper_scale": round(helper_scale, 6),
                "eligible_factor_count": len(candidate_rows),
                "weighted_factor_count": sum(1 for row in weighted_factors if float(row["normalized_weight"]) > 0.0),
                "train_row_count": len(train_hazard),
                "last_train_hazard": round(float(train_hazard[-1]) if train_hazard else 0.0, 6),
            }
        )

    return {
        "helper_rows": helper_rows,
        "model_rows": model_rows,
    }, quarter_signals


def _simulate_holdout_with_helpers(
    *,
    train_state_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    train_transition_rows: list[dict[str, Any]],
    helper_models: dict[str, dict[str, float]],
    quarter_signals: dict[str, dict[str, float]],
    testing_share_mean: float,
    eps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_state = {state_name: float(train_state_rows[-1]["state_values"][state_name]) for state_name in STATE_NAMES}
    last_hazard = {
        transition: float(next(row for row in reversed(train_transition_rows) if transition in row["hazards"])["hazards"][transition])
        for transition in TRANSITION_NAMES
    }
    forecast_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []

    for target_row in holdout_rows:
        quarter = str(target_row["quarter"])
        predicted_hazards: dict[str, float] = {}
        for transition in TRANSITION_NAMES:
            model = dict(helper_models[transition])
            helper_signal = float(quarter_signals.get(transition, {}).get(quarter) or 0.0)
            hazard = (
                float(model["intercept"])
                + float(model["persistence"]) * float(last_hazard[transition])
                + float(model["helper_scale"]) * helper_signal
            )
            predicted_hazards[transition] = float(np.clip(hazard, 0.0, 1.0))

        u_to_d = min(float(current_state["U"]), predicted_hazards["U_to_D"] * float(current_state["U"]))
        d_to_a = min(float(current_state["D"]), predicted_hazards["D_to_A"] * float(current_state["D"]))
        l_to_a = min(float(current_state["L"]), predicted_hazards["L_to_A"] * float(current_state["L"]))
        raw_a_to_v = predicted_hazards["A_to_V"] * float(current_state["A"])
        raw_a_to_l = predicted_hazards["A_to_L"] * float(current_state["A"])
        raw_outflow = raw_a_to_v + raw_a_to_l
        a_scale = min(1.0, float(current_state["A"]) / max(raw_outflow, eps)) if raw_outflow > 0.0 else 1.0
        a_to_v = raw_a_to_v * a_scale
        a_to_l = raw_a_to_l * a_scale

        next_state = {
            "U": max(float(current_state["U"]) - u_to_d, 0.0),
            "D": max(float(current_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(current_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(float(current_state["V"]) + a_to_v, 0.0),
            "L": max(float(current_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        forecast_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(next_state["D"] + next_state["A"] + next_state["V"] + next_state["L"]),
                "alive_on_art": float(next_state["A"] + next_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": float(testing_share_mean * (next_state["A"] + next_state["V"])),
                "virally_suppressed": float(next_state["V"]),
                "state_values": {state_name: float(next_state[state_name]) for state_name in STATE_NAMES},
            }
        )
        hazard_rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: round(float(predicted_hazards[transition]), 6) for transition in TRANSITION_NAMES},
                "flows": {
                    "U_to_D": round(float(u_to_d), 6),
                    "D_to_A": round(float(d_to_a), 6),
                    "A_to_V": round(float(a_to_v), 6),
                    "A_to_L": round(float(a_to_l), 6),
                    "L_to_A": round(float(l_to_a), 6),
                },
                "helper_signals": {transition: round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6) for transition in TRANSITION_NAMES},
            }
        )
        current_state = next_state
        last_hazard = predicted_hazards

    return forecast_rows, hazard_rows


def _baseline_holdout_hazards(mech_01a_reference: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary = dict(mech_01a_reference.get("transition_hazard_summary") or {})
    rows = [dict(row) for row in list(summary.get("rows") or [])]
    holdout_quarters = {str(value) for value in list((mech_01a_reference.get("evaluation") or {}).get("holdout_quarters") or []) if str(value)}
    return {
        str(row["quarter"]): {transition: float(row["hazards"][transition]) for transition in TRANSITION_NAMES}
        for row in rows
        if str(row.get("quarter") or "") in holdout_quarters
    }


def _reference_holdout_hazards(reference: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary = dict(reference.get("transition_hazard_summary") or {})
    baseline_map = {
        str(quarter): {transition: float(value) for transition, value in dict(values).items()}
        for quarter, values in dict(summary.get("baseline_holdout_hazards") or {}).items()
    }
    for row in list(summary.get("holdout_rows") or []):
        quarter = str(row.get("quarter") or "")
        if not quarter:
            continue
        baseline_map[quarter] = {
            transition: float((row.get("hazards") or {}).get(transition) or 0.0)
            for transition in TRANSITION_NAMES
        }
    return baseline_map


def _residual_helper_model_summary(
    *,
    ctx: TransitionResearchContext,
    train_transition_rows: list[dict[str, Any]],
    eps: float,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    factor_surface = _quarter_factor_surface(ctx)
    transition_rows = _factor_transition_rows(load_transition_research_inputs(ctx))
    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    helper_rows: list[dict[str, Any]] = []
    quarter_signals: dict[str, dict[str, float]] = {transition: {} for transition in TRANSITION_NAMES}
    model_rows: list[dict[str, Any]] = []

    for transition in TRANSITION_NAMES:
        candidate_rows = [
            dict(row)
            for row in transition_rows
            if str(row.get("transition") or "") == transition and float(row.get("relevance") or 0.0) > 0.0
        ]
        hazard_targets = {
            str(row["quarter"]): float(row["hazards"][transition])
            for row in train_transition_rows
        }
        hazard_values = [float(hazard_targets[quarter]) for quarter in train_quarters]
        hazard_mean = float(np.mean(hazard_values)) if hazard_values else 0.0
        hazard_residuals = {quarter: float(hazard_targets[quarter] - hazard_mean) for quarter in train_quarters}
        weighted_factors: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            factor_id = str(candidate["factor_id"])
            values_by_quarter = factor_surface.get(factor_id, {})
            standardized_map, mean_value, std_value = _zscore_map(values_by_quarter, train_quarters, eps)
            corr = _pearson_corr(
                [float(standardized_map.get(quarter) or 0.0) for quarter in train_quarters],
                [float(hazard_residuals[quarter]) for quarter in train_quarters],
                eps,
            )
            pre_weight = float(candidate["relevance"]) * abs(float(corr))
            weighted_factors.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(candidate.get("factor_name") or factor_id),
                    "transition_relevance": float(candidate["relevance"]),
                    "empirical_correlation": float(corr),
                    "pre_weight": float(pre_weight),
                    "train_mean": float(mean_value),
                    "train_std": float(std_value),
                    "standardized_map": standardized_map,
                }
            )
        weight_total = float(sum(row["pre_weight"] for row in weighted_factors))
        for row in weighted_factors:
            normalized_weight = float(row["pre_weight"] / weight_total) if weight_total > eps else 0.0
            row["normalized_weight"] = normalized_weight
            helper_rows.append(
                {
                    "transition": transition,
                    "factor_id": row["factor_id"],
                    "factor_name": row["factor_name"],
                    "transition_relevance": round(float(row["transition_relevance"]), 6),
                    "empirical_correlation": round(float(row["empirical_correlation"]), 6),
                    "pre_weight": round(float(row["pre_weight"]), 6),
                    "normalized_weight": round(normalized_weight, 6),
                    "train_mean": round(float(row["train_mean"]), 6),
                    "train_std": round(float(row["train_std"]), 6),
                }
            )
        all_quarters = sorted(
            {
                quarter
                for row in weighted_factors
                for quarter in row["standardized_map"].keys()
            },
            key=quarter_sort_key,
        )
        for quarter in all_quarters:
            quarter_signals[transition][quarter] = float(
                sum(float(row["normalized_weight"]) * float(row["standardized_map"].get(quarter) or 0.0) for row in weighted_factors)
            )
        signal_values = np.asarray([float(quarter_signals[transition].get(quarter) or 0.0) for quarter in train_quarters], dtype=np.float64)
        residual_values = np.asarray([float(hazard_residuals[quarter]) for quarter in train_quarters], dtype=np.float64)
        signal_variance = float(np.var(signal_values)) if signal_values.size else 0.0
        residual_scale = float(np.cov(signal_values, residual_values, bias=True)[0, 1] / signal_variance) if signal_variance > eps else 0.0
        residual_std = float(np.std(residual_values)) if residual_values.size else 0.0
        model_rows.append(
            {
                "transition": transition,
                "hazard_mean": round(hazard_mean, 6),
                "residual_scale": round(residual_scale, 6),
                "residual_std": round(residual_std, 6),
                "eligible_factor_count": len(candidate_rows),
                "weighted_factor_count": sum(1 for row in weighted_factors if float(row["normalized_weight"]) > 0.0),
                "train_row_count": len(train_quarters),
            }
        )
    return {
        "helper_rows": helper_rows,
        "model_rows": model_rows,
    }, quarter_signals


def _kp_overlay_model_summary(
    *,
    ctx: TransitionResearchContext,
    train_transition_rows: list[dict[str, Any]],
    eps: float,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], float, list[str]]:
    inputs = load_transition_research_inputs(ctx)
    factor_surface = _quarter_factor_surface(ctx)
    transition_rows = _factor_transition_rows(inputs)
    collapsed_distribution, low_confidence_reasons = _collapse_kp_distribution(inputs)
    train_quarters = [str(row["quarter"]) for row in train_transition_rows]
    subgroup_factor_lookup = {
        factor_id: "subgroup_allocation_priors" in list((inputs.retained_factor_lookup.get(factor_id) or {}).get("transition_hooks", []))
        for factor_id in inputs.retained_factor_lookup
    }
    positive_group_count = sum(1 for kp_name in KP_COLLAPSED_NAMES if float(collapsed_distribution.get(kp_name, 0.0)) > eps)
    kp_support_strength = float(positive_group_count) / float(max(len(KP_COLLAPSED_NAMES), 1))
    helper_rows: list[dict[str, Any]] = []
    quarter_signals: dict[str, dict[str, float]] = {transition: {} for transition in TRANSITION_NAMES}
    model_rows: list[dict[str, Any]] = []

    for transition in TRANSITION_NAMES:
        if transition not in DOWNSTREAM_TRANSITIONS:
            model_rows.append(
                {
                    "transition": transition,
                    "hazard_mean": 0.0,
                    "residual_scale": 0.0,
                    "residual_std": 0.0,
                    "eligible_factor_count": 0,
                    "weighted_factor_count": 0,
                    "train_row_count": len(train_quarters),
                    "kp_support_strength": round(kp_support_strength, 6),
                    "positive_group_count": positive_group_count,
                    "low_confidence_reasons": list(low_confidence_reasons),
                }
            )
            continue

        candidate_rows = [
            dict(row)
            for row in transition_rows
            if str(row.get("transition") or "") == transition
            and float(row.get("relevance") or 0.0) > 0.0
            and bool(subgroup_factor_lookup.get(str(row.get("factor_id") or ""), False))
        ]
        hazard_targets = {
            str(row["quarter"]): float(row["hazards"][transition])
            for row in train_transition_rows
        }
        hazard_values = [float(hazard_targets[quarter]) for quarter in train_quarters]
        hazard_mean = float(np.mean(hazard_values)) if hazard_values else 0.0
        hazard_residuals = {quarter: float(hazard_targets[quarter] - hazard_mean) for quarter in train_quarters}
        weighted_factors: list[dict[str, Any]] = []
        for candidate in candidate_rows:
            factor_id = str(candidate["factor_id"])
            values_by_quarter = factor_surface.get(factor_id, {})
            standardized_map, mean_value, std_value = _zscore_map(values_by_quarter, train_quarters, eps)
            corr = _pearson_corr(
                [float(standardized_map.get(quarter) or 0.0) for quarter in train_quarters],
                [float(hazard_residuals[quarter]) for quarter in train_quarters],
                eps,
            )
            pre_weight = float(candidate["relevance"]) * kp_support_strength * abs(float(corr))
            weighted_factors.append(
                {
                    "factor_id": factor_id,
                    "factor_name": str(candidate.get("factor_name") or factor_id),
                    "transition_relevance": float(candidate["relevance"]),
                    "empirical_correlation": float(corr),
                    "pre_weight": float(pre_weight),
                    "train_mean": float(mean_value),
                    "train_std": float(std_value),
                    "standardized_map": standardized_map,
                }
            )
        weight_total = float(sum(row["pre_weight"] for row in weighted_factors))
        for row in weighted_factors:
            normalized_weight = float(row["pre_weight"] / weight_total) if weight_total > eps else 0.0
            row["normalized_weight"] = normalized_weight
            helper_rows.append(
                {
                    "transition": transition,
                    "factor_id": row["factor_id"],
                    "factor_name": row["factor_name"],
                    "transition_relevance": round(float(row["transition_relevance"]), 6),
                    "empirical_correlation": round(float(row["empirical_correlation"]), 6),
                    "pre_weight": round(float(row["pre_weight"]), 6),
                    "normalized_weight": round(normalized_weight, 6),
                    "train_mean": round(float(row["train_mean"]), 6),
                    "train_std": round(float(row["train_std"]), 6),
                    "kp_support_strength": round(kp_support_strength, 6),
                }
            )
        all_quarters = sorted(
            {
                quarter
                for row in weighted_factors
                for quarter in row["standardized_map"].keys()
            },
            key=quarter_sort_key,
        )
        for quarter in all_quarters:
            quarter_signals[transition][quarter] = float(
                sum(float(row["normalized_weight"]) * float(row["standardized_map"].get(quarter) or 0.0) for row in weighted_factors)
            )
        signal_values = np.asarray([float(quarter_signals[transition].get(quarter) or 0.0) for quarter in train_quarters], dtype=np.float64)
        residual_values = np.asarray([float(hazard_residuals[quarter]) for quarter in train_quarters], dtype=np.float64)
        signal_variance = float(np.var(signal_values)) if signal_values.size else 0.0
        residual_scale = float(np.cov(signal_values, residual_values, bias=True)[0, 1] / signal_variance) if signal_variance > eps else 0.0
        residual_std = (float(np.std(residual_values)) if residual_values.size else 0.0) * kp_support_strength
        model_rows.append(
            {
                "transition": transition,
                "hazard_mean": round(hazard_mean, 6),
                "residual_scale": round(residual_scale, 6),
                "residual_std": round(residual_std, 6),
                "eligible_factor_count": len(candidate_rows),
                "weighted_factor_count": sum(1 for row in weighted_factors if float(row["normalized_weight"]) > 0.0),
                "train_row_count": len(train_quarters),
                "kp_support_strength": round(kp_support_strength, 6),
                "positive_group_count": positive_group_count,
                "low_confidence_reasons": list(low_confidence_reasons),
            }
        )
    return {
        "helper_rows": helper_rows,
        "model_rows": model_rows,
    }, quarter_signals, kp_support_strength, low_confidence_reasons


def _simulate_holdout_with_residual_helpers(
    *,
    train_state_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    baseline_holdout_hazards: dict[str, dict[str, float]],
    helper_models: dict[str, dict[str, float]],
    quarter_signals: dict[str, dict[str, float]],
    testing_share_mean: float,
    eps: float,
    locked_transitions: set[str] | None = None,
    current_state_override: dict[str, float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_state = (
        {state_name: float(current_state_override[state_name]) for state_name in STATE_NAMES}
        if current_state_override is not None
        else {state_name: float(train_state_rows[-1]["state_values"][state_name]) for state_name in STATE_NAMES}
    )
    forecast_rows: list[dict[str, Any]] = []
    hazard_rows: list[dict[str, Any]] = []
    locked_transition_names = set(locked_transitions or set())

    for target_row in holdout_rows:
        quarter = str(target_row["quarter"])
        predicted_hazards: dict[str, float] = {}
        for transition in TRANSITION_NAMES:
            base_hazard = float(baseline_holdout_hazards.get(quarter, {}).get(transition) or 0.0)
            if transition in locked_transition_names:
                predicted_hazards[transition] = float(np.clip(base_hazard, 0.0, 1.0))
                continue
            helper_signal = float(quarter_signals.get(transition, {}).get(quarter) or 0.0)
            residual_scale = float(helper_models[transition]["residual_scale"])
            residual_std = float(helper_models[transition]["residual_std"])
            correction = float(np.clip(residual_scale * helper_signal, -residual_std, residual_std))
            predicted_hazards[transition] = float(np.clip(base_hazard + correction, 0.0, 1.0))

        u_to_d = min(float(current_state["U"]), predicted_hazards["U_to_D"] * float(current_state["U"]))
        d_to_a = min(float(current_state["D"]), predicted_hazards["D_to_A"] * float(current_state["D"]))
        l_to_a = min(float(current_state["L"]), predicted_hazards["L_to_A"] * float(current_state["L"]))
        raw_a_to_v = predicted_hazards["A_to_V"] * float(current_state["A"])
        raw_a_to_l = predicted_hazards["A_to_L"] * float(current_state["A"])
        raw_outflow = raw_a_to_v + raw_a_to_l
        a_scale = min(1.0, float(current_state["A"]) / max(raw_outflow, eps)) if raw_outflow > 0.0 else 1.0
        a_to_v = raw_a_to_v * a_scale
        a_to_l = raw_a_to_l * a_scale

        next_state = {
            "U": max(float(current_state["U"]) - u_to_d, 0.0),
            "D": max(float(current_state["D"]) + u_to_d - d_to_a, 0.0),
            "A": max(float(current_state["A"]) + d_to_a + l_to_a - a_to_v - a_to_l, 0.0),
            "V": max(float(current_state["V"]) + a_to_v, 0.0),
            "L": max(float(current_state["L"]) + a_to_l - l_to_a, 0.0),
        }
        forecast_rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": float(next_state["D"] + next_state["A"] + next_state["V"] + next_state["L"]),
                "alive_on_art": float(next_state["A"] + next_state["V"]),
                "new_diagnosed_cases_period": float(u_to_d),
                "tested_for_viral_load": float(testing_share_mean * (next_state["A"] + next_state["V"])),
                "virally_suppressed": float(next_state["V"]),
                "state_values": {state_name: float(next_state[state_name]) for state_name in STATE_NAMES},
            }
        )
        hazard_rows.append(
            {
                "quarter": quarter,
                "hazards": {transition: round(float(predicted_hazards[transition]), 6) for transition in TRANSITION_NAMES},
                "base_hazards": {transition: round(float(baseline_holdout_hazards.get(quarter, {}).get(transition) or 0.0), 6) for transition in TRANSITION_NAMES},
                "flows": {
                    "U_to_D": round(float(u_to_d), 6),
                    "D_to_A": round(float(d_to_a), 6),
                    "A_to_V": round(float(a_to_v), 6),
                    "A_to_L": round(float(a_to_l), 6),
                    "L_to_A": round(float(l_to_a), 6),
                },
                "helper_signals": {transition: round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6) for transition in TRANSITION_NAMES},
                "locked_transitions": sorted(locked_transition_names),
            }
        )
        current_state = next_state
    return forecast_rows, hazard_rows


def _holdout_state_rows_from_reference(reference: dict[str, Any], *, reference_label: str) -> list[dict[str, Any]]:
    evaluation = dict(reference.get("evaluation") or {})
    rows = [dict(row) for row in list(evaluation.get("holdout_state_rows") or [])]
    if not rows:
        raise ValueError(f"{reference_label} requires holdout_state_rows for anchored downstream evaluation")
    return rows


def _mech_01a_holdout_state_rows(mech_01a_reference: dict[str, Any]) -> list[dict[str, Any]]:
    return _holdout_state_rows_from_reference(mech_01a_reference, reference_label="MECH-01A reference")


def _holdout_predictions_from_reference(reference: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list((reference.get("evaluation") or {}).get("holdout_rows") or [])
    ]


def _diagnosis_flow_mae(
    *,
    forecast_rows: list[dict[str, Any]],
    diagnosis_flow_targets: dict[str, dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for forecast in forecast_rows:
        quarter = str(forecast["quarter"])
        target = dict(diagnosis_flow_targets.get(quarter) or {})
        estimated_plhiv = float(target.get("estimated_plhiv") or 0.0)
        diagnosed_share = target.get("diagnosed_share")
        if estimated_plhiv <= 0.0 or diagnosed_share is None:
            continue
        predicted_share = float(forecast["new_diagnosed_cases_period"]) / estimated_plhiv
        absolute_error = abs(predicted_share - float(diagnosed_share))
        errors.append(absolute_error)
        rows.append(
            {
                "quarter": quarter,
                "predicted_diagnosed_share": round(float(predicted_share), 6),
                "target_diagnosed_share": round(float(diagnosed_share), 6),
                "absolute_error": round(float(absolute_error), 6),
                "estimated_plhiv_reference": round(float(estimated_plhiv), 6),
            }
        )
    return (float(np.mean(errors)) if errors else float("inf")), rows


def _load_transition_experiment_reference(experiment_id: str) -> dict[str, Any]:
    reference_run_id, experiment_dir = _discover_latest_transition_experiment(experiment_id)
    evaluation = read_json(experiment_dir / "evaluation.json", default={})
    baseline_comparison = read_json(experiment_dir / "baseline_comparison.json", default={})
    fit_artifact = read_json(experiment_dir / "fit_artifact.json", default={})
    transition_hazard_summary = read_json(experiment_dir / "transition_hazard_summary.json", default={})
    return {
        "reference_run_id": reference_run_id,
        "experiment_dir": experiment_dir,
        "evaluation": dict(evaluation or {}),
        "baseline_comparison": dict(baseline_comparison or {}),
        "fit_artifact": dict(fit_artifact or {}),
        "transition_hazard_summary": dict(transition_hazard_summary or {}),
    }


def _load_mech_01a_reference() -> dict[str, Any]:
    return _load_transition_experiment_reference(MECH01A_EXPERIMENT_ID)


def _load_mech_01e_reference() -> dict[str, Any]:
    return _load_transition_experiment_reference(MECH01E_EXPERIMENT_ID)


def run_mech_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    reference = _load_front_half_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(reference)
    observation_rows = _with_holdout_forecasts(list(payload["rows"]), _reference_forecast_rows(reference))
    train_rows = _train_rows(observation_rows, holdout_years)
    holdout_rows = _holdout_rows(observation_rows, holdout_years)
    if not train_rows or not holdout_rows:
        raise ValueError("MECH-01A requires non-empty train and holdout rows")

    coverage_mean = _coverage_mean(reference, dict(payload["estimated_plhiv_by_quarter"]), train_rows)
    suppression_share_mean = _share_mean(train_rows, "virally_suppressed", "alive_on_art")
    testing_share_mean = _share_mean(train_rows, "tested_for_viral_load", "alive_on_art")
    lost_gap_share = _lost_gap_share(train_rows)
    eps = float(np.finfo(np.float32).eps)

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    holdout_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in holdout_rows
    ]
    all_state_rows = train_state_rows + holdout_state_rows
    transition_rows, transition_summary = _transition_rows(all_state_rows, eps)

    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["MECH-01A observed-state reconstruction from archive-supported national rows"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["MECH-01A holdout-state reconstruction from the winning front-half forecast"],
        save_pt=False,
    )

    baseline_comparison = dict(reference.get("baseline_comparison") or {})
    evaluation = dict(reference.get("evaluation") or {})
    evaluation["mode"] = "transition_research_udavl_reference_front_half"
    evaluation["state_names"] = list(STATE_NAMES)
    evaluation["front_half_reference_run_id"] = str(reference["reference_run_id"])
    evaluation["train_quarters"] = [str(row["quarter"]) for row in train_rows]
    evaluation["holdout_quarters"] = [str(row["quarter"]) for row in holdout_rows]
    evaluation["holdout_state_rows"] = _state_trajectory_rows(holdout_state_rows)
    evaluation["service_auxiliary_summary"] = {
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "holdout_rows_with_documented_suppression_target": sum(1 for row in holdout_rows if row.get("virally_suppressed") is not None),
    }

    fit_artifact = {
        "model_family": "transition_research_udavl_reference_front_half",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "front_half_reference_experiment_id": NR01_EXPERIMENT_ID,
        "front_half_reference_experiment_dir": str(reference["experiment_dir"]),
        "observation_source_run_id": ctx.source_run_id,
        "observation_row_count": len(observation_rows),
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "state_estimate_artifact": state_estimate_artifact,
        "forecast_state_artifact": forecast_state_artifact,
        "uses_front_half_reference_forecasts": True,
        "uses_mesoscopic_transition_helpers": False,
        "uses_kp_overlay": False,
    }

    transition_hazard_summary = {
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "rows": transition_rows,
        "summary": transition_summary,
    }
    decision_checks = [
        {
            "name": "front_half_matches_reference_mae",
            "passed": float(evaluation.get("model_mean_absolute_error") or 0.0)
            <= float((reference.get("evaluation") or {}).get("model_mean_absolute_error") or 0.0),
            "actual": float(evaluation.get("model_mean_absolute_error") or 0.0),
            "target": float((reference.get("evaluation") or {}).get("model_mean_absolute_error") or 0.0),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "MECH-01A reconstructs explicit U/D/A/V/L states while preserving the current winning front-half forecast.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(all_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "front_half_reference_run_id": str(reference["reference_run_id"]),
            "front_half_reference_experiment_id": NR01_EXPERIMENT_ID,
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "front_half_reference_run_id": str(reference["reference_run_id"]),
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "state_count": len(STATE_NAMES),
            "transition_count": len(TRANSITION_NAMES),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "NR-01 coverage summary and estimated_plhiv-by-quarter archive support",
                "estimation_method": "mean diagnosed coverage among archive-supported national train quarters",
                "uncertainty": "depends on estimated_plhiv auxiliary prior quality",
                "why_needed": "Reconstructs the latent undiagnosed stock without introducing a manual denominator.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "archive-supported national rows with virally_suppressed and alive_on_art",
                "estimation_method": "mean documented suppression share among ART in train rows",
                "uncertainty": "sensitive to documented suppression ascertainment",
                "why_needed": "Splits ART stock into engaged-unsuppressed and documented-suppressed states with a data-estimated share.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "archive-supported national rows with tested_for_viral_load and alive_on_art",
                "estimation_method": "mean viral-load-testing share among ART in train rows",
                "uncertainty": "sensitive to service documentation coverage",
                "why_needed": "Preserves a defensible service-process reference for the V state without inventing a testing floor.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "consecutive national train rows with diagnosed_plhiv and alive_on_art",
                "estimation_method": "mean negative ART-stock change divided by the prior diagnosed-not-on-ART gap",
                "uncertainty": "limited by sparse direct L observation",
                "why_needed": "Creates a data-derived latent split between diagnosed-not-on-ART and lost-to-follow-up rather than fixing a manual lost-share constant.",
            },
            {
                "name": "transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "exact_hazard_grid_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of the exact HIV transition channels requested in the plan",
                "uncertainty": "none",
                "why_needed": "Defines the mechanistic hazard grid without adding an implicit transition family.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_division_guard",
                why_needed="Prevents undefined hazards when a transition denominator state is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "decision": decision,
    }


def run_mech_01b(ctx: TransitionResearchContext) -> dict[str, Any]:
    reference = _load_front_half_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or not holdout_rows:
        raise ValueError("MECH-01B requires non-empty train and holdout rows")

    coverage_mean = _coverage_mean(reference, dict(payload["estimated_plhiv_by_quarter"]), train_rows)
    suppression_share_mean = _share_mean(train_rows, "virally_suppressed", "alive_on_art")
    testing_share_mean = _share_mean(train_rows, "tested_for_viral_load", "alive_on_art")
    lost_gap_share = _lost_gap_share(train_rows)
    eps = float(np.finfo(np.float32).eps)

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    helper_summary, quarter_signals = _helper_model_summary(
        ctx=ctx,
        train_transition_rows=train_transition_rows,
        eps=eps,
    )
    helper_models = {str(row["transition"]): dict(row) for row in helper_summary["model_rows"]}

    forecast_rows, holdout_transition_rows = _simulate_holdout_with_helpers(
        train_state_rows=train_state_rows,
        holdout_rows=holdout_rows,
        train_transition_rows=train_transition_rows,
        helper_models=helper_models,
        quarter_signals=quarter_signals,
        testing_share_mean=testing_share_mean,
        eps=eps,
    )
    holdout_state_rows = [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(row["state_values"]["U"] + row["state_values"]["D"] + row["state_values"]["A"] + row["state_values"]["V"] + row["state_values"]["L"]),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    reference_baseline = dict(reference.get("baseline_comparison") or {})
    reference_diagnosis_flow = dict(reference.get("diagnosis_flow_evaluation") or {})
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01a_reference_mean_absolute_error": float(reference_baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(reference_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(reference_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01a_reference": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(reference_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(reference_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01a_reference_diagnosis_flow_mean_absolute_error": float(reference_diagnosis_flow.get("mean_absolute_error") or 0.0),
    }
    evaluation = {
        "mode": "transition_research_udavl_mesoscopic_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    fit_artifact = {
        "model_family": "transition_research_udavl_mesoscopic_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "uses_front_half_reference_forecasts": False,
        "uses_mesoscopic_transition_helpers": True,
        "uses_kp_overlay": False,
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["MECH-01B observed-state reconstruction used for mesoscopic helper fitting"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["MECH-01B holdout-state forecast with transition-specific mesoscopic helpers"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    transition_hazard_summary = {
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "train_rows": train_transition_rows,
        "holdout_rows": holdout_transition_rows,
    }
    mesoscopic_transition_helper_summary = {
        "front_half_reference_run_id": str(reference["reference_run_id"]),
        "helper_weight_rows": helper_summary["helper_rows"],
        "transition_model_rows": helper_summary["model_rows"],
        "quarter_signal_rows": [
            {
                "quarter": quarter,
                "transition": transition,
                "helper_signal": round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6),
            }
            for transition in TRANSITION_NAMES
            for quarter in sorted(quarter_signals.get(transition, {}), key=quarter_sort_key)
        ],
    }
    decision_checks = [
        {
            "name": "improves_vs_mech_01a_reference",
            "passed": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
            "actual": model_mae,
            "target": float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        },
        {
            "name": "diagnosis_flow_does_not_regress_vs_mech_01a",
            "passed": float(diagnosis_flow_mae) <= float(reference_diagnosis_flow.get("mean_absolute_error") or float("inf")),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": float(reference_diagnosis_flow.get("mean_absolute_error") or float("inf")),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "MECH-01B keeps only transition-assigned mesoscopic helpers and tests whether they improve the U/D/A/V/L engine out of sample.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "mesoscopic_transition_helper_summary.json", mesoscopic_transition_helper_summary)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "front_half_reference_run_id": str(reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "transition_count": len(TRANSITION_NAMES),
            "eligible_helper_factor_count": sum(int(row["eligible_factor_count"]) for row in helper_summary["model_rows"]),
            "weighted_helper_factor_count": sum(int(row["weighted_factor_count"]) for row in helper_summary["model_rows"]),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "NR-01 coverage summary and estimated_plhiv-by-quarter archive support",
                "estimation_method": "mean diagnosed coverage among archive-supported national train quarters",
                "uncertainty": "depends on estimated_plhiv auxiliary prior quality",
                "why_needed": "Reconstructs the latent undiagnosed stock without introducing a manual denominator.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "archive-supported national rows with virally_suppressed and alive_on_art",
                "estimation_method": "mean documented suppression share among ART in train rows",
                "uncertainty": "sensitive to documented suppression ascertainment",
                "why_needed": "Splits ART stock into engaged-unsuppressed and documented-suppressed states with a data-estimated share.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "archive-supported national rows with tested_for_viral_load and alive_on_art",
                "estimation_method": "mean viral-load-testing share among ART in train rows",
                "uncertainty": "sensitive to service documentation coverage",
                "why_needed": "Produces a data-estimated tested-for-VL proxy when the helper model simulates holdout ART states.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "consecutive national train rows with diagnosed_plhiv and alive_on_art",
                "estimation_method": "mean negative ART-stock change divided by the prior diagnosed-not-on-ART gap",
                "uncertainty": "limited by sparse direct L observation",
                "why_needed": "Creates a data-derived latent split between diagnosed-not-on-ART and lost-to-follow-up rather than fixing a manual lost-share constant.",
            },
            {
                "name": "helper_transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "transition_helper_model_count",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of the exact HIV transition channels requested in the plan",
                "uncertainty": "none",
                "why_needed": "Creates one helper model per transition, with no hidden transition families.",
            },
            {
                "name": "helper_regression_design_width",
                "value": 3,
                "role": "transition_helper_regression_width",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01B transition helper regression design",
                "estimation_method": "count of fitted regression columns: intercept, lagged hazard, helper signal",
                "uncertainty": "none",
                "why_needed": "Defines the helper regression exactly, without hidden covariates or manual penalty constants.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_and_standardization_guard",
                why_needed="Prevents undefined standardization and hazard division when train variation or denominator state mass is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "mesoscopic_transition_helper_summary": mesoscopic_transition_helper_summary,
        "decision": decision,
    }


def run_mech_01c(ctx: TransitionResearchContext) -> dict[str, Any]:
    mech_01a_reference = _load_mech_01a_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(mech_01a_reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or not holdout_rows:
        raise ValueError("MECH-01C requires non-empty train and holdout rows")

    fit_reference = dict(mech_01a_reference.get("fit_artifact") or {})
    coverage_mean = float(fit_reference.get("coverage_mean") or 0.0)
    suppression_share_mean = float(fit_reference.get("suppression_share_train_mean") or 0.0)
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    lost_gap_share = float(fit_reference.get("lost_gap_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    residual_helper_summary, quarter_signals = _residual_helper_model_summary(
        ctx=ctx,
        train_transition_rows=train_transition_rows,
        eps=eps,
    )
    helper_models = {str(row["transition"]): dict(row) for row in residual_helper_summary["model_rows"]}
    baseline_holdout_hazards = _baseline_holdout_hazards(mech_01a_reference)

    forecast_rows, holdout_transition_rows = _simulate_holdout_with_residual_helpers(
        train_state_rows=train_state_rows,
        holdout_rows=holdout_rows,
        baseline_holdout_hazards=baseline_holdout_hazards,
        helper_models=helper_models,
        quarter_signals=quarter_signals,
        testing_share_mean=testing_share_mean,
        eps=eps,
    )
    holdout_state_rows = [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(
                row["state_values"]["U"]
                + row["state_values"]["D"]
                + row["state_values"]["A"]
                + row["state_values"]["V"]
                + row["state_values"]["L"]
            ),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01a_holdout_predictions = [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list((mech_01a_reference.get("evaluation") or {}).get("holdout_rows") or [])
    ]
    mech_01a_diagnosis_flow_mae, _ = _diagnosis_flow_mae(
        forecast_rows=mech_01a_holdout_predictions,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    reference_baseline = dict(mech_01a_reference.get("baseline_comparison") or {})
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01a_reference_mean_absolute_error": float(reference_baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(reference_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(reference_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01a_reference": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(reference_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(reference_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01a_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01a_diagnosis_flow_mae), 6),
    }
    evaluation = {
        "mode": "transition_research_udavl_residual_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "front_half_reference_run_id": str((mech_01a_reference.get("evaluation") or {}).get("front_half_reference_run_id") or ""),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    fit_artifact = {
        "model_family": "transition_research_udavl_residual_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "uses_front_half_reference_forecasts": False,
        "uses_mech_01a_transition_baseline": True,
        "uses_mesoscopic_transition_helpers": True,
        "uses_kp_overlay": False,
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["MECH-01C observed-state reconstruction used for residual helper fitting"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["MECH-01C holdout-state forecast with residual transition helper corrections"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    transition_hazard_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "train_rows": train_transition_rows,
        "baseline_holdout_hazards": baseline_holdout_hazards,
        "holdout_rows": holdout_transition_rows,
    }
    residual_transition_helper_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "helper_weight_rows": residual_helper_summary["helper_rows"],
        "transition_model_rows": residual_helper_summary["model_rows"],
        "quarter_signal_rows": [
            {
                "quarter": quarter,
                "transition": transition,
                "helper_signal": round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6),
            }
            for transition in TRANSITION_NAMES
            for quarter in sorted(quarter_signals.get(transition, {}), key=quarter_sort_key)
        ],
        "holdout_correction_rows": [
            {
                "quarter": str(row["quarter"]),
                "transition": transition,
                "base_hazard": round(float(row["base_hazards"][transition]), 6),
                "corrected_hazard": round(float(row["hazards"][transition]), 6),
                "hazard_correction": round(float(row["hazards"][transition]) - float(row["base_hazards"][transition]), 6),
            }
            for row in holdout_transition_rows
            for transition in TRANSITION_NAMES
        ],
    }
    decision_checks = [
        {
            "name": "improves_vs_mech_01a_reference",
            "passed": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
            "actual": model_mae,
            "target": float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        },
        {
            "name": "diagnosis_flow_does_not_regress_vs_mech_01a",
            "passed": float(diagnosis_flow_mae) <= float(mech_01a_diagnosis_flow_mae),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(mech_01a_diagnosis_flow_mae), 6),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "MECH-01C keeps the MECH-01A hazard path fixed and only allows train-estimated residual mesoscopic corrections on top of it.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "residual_transition_helper_summary.json", residual_transition_helper_summary)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "transition_count": len(TRANSITION_NAMES),
            "eligible_helper_factor_count": sum(int(row["eligible_factor_count"]) for row in residual_helper_summary["model_rows"]),
            "weighted_helper_factor_count": sum(int(row["weighted_factor_count"]) for row in residual_helper_summary["model_rows"]),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated diagnosed coverage mean so residual helper scoring is measured against the same state reconstruction",
                "uncertainty": "inherits MECH-01A coverage uncertainty",
                "why_needed": "Keeps the latent state reconstruction fixed while testing only residual mesoscopic corrections.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated documented suppression share among ART",
                "uncertainty": "inherits MECH-01A service ascertainment uncertainty",
                "why_needed": "Prevents the residual experiment from confounding transition corrections with a different A/V state split.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A mean viral-load-testing share among ART",
                "uncertainty": "inherits MECH-01A service documentation uncertainty",
                "why_needed": "Keeps the tested-for-VL auxiliary process aligned with the MECH-01A baseline while evaluating only hazard residuals.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "MECH-01A fit artifact inherited from consecutive national train rows",
                "estimation_method": "reuse the MECH-01A semi-Markov lost-gap share estimate",
                "uncertainty": "inherits sparse direct L-observation uncertainty",
                "why_needed": "Keeps the D/L latent split fixed so the residual branch isolates hazard corrections rather than state-definition drift.",
            },
            {
                "name": "residual_helper_transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "transition_residual_model_count",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of the exact HIV transition channels requested in the plan",
                "uncertainty": "none",
                "why_needed": "Creates one residual correction model per transition with no hidden transition family.",
            },
            {
                "name": "residual_cap_source",
                "value": 1,
                "role": "residual_cap_definition_flag",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01C residual helper design",
                "estimation_method": "each correction is capped by one train-estimated residual standard deviation for the same transition",
                "uncertainty": "cap magnitude varies by transition through the estimated residual_std term",
                "why_needed": "Prevents residual helpers from overpowering the MECH-01A baseline without introducing a manual transition-specific cap.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_and_standardization_guard",
                why_needed="Prevents undefined standardization and zero-denominator hazards when train variation or state mass is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "residual_transition_helper_summary": residual_transition_helper_summary,
        "decision": decision,
    }


def run_mech_01d(ctx: TransitionResearchContext) -> dict[str, Any]:
    mech_01a_reference = _load_mech_01a_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(mech_01a_reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or not holdout_rows:
        raise ValueError("MECH-01D requires non-empty train and holdout rows")

    fit_reference = dict(mech_01a_reference.get("fit_artifact") or {})
    coverage_mean = float(fit_reference.get("coverage_mean") or 0.0)
    suppression_share_mean = float(fit_reference.get("suppression_share_train_mean") or 0.0)
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    lost_gap_share = float(fit_reference.get("lost_gap_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)
    locked_transitions = {"U_to_D"}

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    residual_helper_summary, quarter_signals = _residual_helper_model_summary(
        ctx=ctx,
        train_transition_rows=train_transition_rows,
        eps=eps,
    )
    helper_models = {str(row["transition"]): dict(row) for row in residual_helper_summary["model_rows"]}
    baseline_holdout_hazards = _baseline_holdout_hazards(mech_01a_reference)

    forecast_rows, holdout_transition_rows = _simulate_holdout_with_residual_helpers(
        train_state_rows=train_state_rows,
        holdout_rows=holdout_rows,
        baseline_holdout_hazards=baseline_holdout_hazards,
        helper_models=helper_models,
        quarter_signals=quarter_signals,
        testing_share_mean=testing_share_mean,
        eps=eps,
        locked_transitions=locked_transitions,
    )
    holdout_state_rows = [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(
                row["state_values"]["U"]
                + row["state_values"]["D"]
                + row["state_values"]["A"]
                + row["state_values"]["V"]
                + row["state_values"]["L"]
            ),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01a_holdout_predictions = [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list((mech_01a_reference.get("evaluation") or {}).get("holdout_rows") or [])
    ]
    mech_01a_diagnosis_flow_mae, _ = _diagnosis_flow_mae(
        forecast_rows=mech_01a_holdout_predictions,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    reference_baseline = dict(mech_01a_reference.get("baseline_comparison") or {})
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01a_reference_mean_absolute_error": float(reference_baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(reference_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(reference_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01a_reference": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(reference_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(reference_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01a_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01a_diagnosis_flow_mae), 6),
    }
    evaluation = {
        "mode": "transition_research_udavl_residual_helpers_diagnosis_locked",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "front_half_reference_run_id": str((mech_01a_reference.get("evaluation") or {}).get("front_half_reference_run_id") or ""),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    fit_artifact = {
        "model_family": "transition_research_udavl_residual_helpers_diagnosis_locked",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "uses_front_half_reference_forecasts": False,
        "uses_mech_01a_transition_baseline": True,
        "uses_mesoscopic_transition_helpers": True,
        "uses_kp_overlay": False,
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["MECH-01D observed-state reconstruction used for diagnosis-locked residual helper fitting"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["MECH-01D holdout-state forecast with diagnosis-locked residual helper corrections"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    transition_hazard_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "train_rows": train_transition_rows,
        "baseline_holdout_hazards": baseline_holdout_hazards,
        "holdout_rows": holdout_transition_rows,
    }
    residual_transition_helper_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "helper_weight_rows": residual_helper_summary["helper_rows"],
        "transition_model_rows": residual_helper_summary["model_rows"],
        "quarter_signal_rows": [
            {
                "quarter": quarter,
                "transition": transition,
                "helper_signal": round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6),
            }
            for transition in TRANSITION_NAMES
            for quarter in sorted(quarter_signals.get(transition, {}), key=quarter_sort_key)
        ],
        "holdout_correction_rows": [
            {
                "quarter": str(row["quarter"]),
                "transition": transition,
                "base_hazard": round(float(row["base_hazards"][transition]), 6),
                "corrected_hazard": round(float(row["hazards"][transition]), 6),
                "hazard_correction": round(float(row["hazards"][transition]) - float(row["base_hazards"][transition]), 6),
            }
            for row in holdout_transition_rows
            for transition in TRANSITION_NAMES
        ],
    }
    decision_checks = [
        {
            "name": "improves_vs_mech_01a_reference",
            "passed": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
            "actual": model_mae,
            "target": float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        },
        {
            "name": "diagnosis_flow_matches_mech_01a_when_locked",
            "passed": float(diagnosis_flow_mae) <= float(mech_01a_diagnosis_flow_mae),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(mech_01a_diagnosis_flow_mae), 6),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "MECH-01D fixes U_to_D at the MECH-01A baseline and tests whether mesoscopic residual helpers improve only the downstream transitions.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "residual_transition_helper_summary.json", residual_transition_helper_summary)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "transition_count": len(TRANSITION_NAMES),
            "locked_transition_count": len(locked_transitions),
            "eligible_helper_factor_count": sum(int(row["eligible_factor_count"]) for row in residual_helper_summary["model_rows"]),
            "weighted_helper_factor_count": sum(int(row["weighted_factor_count"]) for row in residual_helper_summary["model_rows"]),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated diagnosed coverage mean so the diagnosis-locked residual test only changes downstream hazards",
                "uncertainty": "inherits MECH-01A coverage uncertainty",
                "why_needed": "Keeps the latent state reconstruction fixed while testing only downstream residual mesoscopic corrections.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated documented suppression share among ART",
                "uncertainty": "inherits MECH-01A service ascertainment uncertainty",
                "why_needed": "Prevents the diagnosis-locked residual experiment from confounding downstream hazard corrections with a different A/V state split.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A mean viral-load-testing share among ART",
                "uncertainty": "inherits MECH-01A service documentation uncertainty",
                "why_needed": "Keeps the tested-for-VL auxiliary process aligned with the MECH-01A baseline while evaluating only downstream hazard residuals.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "MECH-01A fit artifact inherited from consecutive national train rows",
                "estimation_method": "reuse the MECH-01A semi-Markov lost-gap share estimate",
                "uncertainty": "inherits sparse direct L-observation uncertainty",
                "why_needed": "Keeps the D/L latent split fixed so the diagnosis-locked branch isolates downstream hazard corrections rather than state-definition drift.",
            },
            {
                "name": "residual_helper_transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "transition_residual_model_count",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of the exact HIV transition channels requested in the plan",
                "uncertainty": "none",
                "why_needed": "Creates one residual correction model per transition with no hidden transition family.",
            },
            {
                "name": "locked_transition_count",
                "value": len(locked_transitions),
                "role": "diagnosis_lock_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01D experiment design",
                "estimation_method": "count of transitions constrained to the MECH-01A baseline hazard path",
                "uncertainty": "none",
                "why_needed": "Records that only the diagnosis transition is fixed while downstream transitions remain eligible for residual correction.",
            },
            {
                "name": "residual_cap_source",
                "value": 1,
                "role": "residual_cap_definition_flag",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01D residual helper design",
                "estimation_method": "each correction is capped by one train-estimated residual standard deviation for the same transition",
                "uncertainty": "cap magnitude varies by transition through the estimated residual_std term",
                "why_needed": "Prevents downstream residual helpers from overpowering the MECH-01A baseline without introducing a manual transition-specific cap.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_and_standardization_guard",
                why_needed="Prevents undefined standardization and zero-denominator hazards when train variation or state mass is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "residual_transition_helper_summary": residual_transition_helper_summary,
        "decision": decision,
    }


def run_mech_01e(ctx: TransitionResearchContext) -> dict[str, Any]:
    mech_01a_reference = _load_mech_01a_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(mech_01a_reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or len(holdout_rows) < 2:
        raise ValueError("MECH-01E requires non-empty train rows and at least two holdout rows")

    fit_reference = dict(mech_01a_reference.get("fit_artifact") or {})
    coverage_mean = float(fit_reference.get("coverage_mean") or 0.0)
    suppression_share_mean = float(fit_reference.get("suppression_share_train_mean") or 0.0)
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    lost_gap_share = float(fit_reference.get("lost_gap_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)
    locked_transitions = {"U_to_D"}

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    residual_helper_summary, quarter_signals = _residual_helper_model_summary(
        ctx=ctx,
        train_transition_rows=train_transition_rows,
        eps=eps,
    )
    helper_models = {str(row["transition"]): dict(row) for row in residual_helper_summary["model_rows"]}
    baseline_holdout_hazards = _baseline_holdout_hazards(mech_01a_reference)
    mech_01a_holdout_states = _mech_01a_holdout_state_rows(mech_01a_reference)
    mech_01a_holdout_predictions = [
        dict((row.get("prediction") or {}), quarter=str(row.get("quarter") or ""))
        for row in list((mech_01a_reference.get("evaluation") or {}).get("holdout_rows") or [])
    ]
    if len(mech_01a_holdout_states) != len(mech_01a_holdout_predictions):
        raise ValueError("MECH-01E requires aligned MECH-01A holdout states and prediction rows")

    anchored_state_row = dict(mech_01a_holdout_states[0])
    anchored_prediction_row = dict(mech_01a_holdout_predictions[0])
    anchored_forecast_row = {
        "quarter": str(anchored_state_row["quarter"]),
        "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
        "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
        "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
        "tested_for_viral_load": float(testing_share_mean * (float(anchored_state_row["state_values"]["A"]) + float(anchored_state_row["state_values"]["V"]))),
        "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
        "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    }

    simulated_forecast_rows, simulated_holdout_transition_rows = _simulate_holdout_with_residual_helpers(
        train_state_rows=train_state_rows,
        holdout_rows=holdout_rows[1:],
        baseline_holdout_hazards=baseline_holdout_hazards,
        helper_models=helper_models,
        quarter_signals=quarter_signals,
        testing_share_mean=testing_share_mean,
        eps=eps,
        locked_transitions=locked_transitions,
        current_state_override={state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    )
    forecast_rows = [anchored_forecast_row] + simulated_forecast_rows
    holdout_state_rows = [
        {
            "quarter": str(anchored_state_row["quarter"]),
            "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
            "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
            "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(anchored_forecast_row["tested_for_viral_load"]),
            "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
            "estimated_plhiv": float(sum(float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
    ] + [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(
                row["state_values"]["U"]
                + row["state_values"]["D"]
                + row["state_values"]["A"]
                + row["state_values"]["V"]
                + row["state_values"]["L"]
            ),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in simulated_forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01a_diagnosis_flow_mae, _ = _diagnosis_flow_mae(
        forecast_rows=mech_01a_holdout_predictions,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    reference_baseline = dict(mech_01a_reference.get("baseline_comparison") or {})
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01a_reference_mean_absolute_error": float(reference_baseline.get("model_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(reference_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(reference_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01a_reference": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(reference_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(reference_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01a_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01a_diagnosis_flow_mae), 6),
    }
    evaluation = {
        "mode": "transition_research_udavl_anchored_downstream_residual_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "front_half_reference_run_id": str((mech_01a_reference.get("evaluation") or {}).get("front_half_reference_run_id") or ""),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    fit_artifact = {
        "model_family": "transition_research_udavl_anchored_downstream_residual_helpers",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "uses_front_half_reference_forecasts": False,
        "uses_mech_01a_transition_baseline": True,
        "uses_mesoscopic_transition_helpers": True,
        "uses_kp_overlay": False,
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["MECH-01E observed-state reconstruction used for anchored downstream residual helper fitting"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["MECH-01E holdout-state forecast anchored to the first MECH-01A holdout state with downstream residual helper corrections"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    transition_hazard_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "train_rows": train_transition_rows,
        "baseline_holdout_hazards": baseline_holdout_hazards,
        "holdout_rows": simulated_holdout_transition_rows,
    }
    residual_transition_helper_summary = {
        "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "helper_weight_rows": residual_helper_summary["helper_rows"],
        "transition_model_rows": residual_helper_summary["model_rows"],
        "quarter_signal_rows": [
            {
                "quarter": quarter,
                "transition": transition,
                "helper_signal": round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6),
            }
            for transition in TRANSITION_NAMES
            for quarter in sorted(quarter_signals.get(transition, {}), key=quarter_sort_key)
        ],
        "holdout_correction_rows": [
            {
                "quarter": str(row["quarter"]),
                "transition": transition,
                "base_hazard": round(float(row["base_hazards"][transition]), 6),
                "corrected_hazard": round(float(row["hazards"][transition]), 6),
                "hazard_correction": round(float(row["hazards"][transition]) - float(row["base_hazards"][transition]), 6),
            }
            for row in simulated_holdout_transition_rows
            for transition in TRANSITION_NAMES
        ],
    }
    decision_checks = [
        {
            "name": "improves_vs_mech_01a_reference",
            "passed": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
            "actual": model_mae,
            "target": float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        },
        {
            "name": "diagnosis_flow_matches_mech_01a_when_locked_and_anchored",
            "passed": float(diagnosis_flow_mae) <= float(mech_01a_diagnosis_flow_mae),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(mech_01a_diagnosis_flow_mae), 6),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "MECH-01E anchors the first holdout state to the MECH-01A baseline, locks diagnosis, and tests only downstream residual mesoscopic corrections.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "residual_transition_helper_summary.json", residual_transition_helper_summary)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "mech_01a_reference_run_id": str(mech_01a_reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "transition_count": len(TRANSITION_NAMES),
            "locked_transition_count": len(locked_transitions),
            "anchored_holdout_quarter_count": 1,
            "eligible_helper_factor_count": sum(int(row["eligible_factor_count"]) for row in residual_helper_summary["model_rows"]),
            "weighted_helper_factor_count": sum(int(row["weighted_factor_count"]) for row in residual_helper_summary["model_rows"]),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated diagnosed coverage mean so the anchored downstream residual test only changes downstream hazards",
                "uncertainty": "inherits MECH-01A coverage uncertainty",
                "why_needed": "Keeps the latent state reconstruction fixed while testing only anchored downstream residual mesoscopic corrections.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A data-estimated documented suppression share among ART",
                "uncertainty": "inherits MECH-01A service ascertainment uncertainty",
                "why_needed": "Prevents the anchored downstream experiment from confounding downstream hazard corrections with a different A/V state split.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "MECH-01A fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the MECH-01A mean viral-load-testing share among ART",
                "uncertainty": "inherits MECH-01A service documentation uncertainty",
                "why_needed": "Keeps the tested-for-VL auxiliary process aligned with the MECH-01A baseline while evaluating only downstream hazard residuals.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "MECH-01A fit artifact inherited from consecutive national train rows",
                "estimation_method": "reuse the MECH-01A semi-Markov lost-gap share estimate",
                "uncertainty": "inherits sparse direct L-observation uncertainty",
                "why_needed": "Keeps the D/L latent split fixed so the anchored branch isolates downstream hazard corrections rather than state-definition drift.",
            },
            {
                "name": "residual_helper_transition_count",
                "value": len(TRANSITION_NAMES),
                "role": "transition_residual_model_count",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of the exact HIV transition channels requested in the plan",
                "uncertainty": "none",
                "why_needed": "Creates one residual correction model per transition with no hidden transition family.",
            },
            {
                "name": "locked_transition_count",
                "value": len(locked_transitions),
                "role": "diagnosis_lock_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01E experiment design",
                "estimation_method": "count of transitions constrained to the MECH-01A baseline hazard path",
                "uncertainty": "none",
                "why_needed": "Records that only the diagnosis transition is fixed while downstream transitions remain eligible for residual correction.",
            },
            {
                "name": "anchored_holdout_quarter_count",
                "value": 1,
                "role": "holdout_anchor_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01E experiment design",
                "estimation_method": "count of holdout quarters copied exactly from the MECH-01A baseline before downstream simulation begins",
                "uncertainty": "none",
                "why_needed": "Makes the downstream helper test commensurate with the MECH-01A baseline by fixing the first holdout initial condition.",
            },
            {
                "name": "residual_cap_source",
                "value": 1,
                "role": "residual_cap_definition_flag",
                "source_type": "physical_constraint",
                "estimation_data": "MECH-01E residual helper design",
                "estimation_method": "each correction is capped by one train-estimated residual standard deviation for the same transition",
                "uncertainty": "cap magnitude varies by transition through the estimated residual_std term",
                "why_needed": "Prevents downstream residual helpers from overpowering the MECH-01A baseline without introducing a manual transition-specific cap.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_and_standardization_guard",
                why_needed="Prevents undefined standardization and zero-denominator hazards when train variation or state mass is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "residual_transition_helper_summary": residual_transition_helper_summary,
        "decision": decision,
    }


def run_kp_01a(ctx: TransitionResearchContext) -> dict[str, Any]:
    mech_01e_reference = _load_mech_01e_reference()
    payload = _build_observation_payload(ctx)
    holdout_years = _holdout_years_from_reference(mech_01e_reference)
    actual_rows = [dict(row) for row in list(payload["rows"])]
    train_rows = _train_rows(actual_rows, holdout_years)
    holdout_rows = _holdout_rows(actual_rows, holdout_years)
    if not train_rows or len(holdout_rows) < 2:
        raise ValueError("KP-01A requires non-empty train rows and at least two holdout rows")

    fit_reference = dict(mech_01e_reference.get("fit_artifact") or {})
    coverage_mean = float(fit_reference.get("coverage_mean") or 0.0)
    suppression_share_mean = float(fit_reference.get("suppression_share_train_mean") or 0.0)
    testing_share_mean = float(fit_reference.get("testing_share_train_mean") or 0.0)
    lost_gap_share = float(fit_reference.get("lost_gap_share_train_mean") or 0.0)
    eps = float(np.finfo(np.float32).eps)
    locked_transitions = {"U_to_D"}

    train_state_rows = [
        _state_row(
            row=row,
            estimated_plhiv_by_quarter=dict(payload["estimated_plhiv_by_quarter"]),
            coverage_mean=coverage_mean,
            suppression_share_mean=suppression_share_mean,
            lost_gap_share=lost_gap_share,
        )
        for row in train_rows
    ]
    train_transition_rows = _train_transition_rows(train_state_rows, holdout_years, eps)
    kp_overlay_summary, quarter_signals, kp_support_strength, low_confidence_reasons = _kp_overlay_model_summary(
        ctx=ctx,
        train_transition_rows=train_transition_rows,
        eps=eps,
    )
    overlay_inputs = load_transition_research_inputs(ctx)
    collapsed_distribution, _ = _collapse_kp_distribution(overlay_inputs)
    kp_positive_group_count = sum(1 for kp_name in KP_COLLAPSED_NAMES if float(collapsed_distribution.get(kp_name, 0.0)) > eps)
    helper_models = {str(row["transition"]): dict(row) for row in kp_overlay_summary["model_rows"]}
    baseline_holdout_hazards = _reference_holdout_hazards(mech_01e_reference)
    mech_01e_holdout_states = _holdout_state_rows_from_reference(mech_01e_reference, reference_label="MECH-01E reference")
    mech_01e_holdout_predictions = _holdout_predictions_from_reference(mech_01e_reference)
    if len(mech_01e_holdout_states) != len(mech_01e_holdout_predictions):
        raise ValueError("KP-01A requires aligned MECH-01E holdout states and prediction rows")

    anchored_state_row = dict(mech_01e_holdout_states[0])
    anchored_prediction_row = dict(mech_01e_holdout_predictions[0])
    anchored_forecast_row = {
        "quarter": str(anchored_state_row["quarter"]),
        "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
        "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
        "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
        "tested_for_viral_load": float(testing_share_mean * (float(anchored_state_row["state_values"]["A"]) + float(anchored_state_row["state_values"]["V"]))),
        "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
        "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    }

    simulated_forecast_rows, simulated_holdout_transition_rows = _simulate_holdout_with_residual_helpers(
        train_state_rows=train_state_rows,
        holdout_rows=holdout_rows[1:],
        baseline_holdout_hazards=baseline_holdout_hazards,
        helper_models=helper_models,
        quarter_signals=quarter_signals,
        testing_share_mean=testing_share_mean,
        eps=eps,
        locked_transitions=locked_transitions,
        current_state_override={state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
    )
    forecast_rows = [anchored_forecast_row] + simulated_forecast_rows
    holdout_state_rows = [
        {
            "quarter": str(anchored_state_row["quarter"]),
            "diagnosed_plhiv": float(anchored_prediction_row["diagnosed_plhiv"]),
            "alive_on_art": float(anchored_prediction_row["alive_on_art"]),
            "new_diagnosed_cases_period": float(anchored_prediction_row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(anchored_forecast_row["tested_for_viral_load"]),
            "virally_suppressed": float(anchored_state_row["state_values"]["V"]),
            "estimated_plhiv": float(sum(float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES)),
            "state_values": {state_name: float(anchored_state_row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
    ] + [
        {
            "quarter": str(row["quarter"]),
            "diagnosed_plhiv": float(row["diagnosed_plhiv"]),
            "alive_on_art": float(row["alive_on_art"]),
            "new_diagnosed_cases_period": float(row["new_diagnosed_cases_period"]),
            "tested_for_viral_load": float(row["tested_for_viral_load"]),
            "virally_suppressed": float(row["virally_suppressed"]),
            "estimated_plhiv": float(
                row["state_values"]["U"]
                + row["state_values"]["D"]
                + row["state_values"]["A"]
                + row["state_values"]["V"]
                + row["state_values"]["L"]
            ),
            "state_values": {state_name: float(row["state_values"][state_name]) for state_name in STATE_NAMES},
        }
        for row in simulated_forecast_rows
    ]

    metric_scales = _metric_scales(train_rows, eps)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales, eps), 6)
    model_smape = round(_smape(forecast_rows, holdout_rows, eps), 6)
    diagnosis_flow_mae, diagnosis_flow_rows = _diagnosis_flow_mae(
        forecast_rows=forecast_rows,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    mech_01e_diagnosis_flow_mae, _ = _diagnosis_flow_mae(
        forecast_rows=mech_01e_holdout_predictions,
        diagnosis_flow_targets=dict(payload["diagnosis_flow_targets_by_quarter"]),
    )
    reference_baseline = dict(mech_01e_reference.get("baseline_comparison") or {})
    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "mech_01e_reference_mean_absolute_error": float(reference_baseline.get("model_mean_absolute_error") or 0.0),
        "mech_01a_reference_mean_absolute_error": float(reference_baseline.get("mech_01a_reference_mean_absolute_error") or 0.0),
        "carry_forward_mean_absolute_error": float(reference_baseline.get("carry_forward_mean_absolute_error") or 0.0),
        "simple_compartmental_mean_absolute_error": float(reference_baseline.get("simple_compartmental_mean_absolute_error") or 0.0),
        "model_beats_mech_01e_reference": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        "model_beats_carry_forward": model_mae < float(reference_baseline.get("carry_forward_mean_absolute_error") or float("inf")),
        "model_beats_simple_compartmental": model_mae < float(reference_baseline.get("simple_compartmental_mean_absolute_error") or float("inf")),
        "diagnosis_flow_mean_absolute_error": round(float(diagnosis_flow_mae), 6),
        "mech_01e_reference_diagnosis_flow_mean_absolute_error": round(float(mech_01e_diagnosis_flow_mae), 6),
    }
    evaluation = {
        "mode": "transition_research_udavl_mech_01e_kp_overlay",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "downstream_overlay_transitions": list(DOWNSTREAM_TRANSITIONS),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {name: round(float(value), 6) for name, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": [
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
            for prediction, target in zip(forecast_rows, holdout_rows)
        ],
        "holdout_state_rows": _state_trajectory_rows(holdout_state_rows),
        "diagnosis_flow_evaluation": {
            "mean_absolute_error": round(float(diagnosis_flow_mae), 6),
            "rows": diagnosis_flow_rows,
        },
    }
    fit_artifact = {
        "model_family": "transition_research_udavl_mech_01e_kp_overlay",
        "state_names": list(STATE_NAMES),
        "transition_names": list(TRANSITION_NAMES),
        "locked_transitions": sorted(locked_transitions),
        "downstream_overlay_transitions": list(DOWNSTREAM_TRANSITIONS),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "observation_source_run_id": ctx.source_run_id,
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "coverage_mean": round(float(coverage_mean), 6),
        "suppression_share_train_mean": round(float(suppression_share_mean), 6),
        "testing_share_train_mean": round(float(testing_share_mean), 6),
        "lost_gap_share_train_mean": round(float(lost_gap_share), 6),
        "kp_support_strength": round(float(kp_support_strength), 6),
        "uses_front_half_reference_forecasts": False,
        "uses_mech_01a_transition_baseline": False,
        "uses_mech_01e_transition_baseline": True,
        "uses_mesoscopic_transition_helpers": True,
        "uses_kp_overlay": True,
    }
    state_estimate_artifact = save_tensor_artifact(
        array=_state_tensor(train_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["KP-01A observed-state reconstruction used for weak downstream KP overlay fitting"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=_state_tensor(holdout_state_rows),
        axis_names=["quarter", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["KP-01A holdout-state forecast anchored to MECH-01E with weak downstream KP modulation"],
        save_pt=False,
    )
    fit_artifact["state_estimate_artifact"] = state_estimate_artifact
    fit_artifact["forecast_state_artifact"] = forecast_state_artifact

    transition_hazard_summary = {
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "train_rows": train_transition_rows,
        "baseline_holdout_hazards": baseline_holdout_hazards,
        "holdout_rows": simulated_holdout_transition_rows,
    }
    kp_overlay_artifact = {
        "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
        "locked_transitions": sorted(locked_transitions),
        "downstream_overlay_transitions": list(DOWNSTREAM_TRANSITIONS),
        "anchored_holdout_quarter": str(anchored_state_row["quarter"]),
        "kp_support_strength": round(float(kp_support_strength), 6),
        "kp_positive_group_count": kp_positive_group_count,
        "low_confidence_reasons": list(low_confidence_reasons),
        "helper_weight_rows": kp_overlay_summary["helper_rows"],
        "transition_model_rows": kp_overlay_summary["model_rows"],
        "quarter_signal_rows": [
            {
                "quarter": quarter,
                "transition": transition,
                "kp_overlay_signal": round(float(quarter_signals.get(transition, {}).get(quarter) or 0.0), 6),
            }
            for transition in TRANSITION_NAMES
            for quarter in sorted(quarter_signals.get(transition, {}), key=quarter_sort_key)
        ],
        "holdout_correction_rows": [
            {
                "quarter": str(row["quarter"]),
                "transition": transition,
                "base_hazard": round(float(row["base_hazards"][transition]), 6),
                "corrected_hazard": round(float(row["hazards"][transition]), 6),
                "hazard_correction": round(float(row["hazards"][transition]) - float(row["base_hazards"][transition]), 6),
            }
            for row in simulated_holdout_transition_rows
            for transition in TRANSITION_NAMES
        ],
    }
    decision_checks = [
        {
            "name": "improves_vs_mech_01e_reference",
            "passed": model_mae < float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
            "actual": model_mae,
            "target": float(reference_baseline.get("model_mean_absolute_error") or float("inf")),
        },
        {
            "name": "diagnosis_flow_matches_mech_01e_when_locked_and_anchored",
            "passed": float(diagnosis_flow_mae) <= float(mech_01e_diagnosis_flow_mae),
            "actual": round(float(diagnosis_flow_mae), 6),
            "target": round(float(mech_01e_diagnosis_flow_mae), 6),
        },
        {
            "name": "all_transition_hazards_finite",
            "passed": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "actual": bool(
                all(
                    np.isfinite(float(row["hazards"][transition]))
                    for row in simulated_holdout_transition_rows
                    for transition in TRANSITION_NAMES
                )
            ),
            "target": True,
        },
    ]
    decision = {
        "completed": True,
        "keep": all(bool(check["passed"]) for check in decision_checks),
        "reason": "KP-01A keeps MECH-01E fixed at the diagnosis and first holdout state, then tests whether subgroup-active downstream hazard modulation improves the helper-aware branch.",
        "checks": decision_checks,
    }

    write_json(ctx.experiment_dir / "fit_artifact.json", fit_artifact)
    write_json(ctx.experiment_dir / "evaluation.json", evaluation)
    write_json(ctx.experiment_dir / "baseline_comparison.json", baseline_comparison)
    write_json(ctx.experiment_dir / "transition_hazard_summary.json", transition_hazard_summary)
    write_json(ctx.experiment_dir / "kp_overlay_summary.json", kp_overlay_artifact)
    write_json(ctx.experiment_dir / "state_trajectory_rows.json", _state_trajectory_rows(train_state_rows + holdout_state_rows))

    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "expected_outputs": list(ctx.experiment.expected_outputs),
            "mech_01e_reference_run_id": str(mech_01e_reference["reference_run_id"]),
            "holdout_years": holdout_years,
        },
        coverage_summary={
            "source_run_id": ctx.source_run_id,
            "train_row_count": len(train_rows),
            "holdout_row_count": len(holdout_rows),
            "transition_count": len(TRANSITION_NAMES),
            "locked_transition_count": len(locked_transitions),
            "anchored_holdout_quarter_count": 1,
            "downstream_overlay_transition_count": len(DOWNSTREAM_TRANSITIONS),
            "eligible_helper_factor_count": sum(int(row["eligible_factor_count"]) for row in kp_overlay_summary["model_rows"]),
            "weighted_helper_factor_count": sum(int(row["weighted_factor_count"]) for row in kp_overlay_summary["model_rows"]),
            "kp_support_strength": round(float(kp_support_strength), 6),
            "kp_positive_group_count": kp_positive_group_count,
            "low_confidence_reasons": list(low_confidence_reasons),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "coverage_mean",
                "value": round(float(coverage_mean), 6),
                "role": "undiagnosed_state_reconstruction",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E diagnosed coverage mean so the KP branch only tests downstream hazard modulation",
                "uncertainty": "inherits MECH-01E coverage uncertainty",
                "why_needed": "Keeps the latent state reconstruction fixed while evaluating only downstream KP modulation.",
            },
            {
                "name": "suppression_share_train_mean",
                "value": round(float(suppression_share_mean), 6),
                "role": "art_state_split_between_A_and_V",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E documented suppression share among ART",
                "uncertainty": "inherits MECH-01E service ascertainment uncertainty",
                "why_needed": "Prevents the KP overlay branch from changing the latent A/V split while testing only downstream hazards.",
            },
            {
                "name": "testing_share_train_mean",
                "value": round(float(testing_share_mean), 6),
                "role": "service_auxiliary_reference",
                "source_type": "estimated",
                "estimation_data": "MECH-01E fit artifact inherited from archive-supported national train rows",
                "estimation_method": "reuse the kept MECH-01E mean viral-load-testing share among ART",
                "uncertainty": "inherits MECH-01E documentation uncertainty",
                "why_needed": "Keeps the tested-for-VL auxiliary process aligned with the helper-aware baseline.",
            },
            {
                "name": "lost_gap_share_train_mean",
                "value": round(float(lost_gap_share), 6),
                "role": "diagnosed_gap_split_between_D_and_L",
                "source_type": "semi_markov",
                "estimation_data": "MECH-01E fit artifact inherited from consecutive national train rows",
                "estimation_method": "reuse the kept MECH-01E semi-Markov lost-gap share estimate",
                "uncertainty": "inherits sparse direct L-observation uncertainty",
                "why_needed": "Keeps the D/L latent split fixed so the KP overlay isolates downstream hazard changes.",
            },
            {
                "name": "kp_support_strength",
                "value": round(float(kp_support_strength), 6),
                "role": "weak_overlay_shrinkage",
                "source_type": "estimated",
                "estimation_data": "collapsed KP distribution from subgroup_weight_summary",
                "estimation_method": "fraction of collapsed KP groups with positive explicit support mass in the current Phase 3 subgroup output",
                "uncertainty": "reduced by missing explicit TGW mass and missing national KP profile",
                "why_needed": "Weakens the overlay in proportion to the amount of explicit collapsed KP support available in the current branch.",
            },
            {
                "name": "kp_positive_group_count",
                "value": kp_positive_group_count,
                "role": "collapsed_kp_support_cardinality",
                "source_type": "estimated",
                "estimation_data": "collapsed KP registry and current subgroup support summary",
                "estimation_method": "count of collapsed KP groups with positive explicit support mass",
                "uncertainty": "none",
                "why_needed": "Records how many collapsed KP groups actively support the weak overlay branch.",
            },
            {
                "name": "locked_transition_count",
                "value": len(locked_transitions),
                "role": "diagnosis_lock_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "KP-01A experiment design",
                "estimation_method": "count of transitions constrained to the MECH-01E baseline hazard path",
                "uncertainty": "none",
                "why_needed": "Records that diagnosis remains fixed while the KP overlay only modulates downstream transitions.",
            },
            {
                "name": "downstream_overlay_transition_count",
                "value": len(DOWNSTREAM_TRANSITIONS),
                "role": "overlay_transition_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "transition research registry",
                "estimation_method": "count of downstream HIV transitions eligible for weak KP modulation",
                "uncertainty": "none",
                "why_needed": "Restricts the KP overlay to downstream hazards only, as required by the plan.",
            },
            {
                "name": "anchored_holdout_quarter_count",
                "value": 1,
                "role": "holdout_anchor_design_cardinality",
                "source_type": "physical_constraint",
                "estimation_data": "KP-01A experiment design",
                "estimation_method": "count of holdout quarters copied exactly from the kept MECH-01E baseline before downstream KP simulation begins",
                "uncertainty": "none",
                "why_needed": "Makes the KP overlay test commensurate with the kept helper-aware baseline by fixing the first holdout state.",
            },
            numerical_guard_entry(
                name="float32_epsilon",
                role="hazard_and_standardization_guard",
                why_needed="Prevents undefined standardization and zero-denominator hazards when train variation or state mass is numerically zero.",
            ),
        ],
    )
    return {
        "artifacts": artifacts,
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "transition_hazard_summary": transition_hazard_summary,
        "kp_overlay_summary": kp_overlay_artifact,
        "decision": decision,
    }
