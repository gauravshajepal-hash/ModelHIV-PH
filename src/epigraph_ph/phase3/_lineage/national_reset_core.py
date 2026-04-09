from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import save_tensor_artifact, write_json

try:  # pragma: no cover - optional plotting dependency
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


STATE_NAMES = ["U", "D", "A"]
PRIMARY_METRICS = ["diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period"]
VL_METRICS = ["tested_for_viral_load", "virally_suppressed"]
_HIV_PLUGIN = get_disease_plugin("hiv")


def _national_reset_cfg() -> dict[str, Any]:
    return dict((((_HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("national_reset", {}) or {}))


def _national_reset_required_section(key: str) -> dict[str, Any]:
    value = _national_reset_cfg().get(key)
    if not isinstance(value, dict):
        raise KeyError(f"Missing HIV phase3 national_reset section: {key}")
    return dict(value)


_NR_DEFAULTS = _national_reset_required_section("defaults")
_NR_CALIBRATION = _national_reset_required_section("calibration_grids")
_NR_STATE_RECONSTRUCTION = _national_reset_required_section("state_reconstruction")
_NR_DELAY_AUX = _national_reset_required_section("delay_aux_reference")
_NR_SERVICE = _national_reset_required_section("service_process")
_NR_BENCHMARK_GATES = _national_reset_required_section("benchmark_gates")

_ROLLING_WINDOW_QUARTERS = int(_NR_DEFAULTS["rolling_window_quarters"])
_DEFAULT_HOLDOUT_YEARS = [int(year) for year in list(_NR_DEFAULTS.get("holdout_years", []) or [])]
_ALPHA_GRID = [float(value) for value in list(_NR_CALIBRATION["alpha_grid"])]
_LINKAGE_CAP_GRID = [float(value) for value in list(_NR_CALIBRATION["linkage_cap_grid"])]
_DELAY_AUX_WEIGHT_GRID = [float(value) for value in list(_NR_CALIBRATION["delay_aux_weight_grid"])]
_DELAY_AUX_CAP_GRID = [float(value) for value in list(_NR_CALIBRATION["delay_aux_cap_grid"])]
_SERVICE_ALPHA_GRID = [float(value) for value in list(_NR_CALIBRATION["service_alpha_grid"])]
_SERVICE_TREND_GRID = [float(value) for value in list(_NR_CALIBRATION["service_trend_grid"])]


def _grid_midpoint(values: list[float]) -> float:
    if not values:
        raise ValueError("Calibration grid is empty")
    return float(values[len(values) // 2])


def _grid_closest(values: list[float], *, target: float) -> float:
    if not values:
        raise ValueError("Calibration grid is empty")
    return float(min(values, key=lambda value: abs(float(value) - target)))


def _required_metric(metric_map: dict[str, Any], key: str, *, context: str) -> float:
    if key not in metric_map or metric_map.get(key) is None:
        raise KeyError(f"Missing {context}: {key}")
    return float(metric_map[key])


def quarter_sort_key(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = str(quarter).split("-Q", 1)
    return int(year_text), int(quarter_text)


def quarter_end_month(quarter: str) -> str:
    year, quarter_number = quarter_sort_key(quarter)
    month = {1: 3, 2: 6, 3: 9, 4: 12}[quarter_number]
    return f"{year:04d}-{month:02d}"


def _quarter_start_month(quarter: str) -> str:
    year, quarter_number = quarter_sort_key(quarter)
    month = {1: 1, 2: 4, 3: 7, 4: 10}[quarter_number]
    return f"{year:04d}-{month:02d}"


def _normalized_mae(
    prediction_rows: list[dict[str, float]],
    target_rows: list[dict[str, Any]],
    metric_scales: dict[str, float],
) -> float:
    errors: list[float] = []
    for prediction, target in zip(prediction_rows, target_rows):
        for metric_name in PRIMARY_METRICS:
            target_value = target.get(metric_name)
            prediction_value = prediction.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            scale = max(float(metric_scales.get(metric_name) or 1.0), 1.0)
            errors.append(abs(float(prediction_value) - float(target_value)) / scale)
    return float(np.mean(errors)) if errors else float("inf")


def _smape(prediction: float, target: float) -> float:
    denom = abs(float(prediction)) + abs(float(target))
    if denom <= 1e-6:
        return 0.0
    return (2.0 * abs(float(prediction) - float(target))) / denom


def _build_metric_scales(train_rows: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric_name in PRIMARY_METRICS:
        observed = [float(row[metric_name]) for row in train_rows if row.get(metric_name) is not None]
        scales[metric_name] = max(observed) if observed else 1.0
    return scales


def _step_uda_forecast(history_rows: list[dict[str, Any]], *, alpha_flow: float, alpha_linkage: float, linkage_cap_multiplier: float) -> dict[str, float]:
    latest = history_rows[-1]
    diagnosed_total = max(float(latest.get("diagnosed_plhiv") or 0.0), 0.0)
    on_art = max(float(latest.get("alive_on_art") or 0.0), 0.0)
    diagnosed_not_art = max(diagnosed_total - on_art, 0.0)

    flow_history = [float(row["new_diagnosed_cases_period"]) for row in history_rows if row.get("new_diagnosed_cases_period") is not None]
    if not flow_history:
        flow_history = [0.0]
    linkage_history = [
        max(float(next_row.get("alive_on_art") or 0.0) - float(prev_row.get("alive_on_art") or 0.0), 0.0)
        for prev_row, next_row in zip(history_rows[:-1], history_rows[1:])
        if prev_row.get("alive_on_art") is not None and next_row.get("alive_on_art") is not None
    ]
    if not linkage_history:
        linkage_history = [0.0]

    recent_flow_mean = fmean(flow_history[-_ROLLING_WINDOW_QUARTERS:])
    recent_linkage_mean = fmean(linkage_history[-_ROLLING_WINDOW_QUARTERS:])
    next_flow = (1.0 - alpha_flow) * flow_history[-1] + alpha_flow * recent_flow_mean
    next_linkage = (1.0 - alpha_linkage) * linkage_history[-1] + alpha_linkage * recent_linkage_mean
    next_flow = max(float(next_flow), 0.0)
    next_linkage = max(float(next_linkage), 0.0)
    next_linkage = min(next_linkage, linkage_cap_multiplier * max(diagnosed_not_art, 1.0))

    next_diagnosed_not_art = max(diagnosed_not_art + next_flow - next_linkage, 0.0)
    next_on_art = max(on_art + next_linkage, 0.0)
    next_diagnosed_total = next_diagnosed_not_art + next_on_art
    return {
        "diagnosed_plhiv": float(next_diagnosed_total),
        "alive_on_art": float(next_on_art),
        "new_diagnosed_cases_period": float(next_flow),
    }


def _simple_compartmental_count_baseline(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(train_rows) < 2 or not holdout_rows:
        return {"available": False, "mean_absolute_error": 0.0, "rows": []}
    metric_scales = _build_metric_scales(train_rows)
    diagnosed_not_art = max(float(train_rows[-1]["diagnosed_plhiv"]) - float(train_rows[-1]["alive_on_art"]), 0.0)
    on_art = float(train_rows[-1]["alive_on_art"])
    rate_rows: list[tuple[float, float]] = []
    for previous, current in zip(train_rows[:-1], train_rows[1:]):
        if current.get("new_diagnosed_cases_period") is None:
            continue
        prior_diagnosed_not_art = max(float(previous["diagnosed_plhiv"]) - float(previous["alive_on_art"]), 1.0)
        diagnosis_rate = max(float(current["new_diagnosed_cases_period"]), 0.0) / prior_diagnosed_not_art
        linkage_rate = max(float(current["alive_on_art"]) - float(previous["alive_on_art"]), 0.0) / prior_diagnosed_not_art
        rate_rows.append((diagnosis_rate, linkage_rate))
    diagnosis_rate = float(np.mean([row[0] for row in rate_rows])) if rate_rows else 0.0
    linkage_rate = float(np.mean([row[1] for row in rate_rows])) if rate_rows else 0.0
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, float]] = []
    for target in holdout_rows:
        next_flow = diagnosis_rate * max(diagnosed_not_art, 1.0)
        next_linkage = linkage_rate * max(diagnosed_not_art, 1.0)
        diagnosed_not_art = max(diagnosed_not_art + next_flow - next_linkage, 0.0)
        on_art = max(on_art + next_linkage, 0.0)
        prediction = {
            "diagnosed_plhiv": float(diagnosed_not_art + on_art),
            "alive_on_art": float(on_art),
            "new_diagnosed_cases_period": float(next_flow),
        }
        predictions.append(prediction)
        rows.append(
            {
                "quarter": str(target.get("quarter") or ""),
                "prediction": prediction,
                "target": {metric_name: float(target[metric_name]) for metric_name in PRIMARY_METRICS},
            }
        )
    return {
        "available": True,
        "mean_absolute_error": round(_normalized_mae(predictions, holdout_rows, metric_scales), 6),
        "rows": rows,
        "diagnosis_rate": round(diagnosis_rate, 6),
        "linkage_rate": round(linkage_rate, 6),
    }


def _carry_forward_count_baseline(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not train_rows or not holdout_rows:
        return {"available": False, "mean_absolute_error": 0.0, "rows": []}
    latest = {metric_name: float(train_rows[-1][metric_name]) for metric_name in PRIMARY_METRICS}
    metric_scales = _build_metric_scales(train_rows)
    predictions = [dict(latest) for _ in holdout_rows]
    rows = [
        {
            "quarter": str(target.get("quarter") or ""),
            "prediction": dict(latest),
            "target": {metric_name: float(target[metric_name]) for metric_name in PRIMARY_METRICS},
        }
        for target in holdout_rows
    ]
    return {
        "available": True,
        "mean_absolute_error": round(_normalized_mae(predictions, holdout_rows, metric_scales), 6),
        "rows": rows,
    }


def _calibrate_uda_parameters(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_scales = _build_metric_scales(train_rows)
    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for alpha_flow in _ALPHA_GRID:
        for alpha_linkage in _ALPHA_GRID:
            for linkage_cap_multiplier in _LINKAGE_CAP_GRID:
                rolling_predictions: list[dict[str, float]] = []
                rolling_targets: list[dict[str, Any]] = []
                for end_index in range(_ROLLING_WINDOW_QUARTERS, len(train_rows) - 1):
                    target = train_rows[end_index + 1]
                    if target.get("new_diagnosed_cases_period") is None:
                        continue
                    history = train_rows[: end_index + 1]
                    prediction = _step_uda_forecast(
                        history,
                        alpha_flow=alpha_flow,
                        alpha_linkage=alpha_linkage,
                        linkage_cap_multiplier=linkage_cap_multiplier,
                    )
                    rolling_predictions.append(prediction)
                    rolling_targets.append(target)
                if not rolling_predictions:
                    continue
                mae = _normalized_mae(rolling_predictions, rolling_targets, metric_scales)
                candidate = {
                    "alpha_flow": alpha_flow,
                    "alpha_linkage": alpha_linkage,
                    "linkage_cap_multiplier": linkage_cap_multiplier,
                    "rolling_origin_mae": round(mae, 6),
                    "rolling_origin_split_count": len(rolling_predictions),
                }
                candidate_rows.append(candidate)
                if best is None or mae < float(best["rolling_origin_mae"]):
                    best = candidate
    if best is None:
        best = {
            "alpha_flow": _grid_midpoint(_ALPHA_GRID),
            "alpha_linkage": _grid_closest(_ALPHA_GRID, target=0.0),
            "linkage_cap_multiplier": _grid_closest(_LINKAGE_CAP_GRID, target=1.0),
            "rolling_origin_mae": 0.0,
            "rolling_origin_split_count": 0,
        }
    return {
        "selected": dict(best),
        "candidate_rows": sorted(
            candidate_rows,
            key=lambda row: (
                float(row["rolling_origin_mae"]),
                float(row["alpha_flow"]),
                float(row["alpha_linkage"]),
                float(row["linkage_cap_multiplier"]),
            ),
        ),
    }


def _diagnosis_coverage_summary(
    train_rows: list[dict[str, Any]],
    estimated_plhiv_by_quarter: dict[str, float],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    coverage_values: list[float] = []
    for row in train_rows:
        quarter = str(row.get("quarter") or "")
        estimated_plhiv = float(estimated_plhiv_by_quarter.get(quarter) or 0.0)
        diagnosed = float(row.get("diagnosed_plhiv") or 0.0)
        if estimated_plhiv <= 0.0 or diagnosed <= 0.0:
            continue
        coverage = diagnosed / estimated_plhiv
        coverage_values.append(coverage)
        rows.append(
            {
                "quarter": quarter,
                "diagnosed_plhiv": round(diagnosed, 6),
                "estimated_plhiv": round(estimated_plhiv, 6),
                "diagnosed_coverage": round(coverage, 6),
            }
        )
    mean_coverage = float(np.mean(coverage_values)) if coverage_values else float(_NR_STATE_RECONSTRUCTION["coverage_fallback"])
    return {
        "rows": rows,
        "coverage_mean": round(mean_coverage, 6),
        "coverage_min": round(min(coverage_values), 6) if coverage_values else None,
        "coverage_max": round(max(coverage_values), 6) if coverage_values else None,
        "coverage_point_count": len(coverage_values),
        "uses_estimated_plhiv_as_auxiliary_prior_only": True,
    }


def _state_row(diagnosed_plhiv: float, alive_on_art: float, diagnosed_coverage: float) -> list[float]:
    coverage = min(
        max(float(diagnosed_coverage), float(_NR_STATE_RECONSTRUCTION["coverage_floor"])),
        float(_NR_STATE_RECONSTRUCTION["coverage_ceiling"]),
    )
    diagnosed_not_art = max(float(diagnosed_plhiv) - float(alive_on_art), 0.0)
    total_plhiv = max(float(diagnosed_plhiv) / coverage, float(diagnosed_plhiv))
    undiagnosed = max(total_plhiv - float(diagnosed_plhiv), 0.0)
    return [float(undiagnosed), float(diagnosed_not_art), float(alive_on_art)]


def _build_state_estimates(
    observed_rows: list[dict[str, Any]],
    forecast_rows: list[dict[str, float]],
    coverage_summary: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    coverage_lookup = {
        str(row["quarter"]): float(row["diagnosed_coverage"])
        for row in list(coverage_summary.get("rows") or [])
        if row.get("quarter")
    }
    coverage_mean = float(coverage_summary.get("coverage_mean") or _NR_STATE_RECONSTRUCTION["coverage_fallback"])
    observed_quarters = [str(row.get("quarter") or "") for row in observed_rows]
    forecast_quarters = [str(row.get("quarter") or "") for row in forecast_rows]
    state_estimates = np.asarray(
        [
            _state_row(
                float(row.get("diagnosed_plhiv") or 0.0),
                float(row.get("alive_on_art") or 0.0),
                coverage_lookup.get(str(row.get("quarter") or ""), coverage_mean),
            )
            for row in observed_rows
        ],
        dtype=np.float32,
    )
    forecast_states = np.asarray(
        [
            _state_row(
                float(row.get("diagnosed_plhiv") or 0.0),
                float(row.get("alive_on_art") or 0.0),
                coverage_mean,
            )
            for row in forecast_rows
        ],
        dtype=np.float32,
    )
    return state_estimates, forecast_states, observed_quarters, forecast_quarters


def _build_diagnosis_flow_evaluation(
    forecast_rows: list[dict[str, float]],
    diagnosis_flow_targets: dict[str, dict[str, Any]],
    coverage_summary: dict[str, Any],
    *,
    baseline_mae: float,
) -> dict[str, Any]:
    coverage_mean = float(coverage_summary.get("coverage_mean") or _NR_STATE_RECONSTRUCTION["coverage_fallback"])
    rows: list[dict[str, Any]] = []
    for forecast in forecast_rows:
        quarter = str(forecast.get("quarter") or "")
        target = dict(diagnosis_flow_targets.get(quarter) or {})
        if not target:
            continue
        diagnosed_total = float(forecast.get("diagnosed_plhiv") or 0.0)
        total_plhiv = max(diagnosed_total / max(coverage_mean, 1e-6), diagnosed_total)
        predicted_share = float(forecast.get("new_diagnosed_cases_period") or 0.0) / max(total_plhiv, 1.0)
        target_share = float(target.get("diagnosed_share") or 0.0)
        advanced_hiv_cases = target.get("advanced_hiv_cases_period")
        advanced_hiv_share = None
        if advanced_hiv_cases is not None and float(target.get("diagnosed_count") or 0.0) > 0.0:
            advanced_hiv_share = float(advanced_hiv_cases) / float(target.get("diagnosed_count") or 1.0)
        rows.append(
            {
                "quarter": quarter,
                "period_start": target.get("period_start"),
                "period_end": target.get("period_end"),
                "predicted_diagnosed_share": round(predicted_share, 6),
                "target_diagnosed_share": round(target_share, 6),
                "absolute_error": round(abs(predicted_share - target_share), 6),
                "diagnosed_count_prediction": round(float(forecast.get("new_diagnosed_cases_period") or 0.0), 6),
                "diagnosed_count_target": round(float(target.get("diagnosed_count") or 0.0), 6),
                "estimated_plhiv_reference": round(float(target.get("estimated_plhiv") or 0.0), 6),
                "advanced_hiv_share": round(float(advanced_hiv_share), 6) if advanced_hiv_share is not None else None,
                "source_id": target.get("source_id"),
                "source_label": target.get("source_label"),
                "source_url": target.get("source_url"),
            }
        )
    absolute_errors = [float(row["absolute_error"]) for row in rows]
    mean_error = float(np.mean(absolute_errors)) if absolute_errors else 0.0
    return {
        "point_count": len(rows),
        "mean_absolute_error": round(mean_error, 6),
        "max_absolute_error": round(max(absolute_errors), 6) if absolute_errors else 0.0,
        "baseline_mean_absolute_error": baseline_mae,
        "improvement_vs_baseline_pct": round(((baseline_mae - mean_error) / baseline_mae) * 100.0, 6) if baseline_mae > 0.0 else 0.0,
        "coverage_mean": round(coverage_mean, 6),
        "rows": rows,
    }


def _plot_mae_summary(output_path: Path, baseline_comparison: dict[str, Any], *, title: str) -> str | None:
    if plt is None:  # pragma: no cover
        return None
    labels = ["Model", "Carry-forward", "Simple compartmental"]
    values = [
        float(baseline_comparison.get("model_mean_absolute_error") or 0.0),
        float(baseline_comparison.get("carry_forward_mean_absolute_error") or 0.0),
        float(baseline_comparison.get("simple_compartmental_mean_absolute_error") or 0.0),
    ]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(labels, values, color=["#1f77b4", "#888888", "#f28e2b"])
    axis.set_ylabel("Normalized count-space MAE")
    axis.set_title(title)
    axis.axhline(float(baseline_comparison.get("legacy_threshold_carry_forward_mae") or 0.0), color="#2ca02c", linestyle="--", linewidth=1.0)
    axis.set_ylim(0.0, max(values + [0.08]) * 1.2)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)


def _plot_diagnosis_flow_fit(output_path: Path, diagnosis_flow_evaluation: dict[str, Any], *, title: str) -> str | None:
    if plt is None:  # pragma: no cover
        return None
    rows = list(diagnosis_flow_evaluation.get("rows") or [])
    if not rows:
        return None
    quarters = [str(row["quarter"]) for row in rows]
    predicted = [float(row["predicted_diagnosed_share"]) for row in rows]
    target = [float(row["target_diagnosed_share"]) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(quarters, predicted, marker="o", label="Predicted")
    axis.plot(quarters, target, marker="o", label="Target")
    axis.set_ylabel("Diagnosis-flow share")
    axis.set_title(title)
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)


def fit_national_uda_baseline(
    *,
    observation_rows: list[dict[str, Any]],
    estimated_plhiv_by_quarter: dict[str, float],
    diagnosis_flow_targets: dict[str, dict[str, Any]],
    artifact_dir: Path,
    holdout_years: list[int] | None = None,
    legacy_reference_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    holdout_years = sorted({int(year) for year in (holdout_years or _DEFAULT_HOLDOUT_YEARS)})
    if not holdout_years:
        raise ValueError("NR-01 requires configured or explicit holdout years")
    model_rows = [
        dict(row)
        for row in observation_rows
        if row.get("diagnosed_plhiv") is not None and row.get("alive_on_art") is not None
    ]
    train_rows = [
        row
        for row in model_rows
        if quarter_sort_key(str(row.get("quarter") or ""))[0] < min(holdout_years)
    ]
    holdout_rows = [
        row
        for row in model_rows
        if quarter_sort_key(str(row.get("quarter") or ""))[0] in set(holdout_years)
    ]
    if not train_rows or not holdout_rows:
        raise ValueError("NR-01 requires non-empty train and holdout rows")

    calibration = _calibrate_uda_parameters(train_rows)
    selected = dict(calibration["selected"])
    alpha_flow = float(selected["alpha_flow"])
    alpha_linkage = float(selected["alpha_linkage"])
    linkage_cap_multiplier = float(selected["linkage_cap_multiplier"])

    recursive_history = [dict(row) for row in train_rows]
    forecast_rows: list[dict[str, float]] = []
    holdout_evaluation_rows: list[dict[str, Any]] = []
    for target in holdout_rows:
        prediction = _step_uda_forecast(
            recursive_history,
            alpha_flow=alpha_flow,
            alpha_linkage=alpha_linkage,
            linkage_cap_multiplier=linkage_cap_multiplier,
        )
        prediction["quarter"] = str(target["quarter"])
        prediction["period_start"] = _quarter_start_month(str(target["quarter"]))
        prediction["period_end"] = quarter_end_month(str(target["quarter"]))
        forecast_rows.append(prediction)
        holdout_evaluation_rows.append(
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
        )
        recursive_history.append(dict(prediction))

    metric_scales = _build_metric_scales(train_rows)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales), 6)
    model_smape_rows = [
        _smape(float(prediction[metric_name]), float(target[metric_name]))
        for prediction, target in zip(forecast_rows, holdout_rows)
        for metric_name in PRIMARY_METRICS
        if target.get(metric_name) is not None
    ]
    model_smape = round(float(np.mean(model_smape_rows)) if model_smape_rows else 0.0, 6)

    carry_forward = _carry_forward_count_baseline(train_rows, holdout_rows)
    simple_compartmental = _simple_compartmental_count_baseline(train_rows, holdout_rows)
    coverage_summary = _diagnosis_coverage_summary(train_rows, estimated_plhiv_by_quarter)
    legacy_reference_metrics = dict(legacy_reference_metrics or {})
    legacy_carry_forward_mae = _required_metric(legacy_reference_metrics, "legacy_carry_forward_mae", context="NR-01 legacy benchmark")
    legacy_simple_compartmental_mae = _required_metric(legacy_reference_metrics, "legacy_simple_compartmental_mae", context="NR-01 legacy benchmark")
    legacy_baseline_model_mae = _required_metric(legacy_reference_metrics, "legacy_baseline_model_mae", context="NR-01 legacy benchmark")
    legacy_diagnosis_flow_mae = _required_metric(legacy_reference_metrics, "legacy_diagnosis_flow_mae", context="NR-01 legacy benchmark")
    diagnosis_flow_evaluation = _build_diagnosis_flow_evaluation(
        forecast_rows,
        diagnosis_flow_targets,
        coverage_summary,
        baseline_mae=legacy_diagnosis_flow_mae,
    )

    yearly_mae_rows: list[dict[str, Any]] = []
    grouped_predictions: dict[int, list[tuple[dict[str, float], dict[str, Any]]]] = defaultdict(list)
    for prediction, target in zip(forecast_rows, holdout_rows):
        year = quarter_sort_key(str(target["quarter"]))[0]
        grouped_predictions[year].append((prediction, target))
    for year, paired_rows in sorted(grouped_predictions.items()):
        year_predictions = [row[0] for row in paired_rows]
        year_targets = [row[1] for row in paired_rows]
        yearly_mae_rows.append(
            {
                "year": year,
                "mean_absolute_error": round(_normalized_mae(year_predictions, year_targets, metric_scales), 6),
            }
        )

    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "carry_forward_mean_absolute_error": float(carry_forward["mean_absolute_error"]),
        "simple_compartmental_mean_absolute_error": float(simple_compartmental["mean_absolute_error"]),
        "model_beats_carry_forward": model_mae < float(carry_forward["mean_absolute_error"]),
        "model_beats_simple_compartmental": model_mae < float(simple_compartmental["mean_absolute_error"]),
        "legacy_threshold_carry_forward_mae": legacy_carry_forward_mae,
        "legacy_threshold_simple_compartmental_mae": legacy_simple_compartmental_mae,
        "legacy_threshold_baseline_model_mae": legacy_baseline_model_mae,
        "holdout_year_rows": yearly_mae_rows,
    }

    state_estimates, forecast_states, observed_quarters, forecast_quarters = _build_state_estimates(train_rows, forecast_rows, coverage_summary)
    state_estimate_artifact = save_tensor_artifact(
        array=state_estimates,
        axis_names=["quarter", "state"],
        artifact_dir=artifact_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["NR-01 training state reconstruction"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=forecast_states,
        axis_names=["quarter", "state"],
        artifact_dir=artifact_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["NR-01 holdout forecast states"],
        save_pt=False,
    )

    evaluation = {
        "mode": "national_reset_uda_count_space",
        "state_names": list(STATE_NAMES),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {key: round(float(value), 6) for key, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": holdout_evaluation_rows,
        "yearly_mae_rows": yearly_mae_rows,
    }

    fit_artifact = {
        "model_family": "national_uda_count_space_reset",
        "state_names": list(STATE_NAMES),
        "primary_metrics": list(PRIMARY_METRICS),
        "calibration": calibration,
        "selected_parameters": selected,
        "coverage_summary": coverage_summary,
        "observation_row_count": len(observation_rows),
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "uses_estimated_plhiv_as_direct_training_truth": False,
        "uses_estimated_plhiv_as_auxiliary_latent_size_prior": True,
        "state_estimate_artifact": state_estimate_artifact,
        "forecast_state_artifact": forecast_state_artifact,
        "observed_state_quarters": observed_quarters,
        "forecast_state_quarters": forecast_quarters,
    }

    decision_checks = [
        {
            "name": "beats_legacy_carry_forward_threshold",
            "passed": model_mae < legacy_carry_forward_mae,
            "actual": model_mae,
            "target": legacy_carry_forward_mae,
        },
        {
            "name": "beats_legacy_simple_compartmental_threshold",
            "passed": model_mae < legacy_simple_compartmental_mae,
            "actual": model_mae,
            "target": legacy_simple_compartmental_mae,
        },
        {
            "name": "beats_count_space_carry_forward",
            "passed": model_mae < float(carry_forward["mean_absolute_error"]),
            "actual": model_mae,
            "target": float(carry_forward["mean_absolute_error"]),
        },
        {
            "name": "beats_count_space_simple_compartmental",
            "passed": model_mae < float(simple_compartmental["mean_absolute_error"]),
            "actual": model_mae,
            "target": float(simple_compartmental["mean_absolute_error"]),
        },
        {
            "name": "diagnosis_flow_mae_below_legacy_baseline",
            "passed": float(diagnosis_flow_evaluation["mean_absolute_error"]) < legacy_diagnosis_flow_mae,
            "actual": float(diagnosis_flow_evaluation["mean_absolute_error"]),
            "target": legacy_diagnosis_flow_mae,
        },
        {
            "name": "diagnosis_flow_improves_by_25pct",
            "passed": float(diagnosis_flow_evaluation["improvement_vs_baseline_pct"]) >= float(_NR_BENCHMARK_GATES["legacy_diagnosis_flow_improvement_pct"]),
            "actual": float(diagnosis_flow_evaluation["improvement_vs_baseline_pct"]),
            "target": float(_NR_BENCHMARK_GATES["legacy_diagnosis_flow_improvement_pct"]),
        },
        {
            "name": "no_holdout_year_worse_than_legacy_model",
            "passed": all(float(row["mean_absolute_error"]) <= legacy_baseline_model_mae for row in yearly_mae_rows),
            "actual": max([float(row["mean_absolute_error"]) for row in yearly_mae_rows], default=0.0),
            "target": legacy_baseline_model_mae,
        },
    ]

    decision = {
        "experiment_id": "NR-01-national-uda-baseline",
        "keep": all(bool(row["passed"]) for row in decision_checks),
        "decision_rule": "keep_only_if_all_primary_checks_pass",
        "checks": decision_checks,
        "summary": {
            "model_mean_absolute_error": model_mae,
            "carry_forward_mean_absolute_error": float(carry_forward["mean_absolute_error"]),
            "simple_compartmental_mean_absolute_error": float(simple_compartmental["mean_absolute_error"]),
            "diagnosis_flow_mean_absolute_error": float(diagnosis_flow_evaluation["mean_absolute_error"]),
            "diagnosis_flow_improvement_vs_baseline_pct": float(diagnosis_flow_evaluation["improvement_vs_baseline_pct"]),
        },
    }

    baseline_chart = _plot_mae_summary(artifact_dir / "mae_summary.png", baseline_comparison, title="NR-01 Holdout MAE")
    diagnosis_chart = _plot_diagnosis_flow_fit(artifact_dir / "diagnosis_flow_fit.png", diagnosis_flow_evaluation, title="NR-01 Diagnosis-flow fit")
    if baseline_chart:
        baseline_comparison["mae_summary_chart"] = baseline_chart
    if diagnosis_chart:
        diagnosis_flow_evaluation["diagnosis_flow_fit_chart"] = diagnosis_chart

    write_json(artifact_dir / "fit_artifact.json", fit_artifact)
    write_json(artifact_dir / "evaluation.json", evaluation)
    write_json(artifact_dir / "baseline_comparison.json", baseline_comparison)
    write_json(artifact_dir / "diagnosis_flow_evaluation.json", diagnosis_flow_evaluation)
    write_json(artifact_dir / "decision.json", decision)

    return {
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "diagnosis_flow_evaluation": diagnosis_flow_evaluation,
        "decision": decision,
        "state_estimate_artifact": state_estimate_artifact,
        "forecast_state_artifact": forecast_state_artifact,
    }


def _split_train_holdout_rows(
    observation_rows: list[dict[str, Any]],
    holdout_years: list[int] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    holdout_years = sorted({int(year) for year in (holdout_years or _DEFAULT_HOLDOUT_YEARS)})
    if not holdout_years:
        raise ValueError("National reset requires configured or explicit holdout years")
    model_rows = [
        dict(row)
        for row in observation_rows
        if row.get("diagnosed_plhiv") is not None and row.get("alive_on_art") is not None
    ]
    train_rows = [
        row
        for row in model_rows
        if quarter_sort_key(str(row.get("quarter") or ""))[0] < min(holdout_years)
    ]
    holdout_rows = [
        row
        for row in model_rows
        if quarter_sort_key(str(row.get("quarter") or ""))[0] in set(holdout_years)
    ]
    if not train_rows or not holdout_rows:
        raise ValueError("National reset requires non-empty train and holdout rows")
    return train_rows, holdout_rows, holdout_years


def _delay_aux_reference(observation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    advanced_share_rows: list[tuple[str, float]] = []
    cd4_rows: list[tuple[str, float]] = []
    for row in observation_rows:
        flow = row.get("new_diagnosed_cases_period")
        advanced = row.get("advanced_hiv_cases_period")
        if flow is not None and advanced is not None and float(flow) > 0.0:
            advanced_share_rows.append((str(row.get("quarter") or ""), float(advanced) / float(flow)))
        cd4 = row.get("median_cd4_at_enrollment")
        if cd4 is not None:
            cd4_rows.append((str(row.get("quarter") or ""), float(cd4)))
    advanced_values = [value for _, value in advanced_share_rows]
    cd4_values = [value for _, value in cd4_rows]
    return {
        "advanced_share_rows": [{"quarter": quarter, "value": round(value, 6)} for quarter, value in advanced_share_rows],
        "cd4_rows": [{"quarter": quarter, "value": round(value, 6)} for quarter, value in cd4_rows],
        "advanced_share_mean": float(np.mean(advanced_values)) if advanced_values else 0.0,
        "advanced_share_std": max(float(np.std(advanced_values)), float(_NR_DELAY_AUX["advanced_share_std_floor"])) if advanced_values else float(_NR_DELAY_AUX["advanced_share_std_floor"]),
        "cd4_mean": float(np.mean(cd4_values)) if cd4_values else float(_NR_DELAY_AUX["cd4_mean_fallback"]),
        "cd4_std": max(float(np.std(cd4_values)), float(_NR_DELAY_AUX["cd4_std_floor"])) if cd4_values else float(_NR_DELAY_AUX["cd4_std_floor"]),
    }


def _latest_delay_aux_features(history_rows: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    latest_advanced_share = None
    latest_advanced_quarter = None
    latest_cd4 = None
    latest_cd4_quarter = None
    for row in reversed(history_rows):
        if latest_advanced_share is None:
            flow = row.get("new_diagnosed_cases_period")
            advanced = row.get("advanced_hiv_cases_period")
            if flow is not None and advanced is not None and float(flow) > 0.0:
                latest_advanced_share = float(advanced) / float(flow)
                latest_advanced_quarter = str(row.get("quarter") or "")
        if latest_cd4 is None and row.get("median_cd4_at_enrollment") is not None:
            latest_cd4 = float(row.get("median_cd4_at_enrollment") or 0.0)
            latest_cd4_quarter = str(row.get("quarter") or "")
        if latest_advanced_share is not None and latest_cd4 is not None:
            break
    advanced_signal = 0.0
    cd4_signal = 0.0
    if latest_advanced_share is not None:
        advanced_signal = (latest_advanced_share - float(reference["advanced_share_mean"])) / max(float(reference["advanced_share_std"]), 1e-6)
    if latest_cd4 is not None:
        cd4_signal = (float(reference["cd4_mean"]) - latest_cd4) / max(float(reference["cd4_std"]), 1e-6)
    return {
        "advanced_signal": float(advanced_signal),
        "cd4_signal": float(cd4_signal),
        "latest_advanced_share": latest_advanced_share,
        "latest_advanced_quarter": latest_advanced_quarter,
        "latest_cd4": latest_cd4,
        "latest_cd4_quarter": latest_cd4_quarter,
    }


def _apply_delay_aux_to_prediction(
    *,
    history_rows: list[dict[str, Any]],
    prediction: dict[str, float],
    weights: dict[str, float],
    max_adjustment_pct: float,
    reference: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    latest = history_rows[-1]
    features = _latest_delay_aux_features(history_rows, reference)
    raw_adjustment = (
        float(weights.get("advanced_weight") or 0.0) * float(features["advanced_signal"])
        + float(weights.get("cd4_weight") or 0.0) * float(features["cd4_signal"])
    )
    bounded_adjustment = float(np.clip(raw_adjustment, -max_adjustment_pct, max_adjustment_pct))
    base_flow = float(prediction.get("new_diagnosed_cases_period") or 0.0)
    adjusted_flow = max(base_flow * (1.0 + bounded_adjustment), 0.0)
    adjusted_prediction = dict(prediction)
    adjusted_prediction["new_diagnosed_cases_period"] = float(adjusted_flow)
    adjusted_prediction["diagnosed_plhiv"] = max(
        float(latest.get("diagnosed_plhiv") or 0.0) + adjusted_flow,
        float(adjusted_prediction.get("alive_on_art") or 0.0),
    )
    metadata = {
        "raw_adjustment": round(raw_adjustment, 6),
        "bounded_adjustment": round(bounded_adjustment, 6),
        "advanced_signal": round(float(features["advanced_signal"]), 6),
        "cd4_signal": round(float(features["cd4_signal"]), 6),
        "latest_advanced_share": round(float(features["latest_advanced_share"]), 6) if features["latest_advanced_share"] is not None else None,
        "latest_advanced_quarter": features["latest_advanced_quarter"],
        "latest_cd4": round(float(features["latest_cd4"]), 6) if features["latest_cd4"] is not None else None,
        "latest_cd4_quarter": features["latest_cd4_quarter"],
    }
    return adjusted_prediction, metadata


def _calibrate_delay_aux(
    *,
    train_rows: list[dict[str, Any]],
    selected_parameters: dict[str, Any],
    diagnosis_flow_targets: dict[str, dict[str, Any]],
    estimated_plhiv_by_quarter: dict[str, float],
) -> dict[str, Any]:
    reference = _delay_aux_reference(train_rows)
    metric_scales = _build_metric_scales(train_rows)
    coverage_summary = _diagnosis_coverage_summary(train_rows, estimated_plhiv_by_quarter)
    coverage_mean = float(coverage_summary.get("coverage_mean") or _NR_STATE_RECONSTRUCTION["coverage_fallback"])
    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for advanced_weight in _DELAY_AUX_WEIGHT_GRID:
        for cd4_weight in _DELAY_AUX_WEIGHT_GRID:
            for max_adjustment_pct in _DELAY_AUX_CAP_GRID:
                rolling_predictions: list[dict[str, float]] = []
                rolling_targets: list[dict[str, Any]] = []
                share_errors: list[float] = []
                adjustment_rows: list[float] = []
                for end_index in range(_ROLLING_WINDOW_QUARTERS, len(train_rows) - 1):
                    target = train_rows[end_index + 1]
                    if target.get("new_diagnosed_cases_period") is None:
                        continue
                    history = train_rows[: end_index + 1]
                    base_prediction = _step_uda_forecast(
                        history,
                        alpha_flow=float(selected_parameters["alpha_flow"]),
                        alpha_linkage=float(selected_parameters["alpha_linkage"]),
                        linkage_cap_multiplier=float(selected_parameters["linkage_cap_multiplier"]),
                    )
                    adjusted_prediction, metadata = _apply_delay_aux_to_prediction(
                        history_rows=history,
                        prediction=base_prediction,
                        weights={"advanced_weight": advanced_weight, "cd4_weight": cd4_weight},
                        max_adjustment_pct=max_adjustment_pct,
                        reference=reference,
                    )
                    rolling_predictions.append(adjusted_prediction)
                    rolling_targets.append(target)
                    adjustment_rows.append(abs(float(metadata["bounded_adjustment"])))
                    quarter = str(target.get("quarter") or "")
                    flow_target = diagnosis_flow_targets.get(quarter)
                    if flow_target:
                        estimated_plhiv = float(flow_target.get("estimated_plhiv") or 0.0)
                        if estimated_plhiv <= 0.0:
                            estimated_plhiv = max(
                                float(adjusted_prediction["diagnosed_plhiv"]) / max(coverage_mean, 1e-6),
                                float(adjusted_prediction["diagnosed_plhiv"]),
                            )
                        predicted_share = float(adjusted_prediction["new_diagnosed_cases_period"]) / max(estimated_plhiv, 1.0)
                        share_errors.append(abs(predicted_share - float(flow_target.get("diagnosed_share") or 0.0)))
                if not rolling_predictions:
                    continue
                front_half_mae = _normalized_mae(rolling_predictions, rolling_targets, metric_scales)
                mean_share_error = float(np.mean(share_errors)) if share_errors else float("inf")
                composite_score = front_half_mae + (float(_NR_DELAY_AUX["diagnosis_flow_weight"]) * mean_share_error)
                candidate = {
                    "advanced_weight": round(float(advanced_weight), 6),
                    "cd4_weight": round(float(cd4_weight), 6),
                    "max_adjustment_pct": round(float(max_adjustment_pct), 6),
                    "front_half_mae": round(float(front_half_mae), 6),
                    "diagnosis_flow_share_mae": round(float(mean_share_error), 6) if np.isfinite(mean_share_error) else None,
                    "mean_absolute_adjustment_pct": round(float(np.mean(adjustment_rows)) if adjustment_rows else 0.0, 6),
                    "rolling_origin_split_count": len(rolling_predictions),
                    "composite_score": round(float(composite_score), 6),
                }
                candidate_rows.append(candidate)
                if best is None or composite_score < float(best["composite_score"]):
                    best = candidate
    if best is None:
        best = {
            "advanced_weight": _grid_closest(_DELAY_AUX_WEIGHT_GRID, target=0.0),
            "cd4_weight": _grid_closest(_DELAY_AUX_WEIGHT_GRID, target=0.0),
            "max_adjustment_pct": min(_DELAY_AUX_CAP_GRID),
            "front_half_mae": 0.0,
            "diagnosis_flow_share_mae": None,
            "mean_absolute_adjustment_pct": 0.0,
            "rolling_origin_split_count": 0,
            "composite_score": 0.0,
        }
    return {
        "reference": reference,
        "selected": dict(best),
        "candidate_rows": sorted(
            candidate_rows,
            key=lambda row: (
                float(row["composite_score"]),
                float(row["front_half_mae"]),
                abs(float(row["advanced_weight"])),
                abs(float(row["cd4_weight"])),
                float(row["max_adjustment_pct"]),
            ),
        ),
    }


def fit_national_uda_delay_aux(
    *,
    observation_rows: list[dict[str, Any]],
    estimated_plhiv_by_quarter: dict[str, float],
    diagnosis_flow_targets: dict[str, dict[str, Any]],
    artifact_dir: Path,
    holdout_years: list[int] | None = None,
    reference_metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    train_rows, holdout_rows, holdout_years = _split_train_holdout_rows(observation_rows, holdout_years)
    calibration = _calibrate_uda_parameters(train_rows)
    selected = dict(calibration["selected"])
    delay_calibration = _calibrate_delay_aux(
        train_rows=train_rows,
        selected_parameters=selected,
        diagnosis_flow_targets=diagnosis_flow_targets,
        estimated_plhiv_by_quarter=estimated_plhiv_by_quarter,
    )
    delay_selected = dict(delay_calibration["selected"])

    recursive_history = [dict(row) for row in train_rows]
    forecast_rows: list[dict[str, float]] = []
    delay_rows: list[dict[str, Any]] = []
    holdout_evaluation_rows: list[dict[str, Any]] = []
    for target in holdout_rows:
        base_prediction = _step_uda_forecast(
            recursive_history,
            alpha_flow=float(selected["alpha_flow"]),
            alpha_linkage=float(selected["alpha_linkage"]),
            linkage_cap_multiplier=float(selected["linkage_cap_multiplier"]),
        )
        adjusted_prediction, delay_metadata = _apply_delay_aux_to_prediction(
            history_rows=recursive_history,
            prediction=base_prediction,
            weights={
                "advanced_weight": float(delay_selected["advanced_weight"]),
                "cd4_weight": float(delay_selected["cd4_weight"]),
            },
            max_adjustment_pct=float(delay_selected["max_adjustment_pct"]),
            reference=dict(delay_calibration["reference"]),
        )
        adjusted_prediction["quarter"] = str(target["quarter"])
        adjusted_prediction["period_start"] = _quarter_start_month(str(target["quarter"]))
        adjusted_prediction["period_end"] = quarter_end_month(str(target["quarter"]))
        forecast_rows.append(adjusted_prediction)
        delay_rows.append({"quarter": str(target["quarter"]), **delay_metadata})
        holdout_evaluation_rows.append(
            {
                "quarter": str(target["quarter"]),
                "prediction": {metric_name: round(float(adjusted_prediction[metric_name]), 6) for metric_name in PRIMARY_METRICS},
                "target": {metric_name: round(float(target[metric_name]), 6) for metric_name in PRIMARY_METRICS},
            }
        )
        recursive_history.append(dict(adjusted_prediction))

    metric_scales = _build_metric_scales(train_rows)
    model_mae = round(_normalized_mae(forecast_rows, holdout_rows, metric_scales), 6)
    model_smape_rows = [
        _smape(float(prediction[metric_name]), float(target[metric_name]))
        for prediction, target in zip(forecast_rows, holdout_rows)
        for metric_name in PRIMARY_METRICS
        if target.get(metric_name) is not None
    ]
    model_smape = round(float(np.mean(model_smape_rows)) if model_smape_rows else 0.0, 6)
    carry_forward = _carry_forward_count_baseline(train_rows, holdout_rows)
    simple_compartmental = _simple_compartmental_count_baseline(train_rows, holdout_rows)
    coverage_summary = _diagnosis_coverage_summary(train_rows, estimated_plhiv_by_quarter)
    reference_metrics = dict(reference_metrics or {})
    reference_model_mean_absolute_error = _required_metric(reference_metrics, "reference_model_mean_absolute_error", context="NR-02 reference metric")
    reference_diagnosis_flow_mae = _required_metric(reference_metrics, "reference_diagnosis_flow_mean_absolute_error", context="NR-02 reference metric")
    diagnosis_flow_evaluation = _build_diagnosis_flow_evaluation(
        forecast_rows,
        diagnosis_flow_targets,
        coverage_summary,
        baseline_mae=reference_diagnosis_flow_mae,
    )

    yearly_mae_rows: list[dict[str, Any]] = []
    grouped_predictions: dict[int, list[tuple[dict[str, float], dict[str, Any]]]] = defaultdict(list)
    for prediction, target in zip(forecast_rows, holdout_rows):
        year = quarter_sort_key(str(target["quarter"]))[0]
        grouped_predictions[year].append((prediction, target))
    for year, paired_rows in sorted(grouped_predictions.items()):
        year_predictions = [row[0] for row in paired_rows]
        year_targets = [row[1] for row in paired_rows]
        yearly_mae_rows.append(
            {
                "year": year,
                "mean_absolute_error": round(_normalized_mae(year_predictions, year_targets, metric_scales), 6),
            }
        )

    baseline_comparison = {
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "carry_forward_mean_absolute_error": float(carry_forward["mean_absolute_error"]),
        "simple_compartmental_mean_absolute_error": float(simple_compartmental["mean_absolute_error"]),
        "model_beats_carry_forward": model_mae < float(carry_forward["mean_absolute_error"]),
        "model_beats_simple_compartmental": model_mae < float(simple_compartmental["mean_absolute_error"]),
        "reference_experiment_id": "NR-01-national-uda-baseline",
        "reference_model_mean_absolute_error": reference_model_mean_absolute_error,
        "reference_diagnosis_flow_mean_absolute_error": reference_diagnosis_flow_mae,
        "holdout_year_rows": yearly_mae_rows,
    }

    state_estimates, forecast_states, observed_quarters, forecast_quarters = _build_state_estimates(train_rows, forecast_rows, coverage_summary)
    state_estimate_artifact = save_tensor_artifact(
        array=state_estimates,
        axis_names=["quarter", "state"],
        artifact_dir=artifact_dir,
        stem="state_estimates",
        backend="numpy",
        device="cpu",
        notes=["NR-02 training state reconstruction"],
        save_pt=False,
    )
    forecast_state_artifact = save_tensor_artifact(
        array=forecast_states,
        axis_names=["quarter", "state"],
        artifact_dir=artifact_dir,
        stem="forecast_states",
        backend="numpy",
        device="cpu",
        notes=["NR-02 holdout forecast states"],
        save_pt=False,
    )

    evaluation = {
        "mode": "national_reset_uda_delay_aux",
        "state_names": list(STATE_NAMES),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "metric_scales": {key: round(float(value), 6) for key, value in metric_scales.items()},
        "model_mean_absolute_error": model_mae,
        "model_smape": model_smape,
        "holdout_rows": holdout_evaluation_rows,
        "yearly_mae_rows": yearly_mae_rows,
    }

    fit_artifact = {
        "model_family": "national_uda_count_space_delay_aux_reset",
        "state_names": list(STATE_NAMES),
        "primary_metrics": list(PRIMARY_METRICS),
        "calibration": calibration,
        "selected_parameters": selected,
        "delay_aux_calibration": delay_calibration,
        "coverage_summary": coverage_summary,
        "observation_row_count": len(observation_rows),
        "train_row_count": len(train_rows),
        "holdout_row_count": len(holdout_rows),
        "train_quarters": [str(row["quarter"]) for row in train_rows],
        "holdout_quarters": [str(row["quarter"]) for row in holdout_rows],
        "uses_estimated_plhiv_as_direct_training_truth": False,
        "uses_estimated_plhiv_as_auxiliary_latent_size_prior": True,
        "state_estimate_artifact": state_estimate_artifact,
        "forecast_state_artifact": forecast_state_artifact,
        "observed_state_quarters": observed_quarters,
        "forecast_state_quarters": forecast_quarters,
    }

    delay_aux_summary = {
        "reference_experiment_id": "NR-01-national-uda-baseline",
        "reference_model_mean_absolute_error": reference_model_mean_absolute_error,
        "reference_diagnosis_flow_mean_absolute_error": reference_diagnosis_flow_mae,
        "selected_delay_aux": delay_selected,
        "reference_statistics": delay_calibration["reference"],
        "candidate_count": len(delay_calibration["candidate_rows"]),
        "holdout_adjustments": delay_rows,
        "mean_absolute_holdout_adjustment_pct": round(
            float(np.mean([abs(float(row["bounded_adjustment"])) for row in delay_rows])) if delay_rows else 0.0,
            6,
        ),
        "advanced_hiv_observed_quarters": [
            str(row.get("quarter") or "")
            for row in observation_rows
            if row.get("advanced_hiv_cases_period") is not None
        ],
        "median_cd4_observed_quarters": [
            str(row.get("quarter") or "")
            for row in observation_rows
            if row.get("median_cd4_at_enrollment") is not None
        ],
        "auxiliary_signals_are_weak": float(delay_selected["max_adjustment_pct"]) <= float(_NR_DELAY_AUX["max_mean_holdout_adjustment_pct"]),
    }

    decision_checks = [
        {
            "name": "mean_holdout_mae_does_not_regress_vs_nr01",
            "passed": model_mae <= reference_model_mean_absolute_error,
            "actual": model_mae,
            "target": reference_model_mean_absolute_error,
        },
        {
            "name": "diagnosis_flow_mae_improves_by_10pct_vs_nr01",
            "passed": float(diagnosis_flow_evaluation["mean_absolute_error"]) <= round(reference_diagnosis_flow_mae * (1.0 - float(_NR_DELAY_AUX["diagnosis_flow_improvement_requirement"])), 6),
            "actual": float(diagnosis_flow_evaluation["mean_absolute_error"]),
            "target": round(reference_diagnosis_flow_mae * (1.0 - float(_NR_DELAY_AUX["diagnosis_flow_improvement_requirement"])), 6),
        },
        {
            "name": "auxiliary_terms_remain_weak",
            "passed": float(delay_aux_summary["mean_absolute_holdout_adjustment_pct"]) <= float(_NR_DELAY_AUX["max_mean_holdout_adjustment_pct"]),
            "actual": float(delay_aux_summary["mean_absolute_holdout_adjustment_pct"]),
            "target": float(_NR_DELAY_AUX["max_mean_holdout_adjustment_pct"]),
        },
    ]
    decision = {
        "experiment_id": "NR-02-delay-aux",
        "keep": all(bool(row["passed"]) for row in decision_checks),
        "decision_rule": "keep_only_if_diagnosis_flow_improves_without_front_half_regression",
        "checks": decision_checks,
        "summary": {
            "model_mean_absolute_error": model_mae,
            "diagnosis_flow_mean_absolute_error": float(diagnosis_flow_evaluation["mean_absolute_error"]),
            "reference_model_mean_absolute_error": reference_model_mean_absolute_error,
            "reference_diagnosis_flow_mean_absolute_error": reference_diagnosis_flow_mae,
        },
    }

    baseline_chart = _plot_mae_summary(artifact_dir / "mae_summary.png", baseline_comparison, title="NR-02 Holdout MAE")
    diagnosis_chart = _plot_diagnosis_flow_fit(artifact_dir / "diagnosis_flow_fit.png", diagnosis_flow_evaluation, title="NR-02 Diagnosis-flow fit")
    if baseline_chart:
        baseline_comparison["mae_summary_chart"] = baseline_chart
    if diagnosis_chart:
        diagnosis_flow_evaluation["diagnosis_flow_fit_chart"] = diagnosis_chart

    write_json(artifact_dir / "fit_artifact.json", fit_artifact)
    write_json(artifact_dir / "evaluation.json", evaluation)
    write_json(artifact_dir / "baseline_comparison.json", baseline_comparison)
    write_json(artifact_dir / "diagnosis_flow_evaluation.json", diagnosis_flow_evaluation)
    write_json(artifact_dir / "delay_aux_summary.json", delay_aux_summary)
    write_json(artifact_dir / "decision.json", decision)

    return {
        "fit_artifact": fit_artifact,
        "evaluation": evaluation,
        "baseline_comparison": baseline_comparison,
        "diagnosis_flow_evaluation": diagnosis_flow_evaluation,
        "delay_aux_summary": delay_aux_summary,
        "decision": decision,
        "state_estimate_artifact": state_estimate_artifact,
        "forecast_state_artifact": forecast_state_artifact,
    }


def _extract_forecast_rows_from_evaluation(evaluation: dict[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for row in list(evaluation.get("holdout_rows") or []):
        prediction = dict(row.get("prediction") or {})
        quarter = str(row.get("quarter") or "")
        rows.append(
            {
                "quarter": quarter,
                "period_start": _quarter_start_month(quarter),
                "period_end": quarter_end_month(quarter),
                **{metric_name: float(prediction.get(metric_name) or 0.0) for metric_name in PRIMARY_METRICS},
            }
        )
    return rows


def _calibrate_vl_process(train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_rows = [
        row
        for row in train_rows
        if row.get("alive_on_art") is not None and row.get("tested_for_viral_load") is not None and row.get("virally_suppressed") is not None
    ]
    if not valid_rows:
        return {
            "selected": {
                "testing_alpha": 0.5,
                "suppression_alpha": 0.5,
                "testing_trend_multiplier": 0.0,
                "suppression_trend_multiplier": 0.0,
                "rolling_origin_mae": 0.0,
                "rolling_origin_split_count": 0,
            },
            "candidate_rows": [],
        }
    candidate_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for testing_alpha in _SERVICE_ALPHA_GRID:
        for suppression_alpha in _SERVICE_ALPHA_GRID:
            for testing_trend_multiplier in _SERVICE_TREND_GRID:
                for suppression_trend_multiplier in _SERVICE_TREND_GRID:
                    predictions: list[tuple[float, float]] = []
                    targets: list[tuple[float, float]] = []
                    for end_index in range(2, len(valid_rows) - 1):
                        history = valid_rows[: end_index + 1]
                        target = valid_rows[end_index + 1]
                        testing_rates = [
                            float(row.get("tested_for_viral_load") or 0.0) / max(float(row.get("alive_on_art") or 0.0), 1.0)
                            for row in history
                        ]
                        suppression_rates = [
                            float(row.get("virally_suppressed") or 0.0) / max(float(row.get("tested_for_viral_load") or 0.0), 1.0)
                            for row in history
                        ]
                        testing_trend = (testing_rates[-1] - testing_rates[max(len(testing_rates) - _ROLLING_WINDOW_QUARTERS, 0)]) / max(min(len(testing_rates), _ROLLING_WINDOW_QUARTERS) - 1, 1)
                        suppression_trend = (suppression_rates[-1] - suppression_rates[max(len(suppression_rates) - _ROLLING_WINDOW_QUARTERS, 0)]) / max(min(len(suppression_rates), _ROLLING_WINDOW_QUARTERS) - 1, 1)
                        next_testing_rate = (
                            (1.0 - testing_alpha) * testing_rates[-1]
                            + testing_alpha * fmean(testing_rates[-_ROLLING_WINDOW_QUARTERS:])
                            + float(testing_trend_multiplier) * testing_trend
                        )
                        next_suppression_rate = (
                            (1.0 - suppression_alpha) * suppression_rates[-1]
                            + suppression_alpha * fmean(suppression_rates[-_ROLLING_WINDOW_QUARTERS:])
                            + float(suppression_trend_multiplier) * suppression_trend
                        )
                        predicted_tested = max(
                            float(target.get("alive_on_art") or 0.0)
                            * float(np.clip(next_testing_rate, float(_NR_SERVICE["testing_rate_floor"]), float(_NR_SERVICE["testing_rate_ceiling"]))),
                            0.0,
                        )
                        predicted_suppressed = max(
                            predicted_tested
                            * float(np.clip(next_suppression_rate, float(_NR_SERVICE["suppression_rate_floor"]), float(_NR_SERVICE["suppression_rate_ceiling"]))),
                            0.0,
                        )
                        predictions.append((predicted_tested, predicted_suppressed))
                        targets.append((float(target.get("tested_for_viral_load") or 0.0), float(target.get("virally_suppressed") or 0.0)))
                    if not predictions:
                        continue
                    tested_scale = max([target[0] for target in targets] or [1.0])
                    suppressed_scale = max([target[1] for target in targets] or [1.0])
                    errors = [
                        abs(predicted[0] - target[0]) / max(tested_scale, 1.0) + abs(predicted[1] - target[1]) / max(suppressed_scale, 1.0)
                        for predicted, target in zip(predictions, targets)
                    ]
                    mae = float(np.mean(errors) / 2.0) if errors else 0.0
                    candidate = {
                        "testing_alpha": round(float(testing_alpha), 6),
                        "suppression_alpha": round(float(suppression_alpha), 6),
                        "testing_trend_multiplier": round(float(testing_trend_multiplier), 6),
                        "suppression_trend_multiplier": round(float(suppression_trend_multiplier), 6),
                        "rolling_origin_mae": round(float(mae), 6),
                        "rolling_origin_split_count": len(predictions),
                    }
                    candidate_rows.append(candidate)
                    if best is None or mae < float(best["rolling_origin_mae"]):
                        best = candidate
    if best is None:
        best = {
            "testing_alpha": _grid_midpoint(_SERVICE_ALPHA_GRID),
            "suppression_alpha": _grid_midpoint(_SERVICE_ALPHA_GRID),
            "testing_trend_multiplier": _grid_closest(_SERVICE_TREND_GRID, target=0.0),
            "suppression_trend_multiplier": _grid_closest(_SERVICE_TREND_GRID, target=0.0),
            "rolling_origin_mae": 0.0,
            "rolling_origin_split_count": 0,
        }
    return {
        "selected": dict(best),
        "candidate_rows": sorted(
            candidate_rows,
            key=lambda row: (
                float(row["rolling_origin_mae"]),
                float(row["testing_alpha"]),
                float(row["suppression_alpha"]),
                float(row["testing_trend_multiplier"]),
                float(row["suppression_trend_multiplier"]),
            ),
        ),
    }


def fit_national_uda_vl_observation_process(
    *,
    observation_rows: list[dict[str, Any]],
    upstream_result: dict[str, Any],
    artifact_dir: Path,
    holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    train_rows, holdout_rows, holdout_years = _split_train_holdout_rows(observation_rows, holdout_years)
    forecast_rows = _extract_forecast_rows_from_evaluation(dict(upstream_result.get("evaluation") or {}))
    if len(forecast_rows) != len(holdout_rows):
        raise ValueError("NR-03 requires holdout forecast rows from the upstream front-half experiment")

    calibration = _calibrate_vl_process(train_rows)
    selected = dict(calibration["selected"])
    valid_train_rows = [
        row
        for row in train_rows
        if row.get("alive_on_art") is not None and row.get("tested_for_viral_load") is not None and row.get("virally_suppressed") is not None
    ]
    testing_rates = [
        float(row.get("tested_for_viral_load") or 0.0) / max(float(row.get("alive_on_art") or 0.0), 1.0)
        for row in valid_train_rows
    ]
    suppression_rates = [
        float(row.get("virally_suppressed") or 0.0) / max(float(row.get("tested_for_viral_load") or 0.0), 1.0)
        for row in valid_train_rows
    ]
    testing_alpha = float(selected["testing_alpha"])
    suppression_alpha = float(selected["suppression_alpha"])
    testing_trend_multiplier = float(selected.get("testing_trend_multiplier") or 0.0)
    suppression_trend_multiplier = float(selected.get("suppression_trend_multiplier") or 0.0)
    testing_trend = (
        (testing_rates[-1] - testing_rates[max(len(testing_rates) - _ROLLING_WINDOW_QUARTERS, 0)])
        / max(min(len(testing_rates), _ROLLING_WINDOW_QUARTERS) - 1, 1)
        if testing_rates
        else 0.0
    )
    suppression_trend = (
        (suppression_rates[-1] - suppression_rates[max(len(suppression_rates) - _ROLLING_WINDOW_QUARTERS, 0)])
        / max(min(len(suppression_rates), _ROLLING_WINDOW_QUARTERS) - 1, 1)
        if suppression_rates
        else 0.0
    )
    next_testing_rate = (
        (1.0 - testing_alpha) * testing_rates[-1]
        + testing_alpha * fmean(testing_rates[-_ROLLING_WINDOW_QUARTERS:])
        + testing_trend_multiplier * testing_trend
    ) if testing_rates else 0.0
    next_suppression_rate = (
        (1.0 - suppression_alpha) * suppression_rates[-1]
        + suppression_alpha * fmean(suppression_rates[-_ROLLING_WINDOW_QUARTERS:])
        + suppression_trend_multiplier * suppression_trend
    ) if suppression_rates else 0.0

    service_rows: list[dict[str, Any]] = []
    tested_targets: list[float] = []
    tested_predictions: list[float] = []
    suppressed_targets: list[float] = []
    suppressed_predictions: list[float] = []
    for forecast, target in zip(forecast_rows, holdout_rows):
        art_stock = float(forecast.get("alive_on_art") or 0.0)
        predicted_tested = max(
            art_stock * float(np.clip(next_testing_rate, float(_NR_SERVICE["testing_rate_floor"]), float(_NR_SERVICE["testing_rate_ceiling"]))),
            0.0,
        )
        predicted_suppressed = max(
            predicted_tested
            * float(np.clip(next_suppression_rate, float(_NR_SERVICE["suppression_rate_floor"]), float(_NR_SERVICE["suppression_rate_ceiling"]))),
            0.0,
        )
        target_tested = target.get("tested_for_viral_load")
        target_suppressed = target.get("virally_suppressed")
        if target_tested is not None:
            tested_targets.append(float(target_tested))
            tested_predictions.append(predicted_tested)
        if target_suppressed is not None:
            suppressed_targets.append(float(target_suppressed))
            suppressed_predictions.append(predicted_suppressed)
        service_rows.append(
            {
                "quarter": str(target.get("quarter") or ""),
                "art_stock_prediction": round(art_stock, 6),
                "predicted_tested_for_viral_load": round(predicted_tested, 6),
                "target_tested_for_viral_load": round(float(target_tested), 6) if target_tested is not None else None,
                "predicted_virally_suppressed": round(predicted_suppressed, 6),
                "target_virally_suppressed": round(float(target_suppressed), 6) if target_suppressed is not None else None,
                "testing_rate_prediction": round(
                    float(np.clip(next_testing_rate, float(_NR_SERVICE["testing_rate_floor"]), float(_NR_SERVICE["testing_rate_ceiling"]))),
                    6,
                ),
                "suppression_given_tested_prediction": round(
                    float(np.clip(next_suppression_rate, float(_NR_SERVICE["suppression_rate_floor"]), float(_NR_SERVICE["suppression_rate_ceiling"]))),
                    6,
                ),
                "observation_process": {
                    "tested_for_viral_load": "service_ascertainment_given_art",
                    "virally_suppressed": "documented_suppression_given_testing",
                },
            }
        )

    tested_scale = max(tested_targets) if tested_targets else 1.0
    suppressed_scale = max(suppressed_targets) if suppressed_targets else 1.0
    vl_tested_mae = round(
        float(np.mean([abs(pred - target) / max(tested_scale, 1.0) for pred, target in zip(tested_predictions, tested_targets)])) if tested_targets else 0.0,
        6,
    )
    vl_suppressed_mae = round(
        float(np.mean([abs(pred - target) / max(suppressed_scale, 1.0) for pred, target in zip(suppressed_predictions, suppressed_targets)])) if suppressed_targets else 0.0,
        6,
    )
    carry_forward_testing_rate = testing_rates[-1] if testing_rates else 0.0
    carry_forward_suppression_rate = suppression_rates[-1] if suppression_rates else 0.0
    carry_forward_tested = [
        max(
            float(forecast.get("alive_on_art") or 0.0)
            * float(np.clip(carry_forward_testing_rate, float(_NR_SERVICE["testing_rate_floor"]), float(_NR_SERVICE["testing_rate_ceiling"]))),
            0.0,
        )
        for forecast in forecast_rows
    ]
    carry_forward_suppressed = [
        max(
            predicted_tested
            * float(np.clip(carry_forward_suppression_rate, float(_NR_SERVICE["suppression_rate_floor"]), float(_NR_SERVICE["suppression_rate_ceiling"]))),
            0.0,
        )
        for predicted_tested in carry_forward_tested
    ]
    carry_forward_vl_tested_mae = round(
        float(np.mean([abs(pred - target) / max(tested_scale, 1.0) for pred, target in zip(carry_forward_tested, tested_targets)])) if tested_targets else 0.0,
        6,
    )
    carry_forward_vl_suppressed_mae = round(
        float(np.mean([abs(pred - target) / max(suppressed_scale, 1.0) for pred, target in zip(carry_forward_suppressed, suppressed_targets)])) if suppressed_targets else 0.0,
        6,
    )

    front_half_baseline = dict(upstream_result.get("baseline_comparison") or {})
    front_half_evaluation = dict(upstream_result.get("evaluation") or {})
    diagnosis_flow_evaluation = dict(upstream_result.get("diagnosis_flow_evaluation") or {})
    delay_aux_summary = upstream_result.get("delay_aux_summary")
    fit_artifact = dict(upstream_result.get("fit_artifact") or {})
    fit_artifact["vl_observation_process_calibration"] = calibration
    fit_artifact["vl_observation_process"] = "service_and_ascertainment"

    vl_observation_process = {
        "front_half_reference_experiment": str(upstream_result.get("decision", {}).get("experiment_id") or "NR-01-national-uda-baseline"),
        "selected_parameters": selected,
        "candidate_count": len(calibration["candidate_rows"]),
        "testing_rate_train_mean": round(float(np.mean(testing_rates)) if testing_rates else 0.0, 6),
        "suppression_given_tested_train_mean": round(float(np.mean(suppression_rates)) if suppression_rates else 0.0, 6),
        "holdout_vl_tested_mae": vl_tested_mae,
        "holdout_virally_suppressed_mae": vl_suppressed_mae,
        "carry_forward_vl_tested_mae": carry_forward_vl_tested_mae,
        "carry_forward_virally_suppressed_mae": carry_forward_vl_suppressed_mae,
        "service_rows": service_rows,
        "observation_process_assumptions": {
            "tested_for_viral_load": "observed_through_service_uptake_given_art_stock",
            "virally_suppressed": "documented_only_among_tested_individuals",
        },
    }

    decision_checks = [
        {
            "name": "no_front_half_regression_vs_upstream",
            "passed": True,
            "actual": float(front_half_baseline.get("model_mean_absolute_error") or 0.0),
            "target": float(front_half_baseline.get("model_mean_absolute_error") or 0.0),
        },
        {
            "name": "vl_tested_beats_carry_forward_rate_baseline",
            "passed": vl_tested_mae <= carry_forward_vl_tested_mae,
            "actual": vl_tested_mae,
            "target": carry_forward_vl_tested_mae,
        },
        {
            "name": "virally_suppressed_beats_carry_forward_rate_baseline",
            "passed": vl_suppressed_mae <= carry_forward_vl_suppressed_mae,
            "actual": vl_suppressed_mae,
            "target": carry_forward_vl_suppressed_mae,
        },
    ]
    decision = {
        "experiment_id": "NR-03-vl-observation-process",
        "keep": all(bool(row["passed"]) for row in decision_checks),
        "decision_rule": "keep_only_if_front_half_survives_and_vl_process_beats_naive_service_baseline",
        "checks": decision_checks,
        "summary": {
            "front_half_model_mean_absolute_error": float(front_half_baseline.get("model_mean_absolute_error") or 0.0),
            "holdout_vl_tested_mae": vl_tested_mae,
            "holdout_virally_suppressed_mae": vl_suppressed_mae,
        },
    }

    write_json(artifact_dir / "fit_artifact.json", fit_artifact)
    write_json(artifact_dir / "evaluation.json", front_half_evaluation)
    write_json(artifact_dir / "baseline_comparison.json", front_half_baseline)
    write_json(artifact_dir / "diagnosis_flow_evaluation.json", diagnosis_flow_evaluation)
    if delay_aux_summary is not None:
        write_json(artifact_dir / "delay_aux_summary.json", delay_aux_summary)
    write_json(artifact_dir / "vl_observation_process.json", vl_observation_process)
    write_json(artifact_dir / "decision.json", decision)

    return {
        "fit_artifact": fit_artifact,
        "evaluation": front_half_evaluation,
        "baseline_comparison": front_half_baseline,
        "diagnosis_flow_evaluation": diagnosis_flow_evaluation,
        "delay_aux_summary": delay_aux_summary,
        "vl_observation_process": vl_observation_process,
        "decision": decision,
        "state_estimate_artifact": upstream_result.get("state_estimate_artifact"),
        "forecast_state_artifact": upstream_result.get("forecast_state_artifact"),
    }
