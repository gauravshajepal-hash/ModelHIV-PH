from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_probabilistic_batch as prob
from epigraph_ph.phase3 import tr_v3_shock_plateau_predictability_batch as plateau_batch
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3.tr_v3_05_autoresearch import quarter_sort_key
from epigraph_ph.runtime import ensure_dir, write_json


EXACT_EXPERIMENT_ID = "EXP-R10-M1-F1-C1"
DENSE_EXPERIMENT_ID = "EXP-R10-DENSE-M1-C1-H1"
CALIBRATION_LEVELS: tuple[float, ...] = (0.8, 0.95)
EXACT_SCALE_GRID: tuple[float, ...] = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
ASYM_SCALE_GRID: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5)
JUMP_QUANTILES: tuple[float, ...] = (0.8, 0.9)
HIGH_RISK_PROBABILITY = 0.5
SHARED_REGIME_QUANTILE = 0.8
MIN_REGIME_HISTORY = 8
ERA_MIN_POINTS = 6


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ridge_probability(features: np.ndarray, labels: np.ndarray, current: np.ndarray, *, ridge: float = 0.5) -> float:
    penalty = np.eye(features.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(features.T @ features + penalty, features.T @ labels)
    return float(np.clip(float(np.dot(current, beta)), 0.0, 1.0))


def _year_and_quarter(quarter: str) -> tuple[int, int]:
    year_text, quarter_text = str(quarter).split("Q")
    return int(year_text.rstrip("-")), int(quarter_text)


def _covid_features(point: dict[str, Any]) -> list[float]:
    year, quarter_num = _year_and_quarter(str(point["quarter"]))
    covid = 1.0 if 2020 <= int(year) <= 2021 else 0.0
    recovery = 1.0 if 2022 <= int(year) <= 2025 else 0.0
    return [
        1.0,
        1.0 if str(point["metric_name"]) == "diagnosed_plhiv" else 0.0,
        1.0 if str(point["metric_name"]) == "alive_on_art" else 0.0,
        1.0 if str(point["tier"]) == "bridge_observed" else 0.0,
        1.0 if int(quarter_num) == 1 else 0.0,
        1.0 if int(quarter_num) == 4 else 0.0,
        covid,
        recovery,
        math.sin((2.0 * math.pi * float(quarter_num)) / 4.0),
        math.cos((2.0 * math.pi * float(quarter_num)) / 4.0),
    ]


def _build_labeled_risk_rows(
    points: list[dict[str, Any]],
    *,
    threshold_quantile: float,
    feature_builder: Callable[[dict[str, Any]], list[float]],
) -> list[dict[str, Any]]:
    historical_points: list[dict[str, Any]] = []
    labeled_rows: list[dict[str, Any]] = []
    prev_label_by_stream: dict[tuple[str, str], float] = {}
    for point in points:
        if len(historical_points) < prob.MIN_RISK_HISTORY:
            historical_points.append(dict(point))
            continue
        abs_history = np.asarray([abs(float(row["residual"])) for row in historical_points], dtype=np.float64)
        threshold = float(np.quantile(abs_history, float(threshold_quantile)))
        label = 1.0 if abs(float(point["residual"])) > threshold else 0.0
        history_labels = np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64)
        prevalence = float(np.mean(history_labels)) if history_labels.size else 0.0
        stream_key = (str(point["metric_name"]), str(point["tier"]))
        persistence = float(prev_label_by_stream.get(stream_key, prevalence))
        current_features = np.asarray(feature_builder(point), dtype=np.float64)
        if len(labeled_rows) >= prob.MIN_RISK_HISTORY and int(np.sum(history_labels)) >= prob.MIN_RISK_EVENTS:
            feature_matrix = np.asarray([row["features"] for row in labeled_rows], dtype=np.float64)
            candidate_probability = _ridge_probability(feature_matrix, history_labels, current_features)
        else:
            candidate_probability = prevalence
        labeled_rows.append(
            {
                **point,
                "label": float(label),
                "threshold": float(threshold),
                "features": current_features,
                "candidate_probability": float(candidate_probability),
                "prevalence_probability": float(prevalence),
                "persistence_probability": float(persistence),
            }
        )
        historical_points.append(dict(point))
        prev_label_by_stream[stream_key] = float(label)
    return labeled_rows


def _fit_static_probability_model(
    labeled_rows: list[dict[str, Any]],
    *,
    feature_builder: Callable[[dict[str, Any]], list[float]],
) -> Callable[[dict[str, Any]], float]:
    if len(labeled_rows) < prob.MIN_RISK_HISTORY or sum(int(row["label"]) for row in labeled_rows) < prob.MIN_RISK_EVENTS:
        baseline = float(np.mean(np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64))) if labeled_rows else 0.0
        return lambda point: baseline
    feature_matrix = np.asarray([row["features"] for row in labeled_rows], dtype=np.float64)
    labels = np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64)
    penalty = np.eye(feature_matrix.shape[1], dtype=np.float64) * 0.5
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(feature_matrix.T @ feature_matrix + penalty, feature_matrix.T @ labels)
    return lambda point: float(np.clip(float(np.dot(np.asarray(feature_builder(point), dtype=np.float64), beta)), 0.0, 1.0))


def _fixed_holdout_risk_points(
    fixed_result: dict[str, Any],
    *,
    allowed_tiers: set[str],
) -> list[dict[str, Any]]:
    targets = list(fixed_result.get("holdout_target_rows") or [])
    predictions = list(fixed_result.get("prediction_rows") or [])
    endpoint_audit = dict(fixed_result.get("endpoint_audit") or {})
    train_support = dict(endpoint_audit.get("train_support_counts") or {})
    metric_support_max = {metric_name: 1 for metric_name in prob.PRIMARY_METRICS}
    raw_rows: list[dict[str, Any]] = []
    previous_prediction: dict[str, float] = {}
    for target_row, prediction_row in zip(targets, predictions, strict=False):
        quarter = str(target_row["quarter"])
        _, quarter_num = _year_and_quarter(quarter)
        for metric_name in prob.PRIMARY_METRICS:
            tier_name = prob._metric_tier(target_row, metric_name)
            if tier_name not in allowed_tiers:
                continue
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            support_row = dict(train_support.get(metric_name) or {})
            support_scored = int(support_row.get("scored") or 0)
            previous = previous_prediction.get(metric_name)
            raw_rows.append(
                {
                    "split_idx": None,
                    "quarter": quarter,
                    "quarter_num": quarter_num,
                    "metric_name": metric_name,
                    "tier": tier_name,
                    "target": float(target_value),
                    "prediction": float(prediction_value),
                    "residual": float(prediction_value) - float(target_value),
                    "pred_delta": 0.0 if previous is None else float(prediction_value) - float(previous),
                    "support_scored": int(support_scored),
                }
            )
            metric_support_max[metric_name] = max(metric_support_max[metric_name], int(support_scored))
            previous_prediction[metric_name] = float(prediction_value)
    metric_scales = {metric_name: 1.0 for metric_name in prob.PRIMARY_METRICS}
    for metric_name in prob.PRIMARY_METRICS:
        metric_targets = [abs(float(row["target"])) for row in raw_rows if str(row["metric_name"]) == metric_name]
        if metric_targets:
            metric_scales[metric_name] = max(float(np.median(np.asarray(metric_targets, dtype=np.float64))), 1.0)
    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        metric_name = str(row["metric_name"])
        scale = float(metric_scales[metric_name])
        quarter_num = int(row["quarter_num"])
        rows.append(
            {
                **row,
                "prediction_norm": float(row["prediction"]) / scale,
                "pred_delta_norm": abs(float(row["pred_delta"])) / scale,
                "support_norm": float(int(row["support_scored"])) / float(max(metric_support_max[metric_name], 1)),
                "q_sin": math.sin((2.0 * math.pi * float(quarter_num)) / 4.0),
                "q_cos": math.cos((2.0 * math.pi * float(quarter_num)) / 4.0),
            }
        )
    rows.sort(key=lambda row: (quarter_sort_key(str(row["quarter"])), str(row["metric_name"]), str(row["tier"])))
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], *, level: float) -> dict[str, Any]:
    if not rows:
        return {
            "nominal": float(level),
            "coverage": None,
            "count": 0,
            "mean_normalized_wis": None,
            "mean_normalized_width": None,
        }
    return {
        "nominal": float(level),
        "coverage": float(np.mean(np.asarray([float(row["covered"]) for row in rows], dtype=np.float64))),
        "count": int(len(rows)),
        "mean_normalized_wis": float(np.mean(np.asarray([float(row["normalized_interval_score"]) for row in rows], dtype=np.float64))),
        "mean_normalized_width": float(np.mean(np.asarray([float(row["normalized_width"]) for row in rows], dtype=np.float64))),
    }


def _summarize_level_sets(rows_by_level: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        level_key: _aggregate_rows(level_rows, level=float(level_key) / 100.0)
        for level_key, level_rows in rows_by_level.items()
    }


def _summary_from_level_sets(summary: dict[str, Any]) -> tuple[float | None, float | None]:
    wis_keys = [key for key in ("80", "95") if dict(summary.get(key) or {}).get("mean_normalized_wis") is not None]
    cov_keys = [key for key in ("80", "95") if dict(summary.get(key) or {}).get("coverage") is not None]
    primary_wis = (
        float(np.mean(np.asarray([float(summary[key]["mean_normalized_wis"]) for key in wis_keys], dtype=np.float64)))
        if wis_keys
        else None
    )
    coverage_gap = (
        float(np.mean(np.asarray([abs(float(summary[key]["coverage"]) - (float(key) / 100.0)) for key in cov_keys], dtype=np.float64)))
        if cov_keys
        else None
    )
    return primary_wis, coverage_gap


def _evaluate_scaled_bootstrap(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    scale_by_metric: dict[str, float],
    exclude_split: bool,
) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    for metric_name in prob.PRIMARY_METRICS:
        scale_value = float(scale_by_metric.get(metric_name) or 1.0)
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_scale = prob._metric_scale(calibration_points or target_points)
        tier_payload: dict[str, Any] = {}
        metric_overall_rows: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
        for tier_name in ("overall", "exact_observed", "bridge_observed"):
            tier_targets = target_points if tier_name == "overall" else [point for point in target_points if str(point["tier"]) == tier_name]
            rows_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
            for point in tier_targets:
                pool = prob._pool_points(calibration_points, split_idx=point.get("split_idx"), tier_name=tier_name, exclude_split=exclude_split)
                if pool is None:
                    continue
                draws = prob._distribution_from_pool(pool, method="bootstrap", seed=prob._stable_seed("exact-scale", metric_name, tier_name, point.get("split_idx"), point["quarter"]))
                if draws.size < prob.MIN_POOL_POINTS:
                    continue
                for level in CALIBRATION_LEVELS:
                    alpha = 1.0 - float(level)
                    lower = float(np.quantile(draws, alpha / 2.0)) * scale_value
                    upper = float(np.quantile(draws, 1.0 - (alpha / 2.0))) * scale_value
                    lower_bound = float(point["prediction"]) + lower
                    upper_bound = float(point["prediction"]) + upper
                    covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                    width = float(upper_bound - lower_bound)
                    interval_score = prob._interval_score(target=float(point["target"]), lower_bound=lower_bound, upper_bound=upper_bound, alpha=alpha)
                    rows_by_level[f"{int(level * 100)}"].append(
                        {
                            "covered": covered,
                            "normalized_width": width / float(metric_scale),
                            "normalized_interval_score": interval_score / float(metric_scale),
                        }
                    )
            tier_payload[tier_name] = _summarize_level_sets(rows_by_level)
            if tier_name == "overall":
                for key, values in rows_by_level.items():
                    metric_overall_rows[key].extend(values)
                    overall_by_level[key].extend(values)
        primary_summary = _summarize_level_sets(metric_overall_rows)
        by_metric[metric_name] = {
            "scale_multiplier": scale_value,
            **tier_payload,
            "overall_summary": primary_summary,
        }
    summary = _summarize_level_sets(overall_by_level)
    primary_wis, coverage_gap = _summary_from_level_sets(summary)
    return {"by_metric": by_metric, "summary": summary, "primary_normalized_wis": primary_wis, "coverage_gap": coverage_gap}


def _era_name_for_year(year: int, *, cut_year: int | None) -> str:
    if cut_year is None:
        return "all"
    return "early" if int(year) < int(cut_year) else "late"


def _evaluate_scaled_bootstrap_by_era(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    stratification_by_metric: dict[str, dict[str, Any]],
    exclude_split: bool,
) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    for metric_name in prob.PRIMARY_METRICS:
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_scale = prob._metric_scale(calibration_points or target_points)
        stratification = dict(stratification_by_metric.get(metric_name) or {})
        cut_year = stratification.get("cut_year")
        scale_by_era = dict(stratification.get("scale_by_era") or {"all": 1.0})
        tier_payload: dict[str, Any] = {}
        metric_overall_rows: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
        for tier_name in ("overall", "exact_observed", "bridge_observed"):
            tier_targets = target_points if tier_name == "overall" else [point for point in target_points if str(point["tier"]) == tier_name]
            rows_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
            for point in tier_targets:
                pool = prob._pool_points(calibration_points, split_idx=point.get("split_idx"), tier_name=tier_name, exclude_split=exclude_split)
                if pool is None:
                    continue
                draws = prob._distribution_from_pool(
                    pool,
                    method="bootstrap",
                    seed=prob._stable_seed("exact-era-scale", metric_name, tier_name, point.get("split_idx"), point["quarter"]),
                )
                if draws.size < prob.MIN_POOL_POINTS:
                    continue
                year, _ = _year_and_quarter(str(point["quarter"]))
                era_name = _era_name_for_year(int(year), cut_year=cut_year if isinstance(cut_year, int) else None)
                scale_value = float(scale_by_era.get(era_name) or scale_by_era.get("all") or 1.0)
                for level in CALIBRATION_LEVELS:
                    alpha = 1.0 - float(level)
                    lower = float(np.quantile(draws, alpha / 2.0)) * scale_value
                    upper = float(np.quantile(draws, 1.0 - (alpha / 2.0))) * scale_value
                    lower_bound = float(point["prediction"]) + lower
                    upper_bound = float(point["prediction"]) + upper
                    covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                    width = float(upper_bound - lower_bound)
                    interval_score = prob._interval_score(
                        target=float(point["target"]),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        alpha=alpha,
                    )
                    rows_by_level[f"{int(level * 100)}"].append(
                        {
                            "covered": covered,
                            "normalized_width": width / float(metric_scale),
                            "normalized_interval_score": interval_score / float(metric_scale),
                        }
                    )
            tier_payload[tier_name] = _summarize_level_sets(rows_by_level)
            if tier_name == "overall":
                for key, values in rows_by_level.items():
                    metric_overall_rows[key].extend(values)
                    overall_by_level[key].extend(values)
        by_metric[metric_name] = {
            "cut_year": cut_year,
            "scale_by_era": scale_by_era,
            **tier_payload,
            "overall_summary": _summarize_level_sets(metric_overall_rows),
        }
    summary = _summarize_level_sets(overall_by_level)
    primary_wis, coverage_gap = _summary_from_level_sets(summary)
    return {"by_metric": by_metric, "summary": summary, "primary_normalized_wis": primary_wis, "coverage_gap": coverage_gap}


def _evaluate_asymmetric_bootstrap(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    asymmetric_by_metric: dict[str, dict[str, float]],
    exclude_split: bool,
) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    for metric_name in prob.PRIMARY_METRICS:
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_scale = prob._metric_scale(calibration_points or target_points)
        config = dict(asymmetric_by_metric.get(metric_name) or {})
        center_shift = float(config.get("center_shift") or 0.0)
        lower_scale = float(config.get("lower_scale") or 1.0)
        upper_scale = float(config.get("upper_scale") or 1.0)
        tier_payload: dict[str, Any] = {}
        metric_overall_rows: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
        for tier_name in ("overall", "exact_observed", "bridge_observed"):
            tier_targets = target_points if tier_name == "overall" else [point for point in target_points if str(point["tier"]) == tier_name]
            rows_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
            for point in tier_targets:
                pool = prob._pool_points(calibration_points, split_idx=point.get("split_idx"), tier_name=tier_name, exclude_split=exclude_split)
                if pool is None:
                    continue
                errors = [
                    float(candidate["target"]) - float(candidate["prediction"]) - center_shift
                    for candidate in list(pool["points"])
                ]
                draws = prob._moving_block_draws(
                    errors,
                    seed=prob._stable_seed("exact-asym", metric_name, tier_name, point.get("split_idx"), point["quarter"]),
                )
                if draws.size < prob.MIN_POOL_POINTS:
                    continue
                center = float(point["prediction"]) + center_shift
                for level in CALIBRATION_LEVELS:
                    alpha = 1.0 - float(level)
                    lower = float(np.quantile(draws, alpha / 2.0)) * lower_scale
                    upper = float(np.quantile(draws, 1.0 - (alpha / 2.0))) * upper_scale
                    lower_bound = center + lower
                    upper_bound = center + upper
                    covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                    width = float(upper_bound - lower_bound)
                    interval_score = prob._interval_score(
                        target=float(point["target"]),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        alpha=alpha,
                    )
                    rows_by_level[f"{int(level * 100)}"].append(
                        {
                            "covered": covered,
                            "normalized_width": width / float(metric_scale),
                            "normalized_interval_score": interval_score / float(metric_scale),
                        }
                    )
            tier_payload[tier_name] = _summarize_level_sets(rows_by_level)
            if tier_name == "overall":
                for key, values in rows_by_level.items():
                    metric_overall_rows[key].extend(values)
                    overall_by_level[key].extend(values)
        by_metric[metric_name] = {
            "center_shift": center_shift,
            "lower_scale": lower_scale,
            "upper_scale": upper_scale,
            **tier_payload,
            "overall_summary": _summarize_level_sets(metric_overall_rows),
        }
    summary = _summarize_level_sets(overall_by_level)
    primary_wis, coverage_gap = _summary_from_level_sets(summary)
    return {"by_metric": by_metric, "summary": summary, "primary_normalized_wis": primary_wis, "coverage_gap": coverage_gap}


def _probability_map(rows: list[dict[str, Any]], *, key_mode: str) -> dict[tuple[str, ...], float]:
    mapping: dict[tuple[str, ...], float] = {}
    for row in rows:
        if key_mode == "point":
            key = (str(row["quarter"]), str(row["metric_name"]), str(row["tier"]))
        elif key_mode == "quarter":
            key = (str(row["quarter"]),)
        else:
            raise ValueError(f"Unsupported key_mode: {key_mode}")
        mapping[key] = float(row["candidate_probability"])
    return mapping


def _mixture_draws(
    *,
    calibration_points: list[dict[str, Any]],
    jump_probability: float,
    jump_quantile: float,
    seed_parts: tuple[object, ...],
) -> np.ndarray:
    residuals = np.asarray([float(point["residual"]) for point in calibration_points], dtype=np.float64)
    if residuals.size < prob.MIN_POOL_POINTS:
        return np.asarray([], dtype=np.float64)
    abs_threshold = float(np.quantile(np.abs(residuals), float(jump_quantile)))
    base = [float(value) for value in residuals if abs(float(value)) < abs_threshold]
    jump = [float(value) for value in residuals if abs(float(value)) >= abs_threshold]
    if len(base) < 3 or len(jump) < 3:
        return prob._moving_block_draws(list(residuals), seed=prob._stable_seed("fallback", *seed_parts))
    base_draws = prob._moving_block_draws(base, seed=prob._stable_seed("base", *seed_parts))
    jump_draws = prob._moving_block_draws(jump, seed=prob._stable_seed("jump", *seed_parts))
    rng = np.random.default_rng(prob._stable_seed("mix", *seed_parts))
    flags = rng.random(base_draws.size) < float(np.clip(jump_probability, 0.0, 1.0))
    return np.where(flags, jump_draws, base_draws)


def _evaluate_mixture_sidecar(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    probability_map: dict[tuple[str, ...], float],
    jump_quantile: float,
    exclude_split: bool,
    key_mode: str,
) -> dict[str, Any]:
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    high_risk_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    by_metric: dict[str, Any] = {}
    for metric_name in prob.PRIMARY_METRICS:
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_scale = prob._metric_scale(calibration_points or target_points)
        metric_rows: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
        for point in target_points:
            tier_name = str(point["tier"])
            pool = prob._pool_points(calibration_points, split_idx=point.get("split_idx"), tier_name=tier_name, exclude_split=exclude_split)
            if pool is None:
                continue
            key = (
                (str(point["quarter"]), str(metric_name), tier_name)
                if key_mode == "point"
                else (str(point["quarter"]),)
            )
            jump_probability = float(probability_map.get(key, 0.0))
            draws = _mixture_draws(
                calibration_points=list(pool["points"]),
                jump_probability=jump_probability,
                jump_quantile=jump_quantile,
                seed_parts=(metric_name, tier_name, point.get("split_idx"), point["quarter"], jump_quantile, key_mode),
            )
            if draws.size < prob.MIN_POOL_POINTS:
                continue
            for level in CALIBRATION_LEVELS:
                alpha = 1.0 - float(level)
                lower = float(np.quantile(draws, alpha / 2.0))
                upper = float(np.quantile(draws, 1.0 - (alpha / 2.0)))
                lower_bound = float(point["prediction"]) + lower
                upper_bound = float(point["prediction"]) + upper
                covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                width = float(upper_bound - lower_bound)
                interval_score = prob._interval_score(target=float(point["target"]), lower_bound=lower_bound, upper_bound=upper_bound, alpha=alpha)
                row = {
                    "covered": covered,
                    "normalized_width": width / float(metric_scale),
                    "normalized_interval_score": interval_score / float(metric_scale),
                    "high_risk": 1 if jump_probability >= HIGH_RISK_PROBABILITY else 0,
                }
                metric_rows[f"{int(level * 100)}"].append(row)
                overall_by_level[f"{int(level * 100)}"].append(row)
                if row["high_risk"] == 1:
                    high_risk_by_level[f"{int(level * 100)}"].append(row)
        by_metric[metric_name] = _summarize_level_sets(metric_rows)
    summary = _summarize_level_sets(overall_by_level)
    high_risk_summary = _summarize_level_sets(high_risk_by_level)
    primary_wis, coverage_gap = _summary_from_level_sets(summary)
    return {
        "by_metric": by_metric,
        "summary": summary,
        "high_risk_summary": high_risk_summary,
        "primary_normalized_wis": primary_wis,
        "coverage_gap": coverage_gap,
    }


def _quarter_regime_rows(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_scales: dict[str, float] = {}
    for metric_name in prob.PRIMARY_METRICS:
        metric_points = [row for row in points if str(row["metric_name"]) == metric_name]
        metric_scales[metric_name] = prob._metric_scale(metric_points)
    quarter_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in points:
        grouped.setdefault(str(point["quarter"]), []).append(point)
    ordered_quarters = sorted(grouped.keys(), key=quarter_sort_key)
    history_scores: list[float] = []
    prev_label = 0.0
    labeled_rows: list[dict[str, Any]] = []
    for quarter in ordered_quarters:
        quarter_points = list(grouped[quarter])
        score = float(
            np.mean(
                np.asarray(
                    [
                        abs(float(row["residual"])) / float(metric_scales[str(row["metric_name"])])
                        for row in quarter_points
                    ],
                    dtype=np.float64,
                )
            )
        )
        year, quarter_num = _year_and_quarter(quarter)
        features = np.asarray(
            [
                1.0,
                1.0 if quarter_num == 1 else 0.0,
                1.0 if quarter_num == 4 else 0.0,
                1.0 if 2020 <= int(year) <= 2021 else 0.0,
                1.0 if 2022 <= int(year) <= 2025 else 0.0,
                math.sin((2.0 * math.pi * float(quarter_num)) / 4.0),
                math.cos((2.0 * math.pi * float(quarter_num)) / 4.0),
                float(prev_label),
            ],
            dtype=np.float64,
        )
        if len(history_scores) < MIN_REGIME_HISTORY:
            history_scores.append(float(score))
            continue
        threshold = float(np.quantile(np.asarray(history_scores, dtype=np.float64), SHARED_REGIME_QUANTILE))
        label = 1.0 if float(score) >= threshold else 0.0
        history_labels = np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64)
        prevalence_probability = float(np.mean(history_labels)) if history_labels.size else 0.0
        persistence_probability = float(prev_label)
        if len(labeled_rows) >= MIN_REGIME_HISTORY and int(np.sum(history_labels)) >= 2:
            matrix = np.asarray([row["features"] for row in labeled_rows], dtype=np.float64)
            candidate_probability = _ridge_probability(matrix, history_labels, features)
        else:
            candidate_probability = float(prevalence_probability)
        labeled_rows.append(
            {
                "quarter": quarter,
                "label": float(label),
                "score": float(score),
                "features": features,
                "candidate_probability": float(candidate_probability),
                "prevalence_probability": float(prevalence_probability),
                "persistence_probability": float(persistence_probability),
            }
        )
        history_scores.append(float(score))
        prev_label = float(label)
    return labeled_rows


def _fit_static_quarter_regime_model(labeled_rows: list[dict[str, Any]]) -> Callable[[str, float], float]:
    if len(labeled_rows) < MIN_REGIME_HISTORY or sum(int(row["label"]) for row in labeled_rows) < 2:
        baseline = float(np.mean(np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64))) if labeled_rows else 0.0
        return lambda quarter, prev_label: baseline
    matrix = np.asarray([row["features"] for row in labeled_rows], dtype=np.float64)
    labels = np.asarray([float(row["label"]) for row in labeled_rows], dtype=np.float64)
    penalty = np.eye(matrix.shape[1], dtype=np.float64) * 0.5
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ labels)

    def _predict(quarter: str, prev_label: float) -> float:
        year, quarter_num = _year_and_quarter(quarter)
        features = np.asarray(
            [
                1.0,
                1.0 if quarter_num == 1 else 0.0,
                1.0 if quarter_num == 4 else 0.0,
                1.0 if 2020 <= int(year) <= 2021 else 0.0,
                1.0 if 2022 <= int(year) <= 2025 else 0.0,
                math.sin((2.0 * math.pi * float(quarter_num)) / 4.0),
                math.cos((2.0 * math.pi * float(quarter_num)) / 4.0),
                float(prev_label),
            ],
            dtype=np.float64,
        )
        return float(np.clip(float(np.dot(features, beta)), 0.0, 1.0))

    return _predict


def _shared_regime_payload(
    *,
    result: dict[str, Any],
    fixed_result: dict[str, Any],
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    lockbox_points_by_metric: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    point_rows = prob._risk_point_rows(result, allowed_tiers={"exact_observed", "bridge_observed"})
    quarter_rows = _quarter_regime_rows(point_rows)
    probability_map = {(str(row["quarter"]),): float(row["candidate_probability"]) for row in quarter_rows}
    rolling = _evaluate_mixture_sidecar(
        target_points_by_metric,
        calibration_points_by_metric=calibration_points_by_metric,
        probability_map=probability_map,
        jump_quantile=SHARED_REGIME_QUANTILE,
        exclude_split=True,
        key_mode="quarter",
    )
    predictor = _fit_static_quarter_regime_model(quarter_rows)
    lockbox_quarters = sorted({str(point["quarter"]) for points in lockbox_points_by_metric.values() for point in points}, key=quarter_sort_key)
    lockbox_prob_map: dict[tuple[str, ...], float] = {}
    prev_label = float(quarter_rows[-1]["label"]) if quarter_rows else 0.0
    for quarter in lockbox_quarters:
        probability = predictor(quarter, prev_label)
        lockbox_prob_map[(quarter,)] = float(probability)
        prev_label = 1.0 if probability >= HIGH_RISK_PROBABILITY else 0.0
    lockbox = _evaluate_mixture_sidecar(
        lockbox_points_by_metric,
        calibration_points_by_metric=calibration_points_by_metric,
        probability_map=lockbox_prob_map,
        jump_quantile=SHARED_REGIME_QUANTILE,
        exclude_split=False,
        key_mode="quarter",
    )
    risk_metrics = {
        "candidate_brier": prob._brier(quarter_rows, "candidate_probability"),
        "prevalence_brier": prob._brier(quarter_rows, "prevalence_probability"),
        "persistence_brier": prob._brier(quarter_rows, "persistence_probability"),
    }
    return {"rolling": rolling, "lockbox": lockbox, "quarter_rows": quarter_rows, "risk_metrics": risk_metrics}


def _plateau_probability_map(
    points_by_metric: dict[str, list[dict[str, Any]]],
) -> tuple[dict[tuple[str, ...], float], dict[str, Any], dict[str, list[dict[str, Any]]]]:
    probability_map: dict[tuple[str, ...], float] = {}
    summaries: dict[str, Any] = {}
    rows_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric_name, points in points_by_metric.items():
        rows = [
            {
                "quarter": str(point["quarter"]),
                metric_name: float(point["prediction"]),
                f"{metric_name}_tier": str(point["tier"]),
            }
            for point in points
        ]
        plateau_rows, threshold = plateau_batch._plateau_prediction_rows(rows, metric_name=metric_name, allowed_tiers={"exact_observed", "bridge_observed"})
        rows_by_metric[metric_name] = list(plateau_rows)
        summaries[metric_name] = {
            "threshold": float(threshold),
            "row_count": int(len(plateau_rows)),
        }
        for row in plateau_rows:
            probability_map[(str(row["quarter"]), metric_name)] = float(row["candidate_probability"])
    return probability_map, summaries, rows_by_metric


def _fit_static_plateau_model(plateau_rows: list[dict[str, Any]]) -> Callable[[bool, int], float]:
    if not plateau_rows:
        return lambda current_active, run_bucket: 0.0

    def _predict(current_active: bool, run_bucket: int) -> float:
        same_state = [row for row in plateau_rows if bool(row["current_active"]) == bool(current_active)]
        same_bucket = [row for row in same_state if int(row["run_bucket"]) == int(run_bucket)]
        if len(same_bucket) >= 2:
            return float(np.mean(np.asarray([float(row["label"]) for row in same_bucket], dtype=np.float64)))
        if len(same_state) >= 2:
            return float(np.mean(np.asarray([float(row["label"]) for row in same_state], dtype=np.float64)))
        return float(np.mean(np.asarray([float(row["label"]) for row in plateau_rows], dtype=np.float64)))

    return _predict


def _lockbox_plateau_probability_map(
    points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    plateau_summary: dict[str, Any],
    rows_by_metric: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, ...], float]:
    probability_map: dict[tuple[str, ...], float] = {}
    for metric_name, points in points_by_metric.items():
        ordered_points = sorted(list(points), key=lambda row: quarter_sort_key(str(row["quarter"])))
        if len(ordered_points) < 2:
            continue
        threshold = float(dict(plateau_summary.get(metric_name) or {}).get("threshold") or 0.0)
        predictor = _fit_static_plateau_model(list(rows_by_metric.get(metric_name) or []))
        current_run = 0
        current_active = False
        prev_value = float(ordered_points[0]["prediction"])
        for point in ordered_points[1:]:
            diff = abs(float(point["prediction"]) - prev_value)
            current_active = bool(diff <= threshold)
            current_run = current_run + 1 if current_active else 0
            probability_map[(str(point["quarter"]), metric_name)] = float(predictor(current_active, min(int(current_run), 4)))
            prev_value = float(point["prediction"])
    return probability_map


def _evaluate_plateau_sidecar(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    probability_map: dict[tuple[str, ...], float],
    calibration_probability_map: dict[tuple[str, ...], float],
    exclude_split: bool,
) -> dict[str, Any]:
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    plateau_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in CALIBRATION_LEVELS}
    for metric_name in prob.PRIMARY_METRICS:
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_scale = prob._metric_scale(calibration_points or target_points)
        for point in target_points:
            tier_name = str(point["tier"])
            pool = prob._pool_points(calibration_points, split_idx=point.get("split_idx"), tier_name=tier_name, exclude_split=exclude_split)
            if pool is None:
                continue
            plateau_probability = float(probability_map.get((str(point["quarter"]), metric_name), 0.0))
            pool_points = list(pool["points"])
            plateau_pool = [
                candidate
                for candidate in pool_points
                if float(calibration_probability_map.get((str(candidate["quarter"]), metric_name), 0.0)) >= HIGH_RISK_PROBABILITY
            ]
            base_pool = [
                candidate
                for candidate in pool_points
                if float(calibration_probability_map.get((str(candidate["quarter"]), metric_name), 0.0)) < HIGH_RISK_PROBABILITY
            ]
            if len(base_pool) < 3 or len(plateau_pool) < 3:
                draws = prob._distribution_from_pool(pool, method="bootstrap", seed=prob._stable_seed("plateau-fallback", metric_name, point["quarter"]))
            else:
                normal_draws = prob._moving_block_draws([float(row["residual"]) for row in base_pool], seed=prob._stable_seed("plateau-base", metric_name, point["quarter"]))
                plateau_draws = prob._moving_block_draws([float(row["residual"]) for row in plateau_pool], seed=prob._stable_seed("plateau-plateau", metric_name, point["quarter"]))
                rng = np.random.default_rng(prob._stable_seed("plateau-mix", metric_name, point["quarter"]))
                flags = rng.random(normal_draws.size) < plateau_probability
                draws = np.where(flags, plateau_draws, normal_draws)
            if draws.size < prob.MIN_POOL_POINTS:
                continue
            for level in CALIBRATION_LEVELS:
                alpha = 1.0 - float(level)
                lower = float(np.quantile(draws, alpha / 2.0))
                upper = float(np.quantile(draws, 1.0 - (alpha / 2.0)))
                lower_bound = float(point["prediction"]) + lower
                upper_bound = float(point["prediction"]) + upper
                covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                width = float(upper_bound - lower_bound)
                interval_score = prob._interval_score(target=float(point["target"]), lower_bound=lower_bound, upper_bound=upper_bound, alpha=alpha)
                row = {
                    "covered": covered,
                    "normalized_width": width / float(metric_scale),
                    "normalized_interval_score": interval_score / float(metric_scale),
                }
                overall_by_level[f"{int(level * 100)}"].append(row)
                if plateau_probability >= HIGH_RISK_PROBABILITY:
                    plateau_by_level[f"{int(level * 100)}"].append(row)
    summary = _summarize_level_sets(overall_by_level)
    plateau_summary = _summarize_level_sets(plateau_by_level)
    primary_wis, coverage_gap = _summary_from_level_sets(summary)
    return {"summary": summary, "plateau_summary": plateau_summary, "primary_normalized_wis": primary_wis, "coverage_gap": coverage_gap}


def _exact_scale_search(calibration_points_by_metric: dict[str, list[dict[str, Any]]]) -> dict[str, float]:
    selected: dict[str, float] = {}
    for metric_name in prob.PRIMARY_METRICS:
        target = {metric_name: list(calibration_points_by_metric.get(metric_name) or [])}
        calib = {metric_name: list(calibration_points_by_metric.get(metric_name) or [])}
        best_scale = 1.0
        best_objective = float("inf")
        for scale in EXACT_SCALE_GRID:
            payload = _evaluate_scaled_bootstrap(target, calibration_points_by_metric=calib, scale_by_metric={metric_name: float(scale)}, exclude_split=True)
            summary = dict(payload.get("summary") or {})
            wis, gap = _summary_from_level_sets(summary)
            objective = float((wis or 0.0) + 2.0 * (gap or 1.0))
            if objective < best_objective:
                best_objective = objective
                best_scale = float(scale)
        selected[metric_name] = float(best_scale)
    return selected


def _exact_era_scale_search(calibration_points_by_metric: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for metric_name in prob.PRIMARY_METRICS:
        target = {metric_name: list(calibration_points_by_metric.get(metric_name) or [])}
        calib = {metric_name: list(calibration_points_by_metric.get(metric_name) or [])}
        metric_points = list(calibration_points_by_metric.get(metric_name) or [])
        metric_years = sorted({_year_and_quarter(str(point["quarter"]))[0] for point in metric_points})
        candidate_cut_years: list[int | None] = [None]
        for cut_year in metric_years[1:-1]:
            early_count = sum(1 for point in metric_points if _year_and_quarter(str(point["quarter"]))[0] < int(cut_year))
            late_count = sum(1 for point in metric_points if _year_and_quarter(str(point["quarter"]))[0] >= int(cut_year))
            if early_count >= ERA_MIN_POINTS and late_count >= ERA_MIN_POINTS:
                candidate_cut_years.append(int(cut_year))
        best_payload: dict[str, Any] | None = None
        best_objective = float("inf")
        for cut_year in candidate_cut_years:
            if cut_year is None:
                scale_maps = [{"all": float(scale)} for scale in EXACT_SCALE_GRID]
            else:
                scale_maps = [
                    {"early": float(early_scale), "late": float(late_scale)}
                    for early_scale in EXACT_SCALE_GRID
                    for late_scale in EXACT_SCALE_GRID
                ]
            for scale_map in scale_maps:
                payload = _evaluate_scaled_bootstrap_by_era(
                    target,
                    calibration_points_by_metric=calib,
                    stratification_by_metric={metric_name: {"cut_year": cut_year, "scale_by_era": scale_map}},
                    exclude_split=True,
                )
                summary = dict(payload.get("summary") or {})
                wis, gap = _summary_from_level_sets(summary)
                objective = float((wis or 0.0) + 2.0 * (gap or 1.0))
                if objective < best_objective:
                    best_objective = objective
                    best_payload = {"cut_year": cut_year, "scale_by_era": scale_map}
        selected[metric_name] = dict(best_payload or {"cut_year": None, "scale_by_era": {"all": 1.0}})
    return selected


def _exact_asymmetric_search(calibration_points_by_metric: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    for metric_name in prob.PRIMARY_METRICS:
        metric_points = list(calibration_points_by_metric.get(metric_name) or [])
        errors = np.asarray(
            [float(point["target"]) - float(point["prediction"]) for point in metric_points],
            dtype=np.float64,
        )
        center_shift = float(np.median(errors)) if errors.size else 0.0
        target = {metric_name: metric_points}
        calib = {metric_name: metric_points}
        best_payload = {"center_shift": center_shift, "lower_scale": 1.0, "upper_scale": 1.0}
        best_objective = float("inf")
        for lower_scale in ASYM_SCALE_GRID:
            for upper_scale in ASYM_SCALE_GRID:
                payload = _evaluate_asymmetric_bootstrap(
                    target,
                    calibration_points_by_metric=calib,
                    asymmetric_by_metric={
                        metric_name: {
                            "center_shift": center_shift,
                            "lower_scale": float(lower_scale),
                            "upper_scale": float(upper_scale),
                        }
                    },
                    exclude_split=True,
                )
                wis, gap = _summary_from_level_sets(dict(payload.get("summary") or {}))
                objective = float((wis or 0.0) + 2.0 * (gap or 1.0))
                if objective < best_objective:
                    best_objective = objective
                    best_payload = {
                        "center_shift": center_shift,
                        "lower_scale": float(lower_scale),
                        "upper_scale": float(upper_scale),
                    }
        selected[metric_name] = dict(best_payload)
    return selected


def _save_extension_overview(payload: dict[str, Any], path: Path) -> None:
    labels = [
        "exact-uq02",
        "exact-uq01x",
        "exact-uq01y",
        "exact-uq01z",
        "dense-uq02",
        "dense-uq03",
        "dense-uq04",
        "dense-uq05",
        "dense-plat01",
    ]
    values = [
        float(dict(payload["baseline_exact"]).get("primary_normalized_wis") or np.nan),
        float(dict(payload["exact_uq01x"]).get("primary_normalized_wis") or np.nan),
        float(dict(payload["exact_uq01y"]).get("primary_normalized_wis") or np.nan),
        float(dict(payload["exact_uq01z"]).get("primary_normalized_wis") or np.nan),
        float(dict(payload["baseline_dense"]).get("primary_normalized_wis") or np.nan),
        float(dict(payload["dense_uq03"]).get("rolling", {}).get("primary_normalized_wis") or np.nan),
        float(dict(payload["dense_uq04"]).get("rolling", {}).get("primary_normalized_wis") or np.nan),
        float(dict(payload["dense_uq05"]).get("rolling", {}).get("primary_normalized_wis") or np.nan),
        float(dict(payload["dense_plat01"]).get("rolling", {}).get("primary_normalized_wis") or np.nan),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(labels, values)
    ax.set_ylabel("Primary normalized WIS")
    ax.set_title("TR-V3 probabilistic extension comparison")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_exact_endpoint_tier_coverage_plot(payload: dict[str, Any], path: Path) -> None:
    exact_payload = dict(payload.get("exact_uq01z") or {})
    by_metric = dict(exact_payload.get("by_metric") or {})
    if not by_metric:
        suite._plot_placeholder(path, title="EXP-UQ-01Z exact coverage", body="No exact UQ-01Z payload.")
        return
    metric_names = list(prob.PRIMARY_METRICS)
    tier_names = ["overall", "exact_observed", "bridge_observed"]
    labels = []
    cov80: list[float] = []
    cov95: list[float] = []
    for metric_name in metric_names:
        metric_payload = dict(by_metric.get(metric_name) or {})
        for tier_name in tier_names:
            tier_payload = dict(metric_payload.get(tier_name) or {})
            labels.append(f"{metric_name}\n{tier_name}")
            cov80.append(float(dict(tier_payload.get("80") or {}).get("coverage")) if dict(tier_payload.get("80") or {}).get("coverage") is not None else np.nan)
            cov95.append(float(dict(tier_payload.get("95") or {}).get("coverage")) if dict(tier_payload.get("95") or {}).get("coverage") is not None else np.nan)
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.bar(x - (width / 2.0), cov80, width=width, label="Cov80")
    ax.bar(x + (width / 2.0), cov95, width=width, label="Cov95")
    ax.axhline(0.8, color="tab:blue", linestyle="--", linewidth=1.2, alpha=0.8, label="Nominal 80%")
    ax.axhline(0.95, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8, label="Nominal 95%")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("EXP-UQ-01Z-exact endpoint-by-tier coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Probabilistic Extension Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        "",
        "## Exact Interval Repair",
        "",
        f"- Baseline exact `EXP-UQ-02` primary normalized WIS: `{prob._fmt(dict(payload['baseline_exact']).get('primary_normalized_wis'))}`",
        f"- `EXP-UQ-01X-exact` primary normalized WIS: `{prob._fmt(dict(payload['exact_uq01x']).get('primary_normalized_wis'))}`",
        f"- `EXP-UQ-01X-exact` coverage gap: `{prob._fmt(dict(payload['exact_uq01x']).get('coverage_gap'))}`",
        f"- `EXP-UQ-01X-exact` lockbox primary normalized WIS: `{prob._fmt(dict(dict(payload['exact_uq01x']).get('lockbox') or {}).get('primary_normalized_wis'))}`",
        f"- Scale multipliers: `{dict(payload['exact_uq01x']).get('scale_by_metric')}`",
        f"- `EXP-UQ-01Y-exact` primary normalized WIS: `{prob._fmt(dict(payload['exact_uq01y']).get('primary_normalized_wis'))}`",
        f"- `EXP-UQ-01Y-exact` coverage gap: `{prob._fmt(dict(payload['exact_uq01y']).get('coverage_gap'))}`",
        f"- `EXP-UQ-01Y-exact` lockbox primary normalized WIS: `{prob._fmt(dict(dict(payload['exact_uq01y']).get('lockbox') or {}).get('primary_normalized_wis'))}`",
        f"- Era stratification: `{dict(payload['exact_uq01y']).get('stratification_by_metric')}`",
        f"- `EXP-UQ-01Z-exact` primary normalized WIS: `{prob._fmt(dict(payload['exact_uq01z']).get('primary_normalized_wis'))}`",
        f"- `EXP-UQ-01Z-exact` coverage gap: `{prob._fmt(dict(payload['exact_uq01z']).get('coverage_gap'))}`",
        f"- `EXP-UQ-01Z-exact` lockbox primary normalized WIS: `{prob._fmt(dict(dict(payload['exact_uq01z']).get('lockbox') or {}).get('primary_normalized_wis'))}`",
        f"- Asymmetric settings: `{dict(payload['exact_uq01z']).get('asymmetric_by_metric')}`",
        "",
        "## Dense Extensions",
        "",
        "| Experiment | Rolling primary normalized WIS | Rolling coverage gap | Lockbox primary normalized WIS |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("dense_uq03", "EXP-UQ-03-dense"),
        ("dense_uq04", "EXP-UQ-04-dense"),
        ("dense_uq05", "EXP-UQ-05"),
        ("dense_plat01", "EXP-PLAT-01"),
    ):
        row = dict(payload[key])
        rolling = dict(row.get("rolling") or {})
        lockbox = dict(row.get("lockbox") or {})
        lines.append(
            f"| {label} | {prob._fmt(rolling.get('primary_normalized_wis'))} | {prob._fmt(rolling.get('coverage_gap'))} | {prob._fmt(lockbox.get('primary_normalized_wis'))} |"
    )
    lines.extend(
        [
            "",
            "## Coverage Figure",
            "",
            f"- Endpoint-by-tier exact coverage: `{dict(payload.get('artifacts') or {}).get('exact_uq01z_coverage_plot', '')}`",
            "",
            "## Dense Risk Checks",
            "",
            "| Experiment | Brier | Prevalence Brier | Persistence Brier |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, label in (("dense_uq03", "EXP-UQ-03-dense"), ("dense_uq04", "EXP-UQ-04-dense")):
        risk = dict(dict(payload[key]).get("risk") or {})
        lines.append(
            f"| {label} | {prob._fmt(risk.get('candidate_brier'))} | {prob._fmt(risk.get('prevalence_brier'))} | {prob._fmt(risk.get('persistence_brier'))} |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Exact: `{payload['decisions']['exact_uq01x']}`",
            f"- Exact era-stratified: `{payload['decisions']['exact_uq01y']}`",
            f"- Exact asymmetric: `{payload['decisions']['exact_uq01z']}`",
            f"- Dense jump: `{payload['decisions']['dense_uq03']}`",
            f"- Dense exogenous jump: `{payload['decisions']['dense_uq04']}`",
            f"- Shared regime: `{payload['decisions']['dense_uq05']}`",
            f"- Plateau sidecar: `{payload['decisions']['dense_plat01']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def run_tr_v3_probabilistic_extension_batch(
    *,
    run_id: str,
    archive_run_id: str | None = None,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
    lockbox_holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    archive_run = str(archive_run_id or suite._latest_standard_archive_run())
    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")
    holdout_years = sorted(set(int(year) for year in (lockbox_holdout_years or [2025])))
    exact_payload = prob._run_contract_payload(
        archive_run_id=archive_run,
        contract_name="exact_only",
        experiment_id=EXACT_EXPERIMENT_ID,
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = prob._run_contract_payload(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_id=DENSE_EXPERIMENT_ID,
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    exact_result = prob._suite_result_map(exact_payload)[EXACT_EXPERIMENT_ID]
    dense_result = prob._suite_result_map(dense_payload)[DENSE_EXPERIMENT_ID]
    exact_points = prob._points_by_metric(list(exact_result.get("quarterly_rows") or []), allowed_tiers={"exact_observed"})
    dense_points = prob._points_by_metric(list(dense_result.get("quarterly_rows") or []), allowed_tiers={"exact_observed", "bridge_observed"})
    exact_fixed = prob._fixed_holdout_result(archive_run_id=archive_run, contract_name="exact_only", result=exact_result, holdout_years=holdout_years)
    dense_fixed = prob._fixed_holdout_result(archive_run_id=archive_run, contract_name="purged_dense", result=dense_result, holdout_years=holdout_years)
    exact_lockbox_points = prob._fixed_holdout_points_by_metric(exact_fixed, allowed_tiers={"exact_observed"})
    dense_lockbox_points = prob._fixed_holdout_points_by_metric(dense_fixed, allowed_tiers={"exact_observed", "bridge_observed"})

    baseline_exact = prob._evaluate_interval_method(exact_points, calibration_points_by_metric=exact_points, method="bootstrap", exclude_split=True)
    baseline_dense = prob._evaluate_interval_method(dense_points, calibration_points_by_metric=dense_points, method="bootstrap", exclude_split=True)

    scale_by_metric = _exact_scale_search(exact_points)
    exact_uq01x = _evaluate_scaled_bootstrap(exact_points, calibration_points_by_metric=exact_points, scale_by_metric=scale_by_metric, exclude_split=True)
    exact_uq01x["scale_by_metric"] = scale_by_metric
    exact_uq01x["lockbox"] = _evaluate_scaled_bootstrap(exact_lockbox_points, calibration_points_by_metric=exact_points, scale_by_metric=scale_by_metric, exclude_split=False)
    stratification_by_metric = _exact_era_scale_search(exact_points)
    exact_uq01y = _evaluate_scaled_bootstrap_by_era(
        exact_points,
        calibration_points_by_metric=exact_points,
        stratification_by_metric=stratification_by_metric,
        exclude_split=True,
    )
    exact_uq01y["stratification_by_metric"] = stratification_by_metric
    exact_uq01y["lockbox"] = _evaluate_scaled_bootstrap_by_era(
        exact_lockbox_points,
        calibration_points_by_metric=exact_points,
        stratification_by_metric=stratification_by_metric,
        exclude_split=False,
    )
    asymmetric_by_metric = _exact_asymmetric_search(exact_points)
    exact_uq01z = _evaluate_asymmetric_bootstrap(
        exact_points,
        calibration_points_by_metric=exact_points,
        asymmetric_by_metric=asymmetric_by_metric,
        exclude_split=True,
    )
    exact_uq01z["asymmetric_by_metric"] = asymmetric_by_metric
    exact_uq01z["lockbox"] = _evaluate_asymmetric_bootstrap(
        exact_lockbox_points,
        calibration_points_by_metric=exact_points,
        asymmetric_by_metric=asymmetric_by_metric,
        exclude_split=False,
    )

    dense_point_rows = prob._risk_point_rows(dense_result, allowed_tiers={"exact_observed", "bridge_observed"})
    dense_fixed_point_rows = _fixed_holdout_risk_points(dense_fixed, allowed_tiers={"exact_observed", "bridge_observed"})
    dense_uq03_candidates: list[dict[str, Any]] = []
    dense_uq04_candidates: list[dict[str, Any]] = []
    for jump_quantile in JUMP_QUANTILES:
        residual_rows = _build_labeled_risk_rows(dense_point_rows, threshold_quantile=jump_quantile, feature_builder=prob._risk_features)
        residual_prob_map = _probability_map(residual_rows, key_mode="point")
        rolling = _evaluate_mixture_sidecar(dense_points, calibration_points_by_metric=dense_points, probability_map=residual_prob_map, jump_quantile=jump_quantile, exclude_split=True, key_mode="point")
        residual_predictor = _fit_static_probability_model(residual_rows, feature_builder=prob._risk_features)
        lockbox_prob_map = {(str(point["quarter"]), str(point["metric_name"]), str(point["tier"])): float(residual_predictor(point)) for point in dense_fixed_point_rows}
        lockbox = _evaluate_mixture_sidecar(dense_lockbox_points, calibration_points_by_metric=dense_points, probability_map=lockbox_prob_map, jump_quantile=jump_quantile, exclude_split=False, key_mode="point")
        dense_uq03_candidates.append({"jump_quantile": jump_quantile, "risk": {
            "candidate_brier": prob._brier(residual_rows, "candidate_probability"),
            "prevalence_brier": prob._brier(residual_rows, "prevalence_probability"),
            "persistence_brier": prob._brier(residual_rows, "persistence_probability"),
        }, "rolling": rolling, "lockbox": lockbox})

        exogenous_rows = _build_labeled_risk_rows(dense_point_rows, threshold_quantile=jump_quantile, feature_builder=_covid_features)
        exogenous_prob_map = _probability_map(exogenous_rows, key_mode="point")
        exo_rolling = _evaluate_mixture_sidecar(dense_points, calibration_points_by_metric=dense_points, probability_map=exogenous_prob_map, jump_quantile=jump_quantile, exclude_split=True, key_mode="point")
        exogenous_predictor = _fit_static_probability_model(exogenous_rows, feature_builder=_covid_features)
        exo_lockbox_map = {(str(point["quarter"]), str(point["metric_name"]), str(point["tier"])): float(exogenous_predictor(point)) for point in dense_fixed_point_rows}
        exo_lockbox = _evaluate_mixture_sidecar(dense_lockbox_points, calibration_points_by_metric=dense_points, probability_map=exo_lockbox_map, jump_quantile=jump_quantile, exclude_split=False, key_mode="point")
        dense_uq04_candidates.append({"jump_quantile": jump_quantile, "risk": {
            "candidate_brier": prob._brier(exogenous_rows, "candidate_probability"),
            "prevalence_brier": prob._brier(exogenous_rows, "prevalence_probability"),
            "persistence_brier": prob._brier(exogenous_rows, "persistence_probability"),
        }, "rolling": exo_rolling, "lockbox": exo_lockbox})

    dense_uq03 = min(dense_uq03_candidates, key=lambda row: float(dict(row["rolling"]).get("primary_normalized_wis") or float("inf")))
    dense_uq04 = min(dense_uq04_candidates, key=lambda row: float(dict(row["rolling"]).get("primary_normalized_wis") or float("inf")))

    dense_uq05 = _shared_regime_payload(
        result=dense_result,
        fixed_result=dense_fixed,
        calibration_points_by_metric=dense_points,
        target_points_by_metric=dense_points,
        lockbox_points_by_metric=dense_lockbox_points,
    )
    plateau_probability_map, plateau_summary, plateau_rows_by_metric = _plateau_probability_map(dense_points)
    lockbox_plateau_probability_map = _lockbox_plateau_probability_map(
        dense_lockbox_points,
        plateau_summary=plateau_summary,
        rows_by_metric=plateau_rows_by_metric,
    )
    dense_plat01 = {
        "rolling": _evaluate_plateau_sidecar(
            dense_points,
            calibration_points_by_metric=dense_points,
            probability_map=plateau_probability_map,
            calibration_probability_map=plateau_probability_map,
            exclude_split=True,
        ),
        "lockbox": _evaluate_plateau_sidecar(
            dense_lockbox_points,
            calibration_points_by_metric=dense_points,
            probability_map=lockbox_plateau_probability_map,
            calibration_probability_map=plateau_probability_map,
            exclude_split=False,
        ),
        "plateau_model": plateau_summary,
    }

    decisions = {
        "exact_uq01x": "keep" if (float(exact_uq01x["coverage_gap"] or 1.0) < float(baseline_exact["coverage_gap"] or 1.0) and float(dict(exact_uq01x["lockbox"]).get("primary_normalized_wis") or float("inf")) <= float(dict(baseline_exact).get("primary_normalized_wis") or float("inf")) + 0.25) else "revert",
        "exact_uq01y": "keep" if (
            float(exact_uq01y["coverage_gap"] or float("inf")) <= float(exact_uq01x["coverage_gap"] or float("inf")) + 0.01
            and float(dict(exact_uq01y["lockbox"]).get("primary_normalized_wis") or float("inf")) < float(dict(exact_uq01x["lockbox"]).get("primary_normalized_wis") or float("inf"))
            and float(exact_uq01y["primary_normalized_wis"] or float("inf")) <= float(exact_uq01x["primary_normalized_wis"] or float("inf")) + 0.02
        ) else "revert",
        "exact_uq01z": "keep" if (
            float(exact_uq01z["coverage_gap"] or float("inf")) <= float(exact_uq01x["coverage_gap"] or float("inf")) + 0.01
            and float(dict(exact_uq01z["lockbox"]).get("primary_normalized_wis") or float("inf")) < float(dict(exact_uq01x["lockbox"]).get("primary_normalized_wis") or float("inf"))
            and float(exact_uq01z["primary_normalized_wis"] or float("inf")) <= float(exact_uq01x["primary_normalized_wis"] or float("inf")) + 0.02
        ) else "revert",
        "dense_uq03": "keep" if float(dict(dense_uq03["rolling"]).get("primary_normalized_wis") or float("inf")) < float(baseline_dense["primary_normalized_wis"] or float("inf")) and float(dict(dense_uq03["lockbox"]).get("primary_normalized_wis") or float("inf")) <= float(dict(baseline_dense).get("primary_normalized_wis") or float("inf")) + 0.05 else "revert",
        "dense_uq04": "keep" if float(dict(dense_uq04["risk"]).get("candidate_brier") or float("inf")) < min(float(dict(dense_uq04["risk"]).get("prevalence_brier") or float("inf")), float(dict(dense_uq04["risk"]).get("persistence_brier") or float("inf"))) and float(dict(dense_uq04["rolling"]).get("primary_normalized_wis") or float("inf")) <= float(dict(dense_uq03["rolling"]).get("primary_normalized_wis") or float("inf")) else "revert",
        "dense_uq05": "keep" if float(dict(dense_uq05["rolling"]).get("primary_normalized_wis") or float("inf")) < min(float(baseline_dense["primary_normalized_wis"] or float("inf")), float(dict(dense_uq03["rolling"]).get("primary_normalized_wis") or float("inf"))) else "revert",
        "dense_plat01": "keep" if float(dict(dict(dense_plat01["rolling"]).get("plateau_summary") or {}).get("95", {}).get("coverage") or 0.0) > float(dict(dict(dense_uq03["rolling"]).get("high_risk_summary") or {}).get("95", {}).get("coverage") or 0.0) and float(dict(dense_plat01["rolling"]).get("primary_normalized_wis") or float("inf")) <= float(baseline_dense["primary_normalized_wis"] or float("inf")) + 0.05 else "revert",
    }

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "baseline_exact": baseline_exact,
        "baseline_dense": baseline_dense,
        "exact_uq01x": exact_uq01x,
        "exact_uq01y": exact_uq01y,
        "exact_uq01z": exact_uq01z,
        "dense_uq03": dense_uq03,
        "dense_uq04": dense_uq04,
        "dense_uq05": dense_uq05,
        "dense_plat01": dense_plat01,
        "decisions": decisions,
    }
    overview_path = analysis_dir / "probabilistic_extension_overview.png"
    _save_extension_overview(report_payload, overview_path)
    exact_coverage_path = analysis_dir / "EXP-UQ-01Z-exact_endpoint_tier_coverage.png"
    _save_exact_endpoint_tier_coverage_plot(report_payload, exact_coverage_path)
    report_payload["artifacts"] = {
        "overview_graph": overview_path.name,
        "exact_uq01z_coverage_plot": exact_coverage_path.name,
    }
    write_json(analysis_dir / "tr_v3_probabilistic_extension_batch_report.json", _json_ready(report_payload))
    (analysis_dir / "tr_v3_probabilistic_extension_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded TR-V3 probabilistic extension experiments.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
    args = parser.parse_args()
    run_tr_v3_probabilistic_extension_batch(
        run_id=str(args.run_id),
        archive_run_id=args.archive_run_id,
        quarterly_start_year=int(args.quarterly_start_year),
        quarterly_end_year=int(args.quarterly_end_year),
        quarterly_min_train_years=int(args.quarterly_min_train_years),
        annual_start_year=int(args.annual_start_year),
        annual_end_year=int(args.annual_end_year),
        annual_min_train_years=int(args.annual_min_train_years),
        horizon_years=int(args.horizon_years),
        lockbox_holdout_years=list(args.lockbox_holdout_years),
    )


if __name__ == "__main__":
    main()
