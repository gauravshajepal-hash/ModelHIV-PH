from __future__ import annotations

import argparse
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_eval_hardening_batch as hardening
from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, build_annual_anchor_rows, quarter_sort_key
from epigraph_ph.runtime import ensure_dir, write_json


CALIBRATION_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.95)
RISK_QUANTILES: tuple[float, ...] = (0.8, 0.9)
EXACT_EXPERIMENT_ID = "EXP-R10-M1-F1-C1"
DENSE_EXPERIMENT_ID = "EXP-R10-DENSE-M1-C1-H1"
MIN_POOL_POINTS = 6
BOOTSTRAP_DRAWS = 512
BOOTSTRAP_BLOCK_LENGTH = 4
MIN_RISK_HISTORY = 12
MIN_RISK_EVENTS = 3


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _stable_seed(*parts: object) -> int:
    joined = "|".join(str(part) for part in parts)
    return int(zlib.crc32(joined.encode("utf-8")) & 0xFFFFFFFF)


def _metric_tier(row: dict[str, Any], metric_name: str) -> str:
    metric_tiers = row.get("metric_tiers")
    if isinstance(metric_tiers, dict) and metric_name in metric_tiers:
        return str(metric_tiers[metric_name])
    value = row.get(f"{metric_name}_tier")
    if value:
        return str(value)
    return "exact_observed"


def _metric_scale(points: list[dict[str, Any]]) -> float:
    targets = [abs(float(point["target"])) for point in points if point.get("target") is not None]
    if not targets:
        return 1.0
    scale = float(np.median(np.asarray(targets, dtype=np.float64)))
    return max(scale, 1.0)


def _interval_score(*, target: float, lower_bound: float, upper_bound: float, alpha: float) -> float:
    width = max(float(upper_bound) - float(lower_bound), 0.0)
    if target < lower_bound:
        return width + (2.0 / float(alpha)) * (float(lower_bound) - float(target))
    if target > upper_bound:
        return width + (2.0 / float(alpha)) * (float(target) - float(upper_bound))
    return width


def _run_contract_payload(
    *,
    archive_run_id: str,
    contract_name: str,
    experiment_id: str,
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    return hardening._run_selected_suite_contract(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        experiment_ids=[experiment_id],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )


def _points_by_metric(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
) -> dict[str, list[dict[str, Any]]]:
    return hardening._split_points_by_metric(quarterly_rows, allowed_tiers=allowed_tiers)


def _fixed_holdout_points_by_metric(
    fixed_result: dict[str, Any],
    *,
    allowed_tiers: set[str],
) -> dict[str, list[dict[str, Any]]]:
    points = {metric_name: [] for metric_name in PRIMARY_METRICS}
    targets = list(fixed_result.get("holdout_target_rows") or [])
    predictions = list(fixed_result.get("prediction_rows") or [])
    for target_row, prediction_row in zip(targets, predictions, strict=False):
        for metric_name in PRIMARY_METRICS:
            tier_name = _metric_tier(target_row, metric_name)
            if tier_name not in allowed_tiers:
                continue
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            points[metric_name].append(
                {
                    "split_idx": None,
                    "quarter": str(target_row["quarter"]),
                    "tier": tier_name,
                    "target": float(target_value),
                    "prediction": float(prediction_value),
                    "residual": float(prediction_value) - float(target_value),
                }
            )
    return points


def _pool_points(
    calibration_points: list[dict[str, Any]],
    *,
    split_idx: int | None,
    tier_name: str,
    exclude_split: bool,
    min_points: int = MIN_POOL_POINTS,
) -> dict[str, Any] | None:
    def _filter(points: list[dict[str, Any]], *, tier: str | None) -> list[dict[str, Any]]:
        rows = []
        for point in points:
            if exclude_split and split_idx is not None and point.get("split_idx") == split_idx:
                continue
            if tier is not None and str(point["tier"]) != tier:
                continue
            rows.append(point)
        rows.sort(key=lambda row: (quarter_sort_key(str(row["quarter"])), int(row["split_idx"] or 0)))
        return rows

    tier_points = _filter(calibration_points, tier=None if tier_name == "overall" else tier_name)
    used_fallback = False
    if len(tier_points) < min_points and tier_name != "overall":
        tier_points = _filter(calibration_points, tier=None)
        used_fallback = True
    if len(tier_points) < min_points:
        return None
    return {
        "points": tier_points,
        "used_fallback": used_fallback,
        "pool_size": int(len(tier_points)),
    }


def _moving_block_draws(
    residuals: list[float],
    *,
    seed: int,
    block_length: int = BOOTSTRAP_BLOCK_LENGTH,
    draws: int = BOOTSTRAP_DRAWS,
) -> np.ndarray:
    arr = np.asarray(list(residuals), dtype=np.float64)
    if arr.size <= 0:
        return np.asarray([], dtype=np.float64)
    if arr.size == 1:
        return np.repeat(arr, draws)
    block = min(int(block_length), int(arr.size))
    starts = np.arange(0, int(arr.size) - block + 1, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    sampled: list[float] = []
    while len(sampled) < draws:
        start = int(rng.choice(starts))
        sampled.extend(float(value) for value in arr[start : start + block])
    return np.asarray(sampled[:draws], dtype=np.float64)


def _distribution_from_pool(
    pool: dict[str, Any],
    *,
    method: str,
    seed: int,
) -> np.ndarray:
    residuals = [float(point["residual"]) for point in list(pool["points"])]
    if method == "conformal":
        return np.asarray(residuals, dtype=np.float64)
    if method == "bootstrap":
        return _moving_block_draws(residuals, seed=seed)
    raise ValueError(f"Unsupported method: {method}")


def _aggregate_level_rows(level_rows: list[dict[str, Any]], *, level: float) -> dict[str, Any]:
    if not level_rows:
        return {
            "nominal": float(level),
            "coverage": None,
            "count": 0,
            "mean_width": None,
            "mean_normalized_width": None,
            "mean_interval_score": None,
            "mean_normalized_wis": None,
            "fallback_count": 0,
        }
    return {
        "nominal": float(level),
        "coverage": float(np.mean(np.asarray([float(row["covered"]) for row in level_rows], dtype=np.float64))),
        "count": int(len(level_rows)),
        "mean_width": float(np.mean(np.asarray([float(row["width"]) for row in level_rows], dtype=np.float64))),
        "mean_normalized_width": float(np.mean(np.asarray([float(row["normalized_width"]) for row in level_rows], dtype=np.float64))),
        "mean_interval_score": float(np.mean(np.asarray([float(row["interval_score"]) for row in level_rows], dtype=np.float64))),
        "mean_normalized_wis": float(np.mean(np.asarray([float(row["normalized_interval_score"]) for row in level_rows], dtype=np.float64))),
        "fallback_count": int(sum(int(row["used_fallback"]) for row in level_rows)),
    }


def _evaluate_interval_method(
    target_points_by_metric: dict[str, list[dict[str, Any]]],
    *,
    calibration_points_by_metric: dict[str, list[dict[str, Any]]],
    method: str,
    levels: tuple[float, ...] = CALIBRATION_LEVELS,
    exclude_split: bool,
) -> dict[str, Any]:
    by_metric: dict[str, Any] = {}
    overall_by_level: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in levels}
    for metric_name in PRIMARY_METRICS:
        target_points = list(target_points_by_metric.get(metric_name) or [])
        calibration_points = list(calibration_points_by_metric.get(metric_name) or [])
        scale = _metric_scale(calibration_points or target_points)
        tier_payload: dict[str, Any] = {}
        cache: dict[tuple[str, int | None, str], tuple[np.ndarray, bool, int]] = {}
        for tier_name in ("overall", "exact_observed", "bridge_observed"):
            if tier_name == "overall":
                tier_targets = list(target_points)
            else:
                tier_targets = [point for point in target_points if str(point["tier"]) == tier_name]
            level_rows: dict[str, list[dict[str, Any]]] = {f"{int(level * 100)}": [] for level in levels}
            for point in tier_targets:
                cache_key = (metric_name, point.get("split_idx"), tier_name)
                if cache_key not in cache:
                    pool = _pool_points(
                        calibration_points,
                        split_idx=point.get("split_idx"),
                        tier_name=tier_name,
                        exclude_split=exclude_split,
                    )
                    if pool is None:
                        cache[cache_key] = (np.asarray([], dtype=np.float64), False, 0)
                    else:
                        distribution = _distribution_from_pool(
                            pool,
                            method=method,
                            seed=_stable_seed(method, metric_name, tier_name, point.get("split_idx"), point["quarter"]),
                        )
                        cache[cache_key] = (distribution, bool(pool["used_fallback"]), int(pool["pool_size"]))
                distribution, used_fallback, pool_size = cache[cache_key]
                if distribution.size < MIN_POOL_POINTS:
                    continue
                for level in levels:
                    alpha = 1.0 - float(level)
                    lower_offset = float(np.quantile(distribution, alpha / 2.0))
                    upper_offset = float(np.quantile(distribution, 1.0 - (alpha / 2.0)))
                    lower_bound = float(point["prediction"]) + lower_offset
                    upper_bound = float(point["prediction"]) + upper_offset
                    covered = 1.0 if lower_bound <= float(point["target"]) <= upper_bound else 0.0
                    width = float(upper_bound - lower_bound)
                    interval_score = _interval_score(
                        target=float(point["target"]),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        alpha=alpha,
                    )
                    level_rows[f"{int(level * 100)}"].append(
                        {
                            "covered": covered,
                            "width": width,
                            "normalized_width": width / float(scale),
                            "interval_score": interval_score,
                            "normalized_interval_score": interval_score / float(scale),
                            "used_fallback": 1 if used_fallback else 0,
                            "pool_size": int(pool_size),
                        }
                    )
            tier_payload[tier_name] = {
                level_key: _aggregate_level_rows(rows, level=float(level_key) / 100.0)
                for level_key, rows in level_rows.items()
            }
            if tier_name == "overall":
                for level_key, rows in level_rows.items():
                    overall_by_level[level_key].extend(rows)
        by_metric[metric_name] = {
            "scale": float(scale),
            **tier_payload,
        }
    summary = {
        level_key: _aggregate_level_rows(rows, level=float(level_key) / 100.0)
        for level_key, rows in overall_by_level.items()
    }
    primary_keys = [key for key in ("80", "95") if summary.get(key, {}).get("mean_normalized_wis") is not None]
    coverage_keys = [key for key in ("80", "95") if summary.get(key, {}).get("coverage") is not None]
    primary_wis = (
        float(np.mean(np.asarray([float(summary[key]["mean_normalized_wis"]) for key in primary_keys], dtype=np.float64)))
        if primary_keys
        else None
    )
    coverage_gap = (
        float(
            np.mean(
                np.asarray(
                    [abs(float(summary[key]["coverage"]) - (float(key) / 100.0)) for key in coverage_keys],
                    dtype=np.float64,
                )
            )
        )
        if coverage_keys
        else None
    )
    return {
        "method": str(method),
        "levels": [float(level) for level in levels],
        "by_metric": by_metric,
        "summary": summary,
        "primary_normalized_wis": primary_wis,
        "coverage_gap": coverage_gap,
    }


def _fixed_holdout_result(
    *,
    archive_run_id: str,
    contract_name: str,
    result: dict[str, Any],
    holdout_years: list[int],
) -> dict[str, Any]:
    annual_rows = build_annual_anchor_rows(archive_run_id)
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    return publish._evaluate_fixed_holdout_experiment(
        spec_map[str(result["experiment_id"])],
        observation_rows=publish._observation_rows_for_lockbox(archive_run_id, contract_name=contract_name, holdout_years=holdout_years),
        annual_rows=annual_rows,
        holdout_years=holdout_years,
        scoring_tiers=publish._scoring_tiers(contract_name),
        frozen_config=dict(result.get("best_candidate") or {}),
    )


def _risk_point_rows(
    result: dict[str, Any],
    *,
    allowed_tiers: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quarterly_rows = list(result.get("quarterly_rows") or [])
    raw_rows: list[dict[str, Any]] = []
    metric_scales: dict[str, float] = {metric_name: 1.0 for metric_name in PRIMARY_METRICS}
    metric_support_max: dict[str, int] = {metric_name: 1 for metric_name in PRIMARY_METRICS}
    for split_idx, split in enumerate(quarterly_rows):
        endpoint_audit = dict(split.get("endpoint_audit") or {})
        train_support = dict(endpoint_audit.get("train_support_counts") or {})
        targets = list(split.get("holdout_target_rows") or [])
        predictions = list(split.get("candidate_prediction_rows") or [])
        previous_prediction: dict[str, float] = {}
        for target_row, prediction_row in zip(targets, predictions, strict=False):
            quarter = str(target_row["quarter"])
            quarter_num = int(str(quarter).split("Q")[-1])
            for metric_name in PRIMARY_METRICS:
                tier_name = _metric_tier(target_row, metric_name)
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
                        "split_idx": int(split_idx),
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
    for metric_name in PRIMARY_METRICS:
        metric_targets = [abs(float(row["target"])) for row in raw_rows if str(row["metric_name"]) == metric_name]
        if metric_targets:
            metric_scales[metric_name] = max(float(np.median(np.asarray(metric_targets, dtype=np.float64))), 1.0)
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
    rows.sort(key=lambda row: (quarter_sort_key(str(row["quarter"])), str(row["metric_name"]), str(row["tier"]), int(row["split_idx"])))
    return rows


def _risk_features(row: dict[str, Any]) -> list[float]:
    metric_name = str(row["metric_name"])
    return [
        1.0,
        1.0 if metric_name == "diagnosed_plhiv" else 0.0,
        1.0 if metric_name == "alive_on_art" else 0.0,
        1.0 if str(row["tier"]) == "bridge_observed" else 0.0,
        float(row["q_sin"]),
        float(row["q_cos"]),
        float(row["prediction_norm"]),
        float(row["pred_delta_norm"]),
        float(row["support_norm"]),
    ]


def _brier(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    values = np.asarray([(float(row[key]) - float(row["label"])) ** 2 for row in rows], dtype=np.float64)
    return float(np.mean(values))


def _recall_at_k(rows: list[dict[str, Any]], key: str) -> float | None:
    event_count = int(sum(int(row["label"]) for row in rows))
    if event_count <= 0:
        return None
    ranked = sorted(list(rows), key=lambda row: float(row[key]), reverse=True)
    top = ranked[:event_count]
    return float(sum(int(row["label"]) for row in top)) / float(event_count)


def _risk_payload(
    result: dict[str, Any],
    *,
    contract_name: str,
    allowed_tiers: set[str],
) -> dict[str, Any]:
    points = _risk_point_rows(result, allowed_tiers=allowed_tiers)
    by_quantile: dict[str, Any] = {}
    for threshold_quantile in RISK_QUANTILES:
        rows: list[dict[str, Any]] = []
        historical_points: list[dict[str, Any]] = []
        labeled_history: list[dict[str, Any]] = []
        prev_label_by_stream: dict[tuple[str, str], float] = {}
        for point in points:
            if len(historical_points) < MIN_RISK_HISTORY:
                historical_points.append(dict(point))
                continue
            abs_history = np.asarray([abs(float(entry["residual"])) for entry in historical_points], dtype=np.float64)
            threshold = float(np.quantile(abs_history, threshold_quantile))
            label = 1.0 if abs(float(point["residual"])) > threshold else 0.0
            history_labels = np.asarray([float(entry["label"]) for entry in labeled_history], dtype=np.float64)
            prevalence = float(np.mean(history_labels)) if history_labels.size else 0.0
            stream_key = (str(point["metric_name"]), str(point["tier"]))
            persistence = float(prev_label_by_stream.get(stream_key, prevalence))
            features = np.asarray(_risk_features(point), dtype=np.float64)
            event_count = int(np.sum(history_labels))
            if len(labeled_history) >= MIN_RISK_HISTORY and event_count >= MIN_RISK_EVENTS:
                x = np.asarray([entry["features"] for entry in labeled_history], dtype=np.float64)
                y = history_labels
                penalty = np.eye(x.shape[1], dtype=np.float64) * 0.5
                penalty[0, 0] = 0.0
                beta = np.linalg.solve(x.T @ x + penalty, x.T @ y)
                candidate = float(np.clip(float(np.dot(features, beta)), 0.0, 1.0))
            else:
                candidate = prevalence
            row = {
                "quarter": str(point["quarter"]),
                "metric_name": str(point["metric_name"]),
                "tier": str(point["tier"]),
                "label": float(label),
                "candidate_probability": float(candidate),
                "prevalence_probability": float(prevalence),
                "persistence_probability": float(persistence),
            }
            rows.append(row)
            historical_points.append(dict(point))
            labeled_history.append({**point, "features": _risk_features(point), "label": float(label)})
            prev_label_by_stream[(str(point["metric_name"]), str(point["tier"]))] = float(label)
        event_count = int(sum(int(row["label"]) for row in rows))
        candidate_brier = _brier(rows, "candidate_probability")
        prevalence_brier = _brier(rows, "prevalence_probability")
        persistence_brier = _brier(rows, "persistence_probability")
        if len(rows) < MIN_RISK_HISTORY or event_count < MIN_RISK_EVENTS:
            status = "insufficient_events"
        elif candidate_brier is not None and prevalence_brier is not None and persistence_brier is not None and candidate_brier + 1e-6 < min(prevalence_brier, persistence_brier):
            status = "incremental_risk_signal"
        else:
            status = "no_incremental_risk_signal"
        by_quantile[f"q{int(threshold_quantile * 100)}"] = {
            "status": status,
            "event_count": int(event_count),
            "rows": rows,
            "metrics": {
                "candidate_brier": candidate_brier,
                "prevalence_brier": prevalence_brier,
                "persistence_brier": persistence_brier,
                "candidate_recall_at_k": _recall_at_k(rows, "candidate_probability"),
                "persistence_recall_at_k": _recall_at_k(rows, "persistence_probability"),
            },
        }
    keep = any(str(payload["status"]) == "incremental_risk_signal" for payload in by_quantile.values())
    return {
        "contract": str(contract_name),
        "by_quantile": by_quantile,
        "decision": {
            "status": "keep" if keep else "revert",
            "why": "Candidate risk model beat both prevalence and persistence on at least one threshold."
            if keep
            else "Candidate risk model did not beat both prevalence and persistence baselines.",
        },
    }


def _fmt(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _interval_method_row(contract_payload: dict[str, Any], *, method_name: str) -> dict[str, Any]:
    summary = dict(contract_payload.get("summary") or {})
    lockbox = dict(contract_payload.get("lockbox") or {})
    return {
        "method": method_name,
        "primary_normalized_wis": contract_payload.get("primary_normalized_wis"),
        "coverage_gap": contract_payload.get("coverage_gap"),
        "wis80": dict(summary.get("80") or {}).get("mean_normalized_wis"),
        "wis95": dict(summary.get("95") or {}).get("mean_normalized_wis"),
        "coverage80": dict(summary.get("80") or {}).get("coverage"),
        "coverage95": dict(summary.get("95") or {}).get("coverage"),
        "lockbox_primary_normalized_wis": lockbox.get("primary_normalized_wis"),
    }


def _interval_decision(conformal_payload: dict[str, Any], bootstrap_payload: dict[str, Any]) -> dict[str, Any]:
    conformal_score = conformal_payload.get("primary_normalized_wis")
    bootstrap_score = bootstrap_payload.get("primary_normalized_wis")
    conformal_gap = conformal_payload.get("coverage_gap")
    bootstrap_gap = bootstrap_payload.get("coverage_gap")
    conformal_lockbox = dict(bootstrap_payload.get("lockbox_baselines") or {}).get("conformal_primary_normalized_wis")
    bootstrap_lockbox = dict(bootstrap_payload.get("lockbox") or {}).get("summary", {}).get("primary_normalized_wis")
    if (
        bootstrap_score is not None
        and conformal_score is not None
        and bootstrap_score + 1e-9 < conformal_score
        and (bootstrap_gap is None or conformal_gap is None or bootstrap_gap <= conformal_gap + 0.03)
        and (bootstrap_lockbox is None or conformal_lockbox is None or bootstrap_lockbox <= conformal_lockbox + 0.02)
    ):
        return {
            "winner": "EXP-UQ-02",
            "status": "keep_bootstrap",
            "why": "Moving-block bootstrap improved normalized WIS without materially worsening coverage or lockbox calibration.",
        }
    return {
        "winner": "EXP-UQ-01",
        "status": "keep_conformal",
        "why": "Split-conformal intervals remain the safer primary interval layer under the current coverage and lockbox criteria.",
    }


def _save_coverage_pair(contract_name: str, conformal: dict[str, Any], bootstrap: dict[str, Any], path: Path) -> None:
    levels = [float(level) for level in list(conformal.get("levels") or [])]
    if not levels:
        suite._plot_placeholder(path, title=contract_name, body="No interval levels.")
        return
    fig, axes = plt.subplots(1, len(PRIMARY_METRICS), figsize=(5 * len(PRIMARY_METRICS), 4), constrained_layout=True)
    axes_list = [axes] if not isinstance(axes, np.ndarray) else list(np.asarray(axes).reshape(-1))
    for ax, metric_name in zip(axes_list, PRIMARY_METRICS, strict=False):
        conformal_metric = dict(dict(conformal.get("by_metric") or {}).get(metric_name) or {}).get("overall") or {}
        bootstrap_metric = dict(dict(bootstrap.get("by_metric") or {}).get(metric_name) or {}).get("overall") or {}
        conformal_values = []
        bootstrap_values = []
        for level in levels:
            level_key = f"{int(level * 100)}"
            conformal_values.append(float(dict(conformal_metric.get(level_key) or {}).get("coverage") or np.nan))
            bootstrap_values.append(float(dict(bootstrap_metric.get(level_key) or {}).get("coverage") or np.nan))
        ax.plot(levels, levels, linestyle="--", color="black", label="nominal")
        ax.plot(levels, conformal_values, marker="o", color="#1f77b4", label="conformal")
        ax.plot(levels, bootstrap_values, marker="s", color="#d95f02", label="bootstrap")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(min(levels) - 0.02, max(levels) + 0.02)
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{contract_name} probabilistic interval coverage")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_risk_graph(payload: dict[str, Any], path: Path, *, title: str) -> None:
    by_quantile = dict(payload.get("by_quantile") or {})
    quantiles = list(by_quantile.keys())
    if not quantiles:
        suite._plot_placeholder(path, title=title, body="No risk rows.")
        return
    candidate = [float(dict(dict(by_quantile[key]).get("metrics") or {}).get("candidate_brier") or 0.0) for key in quantiles]
    prevalence = [float(dict(dict(by_quantile[key]).get("metrics") or {}).get("prevalence_brier") or 0.0) for key in quantiles]
    persistence = [float(dict(dict(by_quantile[key]).get("metrics") or {}).get("persistence_brier") or 0.0) for key in quantiles]
    x = np.arange(len(quantiles))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.22, prevalence, width=0.22, label="prevalence")
    ax.bar(x, persistence, width=0.22, label="persistence")
    ax.bar(x + 0.22, candidate, width=0.22, label="candidate")
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.set_ylabel("Brier score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _build_contract_payload(
    *,
    archive_run_id: str,
    contract_name: str,
    result: dict[str, Any],
    holdout_years: list[int],
) -> dict[str, Any]:
    allowed_tiers = publish._scoring_tiers(contract_name)
    calibration_points = _points_by_metric(list(result.get("quarterly_rows") or []), allowed_tiers=allowed_tiers)
    conformal = _evaluate_interval_method(
        calibration_points,
        calibration_points_by_metric=calibration_points,
        method="conformal",
        exclude_split=True,
    )
    bootstrap = _evaluate_interval_method(
        calibration_points,
        calibration_points_by_metric=calibration_points,
        method="bootstrap",
        exclude_split=True,
    )
    fixed_result = _fixed_holdout_result(
        archive_run_id=archive_run_id,
        contract_name=contract_name,
        result=result,
        holdout_years=holdout_years,
    )
    lockbox_points = _fixed_holdout_points_by_metric(fixed_result, allowed_tiers=allowed_tiers)
    lockbox_conformal = _evaluate_interval_method(
        lockbox_points,
        calibration_points_by_metric=calibration_points,
        method="conformal",
        exclude_split=False,
    )
    lockbox_bootstrap = _evaluate_interval_method(
        lockbox_points,
        calibration_points_by_metric=calibration_points,
        method="bootstrap",
        exclude_split=False,
    )
    bootstrap["lockbox"] = lockbox_bootstrap
    bootstrap["lockbox_baselines"] = {"conformal_primary_normalized_wis": lockbox_conformal.get("primary_normalized_wis")}
    conformal["lockbox"] = lockbox_conformal
    risk = _risk_payload(result, contract_name=contract_name, allowed_tiers=allowed_tiers)
    return {
        "experiment_id": str(result["experiment_id"]),
        "contract_name": str(contract_name),
        "quarterly_mean_mae": float(dict(result.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")),
        "conformal": conformal,
        "bootstrap": bootstrap,
        "risk": risk,
        "decision": _interval_decision(conformal, bootstrap),
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Probabilistic Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        "",
        "## Winners",
        "",
        f"- Exact: `{payload['exact']['experiment_id']}`",
        f"- Dense: `{payload['dense']['experiment_id']}`",
        "",
        "## Interval Comparison",
        "",
        "| Contract | Method | Primary normalized WIS | Coverage gap | WIS80 | WIS95 | Cov80 | Cov95 | Lockbox primary normalized WIS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for contract_key in ("exact", "dense"):
        contract_payload = dict(payload[contract_key])
        contract_name = str(contract_payload["contract_name"])
        for method_name, method_payload in (("EXP-UQ-01", dict(contract_payload["conformal"])), ("EXP-UQ-02", dict(contract_payload["bootstrap"]))):
            row = _interval_method_row(method_payload, method_name=method_name)
            lines.append(
                f"| {contract_name} | {method_name} | {_fmt(row['primary_normalized_wis'])} | {_fmt(row['coverage_gap'])} | "
                f"{_fmt(row['wis80'])} | {_fmt(row['wis95'])} | {_fmt(row['coverage80'], digits=3)} | {_fmt(row['coverage95'], digits=3)} | "
                f"{_fmt(row['lockbox_primary_normalized_wis'])} |"
            )
        decision = dict(contract_payload.get("decision") or {})
        lines.extend(
            [
                "",
                f"- `{contract_name}` interval winner: `{decision.get('winner', '')}` ({decision.get('status', '')})",
                f"  - {decision.get('why', '')}",
            ]
        )
    lines.extend(
        [
            "",
            "## High-Error Event Prediction",
            "",
            "| Contract | Threshold | Status | Event count | Candidate Brier | Prevalence Brier | Persistence Brier | Candidate recall@K | Persistence recall@K |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for contract_key in ("exact", "dense"):
        risk = dict(dict(payload[contract_key]).get("risk") or {})
        for threshold_key, threshold_payload in dict(risk.get("by_quantile") or {}).items():
            metrics = dict(threshold_payload.get("metrics") or {})
            lines.append(
                f"| {risk.get('contract', '')} | {threshold_key} | {threshold_payload.get('status', '')} | {int(threshold_payload.get('event_count') or 0)} | "
                f"{_fmt(metrics.get('candidate_brier'))} | {_fmt(metrics.get('prevalence_brier'))} | {_fmt(metrics.get('persistence_brier'))} | "
                f"{_fmt(metrics.get('candidate_recall_at_k'), digits=3)} | {_fmt(metrics.get('persistence_recall_at_k'), digits=3)} |"
            )
        decision = dict(risk.get("decision") or {})
        lines.extend(
            [
                "",
                f"- `{risk.get('contract', '')}` risk decision: `{decision.get('status', '')}`",
                f"  - {decision.get('why', '')}",
            ]
        )
    return "\n".join(lines) + "\n"


def run_tr_v3_probabilistic_batch(
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

    exact_payload = _run_contract_payload(
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
    dense_payload = _run_contract_payload(
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
    exact_result = _suite_result_map(exact_payload)[EXACT_EXPERIMENT_ID]
    dense_result = _suite_result_map(dense_payload)[DENSE_EXPERIMENT_ID]

    exact_contract_payload = _build_contract_payload(
        archive_run_id=archive_run,
        contract_name="exact_only",
        result=exact_result,
        holdout_years=holdout_years,
    )
    dense_contract_payload = _build_contract_payload(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        result=dense_result,
        holdout_years=holdout_years,
    )

    exact_coverage_path = analysis_dir / "uq_coverage_exact.png"
    dense_coverage_path = analysis_dir / "uq_coverage_dense.png"
    exact_risk_path = analysis_dir / "risk_exact.png"
    dense_risk_path = analysis_dir / "risk_dense.png"
    _save_coverage_pair("exact_only", dict(exact_contract_payload["conformal"]), dict(exact_contract_payload["bootstrap"]), exact_coverage_path)
    _save_coverage_pair("purged_dense", dict(dense_contract_payload["conformal"]), dict(dense_contract_payload["bootstrap"]), dense_coverage_path)
    _save_risk_graph(dict(exact_contract_payload["risk"]), exact_risk_path, title="Exact high-error event prediction")
    _save_risk_graph(dict(dense_contract_payload["risk"]), dense_risk_path, title="Dense high-error event prediction")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "exact": exact_contract_payload,
        "dense": dense_contract_payload,
        "artifacts": {
            "exact_coverage_graph": exact_coverage_path.name,
            "dense_coverage_graph": dense_coverage_path.name,
            "exact_risk_graph": exact_risk_path.name,
            "dense_risk_graph": dense_risk_path.name,
        },
    }
    write_json(analysis_dir / "tr_v3_probabilistic_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_probabilistic_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded TR-V3 probabilistic interval and risk experiments.")
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
    run_tr_v3_probabilistic_batch(
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
