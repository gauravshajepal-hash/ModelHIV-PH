from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.phase3._lineage.pipeline import (
    RESCUE_INFERENCE_FAMILY,
    RESCUE_V2_PROFILE_ID,
    _holdout_reference_smape,
    _require_requested_inference_family,
)
from epigraph_ph.phase3._lineage.rescue_core import run_phase3_rescue_core
from epigraph_ph.runtime import read_json


def rolling_origin_splits(years: list[int], *, min_train_years: int = 3) -> list[dict[str, Any]]:
    ordered = sorted({int(year) for year in years})
    if len(ordered) <= min_train_years:
        return []
    splits: list[dict[str, Any]] = []
    for idx in range(min_train_years, len(ordered)):
        train_years = ordered[:idx]
        holdout_year = ordered[idx]
        splits.append(
            {
                "split_label": f"train_{train_years[0]}_{train_years[-1]}__holdout_{holdout_year}",
                "train_years": train_years,
                "holdout_years": [holdout_year],
                "holdout_year": holdout_year,
            }
        )
    return splits


def rank_representation_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[str, dict[str, Any]] = {}
    for row in rows:
        representation = str(row.get("representation") or "")
        bucket = aggregated.setdefault(
            representation,
            {
                "representation": representation,
                "split_count": 0,
                "mean_model_mae": 0.0,
                "mean_model_smape": 0.0,
                "rows": [],
            },
        )
        bucket["rows"].append(dict(row))
    ranking: list[dict[str, Any]] = []
    for representation, bucket in aggregated.items():
        values = list(bucket["rows"])
        ranking.append(
            {
                "representation": representation,
                "split_count": len(values),
                "mean_model_mae": round(float(np.mean([float(item.get("model_mean_absolute_error", 0.0)) for item in values])), 6),
                "mean_model_smape": round(float(np.mean([float(item.get("model_smape", 0.0)) for item in values])), 6),
                "rows": values,
            }
        )
    return sorted(ranking, key=lambda item: (float(item["mean_model_mae"]), float(item["mean_model_smape"]), str(item["representation"])))


def bias_metrics_from_evaluation(evaluation: dict[str, Any]) -> dict[str, float]:
    rows = list(dict(evaluation.get("holdout_reference_check") or {}).get("comparisons", []) or [])
    if not rows:
        return {
            "diagnosed_stock_abs_error": 0.0,
            "art_stock_abs_error": 0.0,
            "documented_suppression_abs_error": 0.0,
            "viral_load_tested_among_art_abs_error": 0.0,
            "suppressed_among_art_abs_error": 0.0,
        }
    first = dict(rows[0].get("errors") or {})
    return {
        "diagnosed_stock_abs_error": round(float(first.get("diagnosed_stock_abs_error", 0.0)), 6),
        "art_stock_abs_error": round(float(first.get("art_stock_abs_error", 0.0)), 6),
        "documented_suppression_abs_error": round(float(first.get("documented_suppression_abs_error", 0.0)), 6),
        "viral_load_tested_among_art_abs_error": round(float(first.get("viral_load_tested_among_art_abs_error", 0.0)), 6),
        "suppressed_among_art_abs_error": round(float(first.get("suppressed_among_art_abs_error", 0.0)), 6),
    }


def run_broad_backtest_trial(
    *,
    prepared: dict[str, Any],
    run_id: str,
    plugin_id: str,
    profile: str,
    inference_family: str = RESCUE_INFERENCE_FAMILY,
    representation: str,
    phase_dir_name: str,
    calibration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = run_phase3_rescue_core(
        run_id=run_id,
        plugin_id=plugin_id,
        profile_id=profile,
        requested_inference_family=inference_family,
        phase_dir_name=phase_dir_name,
        axis_catalogs_override=prepared["filtered_axis_catalogs"],
        normalized_rows_override=prepared["filtered_rows"],
        parameter_catalog_override=prepared["parameter_catalog"],
        standardized_tensor_override=prepared["filtered_tensor"],
        reference_overrides={
            "official_points": prepared["official_points"],
            "harp_points": prepared["backtest_config"]["train_harp_points"],
        },
        backtest_config=prepared["backtest_config"],
        modifier_representation=representation,
        calibration_overrides=calibration_overrides,
    )
    fit_artifact = _require_requested_inference_family(
        manifest,
        inference_family,
        context=f"phase3 broad backtest trial [{representation}]",
    )
    evaluation = read_json(Path(manifest["artifact_paths"]["frozen_history_backtest_evaluation"]), default={})
    evaluation_summary = dict(evaluation.get("summary") or {})
    determinant_summary = fit_artifact.get("determinant_modifier_summary", {}) or {}
    return {
        "representation": representation,
        "phase_dir_name": phase_dir_name,
        "train_years": list(prepared["backtest_config"].get("train_years") or []),
        "holdout_years": list(prepared["backtest_config"].get("holdout_years") or []),
        "model_mean_absolute_error": round(float(evaluation_summary.get("model_mean_absolute_error", 0.0)), 6),
        "model_smape": _holdout_reference_smape(dict(evaluation.get("holdout_reference_check") or {}), eps=1e-6),
        "carry_forward_mean_absolute_error": round(float(evaluation_summary.get("carry_forward_mean_absolute_error", 0.0)), 6),
        "simple_compartmental_mean_absolute_error": round(float(evaluation_summary.get("simple_compartmental_mean_absolute_error", 0.0)), 6),
        "model_beats_carry_forward": bool(evaluation_summary.get("model_beats_carry_forward")),
        "selected_determinant_count": len(list(determinant_summary.get("selected_determinant_modifiers") or [])),
        "bias_metrics": bias_metrics_from_evaluation(evaluation),
        "artifact_paths": dict(manifest.get("artifact_paths", {})),
        "baseline_representation": "clumped" if profile == RESCUE_V2_PROFILE_ID else "unclumped",
    }
