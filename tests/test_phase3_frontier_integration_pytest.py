from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from epigraph_ph.phase3.frontier.artifacts import build_transition_research_context
import epigraph_ph.phase3.frontier.tr_v2 as tr_v2


SLOW_INTEGRATION_ENABLED = os.environ.get("EPIGRAPH_RUN_SLOW_INTEGRATION") == "1"
SMOKE_LATENT_BLOCKS_DIR = Path("D:/EpiGraph_PH/artifacts/runs/smoke-latent-blocks")


def _flatten_explicit_prior_map(explicit_map: dict[str, object]) -> dict[str, dict[str, dict[str, object]]]:
    flattened: dict[str, dict[str, dict[str, object]]] = {}
    for transition, transition_cfg in explicit_map.items():
        transition_cfg = dict(transition_cfg or {})
        source_cfgs: dict[str, dict[str, object]] = {}
        for _target_block, target_cfg in dict(transition_cfg.get("target_blocks") or {}).items():
            target_cfg = dict(target_cfg or {})
            base_cfg = {key: value for key, value in target_cfg.items() if key != "source_blocks"}
            for source_block, source_cfg in dict(target_cfg.get("source_blocks") or {}).items():
                merged = dict(base_cfg)
                merged.update(dict(source_cfg or {}))
                existing = source_cfgs.get(str(source_block))
                candidate = {
                    "lags": [int(value) for value in list(merged.get("lags") or [])],
                    "prior_scale": float(merged.get("prior_scale") or 0.25),
                }
                if existing is None:
                    source_cfgs[str(source_block)] = candidate
                    continue
                existing_lags = set(int(value) for value in list(existing.get("lags") or []))
                existing_lags.update(candidate["lags"])
                existing["lags"] = sorted(existing_lags)
                existing["prior_scale"] = max(float(existing.get("prior_scale") or 0.25), candidate["prior_scale"])
        flattened[str(transition)] = source_cfgs
    return flattened


def _legacy_collapsed_direct_effects(*, structural_inputs: object, dataset: dict[str, object]) -> dict[str, object]:
    frontier_cfg = dict((((tr_v2._HIV_PLUGIN.constraint_settings or {}).get("phase3", {}) or {}).get("frontier", {}) or {}))
    explicit_map = dict(frontier_cfg.get("phase2_transition_prior_map") or {})
    collapsed_map = _flatten_explicit_prior_map(explicit_map)
    quarter_features = tr_v2._quarter_feature_map(structural_inputs)
    support_lookup: dict[tuple[str, int], dict[str, float]] = {}
    for row in structural_inputs.direct_edge_rows:
        key = (str(row.get("source") or ""), int(row.get("lag") or 0))
        candidate = {
            "weight": float(row.get("weight") or 0.0),
            "stability": float(row.get("stability") or 0.0),
            "support_count": int(row.get("support_count") or 0),
        }
        existing = support_lookup.get(key)
        if existing is None or (
            candidate["support_count"],
            abs(candidate["weight"]),
            candidate["stability"],
        ) > (
            existing["support_count"],
            abs(existing["weight"]),
            existing["stability"],
        ):
            support_lookup[key] = candidate
    summaries: dict[str, object] = {}
    predictions: dict[str, dict[str, float]] = defaultdict(dict)
    for transition in tr_v2.TRANSITION_NAMES:
        transition_cfg = dict(collapsed_map.get(transition) or {})
        train_rows = [row for row in list(dataset["train_transition_rows"]) if str(row.get("quarter") or "") in quarter_features]
        if not train_rows:
            summaries[transition] = {"feature_count": 0, "coefficients": []}
            continue
        feature_specs: list[tuple[str, int, float, dict[str, float]]] = []
        for source_block, mapping in transition_cfg.items():
            allowed_lags = [int(value) for value in list(mapping.get("lags") or [])]
            for lag in allowed_lags:
                support = support_lookup.get((str(source_block), lag))
                if support is None:
                    continue
                prior_scale = float(mapping.get("prior_scale") or 0.25)
                feature_specs.append((str(source_block), lag, prior_scale, support))
        if not feature_specs:
            summaries[transition] = {"feature_count": 0, "coefficients": []}
            continue
        y = np.asarray(
            [
                tr_v2._logit(float((row.get("hazards") or {}).get(transition) or 1e-5))
                for row in train_rows
            ],
            dtype=np.float32,
        )
        intercept = float(np.mean(y))
        X = []
        for quarter_row in train_rows:
            quarter = str(quarter_row.get("quarter") or "")
            quarter_position = structural_inputs.quarter_axis.index(quarter) if quarter in structural_inputs.quarter_axis else -1
            if quarter_position < 0:
                continue
            row_values = []
            for block_id, lag, _prior_scale, _support in feature_specs:
                source_idx = quarter_position - lag
                if source_idx < 0:
                    row_values.append(0.0)
                    continue
                source_quarter = structural_inputs.quarter_axis[source_idx]
                row_values.append(float(quarter_features.get(source_quarter, {}).get(block_id) or 0.0))
            X.append(row_values)
        X_matrix = np.asarray(X, dtype=np.float32)
        prior_precisions = np.asarray(
            [
                1.0
                / max(
                    (
                        float(prior_scale)
                        * (1.0 + float(support["support_count"]))
                        * max(float(support["stability"]), 0.05)
                    )
                    ** 2,
                    1e-6,
                )
                for _block_id, _lag, prior_scale, support in feature_specs
            ],
            dtype=np.float32,
        )
        beta = tr_v2._fit_prior_regression(target_values=y - intercept, design_matrix=X_matrix, prior_precisions=prior_precisions)
        summaries[transition] = {
            "feature_count": int(len(feature_specs)),
            "intercept_logit": round(intercept, 6),
            "coefficients": [
                {
                    "block_id": block_id,
                    "lag": lag,
                    "coefficient": round(float(beta[idx]), 6),
                    "prior_scale": round(float(prior_scale), 6),
                    "phase2_weight": round(float(support["weight"]), 6),
                    "support_count": int(support["support_count"]),
                    "stability": round(float(support["stability"]), 6),
                }
                for idx, (block_id, lag, prior_scale, support) in enumerate(feature_specs)
            ],
        }
        for quarter in structural_inputs.quarter_axis:
            quarter_position = structural_inputs.quarter_axis.index(quarter)
            delta = 0.0
            for idx, (block_id, lag, _prior_scale, _support) in enumerate(feature_specs):
                source_idx = quarter_position - lag
                if source_idx < 0:
                    continue
                source_quarter = structural_inputs.quarter_axis[source_idx]
                delta += float(beta[idx]) * float(quarter_features.get(source_quarter, {}).get(block_id) or 0.0)
            predictions[transition][quarter] = float(delta)
    return {"summary": summaries, "quarter_adjustments": {key: dict(value) for key, value in predictions.items()}}


def _evaluate_variant(
    *,
    dataset: dict[str, object],
    baseline_hazards: dict[str, dict[str, float]],
    direct_summary: dict[str, object],
    direct_adjustments: dict[str, dict[str, float]],
    hidden_summary: dict[str, object] | None = None,
    hidden_adjustments: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    _forecast_rows, _hazard_rows, evaluation = tr_v2._simulate_holdout(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_adjustments=direct_adjustments,
        hidden_adjustments=hidden_adjustments or {},
        peak_gates={},
    )
    return {
        "mae": float(evaluation["model_mean_absolute_error"]),
        "smape": float(evaluation["model_smape"]),
        "feature_count": int(
            sum(int((direct_summary.get(transition) or {}).get("feature_count") or 0) for transition in tr_v2.TRANSITION_NAMES)
        ),
        "hidden_rank_used": int(
            sum(int((hidden_summary or {}).get(transition, {}).get("rank_used") or 0) for transition in tr_v2.TRANSITION_NAMES)
        ),
    }


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not SLOW_INTEGRATION_ENABLED, reason="Set EPIGRAPH_RUN_SLOW_INTEGRATION=1 to run slow integration benchmarks")
@pytest.mark.skipif(not SMOKE_LATENT_BLOCKS_DIR.exists(), reason="smoke-latent-blocks artifacts are required for this benchmark")
def test_tr_v2_smoke_latent_blocks_explicit_target_map_benchmark() -> None:
    ctx = build_transition_research_context(
        run_id="pytest-tr-v2-smoke-integration",
        plugin_id="hiv",
        experiment_id="TR-V2-03-phase2-ablation-suite",
        source_run_id="smoke-latent-blocks",
    )
    structural_inputs = tr_v2.load_phase2_structural_inputs(ctx)
    age_lock = tr_v2._load_age01b_lock()
    dataset = tr_v2._empirical_transition_dataset(ctx)
    baseline_hazards = tr_v2._baseline_hazard_map(age_lock)

    no_prior = _evaluate_variant(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_summary={transition: {"feature_count": 0, "coefficients": []} for transition in tr_v2.TRANSITION_NAMES},
        direct_adjustments={},
    )
    legacy = _legacy_collapsed_direct_effects(structural_inputs=structural_inputs, dataset=dataset)
    legacy_metrics = _evaluate_variant(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_summary=legacy["summary"],
        direct_adjustments=legacy["quarter_adjustments"],
    )
    explicit = tr_v2._fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=False)
    explicit_metrics = _evaluate_variant(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_summary=explicit["summary"],
        direct_adjustments=explicit["quarter_adjustments"],
    )
    explicit_multiscale = tr_v2._fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=True)
    explicit_multiscale_metrics = _evaluate_variant(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_summary=explicit_multiscale["summary"],
        direct_adjustments=explicit_multiscale["quarter_adjustments"],
    )
    hidden_only = tr_v2._fit_hidden_transition_effects(
        structural_inputs=structural_inputs,
        dataset=dataset,
        direct_adjustments={},
    )
    hidden_only_metrics = _evaluate_variant(
        dataset=dataset,
        baseline_hazards=baseline_hazards,
        direct_summary={transition: {"feature_count": 0, "coefficients": []} for transition in tr_v2.TRANSITION_NAMES},
        direct_adjustments={},
        hidden_summary=hidden_only["summary"],
        hidden_adjustments=hidden_only["quarter_adjustments"],
    )

    assert explicit_metrics["feature_count"] > 0
    assert legacy_metrics["feature_count"] > 0
    assert explicit_metrics["mae"] < no_prior["mae"]
    assert explicit_metrics["smape"] < no_prior["smape"]
    assert explicit_metrics["mae"] <= legacy_metrics["mae"] + 1e-9
    assert explicit_metrics["smape"] <= legacy_metrics["smape"] + 1e-9
    assert explicit_multiscale_metrics["feature_count"] == explicit_metrics["feature_count"]
    assert any(
        float((explicit_multiscale["summary"].get(transition) or {}).get("multiscale_support", {}).get("multiplier") or 1.0) > 1.0
        for transition in tr_v2.TRANSITION_NAMES
    )
    assert hidden_only_metrics["hidden_rank_used"] > 0
    assert abs(hidden_only_metrics["mae"] - no_prior["mae"]) > 1e-9


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not SLOW_INTEGRATION_ENABLED, reason="Set EPIGRAPH_RUN_SLOW_INTEGRATION=1 to run slow integration benchmarks")
@pytest.mark.skipif(not SMOKE_LATENT_BLOCKS_DIR.exists(), reason="smoke-latent-blocks artifacts are required for this benchmark")
def test_tr_v2_smoke_latent_blocks_rolling_origin_forward_benchmark() -> None:
    ctx = build_transition_research_context(
        run_id="pytest-tr-v2-rolling-origin",
        plugin_id="hiv",
        experiment_id="TR-V2-03-phase2-ablation-suite",
        source_run_id="smoke-latent-blocks",
    )
    structural_inputs = tr_v2.load_phase2_structural_inputs(ctx)
    splits = tr_v2._rolling_origin_holdout_splits(
        ctx,
        start_year=2010,
        end_year=2025,
        min_train_years=5,
        horizon_years=1,
    )
    assert splits

    rows: list[dict[str, float]] = []
    for split in splits:
        dataset = tr_v2._empirical_transition_dataset_for_holdout_years(ctx, list(split["holdout_years"]))
        if not list(dataset["train_transition_rows"]) or not list(dataset["holdout_rows"]):
            continue
        baseline_hazards = tr_v2._train_based_baseline_hazard_map(dataset, mode="last_train")
        no_prior = _evaluate_variant(
            dataset=dataset,
            baseline_hazards=baseline_hazards,
            direct_summary={transition: {"feature_count": 0, "coefficients": []} for transition in tr_v2.TRANSITION_NAMES},
            direct_adjustments={},
        )
        explicit = tr_v2._fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=False)
        explicit_metrics = _evaluate_variant(
            dataset=dataset,
            baseline_hazards=baseline_hazards,
            direct_summary=explicit["summary"],
            direct_adjustments=explicit["quarter_adjustments"],
        )
        explicit_multiscale = tr_v2._fit_direct_transition_effects(structural_inputs=structural_inputs, dataset=dataset, include_multiscale=True)
        explicit_multiscale_metrics = _evaluate_variant(
            dataset=dataset,
            baseline_hazards=baseline_hazards,
            direct_summary=explicit_multiscale["summary"],
            direct_adjustments=explicit_multiscale["quarter_adjustments"],
        )
        rows.append(
            {
                "train_end_year": float(split["train_end_year"]),
                "no_prior_mae": float(no_prior["mae"]),
                "explicit_mae": float(explicit_metrics["mae"]),
                "explicit_multiscale_mae": float(explicit_multiscale_metrics["mae"]),
            }
        )

    assert rows
    assert all(np.isfinite(list(row.values())).all() for row in rows)
    assert any(float(row["explicit_mae"]) < float(row["no_prior_mae"]) for row in rows)
    latest_row = max(rows, key=lambda row: float(row["train_end_year"]))
    assert float(latest_row["explicit_mae"]) < float(latest_row["no_prior_mae"])
    assert float(latest_row["explicit_multiscale_mae"]) < float(latest_row["no_prior_mae"])


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not SLOW_INTEGRATION_ENABLED, reason="Set EPIGRAPH_RUN_SLOW_INTEGRATION=1 to run slow integration benchmarks")
@pytest.mark.skipif(not SMOKE_LATENT_BLOCKS_DIR.exists(), reason="smoke-latent-blocks artifacts are required for this benchmark")
def test_tr_v2_smoke_latent_blocks_rolling_origin_report_artifacts() -> None:
    paths = tr_v2.write_tr_v2_rolling_origin_report(
        run_id="pytest-tr-v2-rolling-origin-report",
        plugin_id="hiv",
        source_run_id="smoke-latent-blocks",
        start_year=2010,
        end_year=2025,
        min_train_years=5,
        horizon_years=1,
    )
    for path_text in paths.values():
        path = Path(str(path_text))
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not SLOW_INTEGRATION_ENABLED, reason="Set EPIGRAPH_RUN_SLOW_INTEGRATION=1 to run slow integration benchmarks")
@pytest.mark.skipif(not SMOKE_LATENT_BLOCKS_DIR.exists(), reason="smoke-latent-blocks artifacts are required for this benchmark")
def test_tr_v2_smoke_latent_blocks_early_history_partial_report_artifacts() -> None:
    paths = tr_v2.write_tr_v2_early_history_partial_report(
        run_id="pytest-tr-v2-early-history-partial-report",
        plugin_id="hiv",
        source_run_id="smoke-latent-blocks",
        start_year=2010,
        end_year=2016,
        min_train_years=3,
        horizon_years=1,
    )
    for path_text in paths.values():
        path = Path(str(path_text))
        assert path.exists()
        assert path.stat().st_size > 0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(not SLOW_INTEGRATION_ENABLED, reason="Set EPIGRAPH_RUN_SLOW_INTEGRATION=1 to run slow integration benchmarks")
@pytest.mark.skipif(not SMOKE_LATENT_BLOCKS_DIR.exists(), reason="smoke-latent-blocks artifacts are required for this benchmark")
def test_tr_v2_smoke_latent_blocks_combined_dashboard_report_artifacts() -> None:
    rolling = tr_v2.write_tr_v2_rolling_origin_report(
        run_id="pytest-tr-v2-rolling-origin-report-dashboard",
        plugin_id="hiv",
        source_run_id="smoke-latent-blocks",
        start_year=2010,
        end_year=2025,
        min_train_years=5,
        horizon_years=1,
    )
    early = tr_v2.write_tr_v2_early_history_partial_report(
        run_id="pytest-tr-v2-early-history-partial-report-dashboard",
        plugin_id="hiv",
        source_run_id="smoke-latent-blocks",
        start_year=2010,
        end_year=2016,
        min_train_years=3,
        horizon_years=1,
    )
    paths = tr_v2.write_tr_v2_benchmark_dashboard_report(
        run_id="pytest-tr-v2-benchmark-dashboard-report",
        plugin_id="hiv",
        source_run_id="smoke-latent-blocks",
        rolling_origin_run_id=Path(str(rolling["json"])).parents[1].name,
        early_history_run_id=Path(str(early["json"])).parents[1].name,
    )
    for path_text in paths.values():
        path = Path(str(path_text))
        assert path.exists()
        assert path.stat().st_size > 0
