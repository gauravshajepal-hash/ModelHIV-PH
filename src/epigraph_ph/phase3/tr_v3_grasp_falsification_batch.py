from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, build_annual_anchor_rows, quarter_sort_key, quarter_year
from epigraph_ph.runtime import ensure_dir, write_json


CONTRACT_CHOICES: tuple[str, ...] = ("exact_only", "legacy_dense", "purged_dense")
BURST_Z_THRESHOLD = 1.5
NULL_TOLERANCE = 0.002
LOCKBOX_HOLDOUT_YEARS = [2025]


def _registered_spec(experiment_id: str) -> suite.ExperimentSpec:
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    return spec_map[experiment_id]


def _make_exact_piecewise_null_spec() -> suite.ExperimentSpec:
    return suite.ExperimentSpec(
        "EXP-G-NULL-01",
        "repair",
        "Piecewise/changepoint observation null on the exact contract using the current exact winner's standard corrections but no shock layer.",
        {},
        transition_model="direct_observation_joint_consistency_bias_corrected",
        repair_params={
            "diagnosed_weight": 1.0,
            "art_weight": 1.0,
            "flow_weight": 0.25,
            "diagnosed_series_model": "piecewise",
            "art_series_model": "piecewise",
            "flow_series_model": "piecewise",
            "diagnosed_recent_blend_weight": 1.0,
            "art_recent_blend_weight": 1.0,
            "flow_recent_blend_weight": 1.0,
            "suppression_carry_weight": 0.75,
            "diagnosed_bias_weight": 1.0,
            "art_bias_weight": 1.0,
            "flow_bias_weight": 1.0,
            "flow_consistency_weight": 1.0,
        },
    )


def _make_dense_piecewise_null_spec() -> suite.ExperimentSpec:
    return suite.ExperimentSpec(
        "EXP-G-NULL-01",
        "repair",
        "Piecewise/changepoint observation null on the dense contract using the current dense winner's standard corrections but no shock layer.",
        {},
        transition_model="direct_observation_joint_consistency",
        repair_params={
            "diagnosed_weight": 1.0,
            "art_weight": 1.0,
            "flow_weight": 1.0,
            "diagnosed_series_model": "piecewise",
            "art_series_model": "piecewise",
            "flow_series_model": "piecewise",
            "diagnosed_recent_blend_weight": 1.0,
            "art_recent_blend_weight": 1.0,
            "flow_recent_blend_weight": 1.0,
            "suppression_carry_weight": 1.0,
            "suppression_fallback_mode": "unclaimed",
        },
    )


def _make_exact_shared_shock_spec() -> suite.ExperimentSpec:
    return suite.ExperimentSpec(
        "EXP-R10-G1-01-min",
        "repair",
        "Minimal shared-shock overlay on the exact winner: one shared scalar shock per quarter, no free lifted states.",
        {},
        transition_model="direct_observation_shared_shock",
        repair_params={
            "diagnosed_weight": 1.0,
            "art_weight": 1.0,
            "flow_weight": 0.25,
            "diagnosed_series_model": "level",
            "art_series_model": "level",
            "flow_series_model": "delta",
            "diagnosed_recent_blend_weight": 1.0,
            "art_recent_blend_weight": 1.0,
            "flow_recent_blend_weight": 0.75,
            "suppression_carry_weight": 0.75,
            "use_joint_consistency": True,
            "use_bias_correction": True,
            "diagnosed_bias_weight": 1.0,
            "art_bias_weight": 1.0,
            "flow_bias_weight": 1.0,
            "flow_consistency_weight": 1.0,
            "shock_min_shared_metrics": 2,
            "shock_threshold_quantile": 0.7,
            "shock_threshold_scale": 1.0,
            "shock_forecast_blend_weight": 0.75,
            "shock_scale": 1.0,
        },
    )


def _make_dense_shared_shock_spec() -> suite.ExperimentSpec:
    return suite.ExperimentSpec(
        "EXP-R10-G1-01-min",
        "repair",
        "Minimal shared-shock overlay on the dense winner: one shared scalar shock per quarter, no free lifted states.",
        {},
        transition_model="direct_observation_shared_shock",
        repair_params={
            "diagnosed_weight": 1.0,
            "art_weight": 1.0,
            "flow_weight": 1.0,
            "diagnosed_series_model": "level",
            "art_series_model": "delta",
            "flow_series_model": "level",
            "diagnosed_recent_blend_weight": 1.0,
            "art_recent_blend_weight": 1.0,
            "flow_recent_blend_weight": 1.0,
            "suppression_carry_weight": 1.0,
            "suppression_fallback_mode": "unclaimed",
            "use_joint_consistency": True,
            "use_bias_correction": False,
            "flow_consistency_weight": 0.0,
            "shock_min_shared_metrics": 2,
            "shock_threshold_quantile": 0.7,
            "shock_threshold_scale": 1.0,
            "shock_forecast_blend_weight": 0.75,
            "shock_scale": 1.0,
        },
    )


def _contract_setup(archive_run_id: str, contract_name: str) -> tuple[list[dict[str, Any]], set[str], bool, list[dict[str, Any]] | None]:
    if contract_name == "exact_only":
        return suite.build_quarterly_observation_rows(archive_run_id), {"exact_observed"}, False, None
    if contract_name in {"legacy_dense", "purged_dense"}:
        dense_payload = suite._build_dense_contract_payload(archive_run_id)
        dense_rows = list(dense_payload["rows"])
        return dense_rows, {"exact_observed", "bridge_observed"}, contract_name == "purged_dense", dense_rows
    raise ValueError(f"Unsupported contract_name: {contract_name}")


def _run_specs_for_contract(
    *,
    archive_run_id: str,
    contract_name: str,
    specs: list[suite.ExperimentSpec],
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    observation_rows, scoring_tiers, split_local_dense, full_dense_rows = _contract_setup(archive_run_id, contract_name)
    availability = suite._build_availability_payload(archive_run_id)
    annual_rows = build_annual_anchor_rows(archive_run_id)
    results = [
        suite._evaluate_experiment_spec(
            spec,
            observation_rows=observation_rows,
            annual_rows=annual_rows,
            availability=availability,
            scoring_tiers=scoring_tiers,
            archive_run_id=archive_run_id,
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
            split_local_dense=split_local_dense,
            full_dense_rows=full_dense_rows,
        )
        for spec in specs
    ]
    return {
        "archive_run_id": archive_run_id,
        "contract_name": contract_name,
        "results": results,
    }


def _result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _allowed_tiers(contract_name: str) -> set[str]:
    return {"exact_observed"} if contract_name == "exact_only" else {"exact_observed", "bridge_observed"}


def _metric_points(result: dict[str, Any], *, metric_name: str, allowed_tiers: set[str]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for split_idx, split in enumerate(list(result.get("quarterly_rows") or [])):
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            if suite._metric_tier(target_row, metric_name) not in allowed_tiers:
                continue
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            points.append(
                {
                    "split_idx": int(split_idx),
                    "quarter": str(target_row["quarter"]),
                    "year": int(quarter_year(str(target_row["quarter"]))),
                    "tier": str(suite._metric_tier(target_row, metric_name)),
                    "target": float(target_value),
                    "prediction": float(prediction_value),
                    "residual": float(prediction_value) - float(target_value),
                }
            )
    return points


def _robust_scale(values: list[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 1.0
    mad = float(np.median(np.abs(arr - np.median(arr))))
    scale = mad * 1.4826
    if scale <= 1e-9:
        scale = float(np.std(arr))
    if scale <= 1e-9:
        scale = 1.0
    return float(scale)


def _shock_concordance_contract_payload(result: dict[str, Any], *, contract_name: str) -> dict[str, Any]:
    allowed_tiers = _allowed_tiers(contract_name)
    points_by_metric = {
        metric_name: _metric_points(result, metric_name=metric_name, allowed_tiers=allowed_tiers)
        for metric_name in PRIMARY_METRICS
    }
    scales = {
        metric_name: _robust_scale([float(point["residual"]) for point in points])
        for metric_name, points in points_by_metric.items()
    }
    quarter_rows: dict[str, dict[str, Any]] = {}
    for metric_name, points in points_by_metric.items():
        scale = float(scales[metric_name])
        for point in points:
            quarter = str(point["quarter"])
            row = quarter_rows.setdefault(
                quarter,
                {"quarter": quarter, "year": int(point["year"]), "metrics": {}, "burst_metrics": []},
            )
            z_score = float(point["residual"]) / float(scale)
            metric_payload = {**point, "z_score": float(z_score), "abs_z": abs(float(z_score))}
            row["metrics"][metric_name] = metric_payload
    for row in quarter_rows.values():
        row["burst_metrics"] = sorted(
            [
                metric_name
                for metric_name, payload in dict(row["metrics"]).items()
                if abs(float(payload["z_score"])) >= float(BURST_Z_THRESHOLD)
            ]
        )
        row["shared_burst"] = bool(len(row["burst_metrics"]) >= 2)
        row["diagnosed_only_burst"] = row["burst_metrics"] == ["diagnosed_plhiv"]
        row["mean_abs_z"] = float(
            np.mean([float(payload["abs_z"]) for payload in dict(row["metrics"]).values()])
        ) if row["metrics"] else 0.0

    correlations: dict[str, float | None] = {}
    for left, right in (("diagnosed_plhiv", "alive_on_art"), ("diagnosed_plhiv", "new_diagnosed_cases_period"), ("alive_on_art", "new_diagnosed_cases_period")):
        left_points = {str(point["quarter"]): float(point["residual"]) for point in points_by_metric[left]}
        right_points = {str(point["quarter"]): float(point["residual"]) for point in points_by_metric[right]}
        shared_quarters = sorted(set(left_points) & set(right_points), key=quarter_sort_key)
        if len(shared_quarters) < 3:
            correlations[f"{left}__{right}"] = None
            continue
        left_values = np.asarray([left_points[quarter] for quarter in shared_quarters], dtype=np.float64)
        right_values = np.asarray([right_points[quarter] for quarter in shared_quarters], dtype=np.float64)
        if float(np.std(left_values)) <= 1e-9 or float(np.std(right_values)) <= 1e-9:
            correlations[f"{left}__{right}"] = None
            continue
        correlations[f"{left}__{right}"] = float(np.corrcoef(left_values, right_values)[0, 1])

    ordered_quarters = sorted(list(quarter_rows.values()), key=lambda row: quarter_sort_key(str(row["quarter"])))
    shared_bursts = [row for row in ordered_quarters if row["shared_burst"]]
    diagnosed_only_bursts = [row for row in ordered_quarters if row["diagnosed_only_burst"]]
    diagnosed_only_share = float(len(diagnosed_only_bursts)) / float(max(len(shared_bursts) + len(diagnosed_only_bursts), 1))
    decision = {
        "status": (
            "supports_shared_shock_hypothesis"
            if len(shared_bursts) >= 2
            and diagnosed_only_share <= 0.5
            and any(abs(float(value)) >= 0.3 for value in correlations.values() if value is not None)
            else "insufficient_shared_shock_signal"
        ),
        "shared_burst_quarter_count": int(len(shared_bursts)),
        "diagnosed_only_share": float(diagnosed_only_share),
    }
    return {
        "contract": contract_name,
        "scales": {metric_name: float(value) for metric_name, value in scales.items()},
        "correlations": correlations,
        "ordered_quarters": ordered_quarters,
        "shared_bursts": shared_bursts,
        "diagnosed_only_bursts": diagnosed_only_bursts,
        "decision": decision,
    }


def _plateau_runs_from_values(quarters: list[str], values: list[float]) -> dict[str, Any]:
    if len(values) < 3:
        return {"run_count": 0, "max_run_length": 0, "mean_run_length": 0.0, "threshold": 0.0, "runs": []}
    diffs = [abs(curr - prev) for prev, curr in zip(values[:-1], values[1:], strict=False)]
    nonzero = [value for value in diffs if value > 1e-9]
    threshold = float(np.median(np.asarray(nonzero, dtype=np.float64))) * 0.25 if nonzero else 0.0
    runs: list[dict[str, Any]] = []
    run_start_idx: int | None = None
    run_length = 1
    for idx, diff in enumerate(diffs, start=1):
        if float(diff) <= float(threshold):
            if run_start_idx is None:
                run_start_idx = idx - 1
                run_length = 2
            else:
                run_length += 1
        else:
            if run_start_idx is not None:
                runs.append(
                    {
                        "start_quarter": str(quarters[run_start_idx]),
                        "end_quarter": str(quarters[idx - 1]),
                        "length": int(run_length),
                    }
                )
            run_start_idx = None
            run_length = 1
    if run_start_idx is not None:
        runs.append(
            {
                "start_quarter": str(quarters[run_start_idx]),
                "end_quarter": str(quarters[-1]),
                "length": int(run_length),
            }
        )
    run_lengths = [int(row["length"]) for row in runs]
    return {
        "run_count": int(len(runs)),
        "max_run_length": int(max(run_lengths) if run_lengths else 0),
        "mean_run_length": float(np.mean(np.asarray(run_lengths, dtype=np.float64))) if run_lengths else 0.0,
        "threshold": float(threshold),
        "runs": runs,
    }


def _plateau_census_contract_payload(archive_run_id: str, *, contract_name: str) -> dict[str, Any]:
    observation_rows, scoring_tiers, _, _ = _contract_setup(archive_run_id, contract_name)
    metrics: dict[str, Any] = {}
    for metric_name in PRIMARY_METRICS:
        supported_rows = [
            row
            for row in sorted(list(observation_rows), key=lambda item: quarter_sort_key(str(item["quarter"])))
            if row.get(metric_name) is not None and suite._metric_tier(row, metric_name) in scoring_tiers
        ]
        quarters = [str(row["quarter"]) for row in supported_rows]
        observed_values = [float(row[metric_name]) for row in supported_rows]
        piecewise = suite._fit_supported_series(
            supported_rows,
            [],
            metric_name=metric_name,
            model_kind="piecewise",
        )
        diagnostics = dict(piecewise["diagnostics"])
        fitted_values = [
            float(value)
            for quarter, value, supported in zip(
                list(diagnostics.get("train_quarters") or []),
                list(diagnostics.get("train_fitted") or []),
                list(diagnostics.get("train_supported") or []),
                strict=False,
            )
            if supported
        ]
        metrics[metric_name] = {
            "observed": _plateau_runs_from_values(quarters, observed_values),
            "piecewise_null": _plateau_runs_from_values(quarters, fitted_values if len(fitted_values) == len(quarters) else observed_values),
            "changepoints": list(diagnostics.get("changepoint_quarters") or []),
            "segment_count": int(diagnostics.get("segment_count") or 0),
        }
    observed_total = sum(int(dict(metric_payload["observed"]).get("run_count") or 0) for metric_payload in metrics.values())
    piecewise_total = sum(int(dict(metric_payload["piecewise_null"]).get("run_count") or 0) for metric_payload in metrics.values())
    status = "plateaus_not_distinct_from_piecewise_null" if piecewise_total >= max(observed_total - 1, 0) else "plateau_signal_nontrivial"
    return {
        "contract": contract_name,
        "metrics": metrics,
        "decision": {
            "status": status,
            "observed_total_runs": int(observed_total),
            "piecewise_total_runs": int(piecewise_total),
        },
    }


def _dense_diagnosed_recalibration_payload(result: dict[str, Any]) -> dict[str, Any]:
    points = sorted(
        _metric_points(result, metric_name="diagnosed_plhiv", allowed_tiers={"exact_observed", "bridge_observed"}),
        key=lambda row: (int(row["year"]), quarter_sort_key(str(row["quarter"]))),
    )
    before_errors: list[float] = []
    after_errors: list[float] = []
    before_residuals: list[float] = []
    after_residuals: list[float] = []
    before_cover = {50: [0, 0], 80: [0, 0], 95: [0, 0]}
    after_cover = {50: [0, 0], 80: [0, 0], 95: [0, 0]}
    calibrated_rows: list[dict[str, Any]] = []
    historical_points: list[dict[str, Any]] = []
    for point in points:
        pool = [row for row in historical_points if row["tier"] == point["tier"]]
        if len(pool) < 8:
            pool = list(historical_points)
        before_prediction = float(point["prediction"])
        target = float(point["target"])
        before_residual = float(point["residual"])
        before_errors.append(abs(before_prediction - target))
        before_residuals.append(before_residual)
        correction = 0.0
        corrected_prediction = float(before_prediction)
        if len(pool) >= 5:
            pool_residuals = np.asarray([float(row["residual"]) for row in pool], dtype=np.float64)
            correction = float(np.median(pool_residuals))
            corrected_prediction = float(before_prediction - correction)
            centered = pool_residuals - float(correction)
            for level, alpha in ((50, 0.5), (80, 0.2), (95, 0.05)):
                lower = float(np.quantile(centered, alpha / 2.0))
                upper = float(np.quantile(centered, 1.0 - (alpha / 2.0)))
                before_cover[level][1] += 1
                if float(before_prediction + lower) <= target <= float(before_prediction + upper):
                    before_cover[level][0] += 1
                after_cover[level][1] += 1
                if float(corrected_prediction + lower) <= target <= float(corrected_prediction + upper):
                    after_cover[level][0] += 1
        after_residual = float(corrected_prediction - target)
        after_errors.append(abs(after_residual))
        after_residuals.append(after_residual)
        calibrated_rows.append(
            {
                **point,
                "correction": float(correction),
                "corrected_prediction": float(corrected_prediction),
                "corrected_residual": float(after_residual),
            }
        )
        historical_points.append(point)
    return {
        "metric": "diagnosed_plhiv",
        "row_count": int(len(points)),
        "before": {
            "raw_mae": float(np.mean(np.asarray(before_errors, dtype=np.float64))) if before_errors else None,
            "mean_residual": float(np.mean(np.asarray(before_residuals, dtype=np.float64))) if before_residuals else None,
            "coverage": {
                str(level): (float(numer) / float(denom)) if denom else None
                for level, (numer, denom) in before_cover.items()
            },
        },
        "after": {
            "raw_mae": float(np.mean(np.asarray(after_errors, dtype=np.float64))) if after_errors else None,
            "mean_residual": float(np.mean(np.asarray(after_residuals, dtype=np.float64))) if after_residuals else None,
            "coverage": {
                str(level): (float(numer) / float(denom)) if denom else None
                for level, (numer, denom) in after_cover.items()
            },
        },
        "rows": calibrated_rows,
        "decision": {
            "status": (
                "calibration_repair_viable"
                if (after_cover[95][1] > 0 and (after_cover[95][0] / max(after_cover[95][1], 1)) > 0.0)
                and (float(np.mean(np.asarray(after_errors, dtype=np.float64))) <= float(np.mean(np.asarray(before_errors, dtype=np.float64))))
                else "calibration_repair_not_yet_sufficient"
            )
        },
    }


def _evaluate_lockbox(
    *,
    archive_run_id: str,
    contract_name: str,
    spec: suite.ExperimentSpec,
    result: dict[str, Any],
    holdout_years: list[int],
) -> dict[str, Any]:
    observation_rows = publish._observation_rows_for_lockbox(
        archive_run_id,
        contract_name=contract_name,
        holdout_years=holdout_years,
    )
    annual_rows = build_annual_anchor_rows(archive_run_id)
    return publish._evaluate_fixed_holdout_experiment(
        spec,
        observation_rows=observation_rows,
        annual_rows=annual_rows,
        holdout_years=holdout_years,
        scoring_tiers=publish._scoring_tiers(contract_name),
        frozen_config=dict(result.get("best_candidate") or {}),
    )


def _shock_gate_decision(
    *,
    check01_exact: dict[str, Any],
    check01_dense: dict[str, Any],
    check02_exact: dict[str, Any],
    check02_dense: dict[str, Any],
    null_exact: dict[str, Any],
    winner_exact: dict[str, Any],
    null_dense: dict[str, Any],
    winner_dense: dict[str, Any],
    cal02: dict[str, Any],
) -> dict[str, Any]:
    exact_shared = int(dict(check01_exact.get("decision") or {}).get("shared_burst_quarter_count") or 0)
    dense_shared = int(dict(check01_dense.get("decision") or {}).get("shared_burst_quarter_count") or 0)
    exact_diagnosed_only_share = float(dict(check01_exact.get("decision") or {}).get("diagnosed_only_share") or 1.0)
    dense_diagnosed_only_share = float(dict(check01_dense.get("decision") or {}).get("diagnosed_only_share") or 1.0)
    plateau_ok = (
        str(dict(check02_exact.get("decision") or {}).get("status")) == "plateau_signal_nontrivial"
        or str(dict(check02_dense.get("decision") or {}).get("status")) == "plateau_signal_nontrivial"
    )
    null_blocks = (
        float(dict(null_exact.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf"))
        <= float(dict(winner_exact.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")) + float(NULL_TOLERANCE)
    ) or (
        float(dict(null_dense.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf"))
        <= float(dict(winner_dense.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")) + float(NULL_TOLERANCE)
    )
    cal_ok = str(dict(cal02.get("decision") or {}).get("status")) == "calibration_repair_viable"
    shock_signal_ok = (
        exact_shared >= 1
        and dense_shared >= 2
        and exact_diagnosed_only_share <= 0.5
        and dense_diagnosed_only_share <= 0.5
    )
    status = "run_minimal_shock_overlay" if shock_signal_ok and plateau_ok and (not null_blocks) and cal_ok else "stop_before_shock_overlay"
    return {
        "status": status,
        "shock_signal_ok": bool(shock_signal_ok),
        "plateau_ok": bool(plateau_ok),
        "null_blocks": bool(null_blocks),
        "calibration_ok": bool(cal_ok),
    }


def _save_residual_burst_graph(payload: dict[str, Any], path: Path) -> None:
    rows = list(payload.get("ordered_quarters") or [])
    if not rows:
        suite._plot_placeholder(path, title=str(payload["contract"]), body="No residual burst data.")
        return
    quarter_labels = [str(row["quarter"]) for row in rows]
    matrix = np.asarray(
        [
            [
                abs(float(dict(row["metrics"]).get(metric_name, {}).get("z_score") or 0.0))
                for metric_name in PRIMARY_METRICS
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    fig, ax = plt.subplots(figsize=(max(10, len(quarter_labels) * 0.4), 4.5))
    im = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", cmap="OrRd")
    ax.set_yticks(range(len(PRIMARY_METRICS)))
    ax.set_yticklabels(PRIMARY_METRICS)
    tick_idx = list(range(0, len(quarter_labels), max(1, len(quarter_labels) // 12 or 1)))
    if tick_idx[-1] != len(quarter_labels) - 1:
        tick_idx.append(len(quarter_labels) - 1)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([quarter_labels[idx] for idx in tick_idx], rotation=45, ha="right")
    ax.set_title(f"{payload['contract']} residual burst heatmap |z|")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_plateau_graph(payloads: list[dict[str, Any]], path: Path) -> None:
    if not payloads:
        suite._plot_placeholder(path, title="Plateau census", body="No plateau payloads.")
        return
    labels: list[str] = []
    observed_counts: list[float] = []
    piecewise_counts: list[float] = []
    for payload in payloads:
        contract_name = str(payload["contract"])
        for metric_name in PRIMARY_METRICS:
            metric_payload = dict(payload["metrics"][metric_name])
            labels.append(f"{contract_name}:{metric_name}")
            observed_counts.append(float(dict(metric_payload["observed"]).get("run_count") or 0))
            piecewise_counts.append(float(dict(metric_payload["piecewise_null"]).get("run_count") or 0))
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.4)))
    ax.barh(y - 0.18, observed_counts, height=0.35, label="Observed")
    ax.barh(y + 0.18, piecewise_counts, height=0.35, label="Piecewise null")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Plateau run count")
    ax.set_title("EXP-G-CHECK-02 plateau census")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_calibration_graph(payload: dict[str, Any], path: Path) -> None:
    before = dict(payload.get("before") or {})
    after = dict(payload.get("after") or {})
    labels = ["MAE", "Mean residual", "95% coverage"]
    before_values = [
        float(before.get("raw_mae") or 0.0),
        abs(float(before.get("mean_residual") or 0.0)),
        float(dict(before.get("coverage") or {}).get("95") or 0.0),
    ]
    after_values = [
        float(after.get("raw_mae") or 0.0),
        abs(float(after.get("mean_residual") or 0.0)),
        float(dict(after.get("coverage") or {}).get("95") or 0.0),
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.18, before_values, width=0.35, label="Before")
    ax.bar(x + 0.18, after_values, width=0.35, label="After")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("EXP-CAL-02 dense diagnosed recalibration")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_shock_check(payload: dict[str, Any]) -> str:
    lines = [
        f"# EXP-G-CHECK-01 {payload['contract']}",
        "",
        f"- Status: `{dict(payload['decision']).get('status', '')}`",
        f"- Shared burst quarters: `{int(dict(payload['decision']).get('shared_burst_quarter_count') or 0)}`",
        f"- Diagnosed-only share: `{float(dict(payload['decision']).get('diagnosed_only_share') or 0.0):.3f}`",
        f"- Correlations: `{dict(payload.get('correlations') or {})}`",
        "",
        "| Quarter | Burst metrics | Mean |z| |",
        "|---|---|---:|",
    ]
    for row in list(payload.get("shared_bursts") or [])[:20]:
        lines.append(f"| {row['quarter']} | `{list(row['burst_metrics'])}` | {float(row['mean_abs_z']):.3f} |")
    return "\n".join(lines) + "\n"


def _markdown_plateau_check(payloads: list[dict[str, Any]]) -> str:
    lines = ["# EXP-G-CHECK-02 Plateau Census", ""]
    for payload in payloads:
        lines.extend(
            [
                f"## {payload['contract']}",
                "",
                f"- Status: `{dict(payload['decision']).get('status', '')}`",
                f"- Observed total runs: `{int(dict(payload['decision']).get('observed_total_runs') or 0)}`",
                f"- Piecewise total runs: `{int(dict(payload['decision']).get('piecewise_total_runs') or 0)}`",
                "",
                "| Metric | Observed runs | Piecewise runs | Changepoints |",
                "|---|---:|---:|---|",
            ]
        )
        for metric_name in PRIMARY_METRICS:
            metric_payload = dict(payload["metrics"][metric_name])
            lines.append(
                f"| `{metric_name}` | {int(dict(metric_payload['observed']).get('run_count') or 0)} | "
                f"{int(dict(metric_payload['piecewise_null']).get('run_count') or 0)} | `{list(metric_payload.get('changepoints') or [])}` |"
            )
        lines.append("")
    return "\n".join(lines)


def _markdown_null_summary(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-G-NULL-01 Piecewise / Changepoint Null",
        "",
        "| Contract | Experiment | Quarterly MAE | Baseline MAE | Lockbox MAE |",
        "|---|---|---:|---:|---:|",
    ]
    for row in list(payload.get("rows") or []):
        lockbox_text = "" if row.get("lockbox_mae") is None else f"{float(row['lockbox_mae']):.6f}"
        lines.append(
            f"| {row['contract']} | {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
            f"{float(row['quarterly_baseline_mae']):.6f} | {lockbox_text} |"
        )
    lines.extend(["", f"- Decision: `{dict(payload['decision']).get('status', '')}`", f"- Why: {dict(payload['decision']).get('why', '')}"])
    return "\n".join(lines) + "\n"


def _markdown_calibration(payload: dict[str, Any]) -> str:
    before = dict(payload.get("before") or {})
    after = dict(payload.get("after") or {})
    return "\n".join(
        [
            "# EXP-CAL-02 Dense Diagnosed Recalibration",
            "",
            f"- Status: `{dict(payload['decision']).get('status', '')}`",
            "",
            "| Stage | Raw MAE | Mean residual | Coverage 50 | Coverage 80 | Coverage 95 |",
            "|---|---:|---:|---:|---:|---:|",
            f"| Before | {float(before.get('raw_mae') or 0.0):.3f} | {float(before.get('mean_residual') or 0.0):.3f} | "
            f"{float(dict(before.get('coverage') or {}).get('50') or 0.0):.3f} | {float(dict(before.get('coverage') or {}).get('80') or 0.0):.3f} | {float(dict(before.get('coverage') or {}).get('95') or 0.0):.3f} |",
            f"| After | {float(after.get('raw_mae') or 0.0):.3f} | {float(after.get('mean_residual') or 0.0):.3f} | "
            f"{float(dict(after.get('coverage') or {}).get('50') or 0.0):.3f} | {float(dict(after.get('coverage') or {}).get('80') or 0.0):.3f} | {float(dict(after.get('coverage') or {}).get('95') or 0.0):.3f} |",
            "",
        ]
    )


def _markdown_shock_overlay(payload: dict[str, Any]) -> str:
    lines = ["# EXP-R10-G1-01-min", ""]
    decision = dict(payload.get("decision") or {})
    lines.append(f"- Status: `{decision.get('status', '')}`")
    lines.append("")
    if decision.get("status") == "skipped_by_gate":
        lines.append(f"- Why: {decision.get('why', '')}")
        lines.append("")
        return "\n".join(lines)
    lines.extend(["| Contract | Quarterly MAE | Baseline MAE | Lockbox MAE |", "|---|---:|---:|---:|"])
    for row in list(payload.get("rows") or []):
        lockbox_text = "" if row.get("lockbox_mae") is None else f"{float(row['lockbox_mae']):.6f}"
        lines.append(
            f"| {row['contract']} | {float(row['quarterly_mean_mae']):.6f} | {float(row['quarterly_baseline_mae']):.6f} | {lockbox_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 GRASP Falsification Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Variant: `{payload['variant']}`",
        "",
        "## Recommendation",
        "",
        f"- Exact benchmark reference: `{payload['recommendation']['exact_candidate_id']}`",
        f"- Dense benchmark reference: `{payload['recommendation']['dense_candidate_id']}`",
        f"- GRASP gate: `{payload['recommendation']['grasp_gate_status']}`",
        "",
        f"- Residual/shock audit: `{payload['artifacts']['check_01_markdown_exact']}`, `{payload['artifacts']['check_01_markdown_dense']}`",
        f"- Plateau census: `{payload['artifacts']['check_02_markdown']}`",
        f"- Piecewise null: `{payload['artifacts']['null_markdown']}`",
        f"- Dense calibration repair: `{payload['artifacts']['calibration_markdown']}`",
        f"- Minimal shock overlay: `{payload['artifacts']['shock_markdown']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_grasp_falsification_batch(
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
    holdout_years = list(lockbox_holdout_years or LOCKBOX_HOLDOUT_YEARS)

    exact_winner_spec = _registered_spec("EXP-R10-M1-F1")
    dense_winner_spec = _registered_spec("EXP-R10-DENSE-M1-H1")
    mechanistic_spec = _registered_spec("EXP-R1")
    exact_null_spec = _make_exact_piecewise_null_spec()
    dense_null_spec = _make_dense_piecewise_null_spec()

    exact_payload = _run_specs_for_contract(
        archive_run_id=archive_run,
        contract_name="exact_only",
        specs=[exact_winner_spec, exact_null_spec, mechanistic_spec],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = _run_specs_for_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        specs=[dense_winner_spec, dense_null_spec, mechanistic_spec],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )

    exact_map = _result_map(exact_payload)
    dense_map = _result_map(dense_payload)
    exact_winner = exact_map["EXP-R10-M1-F1"]
    dense_winner = dense_map["EXP-R10-DENSE-M1-H1"]
    exact_null = exact_map["EXP-G-NULL-01"]
    dense_null = dense_map["EXP-G-NULL-01"]

    check01_exact = _shock_concordance_contract_payload(exact_winner, contract_name="exact_only")
    check01_dense = _shock_concordance_contract_payload(dense_winner, contract_name="purged_dense")
    check02_exact = _plateau_census_contract_payload(archive_run, contract_name="exact_only")
    check02_dense = _plateau_census_contract_payload(archive_run, contract_name="purged_dense")
    cal02 = _dense_diagnosed_recalibration_payload(dense_winner)
    gate = _shock_gate_decision(
        check01_exact=check01_exact,
        check01_dense=check01_dense,
        check02_exact=check02_exact,
        check02_dense=check02_dense,
        null_exact=exact_null,
        winner_exact=exact_winner,
        null_dense=dense_null,
        winner_dense=dense_winner,
        cal02=cal02,
    )

    null_rows = []
    for contract_name, result, spec in (("exact_only", exact_null, exact_null_spec), ("purged_dense", dense_null, dense_null_spec)):
        lockbox_result = _evaluate_lockbox(
            archive_run_id=archive_run,
            contract_name=contract_name,
            spec=spec,
            result=result,
            holdout_years=holdout_years,
        )
        null_rows.append(
            {
                "contract": contract_name,
                "experiment_id": "EXP-G-NULL-01",
                "quarterly_mean_mae": float(dict(result["quarterly_summary"])["candidate_mean_mae"]),
                "quarterly_baseline_mae": float(dict(result["quarterly_summary"])["carry_forward_mean_mae"]),
                "lockbox_mae": float(lockbox_result["quarterly_mean_mae"]),
            }
        )
    null_payload = {
        "rows": null_rows,
        "decision": {
            "status": "null_blocks_grasp" if bool(gate["null_blocks"]) else "null_does_not_block_grasp",
            "why": (
                "The piecewise/changepoint null matches or nearly matches at least one live benchmark, so a richer shock model is not yet justified."
                if bool(gate["null_blocks"])
                else "The piecewise/changepoint null remains materially weaker than the live benchmarks on the frozen contracts."
            ),
        },
    }

    shock_rows: list[dict[str, Any]] = []
    if str(gate["status"]) == "run_minimal_shock_overlay":
        exact_shock_spec = _make_exact_shared_shock_spec()
        dense_shock_spec = _make_dense_shared_shock_spec()
        exact_shock_payload = _run_specs_for_contract(
            archive_run_id=archive_run,
            contract_name="exact_only",
            specs=[exact_shock_spec],
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
        )
        dense_shock_payload = _run_specs_for_contract(
            archive_run_id=archive_run,
            contract_name="purged_dense",
            specs=[dense_shock_spec],
            quarterly_start_year=quarterly_start_year,
            quarterly_end_year=quarterly_end_year,
            quarterly_min_train_years=quarterly_min_train_years,
            annual_start_year=annual_start_year,
            annual_end_year=annual_end_year,
            annual_min_train_years=annual_min_train_years,
            horizon_years=horizon_years,
        )
        exact_shock_result = _result_map(exact_shock_payload)["EXP-R10-G1-01-min"]
        dense_shock_result = _result_map(dense_shock_payload)["EXP-R10-G1-01-min"]
        for contract_name, spec, result in (("exact_only", exact_shock_spec, exact_shock_result), ("purged_dense", dense_shock_spec, dense_shock_result)):
            lockbox_result = _evaluate_lockbox(
                archive_run_id=archive_run,
                contract_name=contract_name,
                spec=spec,
                result=result,
                holdout_years=holdout_years,
            )
            shock_rows.append(
                {
                    "contract": contract_name,
                    "experiment_id": "EXP-R10-G1-01-min",
                    "quarterly_mean_mae": float(dict(result["quarterly_summary"])["candidate_mean_mae"]),
                    "quarterly_baseline_mae": float(dict(result["quarterly_summary"])["carry_forward_mean_mae"]),
                    "lockbox_mae": float(lockbox_result["quarterly_mean_mae"]),
                }
            )
        shock_payload = {"decision": {"status": "executed"}, "rows": shock_rows}
    else:
        shock_payload = {
            "decision": {
                "status": "skipped_by_gate",
                "why": (
                    f"Gate failed: shock_signal_ok={gate['shock_signal_ok']}, plateau_ok={gate['plateau_ok']}, "
                    f"null_blocks={gate['null_blocks']}, calibration_ok={gate['calibration_ok']}."
                ),
            },
            "rows": [],
        }

    check01_exact_graph = analysis_dir / "exp_g_check_01_exact.png"
    check01_dense_graph = analysis_dir / "exp_g_check_01_dense.png"
    check02_graph = analysis_dir / "exp_g_check_02_plateau_census.png"
    cal02_graph = analysis_dir / "exp_cal_02_dense_diagnosed_recalibration.png"
    _save_residual_burst_graph(check01_exact, check01_exact_graph)
    _save_residual_burst_graph(check01_dense, check01_dense_graph)
    _save_plateau_graph([check02_exact, check02_dense], check02_graph)
    _save_calibration_graph(cal02, cal02_graph)

    (analysis_dir / "exp_g_check_01_exact.md").write_text(_markdown_shock_check(check01_exact), encoding="utf-8")
    (analysis_dir / "exp_g_check_01_dense.md").write_text(_markdown_shock_check(check01_dense), encoding="utf-8")
    (analysis_dir / "exp_g_check_02_plateau_census.md").write_text(_markdown_plateau_check([check02_exact, check02_dense]), encoding="utf-8")
    (analysis_dir / "exp_g_null_01_piecewise_baseline.md").write_text(_markdown_null_summary(null_payload), encoding="utf-8")
    (analysis_dir / "exp_cal_02_dense_diagnosed_recalibration.md").write_text(_markdown_calibration(cal02), encoding="utf-8")
    (analysis_dir / "exp_r10_g1_01_min.md").write_text(_markdown_shock_overlay(shock_payload), encoding="utf-8")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "variant": "evidence-to-model-loop",
        "gate": gate,
        "recommendation": {
            "exact_candidate_id": "EXP-R10-M1-F1",
            "dense_candidate_id": "EXP-R10-DENSE-M1-H1",
            "grasp_gate_status": str(gate["status"]),
        },
        "artifacts": {
            "check_01_markdown_exact": "exp_g_check_01_exact.md",
            "check_01_markdown_dense": "exp_g_check_01_dense.md",
            "check_01_graph_exact": check01_exact_graph.name,
            "check_01_graph_dense": check01_dense_graph.name,
            "check_02_markdown": "exp_g_check_02_plateau_census.md",
            "check_02_graph": check02_graph.name,
            "null_markdown": "exp_g_null_01_piecewise_baseline.md",
            "calibration_markdown": "exp_cal_02_dense_diagnosed_recalibration.md",
            "calibration_graph": cal02_graph.name,
            "shock_markdown": "exp_r10_g1_01_min.md",
        },
        "exact_payload": exact_payload,
        "dense_payload": dense_payload,
        "check_01": {"exact_only": check01_exact, "purged_dense": check01_dense},
        "check_02": {"exact_only": check02_exact, "purged_dense": check02_dense},
        "null_baseline": null_payload,
        "calibration": cal02,
        "shock_overlay": shock_payload,
    }
    (analysis_dir / "tr_v3_grasp_falsification_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    write_json(analysis_dir / "tr_v3_grasp_falsification_batch_report.json", payload)
    return payload


def _build_arg_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run the TR-V3 GRASP falsification batch.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=None)
    parser.add_argument("--quarterly-start-year", type=int, default=2010)
    parser.add_argument("--quarterly-end-year", type=int, default=2025)
    parser.add_argument("--quarterly-min-train-years", type=int, default=3)
    parser.add_argument("--annual-start-year", type=int, default=2010)
    parser.add_argument("--annual-end-year", type=int, default=2024)
    parser.add_argument("--annual-min-train-years", type=int, default=5)
    parser.add_argument("--horizon-years", type=int, default=1)
    parser.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=list(LOCKBOX_HOLDOUT_YEARS))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_tr_v3_grasp_falsification_batch(
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
