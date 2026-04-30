from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_publishability_batch as publish
from epigraph_ph.phase3.tr_v3_05_autoresearch import PRIMARY_METRICS, build_annual_anchor_rows, quarter_year
from epigraph_ph.runtime import ensure_dir, write_json


CONTRACT_CHOICES: tuple[str, ...] = ("exact_only", "legacy_dense", "purged_dense")
CALIBRATION_LEVELS: tuple[float, ...] = (0.5, 0.8, 0.95)
ERA_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("pre_covid", 2013, 2019),
    ("covid_shock", 2020, 2021),
    ("recovery_tail", 2022, 2025),
)
EXACT_EXPERIMENT_IDS: list[str] = [
    "EXP-V2",
    "EXP-R10-EXACT-CHAMPION",
    "EXP-R10-M1-B1",
    "EXP-R10-M1-F1",
    "EXP-R1",
]
DENSE_EXPERIMENT_IDS: list[str] = [
    "EXP-V2",
    "EXP-R10-DENSE-CHAMPION",
    "EXP-R10-DENSE-H1",
    "EXP-R10-DENSE-M1-H1",
    "EXP-R10-DENSE-M1-B1-H1",
    "EXP-R10-DENSE-M1-F1-H1",
    "EXP-R1",
]
FROZEN_REFERENCES: dict[str, dict[str, Any]] = {
    "exact_only": {"winner_id": "EXP-R10-M1-F1", "winner_mae": 0.069471},
    "purged_dense": {"winner_id": "EXP-R10-DENSE-M1-H1", "winner_mae": 0.085418},
    "lockbox_exact_only": {"winner_id": "EXP-R10-M1-F1", "winner_mae": 0.045118},
    "lockbox_purged_dense": {"winner_id": "EXP-R10-DENSE-M1-H1", "winner_mae": 0.063772},
}


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _metric_tier(target_row: dict[str, Any], metric_name: str) -> str:
    metric_tiers = target_row.get("metric_tiers")
    if isinstance(metric_tiers, dict) and metric_name in metric_tiers:
        return str(metric_tiers[metric_name])
    tier_name = target_row.get(f"{metric_name}_tier")
    if isinstance(tier_name, str):
        return str(tier_name)
    return "exact_observed"


def _contract_setup(archive_run_id: str, contract_name: str) -> tuple[list[dict[str, Any]], set[str], bool, list[dict[str, Any]] | None]:
    if contract_name == "exact_only":
        return suite.build_quarterly_observation_rows(archive_run_id), {"exact_observed"}, False, None
    if contract_name in {"legacy_dense", "purged_dense"}:
        dense_payload = suite._build_dense_contract_payload(archive_run_id)
        dense_rows = list(dense_payload["rows"])
        return dense_rows, {"exact_observed", "bridge_observed"}, contract_name == "purged_dense", dense_rows
    raise ValueError(f"Unsupported contract_name: {contract_name}")


def _dense_contract_alias(contract_name: str) -> str:
    return "exact_only" if contract_name == "exact_only" else "dense_train_observed_score"


def _run_selected_suite_contract(
    *,
    archive_run_id: str,
    contract_name: str,
    experiment_ids: list[str],
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
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    results = [
        suite._evaluate_experiment_spec(
            spec_map[experiment_id],
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
        for experiment_id in experiment_ids
    ]
    for idx, result in enumerate(results):
        if str(result.get("diagnostic_kind") or "") != "endpoint_tier_audit":
            continue
        results[idx] = {
            **result,
            "decision": "diagnostic_only",
            "endpoint_tier_audit": suite._build_endpoint_tier_audit_payload(
                results,
                quarterly_contract=_dense_contract_alias(contract_name),
            ),
        }
    return {
        "archive_run_id": archive_run_id,
        "contract_name": contract_name,
        "results": results,
    }


def _era_name(year: int) -> str:
    for era_name, start_year, end_year in ERA_BUCKETS:
        if int(start_year) <= int(year) <= int(end_year):
            return era_name
    return "outside_window"


def _empty_metric_bucket() -> dict[str, Any]:
    return {
        "overall_raw_errors": [],
        "exact_raw_errors": [],
        "bridge_raw_errors": [],
    }


def _summarize_metric_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    overall = list(bucket["overall_raw_errors"])
    exact = list(bucket["exact_raw_errors"])
    bridge = list(bucket["bridge_raw_errors"])
    return {
        "overall_raw_mae": float(np.mean(overall)) if overall else None,
        "overall_count": len(overall),
        "exact_raw_mae": float(np.mean(exact)) if exact else None,
        "exact_count": len(exact),
        "bridge_raw_mae": float(np.mean(bridge)) if bridge else None,
        "bridge_count": len(bridge),
    }


def _result_era_audit(result: dict[str, Any]) -> dict[str, Any]:
    era_rows: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        era_name: {
            "candidate": {metric_name: _empty_metric_bucket() for metric_name in suite.AUDIT_METRICS},
            "carry_forward": {metric_name: _empty_metric_bucket() for metric_name in suite.AUDIT_METRICS},
        }
        for era_name, _, _ in ERA_BUCKETS
    }
    for split in list(result.get("quarterly_rows") or []):
        targets = list(split.get("holdout_target_rows") or [])
        candidate_rows = list(split.get("candidate_prediction_rows") or [])
        baseline_rows = list(split.get("carry_forward_prediction_rows") or [])
        for target_row, candidate_row, baseline_row in zip(targets, candidate_rows, baseline_rows, strict=False):
            era_name = _era_name(quarter_year(str(target_row["quarter"])))
            if era_name not in era_rows:
                continue
            for metric_name in suite.AUDIT_METRICS:
                target_value = target_row.get(metric_name)
                if target_value is None:
                    continue
                target_tier = _metric_tier(target_row, metric_name)
                if target_tier not in {"exact_observed", "bridge_observed"}:
                    continue
                for model_name, prediction_row in (("candidate", candidate_row), ("carry_forward", baseline_row)):
                    prediction_value = prediction_row.get(metric_name)
                    if prediction_value is None:
                        continue
                    raw_error = abs(float(prediction_value) - float(target_value))
                    bucket = era_rows[era_name][model_name][metric_name]
                    bucket["overall_raw_errors"].append(raw_error)
                    if target_tier == "exact_observed":
                        bucket["exact_raw_errors"].append(raw_error)
                    elif target_tier == "bridge_observed":
                        bucket["bridge_raw_errors"].append(raw_error)
    return {
        era_name: {
            model_name: {
                metric_name: _summarize_metric_bucket(metric_bucket)
                for metric_name, metric_bucket in metrics.items()
            }
            for model_name, metrics in model_rows.items()
        }
        for era_name, model_rows in era_rows.items()
    }


def _collect_filtered_residuals(
    quarterly_rows: list[dict[str, Any]],
    *,
    metric_name: str,
    allowed_tiers: set[str],
) -> list[float]:
    values: list[float] = []
    for split in quarterly_rows:
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            if _metric_tier(target_row, metric_name) not in allowed_tiers:
                continue
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            values.append(float(prediction_value) - float(target_value))
    return values


def _split_points_by_metric(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
) -> dict[str, list[dict[str, Any]]]:
    points = {metric_name: [] for metric_name in PRIMARY_METRICS}
    for split_idx, split in enumerate(quarterly_rows):
        targets = list(split.get("holdout_target_rows") or [])
        predictions = list(split.get("candidate_prediction_rows") or [])
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
                        "split_idx": int(split_idx),
                        "quarter": str(target_row["quarter"]),
                        "tier": tier_name,
                        "target": float(target_value),
                        "prediction": float(prediction_value),
                        "residual": float(prediction_value) - float(target_value),
                    }
                )
    return points


def _coverage_summary(
    quarterly_rows: list[dict[str, Any]],
    *,
    allowed_tiers: set[str],
    levels: tuple[float, ...] = CALIBRATION_LEVELS,
) -> dict[str, Any]:
    points_by_metric = _split_points_by_metric(quarterly_rows, allowed_tiers=allowed_tiers)
    metrics: dict[str, Any] = {}
    for metric_name, points in points_by_metric.items():
        metric_payload: dict[str, Any] = {"overall": {}, "exact_observed": {}, "bridge_observed": {}}
        for tier_name in metric_payload:
            tier_points = points if tier_name == "overall" else [point for point in points if point["tier"] == tier_name]
            for level in levels:
                alpha = 1.0 - float(level)
                covered = 0
                widths: list[float] = []
                counted = 0
                for point in tier_points:
                    calibration_residuals = [
                        float(other["residual"])
                        for other in tier_points
                        if int(other["split_idx"]) != int(point["split_idx"])
                    ]
                    if len(calibration_residuals) < 5:
                        continue
                    lower = float(np.quantile(calibration_residuals, alpha / 2.0))
                    upper = float(np.quantile(calibration_residuals, 1.0 - (alpha / 2.0)))
                    lower_bound = float(point["prediction"]) + lower
                    upper_bound = float(point["prediction"]) + upper
                    if lower_bound <= float(point["target"]) <= upper_bound:
                        covered += 1
                    widths.append(upper - lower)
                    counted += 1
                metric_payload[tier_name][f"{int(level * 100)}"] = {
                    "nominal": float(level),
                    "coverage": (float(covered) / float(counted)) if counted else None,
                    "count": int(counted),
                    "mean_width": float(np.mean(widths)) if widths else None,
                }
        metrics[metric_name] = metric_payload
    return {"levels": [float(level) for level in levels], "by_metric": metrics}


def _calibration_payload(result: dict[str, Any], *, allowed_tiers: set[str]) -> dict[str, Any]:
    quarterly_rows = list(result.get("quarterly_rows") or [])
    split_rows = [
        {
            "holdout_years": list(split.get("holdout_years") or []),
            "candidate_mae": float(dict(split.get("candidate") or {}).get("mae") or float("inf")),
            "baseline_mae": float(dict(split.get("carry_forward") or {}).get("mae") or float("inf")),
            "suppression_honesty_flag": str(dict(split.get("endpoint_audit") or {}).get("suppression_honesty_flag") or "unsupported_or_unclaimed"),
        }
        for split in quarterly_rows
    ]
    by_metric: dict[str, Any] = {}
    for metric_name in PRIMARY_METRICS:
        residuals = np.asarray(
            _collect_filtered_residuals(quarterly_rows, metric_name=metric_name, allowed_tiers=allowed_tiers),
            dtype=np.float64,
        )
        by_metric[metric_name] = {
            "count": int(residuals.size),
            "mean_residual": float(np.mean(residuals)) if residuals.size else 0.0,
            "median_residual": float(np.median(residuals)) if residuals.size else 0.0,
            "std_residual": float(np.std(residuals)) if residuals.size else 0.0,
            "q10_residual": float(np.quantile(residuals, 0.1)) if residuals.size else 0.0,
            "q90_residual": float(np.quantile(residuals, 0.9)) if residuals.size else 0.0,
        }
    return {
        "experiment_id": str(result["experiment_id"]),
        "quarterly_mean_mae": float(dict(result.get("quarterly_summary") or {}).get("candidate_mean_mae") or float("inf")),
        "quarterly_baseline_mean_mae": float(dict(result.get("quarterly_summary") or {}).get("carry_forward_mean_mae") or float("inf")),
        "split_rows": split_rows,
        "by_metric": by_metric,
        "coverage": _coverage_summary(quarterly_rows, allowed_tiers=allowed_tiers),
        "endpoint_audit_summary": dict(dict(result.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {}),
    }


def _save_coverage_graph(payload: dict[str, Any], path: Path) -> None:
    coverage = dict(payload.get("coverage") or {})
    levels = [float(level) for level in list(coverage.get("levels") or [])]
    by_metric = dict(coverage.get("by_metric") or {})
    metrics = [metric_name for metric_name in PRIMARY_METRICS if metric_name in by_metric]
    if not metrics:
        suite._plot_placeholder(path, title=str(payload["experiment_id"]), body="No coverage data.")
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4), constrained_layout=True)
    axes_list = [axes] if not isinstance(axes, np.ndarray) else list(np.asarray(axes).reshape(-1))
    for ax, metric_name in zip(axes_list, metrics, strict=False):
        metric_payload = dict(by_metric.get(metric_name) or {})
        overall = dict(metric_payload.get("overall") or {})
        empirical = []
        for level in levels:
            row = dict(overall.get(f"{int(level * 100)}") or {})
            empirical.append(float(row["coverage"]) if row.get("coverage") is not None else np.nan)
        ax.plot(levels, levels, linestyle="--", color="black", label="nominal")
        ax.plot(levels, empirical, marker="o", color="#1f77b4", label="empirical")
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(min(levels) - 0.02, max(levels) + 0.02)
        ax.set_title(metric_name.replace("_", " "))
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{payload['experiment_id']} interval coverage")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _result_row(result: dict[str, Any], *, contract_name: str) -> dict[str, Any]:
    audit = dict(dict(result.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {})
    candidate_by_metric = dict(dict(audit.get("candidate") or {}).get("by_metric") or {})
    return {
        "contract": contract_name,
        "experiment_id": str(result["experiment_id"]),
        "quarterly_mean_mae": float(dict(result["quarterly_summary"])["candidate_mean_mae"]),
        "quarterly_baseline_mae": float(dict(result["quarterly_summary"])["carry_forward_mean_mae"]),
        "diagnosed_raw_mae": float(dict(candidate_by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
        "art_raw_mae": float(dict(candidate_by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
        "flow_raw_mae": float(dict(candidate_by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
        "suppression_honesty_flags": dict(audit.get("suppression_honesty_flags") or {}),
    }


def _frozen_protocol_payload(
    *,
    exact_map: dict[str, dict[str, Any]],
    legacy_map: dict[str, dict[str, Any]],
    purged_map: dict[str, dict[str, Any]],
    lockbox_payload: dict[str, Any],
) -> dict[str, Any]:
    rows = [
        _result_row(exact_map["EXP-R10-M1-F1"], contract_name="exact_only"),
        _result_row(exact_map["EXP-R10-M1-B1"], contract_name="exact_only"),
        _result_row(exact_map["EXP-R1"], contract_name="exact_only"),
        _result_row(legacy_map["EXP-R10-DENSE-M1-H1"], contract_name="legacy_dense"),
        _result_row(legacy_map["EXP-R10-DENSE-M1-F1-H1"], contract_name="legacy_dense"),
        _result_row(legacy_map["EXP-R1"], contract_name="legacy_dense"),
        _result_row(purged_map["EXP-R10-DENSE-M1-H1"], contract_name="purged_dense"),
        _result_row(purged_map["EXP-R10-DENSE-M1-F1-H1"], contract_name="purged_dense"),
        _result_row(purged_map["EXP-R1"], contract_name="purged_dense"),
    ]
    exact_lockbox_rows = list(dict(lockbox_payload["contracts"]["exact_only"]).get("rows") or [])
    purged_lockbox_rows = list(dict(lockbox_payload["contracts"]["purged_dense"]).get("rows") or [])
    exact_lockbox_winner = min(exact_lockbox_rows, key=lambda row: float(row["quarterly_mean_mae"]))
    purged_lockbox_winner = min(purged_lockbox_rows, key=lambda row: float(row["quarterly_mean_mae"]))
    purged_best_mae = min(
        float(dict(purged_map["EXP-R10-DENSE-M1-H1"]["quarterly_summary"])["candidate_mean_mae"]),
        float(dict(purged_map["EXP-R10-DENSE-M1-F1-H1"]["quarterly_summary"])["candidate_mean_mae"]),
    )
    purged_best_id = (
        "EXP-R10-DENSE-M1-F1-H1"
        if float(dict(purged_map["EXP-R10-DENSE-M1-F1-H1"]["quarterly_summary"])["candidate_mean_mae"]) <= float(dict(purged_map["EXP-R10-DENSE-M1-H1"]["quarterly_summary"])["candidate_mean_mae"])
        else "EXP-R10-DENSE-M1-H1"
    )
    return {
        "rows": rows,
        "current_winners": {
            "exact_only": {
                "winner_id": "EXP-R10-M1-F1",
                "winner_mae": float(dict(exact_map["EXP-R10-M1-F1"]["quarterly_summary"])["candidate_mean_mae"]),
                "delta_vs_frozen": float(dict(exact_map["EXP-R10-M1-F1"]["quarterly_summary"])["candidate_mean_mae"]) - float(FROZEN_REFERENCES["exact_only"]["winner_mae"]),
            },
            "purged_dense": {
                "winner_id": purged_best_id,
                "winner_mae": purged_best_mae,
                "delta_vs_frozen": purged_best_mae - float(FROZEN_REFERENCES["purged_dense"]["winner_mae"]),
            },
            "lockbox_exact_only": {
                "winner_id": str(exact_lockbox_winner["experiment_id"]),
                "winner_mae": float(exact_lockbox_winner["quarterly_mean_mae"]),
                "delta_vs_frozen": float(exact_lockbox_winner["quarterly_mean_mae"]) - float(FROZEN_REFERENCES["lockbox_exact_only"]["winner_mae"]),
            },
            "lockbox_purged_dense": {
                "winner_id": str(purged_lockbox_winner["experiment_id"]),
                "winner_mae": float(purged_lockbox_winner["quarterly_mean_mae"]),
                "delta_vs_frozen": float(purged_lockbox_winner["quarterly_mean_mae"]) - float(FROZEN_REFERENCES["lockbox_purged_dense"]["winner_mae"]),
            },
        },
    }


def _endpoint_tier_era_payload(contract_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tracked_results: list[dict[str, Any]] = []
    for contract_name, payload in contract_payloads.items():
        result_map = _suite_result_map(payload)
        tracked_ids = (
            ["EXP-R10-M1-F1", "EXP-R10-M1-B1", "EXP-R1"]
            if contract_name == "exact_only"
            else ["EXP-R10-DENSE-M1-H1", "EXP-R10-DENSE-M1-F1-H1", "EXP-R10-DENSE-M1-B1-H1", "EXP-R1"]
        )
        for experiment_id in tracked_ids:
            result = result_map.get(experiment_id)
            if not result:
                continue
            tracked_results.append(
                {
                    "contract": contract_name,
                    "experiment_id": experiment_id,
                    "quarterly_mean_mae": float(dict(result["quarterly_summary"])["candidate_mean_mae"]),
                    "quarterly_baseline_mae": float(dict(result["quarterly_summary"])["carry_forward_mean_mae"]),
                    "overall_endpoint_tier_audit": dict(dict(result.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {}),
                    "era_audit": _result_era_audit(result),
                }
            )
    return {"era_buckets": list(ERA_BUCKETS), "tracked_results": tracked_results}


def _dense_transfer_payload(purged_map: dict[str, dict[str, Any]], legacy_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for contract_name, result_map in (("purged_dense", purged_map), ("legacy_dense", legacy_map)):
        for experiment_id in ("EXP-R10-DENSE-M1-H1", "EXP-R10-DENSE-M1-B1-H1", "EXP-R10-DENSE-M1-F1-H1"):
            result = result_map.get(experiment_id)
            if not result:
                continue
            rows.append(_result_row(result, contract_name=contract_name))
    incumbent = next(row for row in rows if row["contract"] == "purged_dense" and row["experiment_id"] == "EXP-R10-DENSE-M1-H1")
    transfer = next(row for row in rows if row["contract"] == "purged_dense" and row["experiment_id"] == "EXP-R10-DENSE-M1-F1-H1")
    return {
        "rows": rows,
        "decision": {
            "status": "keep_transfer"
            if float(transfer["quarterly_mean_mae"]) < float(incumbent["quarterly_mean_mae"])
            else "revert_transfer",
            "why": (
                "Keep the dense F1 transfer because purged-dense MAE improved while suppression honesty stayed unclaimed."
                if float(transfer["quarterly_mean_mae"]) < float(incumbent["quarterly_mean_mae"])
                else "Revert the dense F1 transfer because it did not beat the current dense winner on the primary purged-dense metric."
            ),
        },
    }


def _save_observation_graph(result: dict[str, Any], path: Path) -> str | None:
    if suite._save_observation_curve_graph(result, path):
        return path.name
    return None


def _save_protocol_overview(payload: dict[str, Any], path: Path) -> None:
    rows = list(payload.get("rows") or [])
    if not rows:
        suite._plot_placeholder(path, title="EXP-EVAL-01", body="No protocol rows.")
        return
    labels = [f"{row['contract']}:{row['experiment_id']}" for row in rows]
    candidate = [float(row["quarterly_mean_mae"]) for row in rows]
    baseline = [float(row["quarterly_baseline_mae"]) for row in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(5.0, len(labels) * 0.45)))
    ax.barh(y - 0.18, baseline, height=0.35, label="Carry-forward")
    ax.barh(y + 0.18, candidate, height=0.35, label="Candidate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Quarterly normalized MAE")
    ax.set_title("EXP-EVAL-01 Frozen Protocol Rebuild")
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_era_audit_graph(payload: dict[str, Any], path: Path) -> None:
    tracked = list(payload.get("tracked_results") or [])
    if not tracked:
        suite._plot_placeholder(path, title="EXP-EVAL-02", body="No tracked results.")
        return
    era_names = [era_name for era_name, _, _ in ERA_BUCKETS]
    labels = [f"{row['contract']}:{row['experiment_id']}" for row in tracked]
    matrix = np.full((len(labels), len(era_names)), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(tracked):
        era_audit = dict(row.get("era_audit") or {})
        for col_idx, era_name in enumerate(era_names):
            metric_payload = dict(dict(dict(era_audit.get(era_name) or {}).get("candidate") or {}).get("diagnosed_plhiv") or {})
            if metric_payload.get("overall_raw_mae") is not None:
                matrix[row_idx, col_idx] = float(metric_payload["overall_raw_mae"])
    fig, ax = plt.subplots(figsize=(10, max(4.5, len(labels) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(len(era_names)))
    ax.set_xticklabels(era_names)
    ax.set_title("EXP-EVAL-02 diagnosed raw MAE by era")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_frozen_protocol(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-EVAL-01 Frozen Protocol Rebuild",
        "",
        "| Contract | Experiment | Quarterly MAE | Baseline MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Suppression honesty |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("rows") or []):
        lines.append(
            f"| {row['contract']} | {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
            f"{float(row['quarterly_baseline_mae']):.6f} | {float(row['diagnosed_raw_mae']):.3f} | "
            f"{float(row['art_raw_mae']):.3f} | {float(row['flow_raw_mae']):.3f} | `{dict(row['suppression_honesty_flags'])}` |"
        )
    lines.extend(["", "## Winner Rebuild", "", "| Protocol | Winner | MAE | Delta vs frozen |", "|---|---|---:|---:|"])
    for protocol_name, row in dict(payload.get("current_winners") or {}).items():
        lines.append(
            f"| {protocol_name} | {row['winner_id']} | {float(row['winner_mae']):.6f} | {float(row['delta_vs_frozen']):.6f} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_endpoint_tier_era(payload: dict[str, Any]) -> str:
    lines = [
        "# EXP-EVAL-02 Endpoint Tier Era Audit",
        "",
        f"- Era buckets: `{list(payload.get('era_buckets') or [])}`",
        "",
    ]
    for row in list(payload.get("tracked_results") or []):
        lines.extend(
            [
                f"## {row['contract']} :: {row['experiment_id']}",
                "",
                f"- Quarterly mean MAE: `{float(row['quarterly_mean_mae']):.6f}`",
                f"- Baseline mean MAE: `{float(row['quarterly_baseline_mae']):.6f}`",
                f"- Suppression honesty flags: `{dict(dict(row['overall_endpoint_tier_audit']).get('suppression_honesty_flags') or {})}`",
                "",
                "| Era | Metric | Candidate raw MAE | Exact count | Bridge count | Baseline raw MAE |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        era_audit = dict(row.get("era_audit") or {})
        for era_name, _, _ in ERA_BUCKETS:
            for metric_name in suite.AUDIT_METRICS:
                candidate_metric = dict(dict(dict(era_audit.get(era_name) or {}).get("candidate") or {}).get(metric_name) or {})
                baseline_metric = dict(dict(dict(era_audit.get(era_name) or {}).get("carry_forward") or {}).get(metric_name) or {})
                candidate_raw = candidate_metric.get("overall_raw_mae")
                baseline_raw = baseline_metric.get("overall_raw_mae")
                candidate_text = "" if candidate_raw is None else f"{float(candidate_raw):.3f}"
                baseline_text = "" if baseline_raw is None else f"{float(baseline_raw):.3f}"
                lines.append(
                    f"| {era_name} | `{metric_name}` | {candidate_text} | "
                    f"{int(candidate_metric.get('exact_count') or 0)} | {int(candidate_metric.get('bridge_count') or 0)} | "
                    f"{baseline_text} |"
                )
        lines.append("")
    return "\n".join(lines)


def _markdown_calibration_coverage(payload: dict[str, Any]) -> str:
    lines = ["# EXP-EVAL-03 Calibration And Interval Coverage", ""]
    for experiment_id, row in dict(payload.get("experiments") or {}).items():
        lines.extend(
            [
                f"## {experiment_id}",
                "",
                f"- Contract: `{row['contract']}`",
                f"- Calibration graph: `{row['calibration_graph_file']}`",
                f"- Coverage graph: `{row['coverage_graph_file']}`",
                f"- Quarterly mean MAE: `{float(row['quarterly_mean_mae']):.6f}`",
                "",
                "| Metric | Count | Mean residual | q10 | q90 | Coverage 50 | Coverage 80 | Coverage 95 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        by_metric = dict(dict(row.get("calibration") or {}).get("by_metric") or {})
        coverage_by_metric = dict(dict(dict(row.get("calibration") or {}).get("coverage") or {}).get("by_metric") or {})
        for metric_name in PRIMARY_METRICS:
            metric_payload = dict(by_metric.get(metric_name) or {})
            coverage_payload = dict(dict(coverage_by_metric.get(metric_name) or {}).get("overall") or {})
            cov_50 = dict(coverage_payload.get("50") or {})
            cov_80 = dict(coverage_payload.get("80") or {})
            cov_95 = dict(coverage_payload.get("95") or {})
            cov_50_text = "" if cov_50.get("coverage") is None else f"{float(cov_50['coverage']):.3f}"
            cov_80_text = "" if cov_80.get("coverage") is None else f"{float(cov_80['coverage']):.3f}"
            cov_95_text = "" if cov_95.get("coverage") is None else f"{float(cov_95['coverage']):.3f}"
            lines.append(
                f"| `{metric_name}` | {int(metric_payload.get('count') or 0)} | {float(metric_payload.get('mean_residual') or 0.0):.3f} | "
                f"{float(metric_payload.get('q10_residual') or 0.0):.3f} | {float(metric_payload.get('q90_residual') or 0.0):.3f} | "
                f"{cov_50_text} | {cov_80_text} | {cov_95_text} |"
            )
        lines.append("")
    return "\n".join(lines)


def _markdown_dense_transfer(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    lines = [
        "# EXP-R10-DENSE-M1-F1-H1",
        "",
        f"- Decision: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "| Contract | Experiment | Quarterly MAE | Baseline MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Suppression honesty |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("rows") or []):
        lines.append(
            f"| {row['contract']} | {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
            f"{float(row['quarterly_baseline_mae']):.6f} | {float(row['diagnosed_raw_mae']):.3f} | "
            f"{float(row['art_raw_mae']):.3f} | {float(row['flow_raw_mae']):.3f} | `{dict(row['suppression_honesty_flags'])}` |"
        )
    return "\n".join(lines) + "\n"


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Eval Hardening Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Variant: `{payload['variant']}`",
        "",
        "## Recommendation",
        "",
        f"- Exact candidate remains: `{payload['recommendation']['exact_candidate_id']}`",
        f"- Dense candidate remains: `{payload['recommendation']['dense_candidate_id']}`",
        f"- Dense transfer status: `{payload['recommendation']['dense_transfer_status']}`",
        "",
        f"- Frozen protocol artifact: `{payload['artifacts']['eval_01_markdown']}`",
        f"- Endpoint/tier/era artifact: `{payload['artifacts']['eval_02_markdown']}`",
        f"- Calibration/coverage artifact: `{payload['artifacts']['eval_03_markdown']}`",
        f"- Dense transfer artifact: `{payload['artifacts']['dense_transfer_markdown']}`",
        "",
    ]
    return "\n".join(lines)


def run_tr_v3_eval_hardening_batch(
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

    exact_payload = _run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="exact_only",
        experiment_ids=list(EXACT_EXPERIMENT_IDS),
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    legacy_payload = _run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="legacy_dense",
        experiment_ids=list(DENSE_EXPERIMENT_IDS),
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    purged_payload = _run_selected_suite_contract(
        archive_run_id=archive_run,
        contract_name="purged_dense",
        experiment_ids=list(DENSE_EXPERIMENT_IDS),
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )

    exact_map = _suite_result_map(exact_payload)
    legacy_map = _suite_result_map(legacy_payload)
    purged_map = _suite_result_map(purged_payload)
    holdout_years = sorted(set(int(year) for year in (lockbox_holdout_years or [2025])))
    annual_rows = build_annual_anchor_rows(archive_run)
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}

    lockbox_contracts: dict[str, Any] = {}
    for contract_name, result_map in (("exact_only", exact_map), ("legacy_dense", legacy_map), ("purged_dense", purged_map)):
        scoring_tiers = publish._scoring_tiers(contract_name)
        observation_rows = publish._observation_rows_for_lockbox(archive_run, contract_name=contract_name, holdout_years=holdout_years)
        tracked_ids = (
            ["EXP-R10-M1-F1", "EXP-R10-M1-B1", "EXP-R1"]
            if contract_name == "exact_only"
            else ["EXP-R10-DENSE-M1-H1", "EXP-R10-DENSE-M1-F1-H1", "EXP-R10-DENSE-M1-B1-H1", "EXP-R1"]
        )
        rows = []
        for experiment_id in tracked_ids:
            frozen = dict(dict(result_map.get(experiment_id) or {}).get("best_candidate") or {})
            if not frozen:
                continue
            rows.append(
                publish._evaluate_fixed_holdout_experiment(
                    spec_map[experiment_id],
                    observation_rows=observation_rows,
                    annual_rows=annual_rows,
                    holdout_years=holdout_years,
                    scoring_tiers=scoring_tiers,
                    frozen_config=frozen,
                )
            )
        rows = sorted(rows, key=lambda row: float(row["quarterly_mean_mae"]))
        graph_path = analysis_dir / f"lockbox_{contract_name}.png"
        publish._save_lockbox_graph({"rows": rows}, graph_path, title=f"TR-V3 eval hardening lockbox ({contract_name})")
        lockbox_contracts[contract_name] = {"rows": rows, "graph_file": graph_path.name}

    frozen_protocol = _frozen_protocol_payload(
        exact_map=exact_map,
        legacy_map=legacy_map,
        purged_map=purged_map,
        lockbox_payload={"contracts": lockbox_contracts},
    )
    eval01_graph = analysis_dir / "exp_eval_01_frozen_protocol_rebuild.png"
    _save_protocol_overview(frozen_protocol, eval01_graph)

    endpoint_tier_era = _endpoint_tier_era_payload(
        {"exact_only": exact_payload, "legacy_dense": legacy_payload, "purged_dense": purged_payload}
    )
    eval02_graph = analysis_dir / "exp_eval_02_endpoint_tier_era_audit.png"
    _save_era_audit_graph(endpoint_tier_era, eval02_graph)

    calibration_rows: dict[str, Any] = {}
    calibration_targets = [
        ("exact_only", exact_map["EXP-R10-M1-F1"]),
        ("purged_dense", purged_map["EXP-R10-DENSE-M1-H1"]),
        ("purged_dense", purged_map["EXP-R10-DENSE-M1-F1-H1"]),
        ("exact_only", exact_map["EXP-R1"]),
    ]
    for contract_name, result in calibration_targets:
        allowed_tiers = publish._scoring_tiers(contract_name)
        calibration = _calibration_payload(result, allowed_tiers=allowed_tiers)
        calibration_graph = analysis_dir / f"{result['experiment_id']}_calibration.png"
        coverage_graph = analysis_dir / f"{result['experiment_id']}_coverage.png"
        payload_for_graph = {**calibration, "_quarterly_rows_internal": list(result.get("quarterly_rows") or [])}
        publish._save_calibration_graph(payload_for_graph, calibration_graph)
        _save_coverage_graph(calibration, coverage_graph)
        calibration_rows[str(result["experiment_id"])] = {
            "contract": contract_name,
            "quarterly_mean_mae": calibration["quarterly_mean_mae"],
            "calibration": calibration,
            "calibration_graph_file": calibration_graph.name,
            "coverage_graph_file": coverage_graph.name,
        }

    dense_transfer = _dense_transfer_payload(purged_map, legacy_map)

    for result in (
        exact_map["EXP-R10-M1-F1"],
        purged_map["EXP-R10-DENSE-M1-H1"],
        purged_map["EXP-R10-DENSE-M1-F1-H1"],
        exact_map["EXP-R1"],
    ):
        graph_path = analysis_dir / f"{result['experiment_id']}_observation_curves.png"
        graph_name = _save_observation_graph(result, graph_path)
        if graph_name:
            result["observation_curve_graph_file"] = graph_name

    eval01_json = analysis_dir / "exp_eval_01_frozen_protocol_rebuild.json"
    eval01_md = analysis_dir / "exp_eval_01_frozen_protocol_rebuild.md"
    write_json(eval01_json, frozen_protocol)
    eval01_md.write_text(_markdown_frozen_protocol(frozen_protocol), encoding="utf-8")

    eval02_json = analysis_dir / "exp_eval_02_endpoint_tier_era_audit.json"
    eval02_md = analysis_dir / "exp_eval_02_endpoint_tier_era_audit.md"
    write_json(eval02_json, endpoint_tier_era)
    eval02_md.write_text(_markdown_endpoint_tier_era(endpoint_tier_era), encoding="utf-8")

    eval03_json = analysis_dir / "exp_eval_03_calibration_interval_coverage.json"
    eval03_md = analysis_dir / "exp_eval_03_calibration_interval_coverage.md"
    write_json(eval03_json, {"experiments": calibration_rows})
    eval03_md.write_text(_markdown_calibration_coverage({"experiments": calibration_rows}), encoding="utf-8")

    dense_transfer_json = analysis_dir / "exp_r10_dense_m1_f1_h1.json"
    dense_transfer_md = analysis_dir / "exp_r10_dense_m1_f1_h1.md"
    write_json(dense_transfer_json, dense_transfer)
    dense_transfer_md.write_text(_markdown_dense_transfer(dense_transfer), encoding="utf-8")

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "archive_run_id": archive_run,
        "variant": "evidence-to-model-loop",
        "recommendation": {
            "exact_candidate_id": "EXP-R10-M1-F1",
            "dense_candidate_id": (
                "EXP-R10-DENSE-M1-F1-H1"
                if dict(dense_transfer.get("decision") or {}).get("status") == "keep_transfer"
                else "EXP-R10-DENSE-M1-H1"
            ),
            "dense_transfer_status": str(dict(dense_transfer.get("decision") or {}).get("status") or ""),
        },
        "artifacts": {
            "eval_01_markdown": eval01_md.name,
            "eval_02_markdown": eval02_md.name,
            "eval_03_markdown": eval03_md.name,
            "dense_transfer_markdown": dense_transfer_md.name,
        },
        "eval_01": frozen_protocol,
        "eval_02": endpoint_tier_era,
        "eval_03": {"experiments": calibration_rows},
        "dense_transfer": dense_transfer,
        "lockbox": {"holdout_years": holdout_years, "contracts": lockbox_contracts},
    }
    report_json = analysis_dir / "tr_v3_eval_hardening_batch_report.json"
    report_md = analysis_dir / "tr_v3_eval_hardening_batch_report.md"
    write_json(report_json, report_payload)
    report_md.write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr_v3_eval_hardening_batch")
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_eval_hardening_batch(
        run_id=args.run_id,
        archive_run_id=args.archive_run_id,
        quarterly_start_year=args.quarterly_start_year,
        quarterly_end_year=args.quarterly_end_year,
        quarterly_min_train_years=args.quarterly_min_train_years,
        annual_start_year=args.annual_start_year,
        annual_end_year=args.annual_end_year,
        annual_min_train_years=args.annual_min_train_years,
        horizon_years=args.horizon_years,
        lockbox_holdout_years=list(args.lockbox_holdout_years),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
