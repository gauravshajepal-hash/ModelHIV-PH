from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3.tr_v3_05_autoresearch import (
    AnnualIncidenceConfig,
    PRIMARY_METRICS,
    build_annual_anchor_rows,
    build_quarterly_dataset,
    carry_forward_hazards,
    quarter_sort_key,
    quarter_year,
    simulate_closed_flow,
)
from epigraph_ph.runtime import ensure_dir, write_json


LOCKBOX_CONTRACT_CHOICES: tuple[str, ...] = ("exact_only", "legacy_dense", "purged_dense")


def _suite_result_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["experiment_id"]): dict(row) for row in list(payload.get("results") or []) if row.get("status") == "executed"}


def _run_selected_suite(
    *,
    archive_run_id: str,
    quarterly_contract: str,
    experiment_ids: list[str],
    quarterly_start_year: int,
    quarterly_end_year: int,
    quarterly_min_train_years: int,
    annual_start_year: int,
    annual_end_year: int,
    annual_min_train_years: int,
    horizon_years: int,
) -> dict[str, Any]:
    if quarterly_contract == "exact_only":
        observation_rows = suite.build_quarterly_observation_rows(archive_run_id)
        scoring_tiers = {"exact_observed"}
        split_local_dense = False
        full_dense_rows = None
    elif quarterly_contract == "dense_train_observed_score":
        dense_payload = suite._build_dense_contract_payload(archive_run_id)
        observation_rows = list(dense_payload["rows"])
        scoring_tiers = {"exact_observed", "bridge_observed"}
        split_local_dense = True
        full_dense_rows = list(dense_payload["rows"])
    else:
        raise ValueError(f"Unsupported quarterly_contract: {quarterly_contract}")
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
            "endpoint_tier_audit": suite._build_endpoint_tier_audit_payload(results, quarterly_contract=quarterly_contract),
        }
    return {
        "archive_run_id": archive_run_id,
        "quarterly_contract": quarterly_contract,
        "results": results,
    }


def _scoring_tiers(contract_name: str) -> set[str]:
    if contract_name == "exact_only":
        return {"exact_observed"}
    if contract_name in {"legacy_dense", "purged_dense", "dense_train_observed_score"}:
        return {"exact_observed", "bridge_observed"}
    raise ValueError(f"Unsupported contract_name: {contract_name}")


def _observation_rows_for_lockbox(
    archive_run_id: str,
    *,
    contract_name: str,
    holdout_years: list[int],
) -> list[dict[str, Any]]:
    if contract_name == "exact_only":
        return suite.build_quarterly_observation_rows(archive_run_id)
    dense_payload = suite._build_dense_contract_payload(archive_run_id)
    full_dense_rows = list(dense_payload["rows"])
    if contract_name == "legacy_dense":
        return full_dense_rows
    if contract_name == "purged_dense":
        train_end_year = min(int(year) for year in holdout_years) - 1
        return suite._build_purged_dense_split_rows(
            archive_run_id,
            full_dense_rows=full_dense_rows,
            train_end_year=train_end_year,
            holdout_years=list(holdout_years),
        )
    raise ValueError(f"Unsupported contract_name: {contract_name}")


def _annual_overlay_error(
    dataset: Any,
    annual_rows: list[dict[str, Any]],
    annual_infections: dict[str | int, float],
) -> float | None:
    if not annual_infections:
        return None
    yearly_rows = suite._yearly_feature_rows_filtered(dataset.train_rows + dataset.holdout_rows, annual_rows)
    holdout_years = sorted({quarter_year(str(row["quarter"])) for row in dataset.holdout_rows})
    return float(
        suite._annual_metric_error(
            yearly_rows,
            holdout_years,
            {int(key): float(value) for key, value in annual_infections.items()},
            "annual_new_infections",
        )
    )


def _evaluate_fixed_holdout_experiment(
    spec: suite.ExperimentSpec,
    *,
    observation_rows: list[dict[str, Any]],
    annual_rows: list[dict[str, Any]],
    holdout_years: list[int],
    scoring_tiers: set[str],
    frozen_config: dict[str, Any],
) -> dict[str, Any]:
    dataset = build_quarterly_dataset(observation_rows, list(holdout_years))
    carry_forward = simulate_closed_flow(
        dict(dataset.train_state_rows[-1]["state_values"]),
        dataset.holdout_rows,
        carry_forward_hazards(dataset),
    )
    baseline_mae = suite._filtered_normalized_mae(
        carry_forward["prediction_rows"],
        dataset.holdout_rows,
        dataset.metric_scales,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    baseline_smape = suite._filtered_smape(
        carry_forward["prediction_rows"],
        dataset.holdout_rows,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    dynamic_cfg = suite.DynamicControlConfig(**dict(frozen_config.get("dynamic_cfg") or {}))
    observation_cfg = suite.ObservationConfig(**dict(frozen_config.get("observation_cfg") or {}))
    candidate = suite._fit_05a_experiment_candidate(dataset, annual_rows, dynamic_cfg, observation_cfg, spec)
    candidate_mae = suite._filtered_normalized_mae(
        candidate["prediction_rows"],
        dataset.holdout_rows,
        dataset.metric_scales,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    candidate_smape = suite._filtered_smape(
        candidate["prediction_rows"],
        dataset.holdout_rows,
        allowed_tiers=scoring_tiers,
        eps=dataset.eps,
    )
    endpoint_audit = {
        "candidate": suite._raw_endpoint_audit(
            candidate["prediction_rows"],
            dataset.holdout_rows,
            dataset.metric_scales,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        ),
        "carry_forward": suite._raw_endpoint_audit(
            carry_forward["prediction_rows"],
            dataset.holdout_rows,
            dataset.metric_scales,
            allowed_tiers=scoring_tiers,
            eps=dataset.eps,
        ),
        "train_support_counts": {
            metric_name: suite._count_metric_support(dataset.train_rows, metric_name, allowed_tiers=scoring_tiers)
            for metric_name in suite.AUDIT_METRICS
        },
        "holdout_support_counts": {
            metric_name: suite._count_metric_support(dataset.holdout_rows, metric_name, allowed_tiers=scoring_tiers)
            for metric_name in suite.AUDIT_METRICS
        },
        "suppression_honesty_flag": suite._suppression_honesty_flag(
            dataset,
            dict(candidate.get("transition_diagnostics") or {}),
            allowed_tiers=scoring_tiers,
        ),
    }
    annual_overlay_error = _annual_overlay_error(dataset, annual_rows, dict(candidate.get("annual_infections") or {}))
    return {
        "experiment_id": spec.experiment_id,
        "best_candidate": dict(frozen_config),
        "quarterly_mean_mae": float(candidate_mae),
        "quarterly_baseline_mae": float(baseline_mae),
        "quarterly_smape": float(candidate_smape),
        "quarterly_baseline_smape": float(baseline_smape),
        "annual_overlay_error": annual_overlay_error,
        "endpoint_audit": dict(endpoint_audit),
        "candidate_count": 1,
        "prediction_rows": list(dict(row) for row in list(candidate.get("prediction_rows") or [])),
        "carry_forward_prediction_rows": list(dict(row) for row in list(carry_forward["prediction_rows"])),
        "holdout_target_rows": [
            {
                "quarter": str(row["quarter"]),
                **{metric_name: row.get(metric_name) for metric_name in PRIMARY_METRICS},
                "metric_tiers": {metric_name: suite._metric_tier(row, metric_name) for metric_name in PRIMARY_METRICS},
            }
            for row in dataset.holdout_rows
        ],
    }


def _collect_residuals(quarterly_rows: list[dict[str, Any]], metric_name: str) -> list[float]:
    values: list[float] = []
    for split in quarterly_rows:
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            values.append(float(prediction_value) - float(target_value))
    return values


def _calibration_payload(result: dict[str, Any]) -> dict[str, Any]:
    quarterly_rows = list(result.get("quarterly_rows") or [])
    split_rows = []
    for split in quarterly_rows:
        split_rows.append(
            {
                "holdout_years": list(split.get("holdout_years") or []),
                "candidate_mae": float(dict(split.get("candidate") or {}).get("mae") or float("inf")),
                "baseline_mae": float(dict(split.get("carry_forward") or {}).get("mae") or float("inf")),
                "suppression_honesty_flag": str(dict(split.get("endpoint_audit") or {}).get("suppression_honesty_flag") or "unsupported_or_unclaimed"),
            }
        )
    by_metric: dict[str, Any] = {}
    for metric_name in PRIMARY_METRICS:
        residuals = np.asarray(_collect_residuals(quarterly_rows, metric_name), dtype=np.float64)
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
        "endpoint_audit_summary": dict(dict(result.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {}),
    }


def _save_calibration_graph(payload: dict[str, Any], path: Path) -> None:
    metrics = [metric_name for metric_name in PRIMARY_METRICS if int(dict(payload["by_metric"][metric_name]).get("count") or 0) > 0]
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), constrained_layout=True)
    axes_list = list(np.asarray(axes).reshape(-1))

    split_rows = list(payload.get("split_rows") or [])
    if split_rows:
        ax = axes_list[0]
        labels = [",".join(str(year) for year in row["holdout_years"]) for row in split_rows]
        candidate = [float(row["candidate_mae"]) for row in split_rows]
        baseline = [float(row["baseline_mae"]) for row in split_rows]
        x = np.arange(len(labels))
        ax.plot(x, baseline, marker="o", label="Carry-forward")
        ax.plot(x, candidate, marker="o", label=str(payload["experiment_id"]))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title("Split MAE")
        ax.grid(alpha=0.3)
        ax.legend()
    else:
        axes_list[0].axis("off")

    for ax, metric_name in zip(axes_list[1:], metrics, strict=False):
        residuals = np.asarray(_collect_residuals_from_payload(payload, metric_name), dtype=np.float64)
        if residuals.size == 0:
            ax.axis("off")
            continue
        ax.hist(residuals, bins=min(10, max(5, residuals.size)), color="#1f77b4", alpha=0.8)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(metric_name.replace("_", " "))
        ax.grid(alpha=0.2)
    for ax in axes_list[1 + len(metrics):]:
        ax.axis("off")
    fig.suptitle(f"{payload['experiment_id']} calibration and uncertainty")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _collect_residuals_from_payload(payload: dict[str, Any], metric_name: str) -> list[float]:
    split_rows = list(payload.get("_quarterly_rows_internal") or [])
    values: list[float] = []
    for split in split_rows:
        for target_row, prediction_row in zip(
            list(split.get("holdout_target_rows") or []),
            list(split.get("candidate_prediction_rows") or []),
            strict=False,
        ):
            target_value = target_row.get(metric_name)
            prediction_value = prediction_row.get(metric_name)
            if target_value is None or prediction_value is None:
                continue
            values.append(float(prediction_value) - float(target_value))
    return values


def _save_lockbox_graph(payload: dict[str, Any], path: Path, *, title: str) -> None:
    rows = list(payload.get("rows") or [])
    if not rows:
        suite._plot_placeholder(path, title=title, body="No lockbox rows were available.")
        return
    labels = [str(row["experiment_id"]) for row in rows]
    candidate = [float(row["quarterly_mean_mae"]) for row in rows]
    baseline = [float(row["quarterly_baseline_mae"]) for row in rows]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(4.5, len(labels) * 0.5)))
    ax.barh(y - 0.18, baseline, height=0.35, label="Carry-forward")
    ax.barh(y + 0.18, candidate, height=0.35, label="Candidate")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Lockbox quarterly normalized MAE")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TR-V3 Publishability Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Variant: `{payload['variant']}`",
        f"- Lockbox note: `{payload['lockbox_note']}`",
        "",
        "## Frozen Contract Audit",
        "",
        "| Contract | Experiment | Quarterly mean MAE | Baseline MAE | Raw diagnosed MAE | Raw ART MAE | Raw flow MAE | Suppression flags |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("frozen_contract_audit", {}).get("rows") or []):
        lines.append(
            f"| {row['contract']} | {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
            f"{float(row['quarterly_baseline_mae']):.6f} | {float(row['diagnosed_raw_mae']):.3f} | "
            f"{float(row['art_raw_mae']):.3f} | {float(row['flow_raw_mae']):.3f} | "
            f"`{dict(row['suppression_honesty_flags'])}` |"
        )
    lines.extend(
        [
            "",
            "## Lockbox",
            "",
            f"- Frozen holdout years: `{payload['lockbox']['holdout_years']}`",
            "",
        ]
    )
    for contract_name, contract_payload in (payload.get("lockbox", {}).get("contracts") or {}).items():
        lines.extend(
            [
                f"### {contract_name}",
                "",
                f"- Graph: `{contract_payload['graph_file']}`",
                "",
                "| Experiment | Quarterly MAE | Baseline MAE | Annual overlay error | Suppression honesty |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in list(contract_payload.get("rows") or []):
            annual_error = row.get("annual_overlay_error")
            annual_text = "" if annual_error is None else f"{float(annual_error):.6f}"
            lines.append(
                f"| {row['experiment_id']} | {float(row['quarterly_mean_mae']):.6f} | "
                f"{float(row['quarterly_baseline_mae']):.6f} | {annual_text} | "
                f"`{row['suppression_honesty_flag']}` |"
            )
        lines.append("")
    lines.extend(
        [
            "## Calibration And Uncertainty",
            "",
        ]
    )
    for experiment_id, calibration in (payload.get("calibration", {}) or {}).items():
        lines.extend(
            [
                f"### {experiment_id}",
                "",
                f"- Graph: `{calibration['graph_file']}`",
                f"- Quarterly mean MAE: `{float(calibration['quarterly_mean_mae']):.6f}`",
                f"- Baseline mean MAE: `{float(calibration['quarterly_baseline_mean_mae']):.6f}`",
                "",
                "| Metric | Count | Mean residual | Median residual | Std | q10 | q90 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for metric_name, metric_payload in (calibration.get("by_metric") or {}).items():
            lines.append(
                f"| `{metric_name}` | `{metric_payload['count']}` | `{metric_payload['mean_residual']:.3f}` | "
                f"`{metric_payload['median_residual']:.3f}` | `{metric_payload['std_residual']:.3f}` | "
                f"`{metric_payload['q10_residual']:.3f}` | `{metric_payload['q90_residual']:.3f}` |"
            )
        lines.append("")
    support_exploration = dict(payload.get("support_exploration") or {})
    if support_exploration:
        susceptible = dict(support_exploration.get("susceptible_contract") or {})
        richer_leakage = dict(support_exploration.get("richer_leakage_contract") or {})
        lines.extend(
            [
                "## Support Exploration",
                "",
            ]
        )
        if susceptible:
            decision = dict(susceptible.get("decision") or {})
            annual_years = dict(susceptible.get("annual_years") or {})
            lines.extend(
                [
                    "### Explicit S(t)",
                    "",
                    f"- Status: `{decision.get('status', '')}`",
                    f"- Why: {decision.get('why', '')}",
                    f"- Joint annual support years: `{annual_years.get('joint_support_years', [])}`",
                    "",
                ]
            )
        if richer_leakage:
            decision = dict(richer_leakage.get("decision") or {})
            lines.extend(
                [
                    "### Richer Leakage",
                    "",
                    f"- Status: `{decision.get('status', '')}`",
                    f"- Why: {decision.get('why', '')}",
                    f"- Annual deaths years: `{richer_leakage.get('annual_aids_deaths_years', [])}`",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def _markdown_susceptible_contract(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    annual_years = dict(payload.get("annual_years") or {})
    lines = [
        "# Explicit S(t) Identification Contract",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Status: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "## Annual Support",
        "",
        f"- `population_total` years: `{annual_years.get('population_total', [])}`",
        f"- `estimated_plhiv` years: `{annual_years.get('estimated_plhiv', [])}`",
        f"- `annual_new_infections` years: `{annual_years.get('annual_new_infections', [])}`",
        f"- Joint support years: `{annual_years.get('joint_support_years', [])}`",
        "",
        "## Allowed Blocks",
        "",
        "| Block | Status |",
        "|---|---|",
    ]
    for name, status in dict(payload.get("allowed_blocks") or {}).items():
        lines.append(f"| {name} | {status} |")
    lines.extend(
        [
            "",
            "## Annual Proxy Rows",
            "",
            "| Year | Population total | Estimated PLHIV | Annual new infections | Susceptible proxy | Susceptible fraction |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(payload.get("proxy_rows") or []):
        lines.append(
            f"| {row['year']} | {float(row['population_total']):.0f} | {float(row['estimated_plhiv']):.0f} | "
            f"{float(row['annual_new_infections']):.0f} | {float(row['susceptible_proxy']):.0f} | {float(row['susceptible_fraction']):.6f} |"
        )
    return "\n".join(lines).strip() + "\n"


def _markdown_richer_leakage_contract(payload: dict[str, Any]) -> str:
    decision = dict(payload.get("decision") or {})
    lines = [
        "# Richer Leakage Identification Contract",
        "",
        f"- Archive run: `{payload['archive_run_id']}`",
        f"- Status: `{decision.get('status', '')}`",
        f"- Why: {decision.get('why', '')}",
        "",
        "## Allowed Blocks",
        "",
        "| Block | Status |",
        "|---|---|",
    ]
    for name, status in dict(payload.get("allowed_blocks") or {}).items():
        lines.append(f"| {name} | {status} |")
    lines.extend(
        [
            "",
            "## Yearly Support",
            "",
            "| Year | Diagnosed stock | ART stock | Suppression | Diagnosis flow | Deaths flow | Net ART leakage | Suppressed leakage separate | Quarterly mortality-coupled leakage |",
            "|---|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in list(payload.get("yearly_support") or []):
        lines.append(
            f"| {row['year']} | {row['diagnosed_stock_quarters']} | {row['art_stock_quarters']} | {row['suppression_quarters']} | "
            f"{row['diagnosis_flow_quarters']} | {row['deaths_flow_quarters']} | {row['can_score_art_leakage_net']} | "
            f"{row['can_score_suppressed_leakage_separately']} | {row['can_score_quarterly_mortality_coupled_leakage']} |"
        )
    return "\n".join(lines).strip() + "\n"


def run_tr_v3_publishability_batch(
    *,
    run_id: str,
    archive_run_id: str,
    quarterly_start_year: int = 2010,
    quarterly_end_year: int = 2025,
    quarterly_min_train_years: int = 3,
    annual_start_year: int = 2010,
    annual_end_year: int = 2024,
    annual_min_train_years: int = 5,
    horizon_years: int = 1,
    lockbox_holdout_years: list[int] | None = None,
) -> dict[str, Any]:
    exact_run_id = f"{run_id}-exact"
    dense_run_id = f"{run_id}-dense"
    exact_payload = _run_selected_suite(
        archive_run_id=archive_run_id,
        quarterly_contract="exact_only",
        experiment_ids=["EXP-V1", "EXP-V2", "EXP-R10-EXACT-CHAMPION", "EXP-R10-M1", "EXP-R10-M1-B1", "EXP-R1"],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )
    dense_payload = _run_selected_suite(
        archive_run_id=archive_run_id,
        quarterly_contract="dense_train_observed_score",
        experiment_ids=[
            "EXP-V2",
            "EXP-R10-EXACT-CHAMPION",
            "EXP-R10-DENSE-CHAMPION",
            "EXP-R10-DENSE-H1",
            "EXP-R10-DENSE-M1",
            "EXP-R10-DENSE-M1-H1",
            "EXP-R10-DENSE-M1-B1-H1",
            "EXP-R10-DENSE-M2",
            "EXP-R11",
            "EXP-R1",
        ],
        quarterly_start_year=quarterly_start_year,
        quarterly_end_year=quarterly_end_year,
        quarterly_min_train_years=quarterly_min_train_years,
        annual_start_year=annual_start_year,
        annual_end_year=annual_end_year,
        annual_min_train_years=annual_min_train_years,
        horizon_years=horizon_years,
    )

    exact_map = _suite_result_map(exact_payload)
    dense_map = _suite_result_map(dense_payload)
    v1_payload = dict(dict(exact_map.get("EXP-V1") or {}).get("purged_dense_contract") or {})

    tracked_exact = ["EXP-R10-EXACT-CHAMPION", "EXP-R10-M1", "EXP-R10-M1-B1", "EXP-R1"]
    tracked_dense = [
        "EXP-R10-DENSE-CHAMPION",
        "EXP-R10-DENSE-H1",
        "EXP-R10-DENSE-M1",
        "EXP-R10-DENSE-M1-H1",
        "EXP-R10-DENSE-M1-B1-H1",
        "EXP-R10-DENSE-M2",
        "EXP-R11",
        "EXP-R1",
    ]
    frozen_rows: list[dict[str, Any]] = []
    for experiment_id in tracked_exact:
        row = exact_map.get(experiment_id)
        if not row:
            continue
        audit = dict(dict(row.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {})
        candidate_by_metric = dict(dict(audit.get("candidate") or {}).get("by_metric") or {})
        frozen_rows.append(
            {
                "contract": "exact_only",
                "experiment_id": experiment_id,
                "quarterly_mean_mae": float(dict(row["quarterly_summary"])["candidate_mean_mae"]),
                "quarterly_baseline_mae": float(dict(row["quarterly_summary"])["carry_forward_mean_mae"]),
                "diagnosed_raw_mae": float(dict(candidate_by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
                "art_raw_mae": float(dict(candidate_by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
                "flow_raw_mae": float(dict(candidate_by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
                "suppression_honesty_flags": dict(audit.get("suppression_honesty_flags") or {}),
            }
        )
    for experiment_id in tracked_dense:
        row = dense_map.get(experiment_id)
        if not row:
            continue
        audit = dict(dict(row.get("quarterly_summary") or {}).get("endpoint_audit_summary") or {})
        candidate_by_metric = dict(dict(audit.get("candidate") or {}).get("by_metric") or {})
        frozen_rows.append(
            {
                "contract": "purged_dense",
                "experiment_id": experiment_id,
                "quarterly_mean_mae": float(dict(row["quarterly_summary"])["candidate_mean_mae"]),
                "quarterly_baseline_mae": float(dict(row["quarterly_summary"])["carry_forward_mean_mae"]),
                "diagnosed_raw_mae": float(dict(candidate_by_metric.get("diagnosed_plhiv") or {}).get("raw_mae") or 0.0),
                "art_raw_mae": float(dict(candidate_by_metric.get("alive_on_art") or {}).get("raw_mae") or 0.0),
                "flow_raw_mae": float(dict(candidate_by_metric.get("new_diagnosed_cases_period") or {}).get("raw_mae") or 0.0),
                "suppression_honesty_flags": dict(audit.get("suppression_honesty_flags") or {}),
            }
        )
    for entry in list(v1_payload.get("comparisons") or []):
        frozen_rows.append(
            {
                "contract": "legacy_dense",
                "experiment_id": str(entry["experiment_id"]),
                "quarterly_mean_mae": float(entry["candidate_mean_mae"]),
                "quarterly_baseline_mae": float(entry["baseline_mean_mae"]),
                "diagnosed_raw_mae": 0.0,
                "art_raw_mae": 0.0,
                "flow_raw_mae": 0.0,
                "suppression_honesty_flags": {},
            }
        )

    holdout_years = sorted(set(int(year) for year in (lockbox_holdout_years or [2025])))
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    annual_rows = build_annual_anchor_rows(archive_run_id)
    lockbox_experiment_ids = [
        "EXP-R10-EXACT-CHAMPION",
        "EXP-R10-DENSE-CHAMPION",
        "EXP-R10-M1",
        "EXP-R10-M1-B1",
        "EXP-R10-DENSE-H1",
        "EXP-R10-DENSE-M1",
        "EXP-R10-DENSE-M1-H1",
        "EXP-R10-DENSE-M1-B1-H1",
        "EXP-R10-DENSE-M2",
        "EXP-R11",
        "EXP-R1",
    ]
    analysis_dir = ensure_dir(suite.repo_root() / "artifacts" / "runs" / run_id / "analysis")
    lockbox_contract_payloads: dict[str, Any] = {}
    for contract_name in LOCKBOX_CONTRACT_CHOICES:
        scoring_tiers = _scoring_tiers(contract_name)
        rows = _observation_rows_for_lockbox(archive_run_id, contract_name=contract_name, holdout_years=holdout_years)
        contract_rows = []
        for experiment_id in lockbox_experiment_ids:
            if contract_name == "exact_only":
                frozen_config = dict(dict(exact_map.get(experiment_id) or {}).get("best_candidate") or {})
            else:
                frozen_config = dict(dict(dense_map.get(experiment_id) or {}).get("best_candidate") or {})
                if not frozen_config:
                    frozen_config = dict(dict(exact_map.get(experiment_id) or {}).get("best_candidate") or {})
            if not frozen_config:
                continue
            contract_rows.append(
                _evaluate_fixed_holdout_experiment(
                    spec_map[experiment_id],
                    observation_rows=rows,
                    annual_rows=annual_rows,
                    holdout_years=holdout_years,
                    scoring_tiers=scoring_tiers,
                    frozen_config=frozen_config,
                )
            )
        contract_rows = sorted(contract_rows, key=lambda row: float(row["quarterly_mean_mae"]))
        graph_path = analysis_dir / f"lockbox_{contract_name}.png"
        contract_payload = {"rows": contract_rows}
        _save_lockbox_graph(contract_payload, graph_path, title=f"TR-V3 lockbox ({contract_name})")
        lockbox_contract_payloads[contract_name] = {
            **contract_payload,
            "graph_file": graph_path.name,
        }

    calibration: dict[str, Any] = {}
    for result in [exact_map["EXP-R10-M1"], dense_map["EXP-R10-DENSE-CHAMPION"]]:
        payload = _calibration_payload(result)
        payload["_quarterly_rows_internal"] = list(result.get("quarterly_rows") or [])
        graph_path = analysis_dir / f"{payload['experiment_id']}_calibration.png"
        _save_calibration_graph(payload, graph_path)
        payload["graph_file"] = graph_path.name
        payload.pop("_quarterly_rows_internal", None)
        calibration[str(payload["experiment_id"])] = payload

    susceptible_contract = suite._build_susceptible_identification_contract_payload(archive_run_id)
    richer_leakage_contract = suite._build_richer_leakage_identification_contract_payload(archive_run_id)
    write_json(analysis_dir / "exp_s1_susceptible_identification_contract.json", susceptible_contract)
    (analysis_dir / "exp_s1_susceptible_identification_contract.md").write_text(
        _markdown_susceptible_contract(susceptible_contract),
        encoding="utf-8",
    )
    write_json(analysis_dir / "exp_l2_richer_leakage_identification_contract.json", richer_leakage_contract)
    (analysis_dir / "exp_l2_richer_leakage_identification_contract.md").write_text(
        _markdown_richer_leakage_contract(richer_leakage_contract),
        encoding="utf-8",
    )

    report_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": "evidence-to-model-loop",
        "archive_run_id": archive_run_id,
        "exact_suite_run_id": exact_run_id,
        "dense_suite_run_id": dense_run_id,
        "lockbox_note": "This is a retroactive paper lockbox frozen from this run forward, not a pristine untouched historical split.",
        "frozen_contract_audit": {
            "rows": frozen_rows,
            "purged_dense_decision": v1_payload.get("decision"),
            "purged_dense_winner": v1_payload.get("purged_winner"),
            "legacy_dense_winner": v1_payload.get("legacy_winner"),
        },
        "lockbox": {
            "holdout_years": holdout_years,
            "contracts": lockbox_contract_payloads,
        },
        "calibration": calibration,
        "support_exploration": {
            "susceptible_contract": susceptible_contract,
            "richer_leakage_contract": richer_leakage_contract,
        },
    }
    write_json(analysis_dir / "tr_v3_publishability_batch_report.json", report_payload)
    (analysis_dir / "tr_v3_publishability_batch_report.md").write_text(_markdown_report(report_payload), encoding="utf-8")
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tr-v3-publishability-batch")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--archive-run-id", default=suite._latest_standard_archive_run())
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
    run_tr_v3_publishability_batch(
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
