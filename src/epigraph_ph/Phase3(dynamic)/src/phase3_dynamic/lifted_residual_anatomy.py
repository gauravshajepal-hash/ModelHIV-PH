from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backhalf_channels import BackHalfChannelContext
from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .hybrid_champion import (
    OBSERVATION_HEAD_METRICS,
    _detect_shock_catalog,
    _evaluate_family_split,
    _load_r10_baseline,
    _load_reference_config,
    _r10_reference_scores,
    _rolling_splits,
    _shock_labels_for_year,
)
from .metrics import SUPPORT_AWARE_BACK_HALF_METRICS, quarter_sort_key, quarter_year
from .monthly_joint_observation import MonthlyJointContext
from .monthly_latent_state import MonthlyLatentContext
from .monthly_shock import MonthlyShockContext
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .runtime import ensure_dir, read_json, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID

LIFTED_RESIDUAL_ANATOMY_SCHEMA_VERSION = "phase3_dynamic_lifted_trajectory_residual_anatomy.v1"

DEFAULT_SOURCE_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-reengagement-sensitivity-20260426-s01-public-stock-flow-proxy"
    / "analysis"
    / "hybrid_champion_search_report.json"
)

ANATOMY_METRICS: tuple[str, ...] = tuple(
    dict.fromkeys(
        (
            *OBSERVATION_HEAD_METRICS,
            *SUPPORT_AWARE_BACK_HALF_METRICS,
            "incident_infections_period",
        )
    )
)

STREAM_LABELS: dict[str, str] = {
    "diagnosed_plhiv": "diagnosis_stock",
    "new_diagnosed_cases_period": "diagnosis_flow_incidence_alignment",
    "alive_on_art": "art_stock",
    "tested_for_viral_load": "vl_testing",
    "virally_suppressed": "suppression",
    "incident_infections_period": "incidence_readout",
}


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _p90(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    if not finite:
        return None
    return float(np.percentile(np.asarray(finite, dtype=np.float64), 90.0))


def _safe_scale(metric_name: str, target_value: float, metric_scales: dict[str, Any]) -> float:
    configured = _finite_float(metric_scales.get(metric_name))
    if configured is not None and configured > 0.0:
        return max(configured, 1.0)
    return max(abs(float(target_value)), 1.0)


def _default_source_report() -> Path:
    if DEFAULT_SOURCE_REPORT.exists():
        return DEFAULT_SOURCE_REPORT
    candidates = sorted(
        (
            sandbox_repo_root()
            / "artifacts"
            / "runs"
        ).glob("p3d-reengagement-sensitivity-*-public-stock-flow-proxy/analysis/hybrid_champion_search_report.json")
    )
    if candidates:
        return candidates[-1]
    candidates = sorted(
        (
            sandbox_repo_root()
            / "artifacts"
            / "runs"
        ).glob("p3d-hybrid-champion-search-*/analysis/hybrid_champion_search_report.json")
    )
    if not candidates:
        raise FileNotFoundError("No hybrid champion report found for lifted residual anatomy.")
    return candidates[-1]


def _selected_family(report: dict[str, Any], requested_family: str | None) -> str:
    if requested_family:
        return requested_family
    champion = dict(report.get("champion_by_claim_aware_promotion") or {})
    if champion.get("family"):
        return str(champion["family"])
    best = dict(report.get("best_family") or {})
    if best.get("family"):
        return str(best["family"])
    families = list(report.get("families") or [])
    if not families:
        raise ValueError("Hybrid champion report has no families.")
    return str(families[0].get("family") or "")


def _r10_metric_reference(r10_baseline: dict[str, Any], reference_contract: str | None) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    source_path = Path(str(r10_baseline.get("source_report") or ""))
    payload = read_json(source_path, default={}) if source_path.exists() else {}
    contracts = list((payload or {}).get("contracts") or [])
    selected: dict[str, Any] | None = None
    for contract in contracts:
        champion = dict((contract or {}).get("merged_current_champion") or {})
        if str(champion.get("contract") or (contract or {}).get("contract") or "") == str(reference_contract or ""):
            selected = champion
            break
    if selected is None:
        for contract in contracts:
            champion = dict((contract or {}).get("merged_current_champion") or {})
            if str(champion.get("archive_variant") or "") == "merged":
                selected = champion
                break
    for row in list((selected or {}).get("residual_rows") or []):
        if str(row.get("tier") or "") != "overall":
            continue
        metric = str(row.get("metric") or "")
        if not metric:
            continue
        mean_value = _finite_float(row.get("abs_residual_mean"))
        p90_value = _finite_float(row.get("abs_residual_p90"))
        rows[metric] = {
            "raw_mean_abs_error": float(mean_value or 0.0),
            "raw_p90_abs_error": float(p90_value or 0.0),
            "count": float(_finite_float(row.get("count")) or 0.0),
        }
    return rows


def _metric_entry_rows(
    evaluated_rows: list[dict[str, Any]],
    shock_catalog: dict[str, Any],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for split in evaluated_rows:
        predictions = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("prediction_rows") or [])
        }
        carry_predictions = {
            str(row.get("quarter") or ""): dict(row)
            for row in list(split.get("carry_forward_prediction_rows") or [])
        }
        holdouts = sorted(list(split.get("holdout_rows") or []), key=lambda row: quarter_sort_key(str(row.get("quarter") or "")))
        metric_scales = dict(split.get("metric_scales") or {})
        horizon_years = int(split.get("gate_horizon_years") or max(1, len(set(quarter_year(str(row.get("quarter") or "0-Q1")) for row in holdouts))))
        max_step = max(len(holdouts) - 1, 0)
        for step_index, target in enumerate(holdouts):
            quarter = str(target.get("quarter") or "")
            if not quarter:
                continue
            candidate = predictions.get(quarter)
            carry = carry_predictions.get(quarter)
            if candidate is None or carry is None:
                continue
            year = quarter_year(quarter)
            labels = _shock_labels_for_year(year, shock_catalog)
            for metric_name in ANATOMY_METRICS:
                target_value = _finite_float(target.get(metric_name))
                candidate_value = _finite_float(candidate.get(metric_name))
                carry_value = _finite_float(carry.get(metric_name))
                if target_value is None or candidate_value is None or carry_value is None:
                    continue
                scale = _safe_scale(metric_name, target_value, metric_scales)
                candidate_abs = abs(candidate_value - target_value)
                carry_abs = abs(carry_value - target_value)
                entries.append(
                    {
                        "family": str(split.get("family") or ""),
                        "train_end_year": int(split["train_end_year"]),
                        "horizon_years": int(horizon_years),
                        "holdout_years": list(split.get("holdout_years") or []),
                        "quarter": quarter,
                        "year": int(year),
                        "step_index": int(step_index),
                        "is_terminal_step": bool(step_index == max_step),
                        "metric_name": metric_name,
                        "stream": STREAM_LABELS.get(metric_name, metric_name),
                        "observed_value": float(target_value),
                        "candidate_value": float(candidate_value),
                        "carry_forward_value": float(carry_value),
                        "candidate_abs_error": float(candidate_abs),
                        "carry_forward_abs_error": float(carry_abs),
                        "candidate_norm_error": float(candidate_abs / scale),
                        "carry_forward_norm_error": float(carry_abs / scale),
                        "candidate_minus_carry_forward_norm_error": float((candidate_abs - carry_abs) / scale),
                        "scale": float(scale),
                        "regime_labels": labels,
                    }
                )
    return entries


def _summarize_group(entries: list[dict[str, Any]], *, group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault(tuple(entry.get(key) for key in group_keys), []).append(entry)
    rows: list[dict[str, Any]] = []
    for key_values, subset in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {key: value for key, value in zip(group_keys, key_values)}
        row.update(
            {
                "entry_count": int(len(subset)),
                "candidate_norm_mae": _mean([float(entry["candidate_norm_error"]) for entry in subset]),
                "carry_forward_norm_mae": _mean([float(entry["carry_forward_norm_error"]) for entry in subset]),
                "candidate_minus_carry_forward_norm_mae": _mean(
                    [float(entry["candidate_minus_carry_forward_norm_error"]) for entry in subset]
                ),
                "candidate_raw_mae": _mean([float(entry["candidate_abs_error"]) for entry in subset]),
                "carry_forward_raw_mae": _mean([float(entry["carry_forward_abs_error"]) for entry in subset]),
                "candidate_norm_p90": _p90([float(entry["candidate_norm_error"]) for entry in subset]),
                "carry_forward_norm_p90": _p90([float(entry["carry_forward_norm_error"]) for entry in subset]),
            }
        )
        rows.append(row)
    return rows


def _attach_r10_metric_proxy(
    metric_rows: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    r10_metric_reference: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    scale_by_metric: dict[str, float] = {}
    for metric_name in sorted({str(entry["metric_name"]) for entry in entries}):
        metric_entries = [entry for entry in entries if str(entry["metric_name"]) == metric_name]
        observed_values = [abs(float(entry["observed_value"])) for entry in metric_entries]
        scale_by_metric[metric_name] = max(observed_values or [1.0], default=1.0)
    output: list[dict[str, Any]] = []
    for row in metric_rows:
        metric_name = str(row.get("metric_name") or "")
        next_row = dict(row)
        reference = dict(r10_metric_reference.get(metric_name) or {})
        if reference:
            scale = max(scale_by_metric.get(metric_name, 1.0), 1.0)
            r10_norm = float(reference.get("raw_mean_abs_error") or 0.0) / scale
            next_row["r10_raw_mean_abs_error"] = float(reference.get("raw_mean_abs_error") or 0.0)
            next_row["r10_raw_p90_abs_error"] = float(reference.get("raw_p90_abs_error") or 0.0)
            next_row["r10_norm_mae_proxy"] = r10_norm
            candidate_norm = _finite_float(next_row.get("candidate_norm_mae"))
            next_row["candidate_minus_r10_norm_mae_proxy"] = (
                None if candidate_norm is None else float(candidate_norm - r10_norm)
            )
        else:
            next_row["r10_raw_mean_abs_error"] = None
            next_row["r10_raw_p90_abs_error"] = None
            next_row["r10_norm_mae_proxy"] = None
            next_row["candidate_minus_r10_norm_mae_proxy"] = None
        output.append(next_row)
    return output


def _top_worst(entries: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda row: (
            float(row.get("candidate_minus_carry_forward_norm_error") or 0.0),
            float(row.get("candidate_norm_error") or 0.0),
        ),
        reverse=True,
    )[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _ordered_metrics(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "diagnosed_plhiv",
        "new_diagnosed_cases_period",
        "alive_on_art",
        "tested_for_viral_load",
        "virally_suppressed",
        "incident_infections_period",
    ]
    present = {str(row.get("metric_name") or "") for row in rows}
    return [metric for metric in preferred if metric in present] + sorted(present.difference(preferred))


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    metric_rows = list(payload.get("metric_rows") or [])
    year_rows = sorted(list(payload.get("year_rows") or []), key=lambda row: int(row.get("year") or 0))
    metric_year_rows = list(payload.get("metric_year_rows") or [])
    worst_rows = list(payload.get("worst_rows") or [])[:8]
    r10_reference = dict(payload.get("r10_reference") or {})
    r10_overall = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 10.5), constrained_layout=True)
    flat = axes.ravel()

    metrics = _ordered_metrics(metric_rows)
    x = np.arange(len(metrics), dtype=np.float64)
    candidate = [
        float(next((row.get("candidate_norm_mae") for row in metric_rows if row.get("metric_name") == metric), 0.0) or 0.0)
        for metric in metrics
    ]
    carry = [
        float(next((row.get("carry_forward_norm_mae") for row in metric_rows if row.get("metric_name") == metric), 0.0) or 0.0)
        for metric in metrics
    ]
    r10 = [
        _finite_float(next((row.get("r10_norm_mae_proxy") for row in metric_rows if row.get("metric_name") == metric), None))
        for metric in metrics
    ]
    width = 0.28
    flat[0].bar(x - width, candidate, width=width, color="#1f77b4", label="Phase3 hybrid")
    flat[0].bar(x, carry, width=width, color="#ff9f1c", label="carry-forward")
    r10_x = [index + width for index, value in enumerate(r10) if value is not None]
    r10_y = [float(value) for value in r10 if value is not None]
    if r10_y:
        flat[0].bar(r10_x, r10_y, width=width, color="#2ca02c", label="R10 metric proxy")
    flat[0].set_xticks(x)
    flat[0].set_xticklabels(metrics, rotation=28, ha="right")
    flat[0].set_ylabel("normalized MAE")
    flat[0].set_title("A. Which stream carries the residual?")
    flat[0].legend(frameon=False, fontsize=8)

    years = [int(row.get("year") or 0) for row in year_rows]
    candidate_y = [float(row.get("candidate_norm_mae") or 0.0) for row in year_rows]
    carry_y = [float(row.get("carry_forward_norm_mae") or 0.0) for row in year_rows]
    flat[1].plot(years, candidate_y, marker="o", linewidth=2.0, color="#1f77b4", label="Phase3 hybrid")
    flat[1].plot(years, carry_y, marker="o", linewidth=1.8, color="#ff9f1c", label="carry-forward")
    if r10_overall is not None:
        flat[1].axhline(r10_overall, color="#2ca02c", linestyle="--", linewidth=1.6, label="R10 scalar threshold")
    flat[1].set_ylabel("normalized MAE")
    flat[1].set_title("B. When does the lifted path fail?")
    flat[1].legend(frameon=False, fontsize=8)

    years_sorted = sorted({int(row.get("year") or 0) for row in metric_year_rows})
    metrics_sorted = _ordered_metrics(metric_year_rows)
    matrix = np.zeros((len(metrics_sorted), len(years_sorted)), dtype=np.float64)
    for i, metric in enumerate(metrics_sorted):
        for j, year in enumerate(years_sorted):
            row = next(
                (
                    item
                    for item in metric_year_rows
                    if str(item.get("metric_name") or "") == metric and int(item.get("year") or 0) == year
                ),
                None,
            )
            matrix[i, j] = float((row or {}).get("candidate_minus_carry_forward_norm_mae") or 0.0)
    vmax = float(max(np.max(np.abs(matrix)), 0.05)) if matrix.size else 0.05
    image = flat[2].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    flat[2].set_xticks(np.arange(len(years_sorted)))
    flat[2].set_xticklabels(years_sorted, rotation=30, ha="right")
    flat[2].set_yticks(np.arange(len(metrics_sorted)))
    flat[2].set_yticklabels(metrics_sorted)
    flat[2].set_title("C. Candidate minus carry-forward by stream/year")
    flat[2].set_xlabel("holdout year")
    flat[2].set_ylabel("metric")
    fig.colorbar(image, ax=flat[2], shrink=0.82, label="delta normalized MAE")

    labels = [
        f"{row.get('metric_name')} {row.get('quarter')}"
        for row in worst_rows
    ]
    values = [float(row.get("candidate_minus_carry_forward_norm_error") or 0.0) for row in worst_rows]
    y = np.arange(len(labels), dtype=np.float64)
    flat[3].barh(y, values, color="#d62728")
    flat[3].set_yticks(y)
    flat[3].set_yticklabels(labels, fontsize=8)
    flat[3].invert_yaxis()
    flat[3].axvline(0.0, color="#111827", linewidth=0.8)
    flat[3].set_xlabel("candidate minus carry-forward normalized error")
    flat[3].set_title("D. Worst individual lifted residuals")

    fig.suptitle("Lifted-Trajectory Residual Anatomy Against R10", fontsize=15, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    r10_reference = dict(payload.get("r10_reference") or {})
    lines = [
        "# Lifted-Trajectory Residual Anatomy",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Source report: `{payload['source_report_path']}`",
        f"- Selected family: `{payload['family']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Baseline run: `{payload['baseline_source_run_id']}`",
        f"- Re-engagement mode: `{payload.get('reengagement_sensitivity_mode')}`",
        f"- R10 reference: `{r10_reference.get('reference_experiment_id')}` / `{r10_reference.get('reference_contract')}` / `{r10_reference.get('reference_quarterly_mean_mae')}`",
        "",
        "## Interpretation Contract",
        "",
        "- Per-year rows are reconstructed from the Phase3 lifted-path replay.",
        "- R10 exposes aggregate quarterly MAE plus aggregate metric residuals, not per-year residual rows in the current archived report.",
        "- Therefore, R10 per-year comparisons are threshold comparisons against the scalar R10 path MAE; metric comparisons use an explicitly marked normalized proxy from R10 aggregate raw residuals.",
        f"- Missing requested streams: `{', '.join(list((payload.get('stream_support') or {}).get('missing_streams') or [])) or 'none'}`.",
        f"- Incidence interpretation: `{(payload.get('stream_support') or {}).get('incidence_interpretation')}`",
        "",
        "## Metric Anatomy",
        "",
        "| metric | stream | entries | candidate norm MAE | carry-forward norm MAE | R10 metric proxy | candidate - R10 proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in list(payload.get("metric_rows") or []):
        r10_proxy = row.get("r10_norm_mae_proxy")
        delta = row.get("candidate_minus_r10_norm_mae_proxy")
        lines.append(
            "| {metric} | {stream} | {count} | {cand:.6f} | {carry:.6f} | {r10} | {delta} |".format(
                metric=row.get("metric_name"),
                stream=row.get("stream"),
                count=int(row.get("entry_count") or 0),
                cand=float(row.get("candidate_norm_mae") or 0.0),
                carry=float(row.get("carry_forward_norm_mae") or 0.0),
                r10="not_available" if r10_proxy is None else f"{float(r10_proxy):.6f}",
                delta="not_available" if delta is None else f"{float(delta):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Year Anatomy",
            "",
            "| year | entries | candidate norm MAE | carry-forward norm MAE | candidate - carry-forward | candidate - R10 scalar |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    r10_scalar = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))
    for row in list(payload.get("year_rows") or []):
        candidate = float(row.get("candidate_norm_mae") or 0.0)
        lines.append(
            "| {year} | {count} | {cand:.6f} | {carry:.6f} | {delta:.6f} | {r10_delta} |".format(
                year=int(row.get("year") or 0),
                count=int(row.get("entry_count") or 0),
                cand=candidate,
                carry=float(row.get("carry_forward_norm_mae") or 0.0),
                delta=float(row.get("candidate_minus_carry_forward_norm_mae") or 0.0),
                r10_delta="not_available" if r10_scalar is None else f"{candidate - r10_scalar:.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Worst Residuals",
            "",
            "| quarter | metric | stream | horizon | candidate norm error | carry-forward norm error | delta | regimes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in list(payload.get("worst_rows") or [])[:12]:
        lines.append(
            "| {quarter} | {metric} | {stream} | {horizon} | {cand:.6f} | {carry:.6f} | {delta:.6f} | {labels} |".format(
                quarter=row.get("quarter"),
                metric=row.get("metric_name"),
                stream=row.get("stream"),
                horizon=int(row.get("horizon_years") or 0),
                cand=float(row.get("candidate_norm_error") or 0.0),
                carry=float(row.get("carry_forward_norm_error") or 0.0),
                delta=float(row.get("candidate_minus_carry_forward_norm_error") or 0.0),
                labels=", ".join(str(label) for label in list(row.get("regime_labels") or [])),
            )
        )
    return "\n".join(lines) + "\n"


def run_lifted_residual_anatomy(
    *,
    run_id: str = "p3d-lifted-residual-anatomy-20260426-s00",
    source_report_path: str | Path | None = None,
    family: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    min_train_years: int | None = None,
) -> dict[str, Any]:
    source_report = Path(source_report_path) if source_report_path is not None else _default_source_report()
    report = dict(read_json(source_report, default={}) or {})
    if not report:
        raise FileNotFoundError(f"Hybrid champion report not found or empty: {source_report}")
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(report.get("source_run_id") or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=str(report.get("baseline_source_run_id") or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    selected_family = _selected_family(report, family)
    benchmark_contract = dict(report.get("benchmark_contract") or {})
    start = int(start_year if start_year is not None else benchmark_contract.get("start_year") or 2010)
    end = int(end_year if end_year is not None else benchmark_contract.get("end_year") or 2025)
    minimum_train = int(min_train_years if min_train_years is not None else benchmark_contract.get("min_train_years") or 5)
    reference_config = _load_reference_config(Path(str(report.get("reference_report_path") or "")) if report.get("reference_report_path") else None)
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    constraint_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    monthly_context = MonthlyShockContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    monthly_latent_context = MonthlyLatentContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    monthly_joint_context = MonthlyJointContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    backhalf_context = BackHalfChannelContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        reengagement_sensitivity_mode=str(report.get("reengagement_sensitivity_mode") or "public_stock_flow_proxy"),
    )
    evaluated_rows: list[dict[str, Any]] = []
    for horizon_years in (1, 5):
        for split in _rolling_splits(
            observation_rows,
            start_year=start,
            end_year=end,
            min_train_years=minimum_train,
            horizon_years=horizon_years,
        ):
            result = _evaluate_family_split(
                family=selected_family,
                observation_rows=observation_rows,
                constraint_rows=constraint_rows,
                split=split,
                reference_config=reference_config,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
                min_train_years=minimum_train,
            )
            if result is not None:
                result["gate_horizon_years"] = int(horizon_years)
                evaluated_rows.append(result)
    shock_catalog = _detect_shock_catalog(observation_rows)
    entries = _metric_entry_rows(evaluated_rows, shock_catalog)
    metric_rows = _summarize_group(entries, group_keys=("metric_name", "stream"))
    requested_streams = sorted(set(STREAM_LABELS.values()))
    observed_streams = sorted({str(entry.get("stream") or "") for entry in entries})
    missing_streams = sorted(set(requested_streams).difference(observed_streams))
    r10_baseline = _load_r10_baseline(epigraph_root)
    r10_reference = _r10_reference_scores(r10_baseline)
    metric_rows = _attach_r10_metric_proxy(
        metric_rows,
        entries,
        _r10_metric_reference(r10_baseline, str(r10_reference.get("reference_contract") or "")),
    )
    year_rows = _summarize_group(entries, group_keys=("year",))
    metric_year_rows = _summarize_group(entries, group_keys=("metric_name", "stream", "year"))
    horizon_rows = _summarize_group(entries, group_keys=("horizon_years",))
    worst_rows = _top_worst(entries)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    payload = {
        "schema_version": LIFTED_RESIDUAL_ANATOMY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_report_path": source_report.as_posix(),
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "family": selected_family,
        "reengagement_sensitivity_mode": str(report.get("reengagement_sensitivity_mode") or ""),
        "contract": {
            "purpose": "Identify which metric stream and holdout year causes the full lifted trajectory to lose against the R10 family.",
            "r10_limitation": "Archived R10 reference exposes scalar quarterly MAE and aggregate metric residual rows, not per-year residual rows.",
            "r10_metric_proxy": "R10 raw aggregate metric residuals are normalized by the active Phase3 observed metric magnitude for rough stream triage only.",
            "not_a_promotion_gate": True,
        },
        "stream_support": {
            "requested_streams": requested_streams,
            "observed_streams": observed_streams,
            "missing_streams": missing_streams,
            "incidence_interpretation": (
                "Direct incident infection rows were not present in the lifted replay; "
                "incidence/readout alignment is represented only by new_diagnosed_cases_period."
            ),
        },
        "benchmark_contract": {
            "start_year": start,
            "end_year": end,
            "min_train_years": minimum_train,
            "horizons": [1, 5],
        },
        "r10_reference": r10_reference,
        "entry_count": len(entries),
        "split_count": len(evaluated_rows),
        "metric_rows": metric_rows,
        "year_rows": year_rows,
        "metric_year_rows": metric_year_rows,
        "horizon_rows": horizon_rows,
        "worst_rows": worst_rows,
        "shock_catalog": shock_catalog,
        "artifact_paths": {
            "json": (analysis_dir / "lifted_residual_anatomy.json").as_posix(),
            "markdown": (analysis_dir / "lifted_residual_anatomy.md").as_posix(),
            "dashboard_png": (analysis_dir / "lifted_residual_anatomy_dashboard.png").as_posix(),
            "metric_year_csv": (analysis_dir / "lifted_residual_metric_year.csv").as_posix(),
            "entry_csv": (analysis_dir / "lifted_residual_entries.csv").as_posix(),
        },
    }
    write_json(Path(payload["artifact_paths"]["json"]), payload)
    Path(payload["artifact_paths"]["markdown"]).write_text(_markdown_report(payload), encoding="utf-8")
    _write_csv(Path(payload["artifact_paths"]["metric_year_csv"]), metric_year_rows)
    _write_csv(Path(payload["artifact_paths"]["entry_csv"]), entries)
    _write_dashboard(payload, Path(payload["artifact_paths"]["dashboard_png"]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lifted trajectory residual anatomy against R10.")
    parser.add_argument("--run-id", default="p3d-lifted-residual-anatomy-20260426-s00")
    parser.add_argument("--source-report-path", default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-train-years", type=int, default=None)
    args = parser.parse_args()
    payload = run_lifted_residual_anatomy(
        run_id=args.run_id,
        source_report_path=args.source_report_path,
        family=args.family,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"]["json"])


if __name__ == "__main__":
    main()
