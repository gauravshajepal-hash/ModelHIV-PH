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
from .diagnosis_incidence_repair import (
    _annual_incidence_targets,
    _default_source_report,
    _evaluate_repair_split,
    _finite_float,
    _r10_annual_incidence_error,
    _selected_family,
)
from .hybrid_champion import _load_r10_baseline, _load_reference_config, _r10_reference_scores, _rolling_splits
from .lifted_residual_anatomy import _r10_metric_reference
from .metrics import quarter_year
from .monthly_joint_observation import MonthlyJointContext
from .monthly_latent_state import MonthlyLatentContext
from .monthly_shock import MonthlyShockContext
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .runtime import ensure_dir, read_json, write_json
from .scenario_lab import DEFAULT_ACTIVE_SOURCE_RUN_ID, DEFAULT_BASELINE_SOURCE_RUN_ID

BACKLOG_RESIDUAL_ANATOMY_SCHEMA_VERSION = "phase3_dynamic_backlog_residual_anatomy.v1"
DEFAULT_REPAIR_FAMILY = "monthly_backlog_late_emission"

STREAM_LABELS: dict[str, str] = {
    "diagnosed_plhiv": "diagnosis_stock",
    "new_diagnosed_cases_period": "diagnosis_flow_incidence_alignment",
    "alive_on_art": "art_stock",
    "tested_for_viral_load": "vl_testing",
    "virally_suppressed": "suppression",
    "annual_new_infections": "incidence_validation",
}


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


def _default_repair_report() -> Path:
    candidates = sorted(
        (sandbox_repo_root() / "artifacts" / "runs").glob(
            "p3d-backlog-late-emission-*/analysis/diagnosis_incidence_repair.json"
        )
    )
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("No backlog-emission diagnosis repair report was found.")


def _entry_rows(split_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for split in split_rows:
        common = {
            "repair_family": str(split.get("repair_family") or ""),
            "base_family": str(split.get("base_family") or ""),
            "delegate_family": str(split.get("delegate_family") or ""),
            "train_end_year": int(split.get("train_end_year") or 0),
            "horizon_years": int(split.get("horizon_years") or 0),
        }
        for row in list(split.get("metric_entries") or []):
            quarter = str(row.get("quarter") or "")
            metric_name = str(row.get("metric_name") or "")
            entries.append(
                {
                    **common,
                    "year": int(row.get("year") or quarter_year(quarter)),
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "stream": STREAM_LABELS.get(metric_name, metric_name),
                    "observed_value": float(row.get("observed_value") or 0.0),
                    "candidate_value": float(row.get("candidate_value") or 0.0),
                    "base_value": float(row.get("base_value") or 0.0),
                    "carry_forward_value": float(row.get("carry_forward_value") or 0.0),
                    "candidate_norm_error": float(row.get("candidate_norm_error") or 0.0),
                    "base_norm_error": float(row.get("base_norm_error") or 0.0),
                    "carry_forward_norm_error": float(row.get("carry_forward_norm_error") or 0.0),
                    "candidate_minus_base_norm_error": float(row.get("candidate_minus_base_norm_error") or 0.0),
                    "candidate_minus_carry_forward_norm_error": float(row.get("candidate_minus_carry_forward_norm_error") or 0.0),
                    "comparison_scope": "quarterly_observation",
                }
            )
        for row in list(split.get("incidence_validation_entries") or []):
            year = int(row.get("year") or 0)
            entries.append(
                {
                    **common,
                    "year": year,
                    "quarter": f"{year:04d}-Q4",
                    "metric_name": "annual_new_infections",
                    "stream": STREAM_LABELS["annual_new_infections"],
                    "observed_value": float(row.get("target_annual_new_infections") or 0.0),
                    "candidate_value": float(row.get("candidate_annual_incidence") or 0.0),
                    "base_value": float(row.get("base_annual_incidence") or 0.0),
                    "carry_forward_value": float(row.get("carry_forward_annual_incidence") or 0.0),
                    "candidate_norm_error": float(row.get("candidate_norm_error") or 0.0),
                    "base_norm_error": float(row.get("base_norm_error") or 0.0),
                    "carry_forward_norm_error": float(row.get("carry_forward_norm_error") or 0.0),
                    "candidate_minus_base_norm_error": float(row.get("candidate_minus_base_norm_error") or 0.0),
                    "candidate_minus_carry_forward_norm_error": float(row.get("candidate_minus_carry_forward_norm_error") or 0.0),
                    "comparison_scope": "validation_only_annual_incidence",
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
                "base_norm_mae": _mean([float(entry["base_norm_error"]) for entry in subset]),
                "carry_forward_norm_mae": _mean([float(entry["carry_forward_norm_error"]) for entry in subset]),
                "candidate_minus_base_norm_mae": _mean([float(entry["candidate_minus_base_norm_error"]) for entry in subset]),
                "candidate_minus_carry_forward_norm_mae": _mean(
                    [float(entry["candidate_minus_carry_forward_norm_error"]) for entry in subset]
                ),
                "candidate_norm_p90": _p90([float(entry["candidate_norm_error"]) for entry in subset]),
                "base_norm_p90": _p90([float(entry["base_norm_error"]) for entry in subset]),
                "carry_forward_norm_p90": _p90([float(entry["carry_forward_norm_error"]) for entry in subset]),
            }
        )
        rows.append(row)
    return rows


def _metric_scales_from_entries(entries: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric_name in sorted({str(entry.get("metric_name") or "") for entry in entries}):
        values = [
            abs(float(entry.get("observed_value") or 0.0))
            for entry in entries
            if str(entry.get("metric_name") or "") == metric_name
        ]
        scales[metric_name] = max(values or [1.0], default=1.0)
    return scales


def _attach_r10_proxy(
    rows: list[dict[str, Any]],
    *,
    entries: list[dict[str, Any]],
    r10_metric_reference: dict[str, dict[str, float]],
    r10_annual_incidence_error: float | None,
) -> list[dict[str, Any]]:
    scales = _metric_scales_from_entries(entries)
    output: list[dict[str, Any]] = []
    for row in rows:
        metric_name = str(row.get("metric_name") or "")
        next_row = dict(row)
        if metric_name == "annual_new_infections":
            next_row["r10_norm_mae_proxy"] = r10_annual_incidence_error
            next_row["r10_proxy_scope"] = "r10_annual_incidence_validation_error"
        else:
            reference = dict(r10_metric_reference.get(metric_name) or {})
            if reference:
                scale = max(scales.get(metric_name, 1.0), 1.0)
                next_row["r10_norm_mae_proxy"] = float(reference.get("raw_mean_abs_error") or 0.0) / scale
                next_row["r10_proxy_scope"] = "r10_aggregate_metric_raw_residual_normalized_by_active_observed_scale"
                next_row["r10_raw_mean_abs_error"] = float(reference.get("raw_mean_abs_error") or 0.0)
                next_row["r10_raw_p90_abs_error"] = float(reference.get("raw_p90_abs_error") or 0.0)
            else:
                next_row["r10_norm_mae_proxy"] = None
                next_row["r10_proxy_scope"] = "not_available"
        r10_proxy = _finite_float(next_row.get("r10_norm_mae_proxy"))
        candidate = _finite_float(next_row.get("candidate_norm_mae"))
        next_row["candidate_minus_r10_norm_mae_proxy"] = (
            None if r10_proxy is None or candidate is None else float(candidate - r10_proxy)
        )
        output.append(next_row)
    return output


def _top_worst(entries: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda row: (
            float(row.get("candidate_minus_base_norm_error") or 0.0),
            float(row.get("candidate_norm_error") or 0.0),
        ),
        reverse=True,
    )[:limit]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not keys:
            handle.write("")
            return
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
        "annual_new_infections",
    ]
    present = {str(row.get("metric_name") or "") for row in rows}
    return [metric for metric in preferred if metric in present] + sorted(present.difference(preferred))


def _write_dashboard(payload: dict[str, Any], path: Path) -> None:
    metric_rows = list(payload.get("metric_rows") or [])
    year_rows = sorted(list(payload.get("year_rows") or []), key=lambda row: int(row.get("year") or 0))
    metric_year_rows = list(payload.get("metric_year_rows") or [])
    worst_rows = list(payload.get("worst_rows") or [])[:8]
    r10_scalar = _finite_float((payload.get("r10_reference") or {}).get("reference_quarterly_mean_mae"))
    repair_label = str(payload.get("repair_family") or "repair_branch").replace("_", " ")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.8), constrained_layout=True)
    flat = axes.ravel()

    metrics = _ordered_metrics(metric_rows)
    x = np.arange(len(metrics), dtype=np.float64)
    candidate = [float(next((row.get("candidate_norm_mae") for row in metric_rows if row.get("metric_name") == metric), 0.0) or 0.0) for metric in metrics]
    base = [float(next((row.get("base_norm_mae") for row in metric_rows if row.get("metric_name") == metric), 0.0) or 0.0) for metric in metrics]
    r10 = [_finite_float(next((row.get("r10_norm_mae_proxy") for row in metric_rows if row.get("metric_name") == metric), None)) for metric in metrics]
    width = 0.28
    flat[0].bar(x - width / 2, candidate, width=width, color="#1f77b4", label=repair_label)
    flat[0].bar(x + width / 2, base, width=width, color="#8c8c8c", label="strict baseline")
    r10_x = [index + width * 1.5 for index, value in enumerate(r10) if value is not None]
    r10_y = [float(value) for value in r10 if value is not None]
    if r10_y:
        flat[0].scatter(r10_x, r10_y, color="#2ca02c", marker="D", s=36, label="R10 proxy")
    flat[0].set_xticks(x)
    flat[0].set_xticklabels(metrics, rotation=26, ha="right")
    flat[0].set_ylabel("normalized MAE")
    flat[0].set_title("A. Metric residuals: repair branch vs strict baseline/R10")
    flat[0].legend(frameon=False, fontsize=8)

    years = [int(row.get("year") or 0) for row in year_rows]
    candidate_y = [float(row.get("candidate_norm_mae") or 0.0) for row in year_rows]
    base_y = [float(row.get("base_norm_mae") or 0.0) for row in year_rows]
    flat[1].plot(years, candidate_y, marker="o", linewidth=2.0, color="#1f77b4", label=repair_label)
    flat[1].plot(years, base_y, marker="o", linewidth=1.8, color="#8c8c8c", label="strict baseline")
    if r10_scalar is not None:
        flat[1].axhline(r10_scalar, color="#2ca02c", linestyle="--", linewidth=1.5, label="R10 scalar")
    flat[1].set_ylabel("normalized MAE")
    flat[1].set_title("B. Year residuals: remaining R10 gap")
    flat[1].legend(frameon=False, fontsize=8)

    years_sorted = sorted({int(row.get("year") or 0) for row in metric_year_rows})
    metrics_sorted = _ordered_metrics(metric_year_rows)
    matrix = np.full((len(metrics_sorted), len(years_sorted)), np.nan, dtype=np.float64)
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
            if row is not None:
                matrix[i, j] = float(row.get("candidate_minus_base_norm_mae") or 0.0)
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size and not np.isnan(matrix).all() else 0.05
    vmax = max(vmax, 0.05)
    image = flat[2].imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    flat[2].set_xticks(np.arange(len(years_sorted)))
    flat[2].set_xticklabels(years_sorted, rotation=30, ha="right")
    flat[2].set_yticks(np.arange(len(metrics_sorted)))
    flat[2].set_yticklabels(metrics_sorted)
    flat[2].set_title("C. Repair branch minus strict baseline by metric/year")
    flat[2].set_xlabel("holdout year")
    flat[2].set_ylabel("metric")
    fig.colorbar(image, ax=flat[2], shrink=0.82, label="delta normalized MAE")

    labels = [f"{row.get('metric_name')} {row.get('quarter')}" for row in worst_rows]
    values = [float(row.get("candidate_minus_base_norm_error") or 0.0) for row in worst_rows]
    y = np.arange(len(labels), dtype=np.float64)
    flat[3].barh(y, values, color="#d62728")
    flat[3].set_yticks(y)
    flat[3].set_yticklabels(labels, fontsize=8)
    flat[3].invert_yaxis()
    flat[3].axvline(0.0, color="#111827", linewidth=0.8)
    flat[3].set_xlabel("repair minus baseline normalized error")
    flat[3].set_title("D. Worst repair side effects")

    fig.suptitle(f"{repair_label.title()} Residual Anatomy Against R10", fontsize=15, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _markdown_report(payload: dict[str, Any]) -> str:
    r10_reference = dict(payload.get("r10_reference") or {})
    repair_family = str(payload.get("repair_family") or "repair_branch")
    lines = [
        "# Repair-Branch Residual Anatomy Against R10",
        "",
        f"- Run ID: `{payload['run_id']}`",
        f"- Repair family: `{payload['repair_family']}`",
        f"- Base family: `{payload['base_family']}`",
        f"- Source run: `{payload['source_run_id']}`",
        f"- Baseline run: `{payload['baseline_source_run_id']}`",
        f"- R10 reference: `{r10_reference.get('reference_experiment_id')}` / `{r10_reference.get('reference_contract')}` / `{r10_reference.get('reference_quarterly_mean_mae')}`",
        "",
        "## Interpretation Contract",
        "",
        f"- This replay evaluates `{repair_family}`, not the generic hybrid family.",
        "- R10 does not expose per-year residual rows in the archived report; metric/year R10 comparisons therefore use aggregate metric proxies where available.",
        "- Annual incidence is validation-only and appears as `annual_new_infections`; it is not used for training this branch.",
        "",
        "## Metric Anatomy",
        "",
        "| metric | stream | entries | candidate norm MAE | baseline norm MAE | R10 proxy | candidate - R10 proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in list(payload.get("metric_rows") or []):
        r10_proxy = row.get("r10_norm_mae_proxy")
        delta = row.get("candidate_minus_r10_norm_mae_proxy")
        lines.append(
            "| {metric} | {stream} | {count} | {candidate:.6f} | {base:.6f} | {r10} | {delta} |".format(
                metric=row.get("metric_name"),
                stream=row.get("stream"),
                count=int(row.get("entry_count") or 0),
                candidate=float(row.get("candidate_norm_mae") or 0.0),
                base=float(row.get("base_norm_mae") or 0.0),
                r10="not_available" if r10_proxy is None else f"{float(r10_proxy):.6f}",
                delta="not_available" if delta is None else f"{float(delta):.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Year Anatomy",
            "",
            "| year | entries | candidate norm MAE | baseline norm MAE | candidate - baseline | candidate - R10 scalar |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    r10_scalar = _finite_float(r10_reference.get("reference_quarterly_mean_mae"))
    for row in list(payload.get("year_rows") or []):
        candidate = float(row.get("candidate_norm_mae") or 0.0)
        lines.append(
            "| {year} | {count} | {candidate:.6f} | {base:.6f} | {delta:.6f} | {r10_delta} |".format(
                year=int(row.get("year") or 0),
                count=int(row.get("entry_count") or 0),
                candidate=candidate,
                base=float(row.get("base_norm_mae") or 0.0),
                delta=float(row.get("candidate_minus_base_norm_mae") or 0.0),
                r10_delta="not_available" if r10_scalar is None else f"{candidate - r10_scalar:.6f}",
            )
        )
    lines.extend(
        [
            "",
            "## Worst Side Effects",
            "",
            "| quarter | metric | stream | horizon | candidate norm error | baseline norm error | candidate - baseline |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(payload.get("worst_rows") or [])[:12]:
        lines.append(
            "| {quarter} | {metric} | {stream} | {horizon} | {candidate:.6f} | {base:.6f} | {delta:.6f} |".format(
                quarter=row.get("quarter"),
                metric=row.get("metric_name"),
                stream=row.get("stream"),
                horizon=int(row.get("horizon_years") or 0),
                candidate=float(row.get("candidate_norm_error") or 0.0),
                base=float(row.get("base_norm_error") or 0.0),
                delta=float(row.get("candidate_minus_base_norm_error") or 0.0),
            )
        )
    return "\n".join(lines) + "\n"


def run_backlog_residual_anatomy(
    *,
    run_id: str = "p3d-backlog-residual-anatomy-20260428-s00",
    repair_report_path: str | Path | None = None,
    source_report_path: str | Path | None = None,
    repair_family: str = DEFAULT_REPAIR_FAMILY,
    family: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    min_train_years: int | None = None,
) -> dict[str, Any]:
    repair_report = Path(repair_report_path) if repair_report_path is not None else _default_repair_report()
    repair_payload = dict(read_json(repair_report, default={}) or {})
    source_report = (
        Path(source_report_path)
        if source_report_path is not None
        else Path(str(repair_payload.get("source_report_path") or _default_source_report()))
    )
    report = dict(read_json(source_report, default={}) or {})
    if not report:
        raise FileNotFoundError(f"Hybrid source report not found or empty: {source_report}")
    epigraph_root = default_epigraph_root()
    source_run_id = resolve_active_source_run_id(
        epigraph_root,
        str(repair_payload.get("source_run_id") or report.get("source_run_id") or DEFAULT_ACTIVE_SOURCE_RUN_ID),
    )
    baseline_source_run_id = resolve_baseline_source_run_id(
        epigraph_root,
        source_run_id=source_run_id,
        preferred=str(repair_payload.get("baseline_source_run_id") or report.get("baseline_source_run_id") or DEFAULT_BASELINE_SOURCE_RUN_ID),
    )
    base_family = str(repair_payload.get("base_family") or _selected_family(report, family))
    benchmark_contract = dict(repair_payload.get("benchmark_contract") or report.get("benchmark_contract") or {})
    start = int(start_year if start_year is not None else benchmark_contract.get("start_year") or 2010)
    end = int(end_year if end_year is not None else benchmark_contract.get("end_year") or 2025)
    minimum_train = int(min_train_years if min_train_years is not None else benchmark_contract.get("min_train_years") or 5)
    reference_config = _load_reference_config(Path(str(report.get("reference_report_path") or "")) if report.get("reference_report_path") else None)
    observation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    validation_rows = build_observation_rows(
        epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    validation_targets = _annual_incidence_targets(validation_rows)
    monthly_context = MonthlyShockContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    monthly_latent_context = MonthlyLatentContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    monthly_joint_context = MonthlyJointContext(epigraph_root=epigraph_root, source_run_id=source_run_id, baseline_source_run_id=baseline_source_run_id)
    backhalf_context = BackHalfChannelContext(
        epigraph_root=epigraph_root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        reengagement_sensitivity_mode=str(report.get("reengagement_sensitivity_mode") or "public_stock_flow_proxy"),
    )
    split_rows: list[dict[str, Any]] = []
    for horizon_years in (1, 5):
        for split in _rolling_splits(
            observation_rows,
            start_year=start,
            end_year=end,
            min_train_years=minimum_train,
            horizon_years=horizon_years,
        ):
            row = _evaluate_repair_split(
                repair_family=repair_family,
                base_family=base_family,
                observation_rows=observation_rows,
                constraint_rows=validation_rows,
                split=split,
                reference_config=reference_config,
                validation_targets=validation_targets,
                monthly_context=monthly_context,
                monthly_latent_context=monthly_latent_context,
                monthly_joint_context=monthly_joint_context,
                backhalf_context=backhalf_context,
                min_train_years=minimum_train,
                horizon_years=horizon_years,
            )
            if row is not None:
                split_rows.append(row)
    entries = _entry_rows(split_rows)
    r10_baseline = _load_r10_baseline(epigraph_root)
    r10_reference = _r10_reference_scores(r10_baseline)
    r10_annual_error = _r10_annual_incidence_error(r10_baseline, r10_reference)
    r10_metric_ref = _r10_metric_reference(r10_baseline, str(r10_reference.get("reference_contract") or ""))
    metric_rows = _attach_r10_proxy(
        _summarize_group(entries, group_keys=("metric_name", "stream")),
        entries=entries,
        r10_metric_reference=r10_metric_ref,
        r10_annual_incidence_error=r10_annual_error,
    )
    metric_year_rows = _attach_r10_proxy(
        _summarize_group(entries, group_keys=("metric_name", "stream", "year")),
        entries=entries,
        r10_metric_reference=r10_metric_ref,
        r10_annual_incidence_error=r10_annual_error,
    )
    year_rows = _summarize_group(entries, group_keys=("year",))
    horizon_rows = _summarize_group(entries, group_keys=("horizon_years",))
    worst_rows = _top_worst(entries)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / run_id / "analysis")
    payload = {
        "schema_version": BACKLOG_RESIDUAL_ANATOMY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "repair_report_path": repair_report.as_posix(),
        "source_report_path": source_report.as_posix(),
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "repair_family": repair_family,
        "base_family": base_family,
        "contract": {
            "purpose": "Residual anatomy for a diagnosis/incidence repair branch against strict baseline and R10 references by metric/year.",
            "r10_limitation": "R10 archived output exposes scalar path MAE and aggregate metric residuals, not aligned per-year residual rows.",
            "annual_incidence_use": "validation_only",
            "not_a_promotion_gate": True,
        },
        "benchmark_contract": {
            "start_year": start,
            "end_year": end,
            "min_train_years": minimum_train,
            "horizons": [1, 5],
        },
        "r10_reference": {**dict(r10_reference), "reference_annual_incidence_error": r10_annual_error},
        "split_count": len(split_rows),
        "entry_count": len(entries),
        "metric_rows": metric_rows,
        "year_rows": year_rows,
        "metric_year_rows": metric_year_rows,
        "horizon_rows": horizon_rows,
        "worst_rows": worst_rows,
        "artifact_paths": {
            "json": (analysis_dir / "backlog_residual_anatomy.json").as_posix(),
            "markdown": (analysis_dir / "backlog_residual_anatomy.md").as_posix(),
            "dashboard_png": (analysis_dir / "backlog_residual_anatomy_dashboard.png").as_posix(),
            "entry_csv": (analysis_dir / "backlog_residual_entries.csv").as_posix(),
            "metric_year_csv": (analysis_dir / "backlog_residual_metric_year.csv").as_posix(),
        },
    }
    write_json(Path(payload["artifact_paths"]["json"]), payload)
    Path(payload["artifact_paths"]["markdown"]).write_text(_markdown_report(payload), encoding="utf-8")
    _write_csv(Path(payload["artifact_paths"]["entry_csv"]), entries)
    _write_csv(Path(payload["artifact_paths"]["metric_year_csv"]), metric_year_rows)
    _write_dashboard(payload, Path(payload["artifact_paths"]["dashboard_png"]))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run backlog-emission residual anatomy against R10 by metric/year.")
    parser.add_argument("--run-id", default="p3d-backlog-residual-anatomy-20260428-s00")
    parser.add_argument("--repair-report-path", default=None)
    parser.add_argument("--source-report-path", default=None)
    parser.add_argument("--repair-family", default=DEFAULT_REPAIR_FAMILY)
    parser.add_argument("--family", default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--min-train-years", type=int, default=None)
    args = parser.parse_args()
    payload = run_backlog_residual_anatomy(
        run_id=args.run_id,
        repair_report_path=args.repair_report_path,
        source_report_path=args.source_report_path,
        repair_family=args.repair_family,
        family=args.family,
        start_year=args.start_year,
        end_year=args.end_year,
        min_train_years=args.min_train_years,
    )
    print(payload["artifact_paths"]["json"])


if __name__ == "__main__":
    main()
