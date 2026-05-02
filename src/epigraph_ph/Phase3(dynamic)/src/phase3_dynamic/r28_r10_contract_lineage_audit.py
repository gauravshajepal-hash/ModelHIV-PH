from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    R10_COMPARABLE_METRICS,
    _finite_float,
    _generated_at,
    _r10_horizon_replay_paths,
)
from .r26_r10_teacher_fusion import (
    _metric_scales,
    _rows_by_quarter,
    _select_reference_result,
)
from .runtime import ensure_dir, read_json, write_json


R28_SCHEMA_VERSION = "phase3_dynamic.r28_r10_contract_lineage_audit.v1"
R28_RUN_ID = "p3d-r28-r10-contract-lineage-audit-20260502-s00"


def _source_rows_by_quarter(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("quarter") or ""): dict(row)
        for row in rows
        if row.get("quarter")
    }


def _provenance_for_metric(source_row: dict[str, Any], metric_name: str) -> dict[str, Any]:
    provenance = dict(source_row.get("metric_provenance") or {}).get(metric_name)
    return dict(provenance) if isinstance(provenance, dict) else {}


def _lineage_id(provenance: dict[str, Any]) -> str:
    pieces = [
        str(provenance.get("source_quality_tier") or provenance.get("source_tier") or "unknown_source_tier"),
        str(provenance.get("measurement_class") or "unknown_measurement_class"),
        str(provenance.get("series_kind") or "unknown_series_kind"),
    ]
    return "|".join(pieces)


def _record_rows_from_split(
    split_row: dict[str, Any],
    *,
    horizon: int,
    source_by_quarter: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    scales = _metric_scales(split_row)
    r10_by_quarter = _rows_by_quarter(list(split_row.get("candidate_prediction_rows") or []))
    carry_by_quarter = _rows_by_quarter(list(split_row.get("carry_forward_prediction_rows") or []))
    records: list[dict[str, Any]] = []
    for target_row in list(split_row.get("holdout_target_rows") or []):
        if not isinstance(target_row, dict):
            continue
        quarter = str(target_row.get("quarter") or "")
        if not quarter:
            continue
        target_year = quarter_year(quarter)
        tiers = dict(target_row.get("metric_tiers") or {})
        source_row = source_by_quarter.get(quarter, {})
        r10_row = r10_by_quarter.get(quarter, {})
        carry_row = carry_by_quarter.get(quarter, {})
        for metric_name in R10_COMPARABLE_METRICS:
            tier = str(tiers.get(metric_name) or "")
            if tier not in {"exact_observed", "bridge_observed"}:
                continue
            target = _finite_float(target_row.get(metric_name))
            r10_value = _finite_float(r10_row.get(metric_name))
            carry_value = _finite_float(carry_row.get(metric_name))
            if target is None or r10_value is None or carry_value is None:
                continue
            scale = max(float(scales.get(metric_name, 1.0)), float(np.finfo(np.float32).eps))
            provenance = _provenance_for_metric(source_row, metric_name)
            unknown_provenance = not provenance
            r10_error = abs(float(r10_value) - float(target)) / scale
            carry_error = abs(float(carry_value) - float(target)) / scale
            records.append(
                {
                    "horizon_years": int(horizon),
                    "train_end_year": int(split_row.get("train_end_year") or 0),
                    "target_year": int(target_year),
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "metric_tier": tier,
                    "source_lineage": _lineage_id(provenance),
                    "unknown_provenance": bool(unknown_provenance),
                    "source_id": str(provenance.get("source_id") or "unknown_source_id"),
                    "source_quality_tier": str(provenance.get("source_quality_tier") or provenance.get("source_tier") or "unknown_source_tier"),
                    "measurement_class": str(provenance.get("measurement_class") or "unknown_measurement_class"),
                    "series_kind": str(provenance.get("series_kind") or "unknown_series_kind"),
                    "support_partition": str(provenance.get("support_partition") or "unknown_support_partition"),
                    "aggregation_mode": str(provenance.get("aggregation_mode") or "unknown_aggregation_mode"),
                    "extraction_method": str(provenance.get("extraction_method") or "unknown_extraction_method"),
                    "target_value": float(target),
                    "r10_value": float(r10_value),
                    "carry_forward_value": float(carry_value),
                    "scale": scale,
                    "r10_norm_error": float(r10_error),
                    "carry_forward_norm_error": float(carry_error),
                    "r10_minus_carry_forward_norm_error": float(r10_error - carry_error),
                    "r10_beats_carry_forward": bool(r10_error < carry_error),
                }
            )
    return records


def _collect_r10_records(
    *,
    epigraph_root: Path,
    horizons: tuple[int, ...],
    source_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    paths = _r10_horizon_replay_paths(epigraph_root, horizons)
    records: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        path = paths.get(int(horizon))
        if path is None:
            reference_rows.append(
                {
                    "horizon_years": int(horizon),
                    "available": False,
                    "blocker": "missing_r10_replay_artifact",
                }
            )
            continue
        report = read_json(Path(path), default={})
        if not isinstance(report, dict):
            reference_rows.append(
                {
                    "horizon_years": int(horizon),
                    "available": False,
                    "blocker": "invalid_r10_replay_artifact",
                    "artifact_path": Path(path).as_posix(),
                }
            )
            continue
        result = _select_reference_result(report)
        reference_rows.append(
            {
                "horizon_years": int(horizon),
                "available": True,
                "artifact_path": Path(path).as_posix(),
                "reference_experiment_id": str(result.get("experiment_id") or ""),
                "reference_family": str(result.get("family") or ""),
                "split_count": len(list(result.get("quarterly_rows") or [])),
                "reported_candidate_mean_mae": _finite_float(dict(result.get("quarterly_summary") or {}).get("candidate_mean_mae")),
                "reported_carry_forward_mean_mae": _finite_float(dict(result.get("quarterly_summary") or {}).get("carry_forward_mean_mae")),
            }
        )
        for split_row in list(result.get("quarterly_rows") or []):
            if isinstance(split_row, dict):
                records.extend(_record_rows_from_split(dict(split_row), horizon=int(horizon), source_by_quarter=source_by_quarter))
    return records, reference_rows


def _aggregate(records: list[dict[str, Any]], group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(record.get(field) for field in group_fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(part) for part in item[0])):
        r10_errors = np.asarray([float(row["r10_norm_error"]) for row in values], dtype=np.float64)
        carry_errors = np.asarray([float(row["carry_forward_norm_error"]) for row in values], dtype=np.float64)
        output = {field: key[index] for index, field in enumerate(group_fields)}
        output.update(
            {
                "entry_count": len(values),
                "r10_mean_norm_error": float(np.mean(r10_errors)),
                "carry_forward_mean_norm_error": float(np.mean(carry_errors)),
                "r10_minus_carry_forward_mean_norm_error": float(np.mean(r10_errors) - np.mean(carry_errors)),
                "r10_worst_norm_error": float(np.max(r10_errors)),
                "carry_forward_worst_norm_error": float(np.max(carry_errors)),
                "r10_better_share": float(np.mean(np.asarray([1.0 if row["r10_beats_carry_forward"] else 0.0 for row in values], dtype=np.float64))),
            }
        )
        rows.append(output)
    return rows


def _provenance_coverage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record.get("horizon_years") or 0)].append(record)
    rows: list[dict[str, Any]] = []
    for horizon, values in sorted(grouped.items()):
        unknown = [row for row in values if bool(row.get("unknown_provenance"))]
        rows.append(
            {
                "horizon_years": int(horizon),
                "entry_count": len(values),
                "unknown_provenance_count": len(unknown),
                "unknown_provenance_share": 0.0 if not values else float(len(unknown) / len(values)),
            }
        )
    return rows


def _r13_results_path(phase3_root: Path) -> Path:
    candidates = [
        phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r19-joint-service-r13-queue-20260502-final-v3"
        / "analysis"
        / "r13_priority_experiment_results.json",
        phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r19-joint-service-r13-queue-20260502-final"
        / "analysis"
        / "r13_priority_experiment_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _r24_results_path(phase3_root: Path) -> Path:
    return phase3_root / "artifacts" / "scientific_audits" / "phase3_r24_matched_r10_fairness_audit_results_20260502.json"


def _phase3_metric_rows(phase3_root: Path) -> list[dict[str, Any]]:
    path = _r13_results_path(phase3_root)
    report = read_json(path, default={})
    if not isinstance(report, dict):
        return []
    output: list[dict[str, Any]] = []
    for result in list(report.get("results") or []):
        if not isinstance(result, dict):
            continue
        experiment_id = str(result.get("experiment_id") or "")
        if experiment_id not in {"R13-006", "R13-050"}:
            continue
        for row in list(result.get("metric_rows") or []):
            if not isinstance(row, dict):
                continue
            if str(row.get("metric_name") or "") not in R10_COMPARABLE_METRICS:
                continue
            output.append(
                {
                    "experiment_id": experiment_id,
                    "row_scope": str(result.get("row_scope") or ""),
                    "family": str(result.get("family") or ""),
                    "horizon_years": int(row.get("horizon_years") or 0),
                    "metric_name": str(row.get("metric_name") or ""),
                    "train_end_year": row.get("train_end_year"),
                    "entry_count": int(row.get("entry_count") or 0),
                    "phase3_candidate_mean_norm_error": _finite_float(row.get("candidate_mean_norm_error")),
                    "phase3_carry_forward_mean_norm_error": _finite_float(row.get("carry_forward_mean_norm_error")),
                }
            )
    return output


def _phase3_metric_summary(phase3_metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in phase3_metric_rows:
        value = _finite_float(row.get("phase3_candidate_mean_norm_error"))
        if value is None:
            continue
        grouped[(str(row.get("row_scope") or ""), int(row.get("horizon_years") or 0), str(row.get("metric_name") or ""))].append(row)
    output: list[dict[str, Any]] = []
    for (row_scope, horizon, metric_name), values in sorted(grouped.items()):
        weights = np.asarray([max(int(row.get("entry_count") or 0), 1) for row in values], dtype=np.float64)
        phase3 = np.asarray([float(row["phase3_candidate_mean_norm_error"]) for row in values], dtype=np.float64)
        carry_values = [
            float(row["phase3_carry_forward_mean_norm_error"])
            for row in values
            if _finite_float(row.get("phase3_carry_forward_mean_norm_error")) is not None
        ]
        output.append(
            {
                "row_scope": row_scope,
                "horizon_years": horizon,
                "metric_name": metric_name,
                "entry_count": int(sum(int(row.get("entry_count") or 0) for row in values)),
                "phase3_candidate_mean_norm_error": float(np.average(phase3, weights=weights)),
                "phase3_carry_forward_mean_norm_error": None
                if not carry_values
                else float(np.mean(np.asarray(carry_values, dtype=np.float64))),
            }
        )
    return output


def _phase3_vs_r10_rows(
    *,
    r10_metric_horizon_rows: list[dict[str, Any]],
    phase3_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    r10_by_key = {
        (int(row.get("horizon_years") or 0), str(row.get("metric_name") or "")): dict(row)
        for row in r10_metric_horizon_rows
    }
    output: list[dict[str, Any]] = []
    for row in phase3_rows:
        key = (int(row.get("horizon_years") or 0), str(row.get("metric_name") or ""))
        r10_row = r10_by_key.get(key)
        if not r10_row:
            continue
        phase3_value = _finite_float(row.get("phase3_candidate_mean_norm_error"))
        r10_value = _finite_float(r10_row.get("r10_mean_norm_error"))
        if phase3_value is None or r10_value is None:
            continue
        output.append(
            {
                "row_scope": str(row.get("row_scope") or ""),
                "horizon_years": key[0],
                "metric_name": key[1],
                "phase3_candidate_mean_norm_error": phase3_value,
                "matched_r10_mean_norm_error": r10_value,
                "carry_forward_mean_norm_error": r10_row.get("carry_forward_mean_norm_error"),
                "phase3_minus_r10_mean_norm_error": float(phase3_value - r10_value),
                "phase3_entry_count": int(row.get("entry_count") or 0),
                "r10_entry_count": int(r10_row.get("entry_count") or 0),
                "comparison_contract": "approximate_metric_horizon_comparison; split scopes differ between R13 and dense R10 replay",
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _top_rows(rows: list[dict[str, Any]], key: str, *, descending: bool = True, limit: int = 8) -> list[dict[str, Any]]:
    return sorted(
        [row for row in rows if _finite_float(row.get(key)) is not None],
        key=lambda row: float(row[key]),
        reverse=descending,
    )[:limit]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R28 Matched R10 Contract Lineage Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## R10 Reference Rows",
        "",
        "| Horizon | Reference | R10 MAE | Carry MAE | Splits |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in list(report.get("r10_reference_rows") or []):
        lines.append(
            f"| {row.get('horizon_years')} | `{row.get('reference_experiment_id')}` | "
            f"{_format_float(row.get('reported_candidate_mean_mae'))} | "
            f"{_format_float(row.get('reported_carry_forward_mean_mae'))} | {row.get('split_count')} |"
        )
    lines.extend(
        [
            "",
            "## Largest Phase3 vs R10 Metric Gaps",
            "",
            "| Scope | Horizon | Metric | Phase3 | R10 | Delta |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in _top_rows(list(report.get("phase3_vs_r10_metric_rows") or []), "phase3_minus_r10_mean_norm_error"):
        lines.append(
            f"| {row.get('row_scope')} | {row.get('horizon_years')} | `{row.get('metric_name')}` | "
            f"{_format_float(row.get('phase3_candidate_mean_norm_error'))} | "
            f"{_format_float(row.get('matched_r10_mean_norm_error'))} | "
            f"{_format_float(row.get('phase3_minus_r10_mean_norm_error'))} |"
        )
    lines.extend(
        [
            "",
            "## R10 Error By Metric And Horizon",
            "",
            "| Horizon | Metric | Entries | R10 | Carry | R10 minus carry | R10 better share |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(report.get("r10_metric_horizon_rows") or []):
        lines.append(
            f"| {row.get('horizon_years')} | `{row.get('metric_name')}` | {row.get('entry_count')} | "
            f"{_format_float(row.get('r10_mean_norm_error'))} | "
            f"{_format_float(row.get('carry_forward_mean_norm_error'))} | "
            f"{_format_float(row.get('r10_minus_carry_forward_mean_norm_error'))} | "
            f"{_format_float(row.get('r10_better_share'))} |"
        )
    lines.extend(
        [
            "",
            "## Dominant Target Lineages",
            "",
            "| Horizon | Source lineage | Entries | R10 | Carry | R10 minus carry |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in _top_rows(list(report.get("r10_lineage_horizon_rows") or []), "entry_count", limit=10):
        lines.append(
            f"| {row.get('horizon_years')} | `{row.get('source_lineage')}` | {row.get('entry_count')} | "
            f"{_format_float(row.get('r10_mean_norm_error'))} | "
            f"{_format_float(row.get('carry_forward_mean_norm_error'))} | "
            f"{_format_float(row.get('r10_minus_carry_forward_mean_norm_error'))} |"
        )
    lines.extend(
        [
            "",
            "## Provenance Coverage",
            "",
            "| Horizon | Entries | Unmapped entries | Unmapped share |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in list(report.get("provenance_coverage_rows") or []):
        lines.append(
            f"| {row.get('horizon_years')} | {row.get('entry_count')} | "
            f"{row.get('unknown_provenance_count')} | {_format_float(row.get('unknown_provenance_share'))} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R28 is not a model branch and cannot promote a champion.",
            "- It reads the frozen horizon-matched R10 replay artifacts and joins scored target rows back to the active ObservationRoleLedger-derived provenance.",
            "- Rows reported as unmapped could not be traced to an active strict-ledger metric provenance row for the same quarter and metric.",
            "- Phase3-vs-R10 metric comparison is labeled approximate because R13 Phase3 rows and dense R10 replay rows do not have identical split scopes.",
            "- The audit is intended to decide whether the next useful action is model dynamics, observation-lineage stratification, or benchmark-contract revision.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    metric_rows = list(report.get("r10_metric_horizon_rows") or [])
    compare_rows = list(report.get("phase3_vs_r10_metric_rows") or [])
    horizons = sorted({int(row.get("horizon_years") or 0) for row in metric_rows})
    metrics = list(R10_COMPARABLE_METRICS)
    r10_matrix = np.full((len(metrics), len(horizons)), np.nan, dtype=np.float64)
    for row in metric_rows:
        metric = str(row.get("metric_name") or "")
        horizon = int(row.get("horizon_years") or 0)
        if metric in metrics and horizon in horizons:
            r10_matrix[metrics.index(metric), horizons.index(horizon)] = float(row.get("r10_mean_norm_error") or np.nan)
    compare_filtered = [row for row in compare_rows if str(row.get("row_scope") or "") == "all"]
    delta_matrix = np.full((len(metrics), len(horizons)), np.nan, dtype=np.float64)
    for row in compare_filtered:
        metric = str(row.get("metric_name") or "")
        horizon = int(row.get("horizon_years") or 0)
        if metric in metrics and horizon in horizons:
            delta_matrix[metrics.index(metric), horizons.index(horizon)] = float(row.get("phase3_minus_r10_mean_norm_error") or np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("R28 Matched R10 Contract Lineage Audit", fontsize=14, fontweight="bold")
    im0 = axes[0].imshow(r10_matrix, aspect="auto", cmap="Blues")
    axes[0].set_title("Matched R10 error")
    axes[0].set_xticks(np.arange(len(horizons)))
    axes[0].set_xticklabels([f"h{h}" for h in horizons])
    axes[0].set_yticks(np.arange(len(metrics)))
    axes[0].set_yticklabels(metrics)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    vmax = float(np.nanmax(np.abs(delta_matrix))) if np.isfinite(delta_matrix).any() else 1.0
    im1 = axes[1].imshow(delta_matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[1].set_title("Phase3 minus R10, full sentinel")
    axes[1].set_xticks(np.arange(len(horizons)))
    axes[1].set_xticklabels([f"h{h}" for h in horizons])
    axes[1].set_yticks(np.arange(len(metrics)))
    axes[1].set_yticklabels(metrics)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r28_r10_contract_lineage_audit(
    *,
    run_id: str = R28_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    source_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    phase3_root = sandbox_repo_root()
    r10_records, reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    r10_metric_horizon_rows = _aggregate(r10_records, ("horizon_years", "metric_name"))
    r10_lineage_horizon_rows = _aggregate(r10_records, ("horizon_years", "source_lineage"))
    r10_tier_horizon_rows = _aggregate(r10_records, ("horizon_years", "metric_tier"))
    r10_metric_year_rows = _aggregate(r10_records, ("horizon_years", "metric_name", "target_year"))
    r10_source_metric_rows = _aggregate(r10_records, ("horizon_years", "metric_name", "source_lineage", "metric_tier"))
    provenance_coverage_rows = _provenance_coverage_rows(r10_records)
    phase3_metrics = _phase3_metric_summary(_phase3_metric_rows(phase3_root))
    phase3_vs_r10 = _phase3_vs_r10_rows(
        r10_metric_horizon_rows=r10_metric_horizon_rows,
        phase3_rows=phase3_metrics,
    )
    positive_gaps = [
        row
        for row in phase3_vs_r10
        if _finite_float(row.get("phase3_minus_r10_mean_norm_error")) is not None
        and float(row["phase3_minus_r10_mean_norm_error"]) > 0.0
    ]
    blockers = sorted(
        {
            f"{row.get('row_scope')}_h{row.get('horizon_years')}_{row.get('metric_name')}_phase3_worse_than_r10"
            for row in positive_gaps
        }
    )
    if any(float(row.get("unknown_provenance_share") or 0.0) > 0.0 for row in provenance_coverage_rows):
        blockers.append("matched_r10_replay_contains_targets_unmapped_to_active_observation_ledger")
    verdict = (
        "R28 is an audit-only artifact. It localizes the matched-R10 blocker to the listed metric/horizon/lineage cells; "
        "no model is promoted."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R28_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "horizons": list(horizons),
        "record_count": len(r10_records),
        "blockers": blockers,
        "verdict": verdict,
        "r10_reference_rows": reference_rows,
        "r10_metric_horizon_rows": r10_metric_horizon_rows,
        "r10_lineage_horizon_rows": r10_lineage_horizon_rows,
        "r10_tier_horizon_rows": r10_tier_horizon_rows,
        "r10_metric_year_rows": r10_metric_year_rows,
        "r10_source_metric_rows": r10_source_metric_rows,
        "provenance_coverage_rows": provenance_coverage_rows,
        "phase3_metric_rows": phase3_metrics,
        "phase3_vs_r10_metric_rows": phase3_vs_r10,
        "artifact_paths": {},
        "contract": "audit_only_no_promotion",
    }
    json_path = analysis_dir / "r28_r10_contract_lineage_audit.json"
    md_path = analysis_dir / "r28_r10_contract_lineage_audit.md"
    dashboard_path = analysis_dir / "r28_r10_contract_lineage_audit_dashboard.png"
    metric_csv_path = analysis_dir / "r28_metric_horizon_rows.csv"
    lineage_csv_path = analysis_dir / "r28_lineage_horizon_rows.csv"
    compare_csv_path = analysis_dir / "r28_phase3_vs_r10_metric_rows.csv"
    provenance_csv_path = analysis_dir / "r28_provenance_coverage_rows.csv"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
        "metric_horizon_csv": metric_csv_path.as_posix(),
        "lineage_horizon_csv": lineage_csv_path.as_posix(),
        "phase3_vs_r10_csv": compare_csv_path.as_posix(),
        "provenance_coverage_csv": provenance_csv_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(metric_csv_path, r10_metric_horizon_rows)
    _write_csv(lineage_csv_path, r10_lineage_horizon_rows)
    _write_csv(compare_csv_path, phase3_vs_r10)
    _write_csv(provenance_csv_path, provenance_coverage_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R28 matched-R10 contract lineage audit.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R28_RUN_ID)
    args = parser.parse_args()
    run_r28_r10_contract_lineage_audit(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
