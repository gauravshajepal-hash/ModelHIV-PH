from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import COUNT_METRICS, _finite_float
from .runtime import ensure_dir, read_json, write_json


R50_SCHEMA_VERSION = "phase3_dynamic.r50_subnational_residual_anatomy.v1"
R50_RUN_ID = "p3d-r50-subnational-residual-anatomy-20260503-s00"
R48_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r48-subnational-proxy-champion-contract-20260503-s00"
    / "analysis"
    / "r48_subnational_proxy_champion_contract_report.json"
)
R49_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r49-subnational-module-champion-gate-20260503-s00"
    / "analysis"
    / "r49_subnational_module_champion_gate_report.json"
)


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {})


def _score_rows_with_module_selector(r48_report: dict[str, Any], r49_report: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(row) for row in list(r48_report.get("score_rows") or [])] + [
        dict(row) for row in list(r49_report.get("module_score_rows") or [])
    ]


def _normalized_error(rows: list[dict[str, Any]]) -> float | None:
    error_sum = 0.0
    target_sum = 0.0
    for row in rows:
        error = _finite_float(row.get("absolute_error"))
        target = _finite_float(row.get("target_abs"))
        if error is None:
            error = _finite_float(row.get("absolute_error_sum"))
        if target is None:
            target = _finite_float(row.get("target_abs_sum"))
        if error is None or target is None:
            continue
        error_sum += float(error)
        target_sum += abs(float(target))
    if target_sum <= 0.0:
        return None
    return float(error_sum / target_sum)


def _region_metric_score_table(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        family = str(row.get("candidate_family") or "")
        region = str(row.get("region") or "")
        metric = str(row.get("metric_name") or "")
        if family and region and metric:
            grouped[(family, region, metric)].append(dict(row))
    table: list[dict[str, Any]] = []
    for (family, region, metric), rows in sorted(grouped.items()):
        score = _normalized_error(rows)
        if score is None:
            continue
        table.append(
            {
                "candidate_family": family,
                "region": region,
                "metric_name": metric,
                "normalized_absolute_error": score,
                "absolute_error_sum": float(sum(float(row.get("absolute_error") or 0.0) for row in rows)),
                "target_abs_sum": float(sum(abs(float(row.get("target_abs") or 0.0)) for row in rows)),
                "scored_period_count": len({str(row.get("holdout_period") or "") for row in rows}),
            }
        )
    return table


def _region_summary_table(region_metric_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in region_metric_score_rows:
        grouped[(str(row.get("candidate_family") or ""), str(row.get("region") or ""))].append(dict(row))
    table: list[dict[str, Any]] = []
    for (family, region), rows in sorted(grouped.items()):
        scores = [float(row.get("normalized_absolute_error") or 0.0) for row in rows]
        table.append(
            {
                "candidate_family": family,
                "region": region,
                "mean_metric_normalized_absolute_error": float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else None,
                "worst_metric_normalized_absolute_error": float(np.max(np.asarray(scores, dtype=np.float64))) if scores else None,
                "scored_metric_count": len(scores),
            }
        )
    return table


def _region_best_family_table(region_summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in region_summary_rows:
        by_region[str(row.get("region") or "")].append(dict(row))
    rows: list[dict[str, Any]] = []
    for region, candidates in sorted(by_region.items()):
        candidates.sort(key=lambda row: float(row.get("mean_metric_normalized_absolute_error") or float("inf")))
        best = candidates[0] if candidates else {}
        by_family = {str(row.get("candidate_family") or ""): row for row in candidates}
        carry = _finite_float((by_family.get("regional_carry_forward") or {}).get("mean_metric_normalized_absolute_error"))
        aggregate = _finite_float((by_family.get("aggregate_log_trend") or {}).get("mean_metric_normalized_absolute_error"))
        similarity = _finite_float((by_family.get("similarity_proxy_log_delta") or {}).get("mean_metric_normalized_absolute_error"))
        module = _finite_float((by_family.get("module_local_selector") or {}).get("mean_metric_normalized_absolute_error"))
        rows.append(
            {
                "region": region,
                "best_candidate_family": best.get("candidate_family"),
                "best_mean_metric_normalized_absolute_error": best.get("mean_metric_normalized_absolute_error"),
                "regional_carry_forward_mean_metric_normalized_absolute_error": carry,
                "aggregate_log_trend_mean_metric_normalized_absolute_error": aggregate,
                "similarity_proxy_log_delta_mean_metric_normalized_absolute_error": similarity,
                "module_local_selector_mean_metric_normalized_absolute_error": module,
                "similarity_beats_aggregate": similarity is not None and aggregate is not None and similarity < aggregate,
                "aggregate_beats_carry_forward": aggregate is not None and carry is not None and aggregate < carry,
            }
        )
    return rows


def _metric_best_family_table(region_metric_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in region_metric_score_rows:
        grouped[(str(row.get("candidate_family") or ""), str(row.get("metric_name") or ""))].append(dict(row))
    rows: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (family, metric), metric_rows in grouped.items():
        score = _normalized_error(metric_rows)
        if score is None:
            continue
        by_metric[metric].append(
            {
                "metric_name": metric,
                "candidate_family": family,
                "normalized_absolute_error": score,
                "scored_region_metric_count": len(metric_rows),
            }
        )
    for metric, candidates in sorted(by_metric.items()):
        candidates.sort(key=lambda row: float(row.get("normalized_absolute_error") or float("inf")))
        best = candidates[0]
        by_family = {str(row.get("candidate_family") or ""): row for row in candidates}
        rows.append(
            {
                "metric_name": metric,
                "best_candidate_family": best.get("candidate_family"),
                "best_normalized_absolute_error": best.get("normalized_absolute_error"),
                "regional_carry_forward_normalized_absolute_error": (by_family.get("regional_carry_forward") or {}).get("normalized_absolute_error"),
                "aggregate_log_trend_normalized_absolute_error": (by_family.get("aggregate_log_trend") or {}).get("normalized_absolute_error"),
                "similarity_proxy_log_delta_normalized_absolute_error": (by_family.get("similarity_proxy_log_delta") or {}).get("normalized_absolute_error"),
                "module_local_selector_normalized_absolute_error": (by_family.get("module_local_selector") or {}).get("normalized_absolute_error"),
            }
        )
    return rows


def _gate(region_best_rows: list[dict[str, Any]], metric_best_rows: list[dict[str, Any]]) -> dict[str, Any]:
    hard_regions = sorted(
        region_best_rows,
        key=lambda row: float(row.get("module_local_selector_mean_metric_normalized_absolute_error") or float("inf")),
        reverse=True,
    )[:5]
    proxy_help_regions = [
        str(row.get("region") or "")
        for row in region_best_rows
        if row.get("similarity_beats_aggregate") is True
    ]
    metric_best_families = {
        str(row.get("metric_name") or ""): str(row.get("best_candidate_family") or "")
        for row in metric_best_rows
    }
    blockers: list[str] = []
    if not hard_regions:
        blockers.append("no_region_residuals_scored")
    if not proxy_help_regions:
        blockers.append("similarity_proxy_never_beats_aggregate_by_region")
    status = "residual_anatomy_ready_for_region_specific_selector" if not blockers else "residual_anatomy_ready_but_proxy_signal_weak"
    return {
        "status": status,
        "blockers": blockers,
        "top_hard_regions_by_module_selector": hard_regions,
        "regions_where_similarity_proxy_beats_aggregate": proxy_help_regions,
        "metric_best_families": metric_best_families,
        "contract": (
            "R50 is an anatomy artifact only. It can nominate regions or metrics for a future train-only "
            "selector, but it cannot promote region-specific equations because the table uses holdout residuals."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("residual_anatomy_gate") or {})
    lines = [
        "# Phase 3 R50 Subnational Residual Anatomy",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        f"- Regions where similarity proxy beats aggregate: `{', '.join(gate.get('regions_where_similarity_proxy_beats_aggregate') or []) or 'none'}`",
        "",
        "## Metric Best Families",
        "",
        "| Metric | Best family | Best NAE | Carry-forward NAE | Aggregate NAE | Similarity NAE | Module selector NAE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in list(report.get("metric_best_family_table") or []):
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('best_candidate_family')}` | "
            f"{float(row.get('best_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('regional_carry_forward_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('aggregate_log_trend_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('similarity_proxy_log_delta_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('module_local_selector_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Hard Regions", "", "| Region | Best family | Module selector NAE | Aggregate NAE | Similarity NAE |", "|---|---|---:|---:|---:|"])
    for row in list(gate.get("top_hard_regions_by_module_selector") or []):
        lines.append(
            f"| `{row.get('region')}` | `{row.get('best_candidate_family')}` | "
            f"{float(row.get('module_local_selector_mean_metric_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('aggregate_log_trend_mean_metric_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('similarity_proxy_log_delta_mean_metric_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(
    path: Path,
    *,
    region_best_rows: list[dict[str, Any]],
    metric_best_rows: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    hard = sorted(
        region_best_rows,
        key=lambda row: float(row.get("module_local_selector_mean_metric_normalized_absolute_error") or 0.0),
        reverse=True,
    )[:10]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    regions = [str(row.get("region") or "") for row in hard]
    module_scores = [float(row.get("module_local_selector_mean_metric_normalized_absolute_error") or 0.0) for row in hard]
    axes[0].barh(np.arange(len(regions)), module_scores, color="#7a514f")
    axes[0].set_yticks(np.arange(len(regions)))
    axes[0].set_yticklabels(regions)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean metric normalized absolute error")
    axes[0].set_title("Hardest regions under R49 module selector")
    metrics = [str(row.get("metric_name") or "") for row in metric_best_rows]
    aggregate = [float(row.get("aggregate_log_trend_normalized_absolute_error") or 0.0) for row in metric_best_rows]
    similarity = [float(row.get("similarity_proxy_log_delta_normalized_absolute_error") or 0.0) for row in metric_best_rows]
    x = np.arange(len(metrics))
    width = 0.35
    axes[1].bar(x - width / 2, aggregate, width, label="aggregate trend", color="#496f9e")
    axes[1].bar(x + width / 2, similarity, width, label="similarity proxy", color="#8a6f2a")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, rotation=25, ha="right")
    axes[1].set_ylabel("normalized absolute error")
    axes[1].set_title("Metric-level aggregate vs similarity proxy")
    axes[1].legend(frameon=False)
    fig.suptitle("R50 subnational residual anatomy", fontsize=14, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r50_subnational_residual_anatomy(
    *,
    run_id: str = R50_RUN_ID,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
) -> dict[str, Any]:
    r48_path = Path(r48_report_path) if r48_report_path is not None else R48_DEFAULT_REPORT
    r49_path = Path(r49_report_path) if r49_report_path is not None else R49_DEFAULT_REPORT
    r48_report = _load_report(r48_path)
    r49_report = _load_report(r49_path)
    score_rows = _score_rows_with_module_selector(r48_report, r49_report)
    region_metric_score_rows = _region_metric_score_table(score_rows)
    region_summary_rows = _region_summary_table(region_metric_score_rows)
    region_best_rows = _region_best_family_table(region_summary_rows)
    metric_best_rows = _metric_best_family_table(region_metric_score_rows)
    gate = _gate(region_best_rows, metric_best_rows)
    verdict = (
        "R50 found region-level proxy signal that can justify a future train-only regional selector."
        if gate["status"] == "residual_anatomy_ready_for_region_specific_selector"
        else "R50 found no robust region-level proxy signal beyond aggregate trend; subnational work should next focus on lineage support and module-specific evidence rather than fitting free regional equations."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r50_subnational_residual_anatomy_report.json"
    md_path = analysis_dir / "r50_subnational_residual_anatomy_report.md"
    region_metric_csv = analysis_dir / "r50_region_metric_score_table.csv"
    region_summary_csv = analysis_dir / "r50_region_summary_table.csv"
    region_best_csv = analysis_dir / "r50_region_best_family_table.csv"
    metric_best_csv = analysis_dir / "r50_metric_best_family_table.csv"
    dashboard_path = analysis_dir / "r50_subnational_residual_anatomy_dashboard.png"
    report = {
        "schema_version": R50_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "claim_limit": "residual anatomy only; region-specific selector must be trained without holdout residual leakage",
        },
        "residual_anatomy_gate": gate,
        "region_metric_score_table": region_metric_score_rows,
        "region_summary_table": region_summary_rows,
        "region_best_family_table": region_best_rows,
        "metric_best_family_table": metric_best_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "region_metric_score_csv": region_metric_csv.as_posix(),
            "region_summary_csv": region_summary_csv.as_posix(),
            "region_best_family_csv": region_best_csv.as_posix(),
            "metric_best_family_csv": metric_best_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(region_metric_csv, region_metric_score_rows)
    _write_csv(region_summary_csv, region_summary_rows)
    _write_csv(region_best_csv, region_best_rows)
    _write_csv(metric_best_csv, metric_best_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, region_best_rows=region_best_rows, metric_best_rows=metric_best_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R50 subnational residual anatomy.")
    parser.add_argument("--run-id", default=R50_RUN_ID)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    args = parser.parse_args()
    run_r50_subnational_residual_anatomy(
        run_id=str(args.run_id),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
    )


if __name__ == "__main__":
    _main()
