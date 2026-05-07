from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .runtime import ensure_dir, read_json, write_json


R49_SCHEMA_VERSION = "phase3_dynamic.r49_subnational_module_champion_gate.v1"
R49_RUN_ID = "p3d-r49-subnational-module-champion-gate-20260503-s00"
R48_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r48-subnational-proxy-champion-contract-20260503-s00"
    / "analysis"
    / "r48_subnational_proxy_champion_contract_report.json"
)


def _load_r48_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {})


def _mean_score(rows: list[dict[str, Any]]) -> float | None:
    values = [_finite_float(row.get("normalized_absolute_error")) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _worst_score(rows: list[dict[str, Any]]) -> float | None:
    values = [_finite_float(row.get("normalized_absolute_error")) for row in rows]
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    return float(np.max(np.asarray(finite, dtype=np.float64)))


def _module_selector_table(split_metric_score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in split_metric_score_rows:
        metric = str(row.get("metric_name") or "")
        family = str(row.get("candidate_family") or "")
        if metric and family:
            grouped[(metric, family)].append(dict(row))
    rows: list[dict[str, Any]] = []
    for metric in COUNT_METRICS:
        candidate_rows: list[dict[str, Any]] = []
        families = sorted({family for candidate_metric, family in grouped if candidate_metric == metric})
        for family in families:
            family_rows = grouped.get((metric, family), [])
            mean_score = _mean_score(family_rows)
            if mean_score is None:
                continue
            candidate_rows.append(
                {
                    "metric_name": metric,
                    "candidate_family": family,
                    "mean_normalized_absolute_error": mean_score,
                    "worst_normalized_absolute_error": _worst_score(family_rows),
                    "scored_split_count": len(family_rows),
                }
            )
        candidate_rows.sort(key=lambda row: float(row.get("mean_normalized_absolute_error") or float("inf")))
        by_family = {str(row["candidate_family"]): row for row in candidate_rows}
        carry_score = _finite_float((by_family.get("regional_carry_forward") or {}).get("mean_normalized_absolute_error"))
        aggregate_score = _finite_float((by_family.get("aggregate_log_trend") or {}).get("mean_normalized_absolute_error"))
        similarity_score = _finite_float((by_family.get("similarity_proxy_log_delta") or {}).get("mean_normalized_absolute_error"))
        best = candidate_rows[0] if candidate_rows else {}
        selected_family = str(best.get("candidate_family") or "regional_carry_forward")
        selected_score = _finite_float(best.get("mean_normalized_absolute_error"))
        blockers: list[str] = []
        if selected_score is None:
            blockers.append("no_scored_candidates_for_metric")
            selected_family = "regional_carry_forward"
        if carry_score is not None and selected_score is not None and not selected_score < carry_score:
            blockers.append("does_not_beat_metric_carry_forward")
            selected_family = "regional_carry_forward"
            selected_score = carry_score
        if selected_family == "regional_carry_forward":
            blockers.append("selected_carry_forward_for_metric")
        status = "module_candidate_promoted" if not blockers else "module_locked_to_carry_forward"
        rows.append(
            {
                "metric_name": metric,
                "status": status,
                "selected_candidate_family": selected_family,
                "selected_mean_normalized_absolute_error": selected_score,
                "regional_carry_forward_mean_normalized_absolute_error": carry_score,
                "aggregate_log_trend_mean_normalized_absolute_error": aggregate_score,
                "similarity_proxy_log_delta_mean_normalized_absolute_error": similarity_score,
                "blockers": blockers,
                "candidate_scores": candidate_rows,
            }
        )
    return rows


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("holdout_period") or ""), str(row.get("region") or ""), str(row.get("candidate_family") or "")): dict(row)
        for row in prediction_rows
        if row.get("holdout_period") and row.get("region") and row.get("candidate_family")
    }


def _module_local_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    module_selector_table: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selector = {
        str(row.get("metric_name") or ""): str(row.get("selected_candidate_family") or "regional_carry_forward")
        for row in module_selector_table
    }
    indexed = _prediction_index(prediction_rows)
    period_regions = sorted({(period, region) for period, region, _family in indexed.keys()})
    selected_rows: list[dict[str, Any]] = []
    for period, region in period_regions:
        row: dict[str, Any] = {
            "candidate_family": "module_local_selector",
            "holdout_period": period,
            "region": region,
        }
        source_families: dict[str, str] = {}
        for metric in COUNT_METRICS:
            family = selector.get(metric, "regional_carry_forward")
            source = indexed.get((period, region, family)) or indexed.get((period, region, "regional_carry_forward"))
            if source is None:
                continue
            row[metric] = _finite_float(source.get(metric)) or 0.0
            source_families[metric] = str(source.get("candidate_family") or family)
            if not row.get("train_previous_period") and source.get("train_previous_period"):
                row["train_previous_period"] = source.get("train_previous_period")
            if not row.get("train_last_period") and source.get("train_last_period"):
                row["train_last_period"] = source.get("train_last_period")
        projected = _projection_cascade(row)
        projection_adjusted = any(
            abs(float(projected.get(metric) or 0.0) - float(row.get(metric) or 0.0)) > 1e-8
            for metric in COUNT_METRICS
        )
        for metric, family in source_families.items():
            projected[f"{metric}_source_candidate_family"] = family
        projected["projection_adjusted"] = projection_adjusted
        selected_rows.append(projected)
    return selected_rows


def _candidate_table(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        family = str(row.get("candidate_family") or "")
        if family:
            grouped[family].append(dict(row))
    rows: list[dict[str, Any]] = []
    for family, family_rows in sorted(grouped.items()):
        rows.append(
            {
                "candidate_family": family,
                "mean_normalized_absolute_error": _mean_score(family_rows),
                "worst_normalized_absolute_error": _worst_score(family_rows),
                "scored_split_metric_count": len(family_rows),
            }
        )
    rows.sort(key=lambda row: float(row.get("mean_normalized_absolute_error") or float("inf")))
    return rows


def _gate(
    *,
    module_selector_table: list[dict[str, Any]],
    combined_candidate_table: list[dict[str, Any]],
) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in combined_candidate_table}
    selector_score = _finite_float((by_family.get("module_local_selector") or {}).get("mean_normalized_absolute_error"))
    carry_score = _finite_float((by_family.get("regional_carry_forward") or {}).get("mean_normalized_absolute_error"))
    aggregate_score = _finite_float((by_family.get("aggregate_log_trend") or {}).get("mean_normalized_absolute_error"))
    r48_best_score = _finite_float((combined_candidate_table[0] if combined_candidate_table else {}).get("mean_normalized_absolute_error"))
    promoted_modules = [
        str(row.get("metric_name") or "")
        for row in module_selector_table
        if str(row.get("status") or "") == "module_candidate_promoted"
    ]
    blockers: list[str] = []
    if not promoted_modules:
        blockers.append("no_metric_promotes_non_carry_forward_candidate")
    if selector_score is None:
        blockers.append("module_selector_not_scored")
    if carry_score is not None and selector_score is not None and not selector_score < carry_score:
        blockers.append("module_selector_does_not_beat_regional_carry_forward")
    if r48_best_score is not None and selector_score is not None and selector_score > r48_best_score:
        blockers.append("module_selector_regresses_vs_best_single_family_r48")
    status = "module_local_subnational_champion_promoted" if not blockers else "module_local_selector_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "promoted_modules": promoted_modules,
        "module_selector_mean_normalized_absolute_error": selector_score,
        "regional_carry_forward_mean_normalized_absolute_error": carry_score,
        "aggregate_log_trend_mean_normalized_absolute_error": aggregate_score,
        "best_single_family_mean_normalized_absolute_error": r48_best_score,
        "contract": (
            "R49 may select different blocked-time candidates for different regional cascade modules, but "
            "combined predictions must be reprojected into the stock cone and must not regress against the "
            "best single-family R48 benchmark. This is a readout/proxy contract, not a causal determinant model."
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


def _flatten_candidate_scores(module_selector_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for selector_row in module_selector_table:
        metric = str(selector_row.get("metric_name") or "")
        for score in list(selector_row.get("candidate_scores") or []):
            rows.append({"metric_name": metric, **dict(score)})
    return rows


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("module_local_gate") or {})
    lines = [
        "# Phase 3 R49 Subnational Module Champion Gate",
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
        f"- Promoted modules: `{', '.join(gate.get('promoted_modules') or []) or 'none'}`",
        f"- Module selector score: `{gate.get('module_selector_mean_normalized_absolute_error')}`",
        f"- Carry-forward score: `{gate.get('regional_carry_forward_mean_normalized_absolute_error')}`",
        f"- Aggregate-trend score: `{gate.get('aggregate_log_trend_mean_normalized_absolute_error')}`",
        f"- Best single-family score: `{gate.get('best_single_family_mean_normalized_absolute_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Module Selection",
        "",
        "| Metric | Status | Selected family | Selected NAE | Carry-forward NAE | Aggregate NAE | Similarity NAE |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in list(report.get("module_selector_table") or []):
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('status')}` | `{row.get('selected_candidate_family')}` | "
            f"{float(row.get('selected_mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('regional_carry_forward_mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('aggregate_log_trend_mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('similarity_proxy_log_delta_mean_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Combined Candidate Table",
            "",
            "| Candidate | Mean normalized AE | Worst normalized AE | Split-metrics |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in list(report.get("combined_candidate_table") or []):
        lines.append(
            f"| `{row.get('candidate_family')}` | {float(row.get('mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('worst_normalized_absolute_error') or 0.0):.6f} | {row.get('scored_split_metric_count')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(
    path: Path,
    *,
    module_selector_table: list[dict[str, Any]],
    combined_candidate_table: list[dict[str, Any]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    metrics = [str(row.get("metric_name") or "") for row in module_selector_table]
    x = np.arange(len(metrics))
    width = 0.25
    carry = [float(row.get("regional_carry_forward_mean_normalized_absolute_error") or 0.0) for row in module_selector_table]
    aggregate = [float(row.get("aggregate_log_trend_mean_normalized_absolute_error") or 0.0) for row in module_selector_table]
    similarity = [float(row.get("similarity_proxy_log_delta_mean_normalized_absolute_error") or 0.0) for row in module_selector_table]
    axes[0].bar(x - width, carry, width, label="carry-forward", color="#727272")
    axes[0].bar(x, aggregate, width, label="aggregate trend", color="#496f9e")
    axes[0].bar(x + width, similarity, width, label="similarity proxy", color="#8a6f2a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=25, ha="right")
    axes[0].set_ylabel("mean normalized absolute error")
    axes[0].set_title("Module-local blocked regional scores")
    axes[0].legend(frameon=False)
    labels = [str(row.get("candidate_family") or "") for row in combined_candidate_table]
    scores = [float(row.get("mean_normalized_absolute_error") or 0.0) for row in combined_candidate_table]
    axes[1].barh(np.arange(len(labels)), scores, color="#516b5f")
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean normalized absolute error")
    axes[1].set_title("Combined stock-cone-safe selector")
    fig.suptitle("R49 subnational module champion gate", fontsize=14, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r49_subnational_module_champion_gate(
    *,
    run_id: str = R49_RUN_ID,
    r48_report_path: Path | None = None,
) -> dict[str, Any]:
    r48_path = Path(r48_report_path) if r48_report_path is not None else R48_DEFAULT_REPORT
    r48_report = _load_r48_report(r48_path)
    r44_path = Path(str(r48_report.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    base_split_scores = [dict(row) for row in list(r48_report.get("split_metric_score_rows") or [])]
    base_prediction_rows = [dict(row) for row in list(r48_report.get("prediction_rows") or [])]
    module_selector_table = _module_selector_table(base_split_scores)
    module_prediction_rows = _module_local_prediction_rows(base_prediction_rows, module_selector_table)
    module_score_rows, module_split_scores = _score_predictions(rows_by_key, module_prediction_rows)
    combined_split_scores = base_split_scores + module_split_scores
    combined_candidate_table = _candidate_table(combined_split_scores)
    gate = _gate(module_selector_table=module_selector_table, combined_candidate_table=combined_candidate_table)
    verdict = (
        "R49 promoted a module-local subnational selector that beats carry-forward without regressing against R48."
        if gate["status"] == "module_local_subnational_champion_promoted"
        else "R49 is diagnostic-only: module-local selection did not clear the non-regression gate."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r49_subnational_module_champion_gate_report.json"
    md_path = analysis_dir / "r49_subnational_module_champion_gate_report.md"
    module_selector_csv = analysis_dir / "r49_module_selector_table.csv"
    module_candidate_score_csv = analysis_dir / "r49_module_candidate_scores.csv"
    combined_candidate_csv = analysis_dir / "r49_combined_candidate_table.csv"
    prediction_csv = analysis_dir / "r49_module_prediction_rows.csv"
    score_csv = analysis_dir / "r49_module_score_rows.csv"
    split_score_csv = analysis_dir / "r49_module_split_metric_scores.csv"
    dashboard_path = analysis_dir / "r49_subnational_module_dashboard.png"
    report = {
        "schema_version": R49_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r48_report_path": r48_path.as_posix(),
        "r48_status": r48_report.get("status"),
        "r44_report_path": r44_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "selector": "select the lowest blocked-time mean normalized absolute error candidate independently per metric, then reproject the combined row into the cascade stock cone",
            "free_region_parameters": 0,
            "claim_limit": "subnational readout/proxy champion only; determinant and causal process claims remain gated separately",
        },
        "module_local_gate": gate,
        "module_selector_table": module_selector_table,
        "module_candidate_score_rows": _flatten_candidate_scores(module_selector_table),
        "combined_candidate_table": combined_candidate_table,
        "module_prediction_rows": module_prediction_rows,
        "module_score_rows": module_score_rows,
        "module_split_metric_score_rows": module_split_scores,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "module_selector_csv": module_selector_csv.as_posix(),
            "module_candidate_scores_csv": module_candidate_score_csv.as_posix(),
            "combined_candidate_csv": combined_candidate_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "score_csv": score_csv.as_posix(),
            "split_metric_scores_csv": split_score_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(module_selector_csv, module_selector_table)
    _write_csv(module_candidate_score_csv, report["module_candidate_score_rows"])
    _write_csv(combined_candidate_csv, combined_candidate_table)
    _write_csv(prediction_csv, module_prediction_rows)
    _write_csv(score_csv, module_score_rows)
    _write_csv(split_score_csv, module_split_scores)
    _write_markdown(md_path, report)
    _write_dashboard(
        dashboard_path,
        module_selector_table=module_selector_table,
        combined_candidate_table=combined_candidate_table,
    )
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R49 subnational module champion gate.")
    parser.add_argument("--run-id", default=R49_RUN_ID)
    parser.add_argument("--r48-report-path", default=None)
    args = parser.parse_args()
    run_r49_subnational_module_champion_gate(
        run_id=str(args.run_id),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
    )


if __name__ == "__main__":
    _main()
