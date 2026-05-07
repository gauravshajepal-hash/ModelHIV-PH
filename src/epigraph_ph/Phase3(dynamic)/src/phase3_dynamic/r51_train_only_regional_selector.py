from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
from .r11_sparse_state_space import _generated_at
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
    _score_predictions,
)
from .r49_subnational_module_champion_gate import _candidate_table
from .runtime import ensure_dir, read_json, write_json


R51_SCHEMA_VERSION = "phase3_dynamic.r51_train_only_regional_selector.v1"
R51_RUN_ID = "p3d-r51-train-only-regional-selector-20260503-s00"
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


def _family_score_from_rows(rows: list[dict[str, Any]]) -> float | None:
    error_sum = 0.0
    target_sum = 0.0
    for row in rows:
        error = _finite_float(row.get("absolute_error"))
        target = _finite_float(row.get("target_abs"))
        if error is None or target is None:
            continue
        error_sum += float(error)
        target_sum += abs(float(target))
    if target_sum <= 0.0:
        return None
    return float(error_sum / target_sum)


def _select_family_from_prior_scores(
    score_rows: list[dict[str, Any]],
    *,
    region: str,
    metric: str,
    holdout_period: str,
    default_family: str = "aggregate_log_trend",
) -> tuple[str, dict[str, Any]]:
    prior_rows = [
        dict(row)
        for row in score_rows
        if str(row.get("region") or "") == region
        and str(row.get("metric_name") or "") == metric
        and quarter_sort_key(str(row.get("holdout_period") or "")) < quarter_sort_key(holdout_period)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in prior_rows:
        family = str(row.get("candidate_family") or "")
        if family:
            grouped[family].append(row)
    candidate_scores: list[dict[str, Any]] = []
    for family, rows in sorted(grouped.items()):
        score = _family_score_from_rows(rows)
        if score is not None:
            candidate_scores.append(
                {
                    "candidate_family": family,
                    "prior_normalized_absolute_error": score,
                    "prior_score_row_count": len(rows),
                }
            )
    candidate_scores.sort(key=lambda row: float(row.get("prior_normalized_absolute_error") or float("inf")))
    if not candidate_scores:
        return default_family, {
            "selection_status": "default_no_prior_region_metric_scores",
            "candidate_scores": [],
            "default_family": default_family,
        }
    selected = str(candidate_scores[0].get("candidate_family") or default_family)
    return selected, {
        "selection_status": "selected_from_prior_region_metric_scores",
        "candidate_scores": candidate_scores,
        "default_family": default_family,
    }


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(row.get("holdout_period") or ""), str(row.get("region") or ""), str(row.get("candidate_family") or "")): dict(row)
        for row in prediction_rows
        if row.get("holdout_period") and row.get("region") and row.get("candidate_family")
    }


def _train_only_selector_prediction_rows(
    *,
    prediction_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    default_family: str = "aggregate_log_trend",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = _prediction_index(prediction_rows)
    period_regions = sorted({(period, region) for period, region, _family in indexed.keys()}, key=lambda item: (quarter_sort_key(item[0]), item[1]))
    selected_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for period, region in period_regions:
        row: dict[str, Any] = {
            "candidate_family": "train_only_region_metric_selector",
            "holdout_period": period,
            "region": region,
        }
        for metric in COUNT_METRICS:
            family, decision = _select_family_from_prior_scores(
                score_rows,
                region=region,
                metric=metric,
                holdout_period=period,
                default_family=default_family,
            )
            source = indexed.get((period, region, family)) or indexed.get((period, region, default_family))
            if source is None:
                continue
            row[metric] = _finite_float(source.get(metric)) or 0.0
            row[f"{metric}_source_candidate_family"] = str(source.get("candidate_family") or family)
            decision_rows.append(
                {
                    "holdout_period": period,
                    "region": region,
                    "metric_name": metric,
                    "selected_candidate_family": str(source.get("candidate_family") or family),
                    "requested_candidate_family": family,
                    "selection_status": decision.get("selection_status"),
                    "prior_candidate_score_count": len(list(decision.get("candidate_scores") or [])),
                    "prior_candidate_scores": decision.get("candidate_scores"),
                }
            )
            if not row.get("train_previous_period") and source.get("train_previous_period"):
                row["train_previous_period"] = source.get("train_previous_period")
            if not row.get("train_last_period") and source.get("train_last_period"):
                row["train_last_period"] = source.get("train_last_period")
        projected = _projection_cascade(row)
        projected["projection_adjusted"] = any(
            abs(float(projected.get(metric) or 0.0) - float(row.get(metric) or 0.0)) > 1e-8
            for metric in COUNT_METRICS
        )
        selected_rows.append(projected)
    return selected_rows, decision_rows


def _gate(candidate_table: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in candidate_table}
    selector_score = _finite_float((by_family.get("train_only_region_metric_selector") or {}).get("mean_normalized_absolute_error"))
    carry_score = _finite_float((by_family.get("regional_carry_forward") or {}).get("mean_normalized_absolute_error"))
    aggregate_score = _finite_float((by_family.get("aggregate_log_trend") or {}).get("mean_normalized_absolute_error"))
    similarity_score = _finite_float((by_family.get("similarity_proxy_log_delta") or {}).get("mean_normalized_absolute_error"))
    non_default_decisions = [
        row for row in decision_rows if str(row.get("selection_status") or "") == "selected_from_prior_region_metric_scores"
    ]
    blockers: list[str] = []
    if selector_score is None:
        blockers.append("selector_not_scored")
    if carry_score is not None and selector_score is not None and not selector_score < carry_score:
        blockers.append("selector_does_not_beat_regional_carry_forward")
    if aggregate_score is not None and selector_score is not None and selector_score > aggregate_score:
        blockers.append("selector_regresses_vs_aggregate_trend")
    if not non_default_decisions:
        blockers.append("no_train_only_region_metric_decisions_available")
    status = "train_only_regional_selector_promoted" if not blockers else "train_only_regional_selector_diagnostic_only"
    return {
        "status": status,
        "blockers": blockers,
        "train_only_decision_count": len(non_default_decisions),
        "selector_mean_normalized_absolute_error": selector_score,
        "regional_carry_forward_mean_normalized_absolute_error": carry_score,
        "aggregate_log_trend_mean_normalized_absolute_error": aggregate_score,
        "similarity_proxy_log_delta_mean_normalized_absolute_error": similarity_score,
        "contract": (
            "R51 can choose region/metric candidate families only from strictly earlier blocked residuals. "
            "It defaults to aggregate trend where no prior regional score exists and must beat carry-forward "
            "without regressing against aggregate trend."
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
    gate = dict(report.get("train_only_selector_gate") or {})
    lines = [
        "# Phase 3 R51 Train-Only Regional Selector",
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
        f"- Selector score: `{gate.get('selector_mean_normalized_absolute_error')}`",
        f"- Carry-forward score: `{gate.get('regional_carry_forward_mean_normalized_absolute_error')}`",
        f"- Aggregate-trend score: `{gate.get('aggregate_log_trend_mean_normalized_absolute_error')}`",
        f"- Similarity-proxy score: `{gate.get('similarity_proxy_log_delta_mean_normalized_absolute_error')}`",
        f"- Train-only decisions: `{gate.get('train_only_decision_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Candidate Table",
        "",
        "| Candidate | Mean normalized AE | Worst normalized AE | Split-metrics |",
        "|---|---:|---:|---:|",
    ]
    for row in list(report.get("candidate_table") or []):
        lines.append(
            f"| `{row.get('candidate_family')}` | {float(row.get('mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('worst_normalized_absolute_error') or 0.0):.6f} | {row.get('scored_split_metric_count')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, candidate_table: list[dict[str, Any]], decision_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    labels = [str(row.get("candidate_family") or "") for row in candidate_table]
    scores = [float(row.get("mean_normalized_absolute_error") or 0.0) for row in candidate_table]
    axes[0].barh(np.arange(len(labels)), scores, color="#516b5f")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("mean normalized absolute error")
    axes[0].set_title("Train-only regional selector gate")
    selected_counts: dict[str, int] = defaultdict(int)
    for row in decision_rows:
        selected_counts[str(row.get("selected_candidate_family") or "")] += 1
    count_labels = sorted(selected_counts)
    counts = [selected_counts[label] for label in count_labels]
    axes[1].barh(np.arange(len(count_labels)), counts, color="#8a6f2a")
    axes[1].set_yticks(np.arange(len(count_labels)))
    axes[1].set_yticklabels(count_labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("region-metric decisions")
    axes[1].set_title("Selected source families")
    fig.suptitle("R51 train-only regional selector", fontsize=14, fontweight="bold")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r51_train_only_regional_selector(
    *,
    run_id: str = R51_RUN_ID,
    r48_report_path: Path | None = None,
    default_family: str = "aggregate_log_trend",
) -> dict[str, Any]:
    r48_path = Path(r48_report_path) if r48_report_path is not None else R48_DEFAULT_REPORT
    r48_report = _load_r48_report(r48_path)
    r44_path = Path(str(r48_report.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    base_prediction_rows = [dict(row) for row in list(r48_report.get("prediction_rows") or [])]
    base_score_rows = [dict(row) for row in list(r48_report.get("score_rows") or [])]
    selector_rows, decision_rows = _train_only_selector_prediction_rows(
        prediction_rows=base_prediction_rows,
        score_rows=base_score_rows,
        default_family=default_family,
    )
    selector_score_rows, selector_split_scores = _score_predictions(rows_by_key, selector_rows)
    combined_split_scores = [dict(row) for row in list(r48_report.get("split_metric_score_rows") or [])] + selector_split_scores
    candidate_table = _candidate_table(combined_split_scores)
    gate = _gate(candidate_table, decision_rows)
    verdict = (
        "R51 promoted a strict train-only regional selector."
        if gate["status"] == "train_only_regional_selector_promoted"
        else "R51 did not promote the train-only regional selector; proxy effects remain diagnostic until more historical regional periods are onboarded."
    )
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r51_train_only_regional_selector_report.json"
    md_path = analysis_dir / "r51_train_only_regional_selector_report.md"
    candidate_csv = analysis_dir / "r51_candidate_table.csv"
    prediction_csv = analysis_dir / "r51_prediction_rows.csv"
    decision_csv = analysis_dir / "r51_decision_rows.csv"
    score_csv = analysis_dir / "r51_score_rows.csv"
    split_score_csv = analysis_dir / "r51_split_metric_scores.csv"
    dashboard_path = analysis_dir / "r51_train_only_regional_selector_dashboard.png"
    report = {
        "schema_version": R51_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "r48_report_path": r48_path.as_posix(),
        "r44_report_path": r44_path.as_posix(),
        "metric_contract": {
            "count_metrics": list(COUNT_METRICS),
            "default_family_without_prior_region_metric_scores": default_family,
            "selection_rule": "for each region, metric, and holdout period choose the lowest-error family using strictly earlier blocked residuals only",
            "free_region_parameters": 0,
            "claim_limit": "selector/readout claim only; not a determinant or mechanistic subnational process claim",
        },
        "train_only_selector_gate": gate,
        "candidate_table": candidate_table,
        "prediction_rows": selector_rows,
        "decision_rows": decision_rows,
        "score_rows": selector_score_rows,
        "split_metric_score_rows": selector_split_scores,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "candidate_csv": candidate_csv.as_posix(),
            "prediction_csv": prediction_csv.as_posix(),
            "decision_csv": decision_csv.as_posix(),
            "score_csv": score_csv.as_posix(),
            "split_metric_scores_csv": split_score_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(candidate_csv, candidate_table)
    _write_csv(prediction_csv, selector_rows)
    _write_csv(decision_csv, decision_rows)
    _write_csv(score_csv, selector_score_rows)
    _write_csv(split_score_csv, selector_split_scores)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, candidate_table, decision_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R51 train-only regional selector.")
    parser.add_argument("--run-id", default=R51_RUN_ID)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--default-family", default="aggregate_log_trend")
    args = parser.parse_args()
    run_r51_train_only_regional_selector(
        run_id=str(args.run_id),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        default_family=str(args.default_family),
    )


if __name__ == "__main__":
    _main()
