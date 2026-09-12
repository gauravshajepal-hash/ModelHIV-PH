from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import R71_METRICS, _compile_feature_rows, _default_evidence_root, _score_prediction_family, _summary_rows
from .r72_service_feature_selector_gate import _family_predictions
from .r73_external_signal_lag_falsification import _select_feature_models, _selector_predictions
from .runtime import ensure_dir, read_json, write_json


R74_SCHEMA_VERSION = "phase3_dynamic.r74_external_tail_risk_selector.v1"
R74_RUN_ID = "p3d-r74-external-tail-risk-selector-20260506-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_summary(score_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = {}
    for row in score_rows:
        value = _finite_float(row.get("normalized_absolute_error"))
        metric = str(row.get("metric_name") or "")
        if value is None or not metric:
            continue
        grouped.setdefault(metric, []).append(float(value))
    return {
        metric: {
            "mean_nae": float(np.mean(np.asarray(values, dtype=np.float64))),
            "p90_nae": float(np.quantile(np.asarray(values, dtype=np.float64), 0.9)),
        }
        for metric, values in grouped.items()
        if values
    }


def _inner_split_years(train_rows: list[dict[str, Any]], *, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    return [year for year in years if len([prior for prior in years if prior < year]) >= int(min_train_years)]


def _tail_risk_metric_policy(
    train_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    *,
    min_train_years: int,
) -> tuple[dict[str, bool], list[dict[str, Any]]]:
    r41_scores_by_metric: dict[str, list[float]] = {}
    r41_p90_pool: dict[str, list[float]] = {}
    r73_scores_by_metric: dict[str, list[float]] = {}
    r73_p90_pool: dict[str, list[float]] = {}
    for inner_year in _inner_split_years(train_rows, min_train_years=min_train_years):
        inner_train = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) < inner_year]
        inner_holdout = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) == inner_year]
        if not inner_train or not inner_holdout:
            continue
        r41_predictions = _family_predictions(inner_train, inner_holdout, feature_rows)["r41_research_champion_reference"]
        r41_scores = _score_prediction_family(inner_holdout, r41_predictions, family="r41_research_champion_reference", train_rows=inner_train)
        selected, _selection_rows = _select_feature_models(
            inner_train,
            feature_rows,
            policy="forecast_origin",
            min_train_years=min_train_years,
        )
        r73_predictions = _selector_predictions(inner_train, inner_holdout, feature_rows, selected, policy="forecast_origin")
        r73_scores = _score_prediction_family(inner_holdout, r73_predictions, family="r73_forecast_origin_external_signal_selector", train_rows=inner_train)
        for row in r41_scores:
            metric = str(row.get("metric_name") or "")
            value = _finite_float(row.get("normalized_absolute_error"))
            if metric and value is not None:
                r41_scores_by_metric.setdefault(metric, []).append(float(value))
                r41_p90_pool.setdefault(metric, []).append(float(value))
        for row in r73_scores:
            metric = str(row.get("metric_name") or "")
            value = _finite_float(row.get("normalized_absolute_error"))
            if metric and value is not None:
                r73_scores_by_metric.setdefault(metric, []).append(float(value))
                r73_p90_pool.setdefault(metric, []).append(float(value))
    policy: dict[str, bool] = {}
    policy_rows: list[dict[str, Any]] = []
    for metric in R71_METRICS:
        r41_values = r41_scores_by_metric.get(metric, [])
        r73_values = r73_scores_by_metric.get(metric, [])
        if not r41_values or not r73_values:
            policy[metric] = False
            policy_rows.append({"metric_name": metric, "use_r73": False, "reason": "insufficient_inner_scores"})
            continue
        r41_mean = float(np.mean(np.asarray(r41_values, dtype=np.float64)))
        r73_mean = float(np.mean(np.asarray(r73_values, dtype=np.float64)))
        r41_p90 = float(np.quantile(np.asarray(r41_p90_pool[metric], dtype=np.float64), 0.9))
        r73_p90 = float(np.quantile(np.asarray(r73_p90_pool[metric], dtype=np.float64), 0.9))
        use_r73 = bool(r73_mean <= r41_mean and r73_p90 < r41_p90)
        policy[metric] = use_r73
        policy_rows.append(
            {
                "metric_name": metric,
                "use_r73": use_r73,
                "r41_inner_mean_nae": r41_mean,
                "r73_inner_mean_nae": r73_mean,
                "r41_inner_p90_nae": r41_p90,
                "r73_inner_p90_nae": r73_p90,
                "contract": "use R73 only if inner-history mean does not regress and p90 improves versus R41",
            }
        )
    return policy, policy_rows


def _tail_selector_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    metric_policy: dict[str, bool],
    *,
    min_train_years: int,
) -> list[dict[str, Any]]:
    selected, _selection_rows = _select_feature_models(
        train_rows,
        feature_rows,
        policy="forecast_origin",
        min_train_years=min_train_years,
    )
    guarded: dict[str, dict[str, Any]] = {}
    for metric in R71_METRICS:
        if metric_policy.get(metric):
            guarded[metric] = selected.get(metric, {"selection": "base_family", "base_family": "r41_research_champion_reference"})
        else:
            guarded[metric] = {"selection": "base_family", "base_family": "r41_research_champion_reference", "policy": "forecast_origin"}
    return _selector_predictions(train_rows, holdout_rows, feature_rows, guarded, policy="forecast_origin")


def _split_years(rows: list[dict[str, Any]], *, start_year: int, end_year: int, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in rows if row.get("quarter")})
    return [
        year
        for year in years
        if int(start_year) <= year <= int(end_year)
        and len([prior for prior in years if prior < year]) >= int(min_train_years)
    ]


def _gate(summary_rows: list[dict[str, Any]], policy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {str(row.get("family")): row for row in summary_rows if row.get("metric_name") == "__overall__"}
    selector = _finite_float((overall.get("r74_p90_safe_external_tail_selector") or {}).get("mean_normalized_absolute_error"))
    selector_p90 = _finite_float((overall.get("r74_p90_safe_external_tail_selector") or {}).get("p90_normalized_absolute_error"))
    carry = _finite_float((overall.get("carry_forward") or {}).get("mean_normalized_absolute_error"))
    r41 = _finite_float((overall.get("r41_research_champion_reference") or {}).get("mean_normalized_absolute_error"))
    r41_p90 = _finite_float((overall.get("r41_research_champion_reference") or {}).get("p90_normalized_absolute_error"))
    selected_metric_count = sum(1 for row in policy_rows if row.get("use_r73") is True)
    strict = (
        selector is not None
        and selector_p90 is not None
        and carry is not None
        and r41 is not None
        and r41_p90 is not None
        and selector <= r41
        and selector < carry
        and selector_p90 < r41_p90
    )
    return {
        "status": "r74_tail_risk_selector_promoted" if strict else "r74_diagnostic_only",
        "selector_mean_nae": selector,
        "selector_p90_nae": selector_p90,
        "carry_forward_mean_nae": carry,
        "r41_reference_mean_nae": r41,
        "r41_reference_p90_nae": r41_p90,
        "selector_beats_carry_forward": None if selector is None or carry is None else bool(selector < carry),
        "selector_mean_nonregression_vs_r41": None if selector is None or r41 is None else bool(selector <= r41),
        "selector_p90_beats_r41": None if selector_p90 is None or r41_p90 is None else bool(selector_p90 < r41_p90),
        "selected_metric_count": selected_metric_count,
        "contract": "R74 can only use R73 external features for metric streams whose inner-history p90 improves without mean regression versus R41.",
    }


def run_r74_external_tail_risk_selector(
    *,
    run_id: str = R74_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    start_year: int = 2019,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    root = Path(epigraph_root) if epigraph_root is not None else _default_evidence_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(root, source_run_id=source_run_id, preferred=baseline_source_run_id)
    observation_rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    feature_rows = _compile_feature_rows(r69)
    all_score_rows: list[dict[str, Any]] = []
    all_policy_rows: list[dict[str, Any]] = []
    for holdout_year in _split_years(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years):
        train_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) < holdout_year]
        holdout_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) == holdout_year]
        base_predictions = _family_predictions(train_rows, holdout_rows, feature_rows)
        for family, rows in base_predictions.items():
            for score in _score_prediction_family(holdout_rows, rows, family=family, train_rows=train_rows):
                score["holdout_year"] = holdout_year
                all_score_rows.append(score)
        metric_policy, policy_rows = _tail_risk_metric_policy(train_rows, feature_rows, min_train_years=min_train_years)
        predictions = _tail_selector_predictions(
            train_rows,
            holdout_rows,
            feature_rows,
            metric_policy,
            min_train_years=min_train_years,
        )
        for score in _score_prediction_family(holdout_rows, predictions, family="r74_p90_safe_external_tail_selector", train_rows=train_rows):
            score["holdout_year"] = holdout_year
            all_score_rows.append(score)
        for row in policy_rows:
            all_policy_rows.append({"holdout_year": holdout_year, "train_row_count": len(train_rows), "holdout_row_count": len(holdout_rows), **row})
    summary = _summary_rows(all_score_rows)
    gate = _gate(summary, all_policy_rows)
    report_path = analysis_dir / "r74_external_tail_risk_selector_report.json"
    markdown_path = analysis_dir / "r74_external_tail_risk_selector_report.md"
    score_csv = analysis_dir / "r74_score_rows.csv"
    summary_csv = analysis_dir / "r74_summary_rows.csv"
    policy_csv = analysis_dir / "r74_policy_rows.csv"
    report = {
        "schema_version": R74_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "tail_risk_gate": gate,
        "score_rows": all_score_rows,
        "summary_rows": summary,
        "policy_rows": all_policy_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "epigraph_root": root.as_posix(),
            "source_run_id": source_run_id,
            "baseline_source_run_id": baseline_source_run_id,
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "score_rows_csv": score_csv.as_posix(),
            "summary_rows_csv": summary_csv.as_posix(),
            "policy_rows_csv": policy_csv.as_posix(),
        },
    }
    _write_csv(score_csv, all_score_rows)
    _write_csv(summary_csv, summary)
    _write_csv(policy_csv, all_policy_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("tail_risk_gate") or {})
    lines = [
        "# Phase 3 R74 External Tail-Risk Selector",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Selector mean NAE: `{gate.get('selector_mean_nae')}`",
        f"- Selector p90 NAE: `{gate.get('selector_p90_nae')}`",
        f"- R41 mean NAE: `{gate.get('r41_reference_mean_nae')}`",
        f"- R41 p90 NAE: `{gate.get('r41_reference_p90_nae')}`",
        f"- Selected metric count: `{gate.get('selected_metric_count')}`",
        "",
        "## Overall Scores",
        "",
        "| Family | Mean NAE | p90 NAE |",
        "|---|---:|---:|",
    ]
    for row in report.get("summary_rows") or []:
        if row.get("metric_name") != "__overall__":
            continue
        lines.append(
            f"| `{row.get('family')}` | {float(row.get('mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('p90_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R74 external tail-risk selector.")
    parser.add_argument("--run-id", default=R74_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    run_r74_external_tail_risk_selector(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_train_years=int(args.min_train_years),
    )


if __name__ == "__main__":
    _main()
