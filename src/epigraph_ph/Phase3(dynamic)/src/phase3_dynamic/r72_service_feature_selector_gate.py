from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    _carry_forward_prediction,
    _finite_float,
    _generated_at,
    _r41_monotone_growth_component_predictions,
    _sha256,
    project_cascade_stock_row,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import (
    R71_METRICS,
    _compile_feature_rows,
    _default_evidence_root,
    _feature_predictions,
    _score_prediction_family,
    _summary_rows,
)
from .runtime import ensure_dir, read_json, write_json


R72_SCHEMA_VERSION = "phase3_dynamic.r72_service_feature_selector_gate.v1"
R72_RUN_ID = "p3d-r72-service-feature-selector-gate-20260506-s00"
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


def _family_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
) -> dict[str, list[dict[str, Any]]]:
    r41, _summary = _r41_monotone_growth_component_predictions(train_rows, holdout_rows)
    r71_safe, _feature_summary = _feature_predictions(
        train_rows,
        holdout_rows,
        feature_rows,
        policy="forecast_safe",
        family="r71_forecast_safe_service_intensity_capacity",
    )
    return {
        "carry_forward": _carry_forward_prediction(train_rows, holdout_rows),
        "r41_research_champion_reference": r41,
        "r71_forecast_safe_service_intensity_capacity": r71_safe,
    }


def _metric_mean_errors(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    predictions_by_family: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, str], float]:
    errors: dict[tuple[str, str], list[float]] = {}
    for family, predictions in predictions_by_family.items():
        for row in _score_prediction_family(holdout_rows, predictions, family=family, train_rows=train_rows):
            errors.setdefault((family, str(row.get("metric_name") or "")), []).append(float(row["normalized_absolute_error"]))
    return {
        key: float(np.mean(np.asarray(values, dtype=np.float64)))
        for key, values in errors.items()
        if values
    }


def _inner_split_years(train_rows: list[dict[str, Any]], *, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    return [year for year in years if len([prior for prior in years if prior < year]) >= int(min_train_years)]


def _select_metric_families(
    train_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    *,
    min_train_years: int,
) -> dict[str, str]:
    candidate_families = (
        "r41_research_champion_reference",
        "carry_forward",
        "r71_forecast_safe_service_intensity_capacity",
    )
    metric_errors: dict[tuple[str, str], list[float]] = {}
    for inner_year in _inner_split_years(train_rows, min_train_years=min_train_years):
        inner_train = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) < inner_year]
        inner_holdout = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) == inner_year]
        if not inner_train or not inner_holdout:
            continue
        predictions = _family_predictions(inner_train, inner_holdout, feature_rows)
        for (family, metric), error in _metric_mean_errors(inner_train, inner_holdout, predictions).items():
            metric_errors.setdefault((family, metric), []).append(float(error))
    selected: dict[str, str] = {}
    for metric in R71_METRICS:
        scored = []
        for family in candidate_families:
            values = metric_errors.get((family, metric), [])
            if values:
                scored.append((family, float(np.mean(np.asarray(values, dtype=np.float64)))))
        if not scored:
            selected[metric] = "r41_research_champion_reference"
            continue
        scored.sort(key=lambda item: (item[1], candidate_families.index(item[0])))
        selected[metric] = scored[0][0]
    return selected


def _selector_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    selected: dict[str, str],
) -> list[dict[str, Any]]:
    by_family = {
        family: {str(row.get("quarter") or ""): dict(row) for row in predictions}
        for family, predictions in _family_predictions(train_rows, holdout_rows, feature_rows).items()
    }
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        prediction = {"quarter": quarter}
        for metric in R71_METRICS:
            family = selected.get(metric, "r41_research_champion_reference")
            value = _finite_float((by_family.get(family) or {}).get(quarter, {}).get(metric))
            if value is None:
                value = _finite_float((by_family.get("r41_research_champion_reference") or {}).get(quarter, {}).get(metric))
            prediction[metric] = value
        projected = project_cascade_stock_row(prediction)
        for metric_name, value in dict(projected.get("projected") or {}).items():
            prediction[metric_name] = value
        prediction["cascade_projection_changed"] = bool(projected.get("changed"))
        rows.append(prediction)
    return rows


def _split_years(rows: list[dict[str, Any]], *, start_year: int, end_year: int, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in rows if row.get("quarter")})
    return [
        year
        for year in years
        if int(start_year) <= year <= int(end_year)
        and len([prior for prior in years if prior < year]) >= int(min_train_years)
    ]


def _gate(summary_rows: list[dict[str, Any]], selector_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {str(row.get("family")): row for row in summary_rows if row.get("metric_name") == "__overall__"}
    selector = _finite_float((overall.get("r72_train_backtested_service_feature_selector") or {}).get("mean_normalized_absolute_error"))
    carry = _finite_float((overall.get("carry_forward") or {}).get("mean_normalized_absolute_error"))
    r41 = _finite_float((overall.get("r41_research_champion_reference") or {}).get("mean_normalized_absolute_error"))
    selected_feature_count = sum(
        1
        for row in selector_rows
        for metric, family in dict(row.get("selected_metric_families") or {}).items()
        if family == "r71_forecast_safe_service_intensity_capacity"
    )
    strict = selector is not None and carry is not None and r41 is not None and selector < carry and selector < r41
    nonregress = selector is not None and r41 is not None and selector <= r41
    return {
        "status": "r72_selector_promoted" if strict else ("r72_selector_nonregression_reference" if nonregress else "r72_diagnostic_only"),
        "selector_mean_nae": selector,
        "carry_forward_mean_nae": carry,
        "r41_reference_mean_nae": r41,
        "selector_beats_carry_forward": None if selector is None or carry is None else bool(selector < carry),
        "selector_beats_r41": None if selector is None or r41 is None else bool(selector < r41),
        "selected_feature_metric_count": selected_feature_count,
        "contract": (
            "R72 can only promote if train-origin selection beats both carry-forward and R41. Feature branch choices are "
            "allowed only when they win inside training history; otherwise metrics fall back to R41/carry-forward."
        ),
    }


def run_r72_service_feature_selector_gate(
    *,
    run_id: str = R72_RUN_ID,
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
    selector_rows: list[dict[str, Any]] = []
    for holdout_year in _split_years(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years):
        train_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) < holdout_year]
        holdout_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) == holdout_year]
        selected = _select_metric_families(train_rows, feature_rows, min_train_years=min_train_years)
        predictions = _family_predictions(train_rows, holdout_rows, feature_rows)
        predictions["r72_train_backtested_service_feature_selector"] = _selector_predictions(train_rows, holdout_rows, feature_rows, selected)
        for family, rows in predictions.items():
            for score in _score_prediction_family(holdout_rows, rows, family=family, train_rows=train_rows):
                score["holdout_year"] = holdout_year
                all_score_rows.append(score)
        selector_rows.append(
            {
                "holdout_year": holdout_year,
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
                "selected_metric_families": selected,
                "selected_feature_metric_count": sum(1 for family in selected.values() if family == "r71_forecast_safe_service_intensity_capacity"),
            }
        )
    summary = _summary_rows(all_score_rows)
    gate = _gate(summary, selector_rows)
    report_path = analysis_dir / "r72_service_feature_selector_gate_report.json"
    markdown_path = analysis_dir / "r72_service_feature_selector_gate_report.md"
    score_csv = analysis_dir / "r72_score_rows.csv"
    summary_csv = analysis_dir / "r72_summary_rows.csv"
    selector_csv = analysis_dir / "r72_selector_rows.csv"
    report = {
        "schema_version": R72_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "selector_gate": gate,
        "score_rows": all_score_rows,
        "summary_rows": summary,
        "selector_rows": selector_rows,
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
            "selector_rows_csv": selector_csv.as_posix(),
        },
    }
    _write_csv(score_csv, all_score_rows)
    _write_csv(summary_csv, summary)
    _write_csv(selector_csv, selector_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("selector_gate") or {})
    lines = [
        "# Phase 3 R72 Service Feature Selector Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Selector mean NAE: `{gate.get('selector_mean_nae')}`",
        f"- Carry-forward mean NAE: `{gate.get('carry_forward_mean_nae')}`",
        f"- R41 reference mean NAE: `{gate.get('r41_reference_mean_nae')}`",
        f"- Selected feature metric count: `{gate.get('selected_feature_metric_count')}`",
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
    parser = argparse.ArgumentParser(description="Run R72 train-backtested service feature selector gate.")
    parser.add_argument("--run-id", default=R72_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    run_r72_service_feature_selector_gate(
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
