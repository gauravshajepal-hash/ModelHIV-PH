from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_ordinal, quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    R11_EVALUATION_METRICS,
    _carry_forward_prediction,
    _finite_float,
    _generated_at,
    _r41_monotone_growth_component_predictions,
    _sha256,
    project_cascade_stock_row,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R71_SCHEMA_VERSION = "phase3_dynamic.r71_service_intensity_capacity_branch.v1"
R71_RUN_ID = "p3d-r71-service-intensity-capacity-branch-20260506-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
R71_METRICS: tuple[str, ...] = tuple(dict.fromkeys(R11_EVALUATION_METRICS + ("tested_for_viral_load", "virally_suppressed")))


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _default_evidence_root() -> Path:
    candidates = [
        Path("/media/gaurav/New_Volume/EpiGraph_PH"),
        sandbox_repo_root().parents[1],
        sandbox_repo_root().parent,
    ]
    for candidate in candidates:
        if (candidate / "artifacts" / "runs").exists():
            try:
                resolve_active_source_run_id(candidate, None)
                return candidate
            except FileNotFoundError:
                continue
    return candidates[0]


def _metric_module(metric_name: str) -> str:
    if metric_name in {"new_diagnosed_cases_period", "diagnosed_plhiv"}:
        return "diagnosis_reporting"
    if metric_name == "alive_on_art":
        return "art_retention"
    if metric_name in {"tested_for_viral_load", "virally_suppressed"}:
        return "vl_suppression_service"
    return "other"


def _r69_paths(r69_report: dict[str, Any]) -> dict[str, Path]:
    paths = dict(r69_report.get("artifact_paths") or {})
    return {key: Path(str(value)) for key, value in paths.items() if value}


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compile_feature_rows(r69_report: dict[str, Any]) -> dict[str, dict[str, float]]:
    paths = _r69_paths(r69_report)
    by_quarter: dict[str, dict[str, float]] = {}
    mobility_path = paths.get("google_mobility_quarterly_csv")
    if mobility_path is not None:
        for row in _read_csv(mobility_path):
            if str(row.get("geography") or "") != "national":
                continue
            quarter = str(row.get("time_period") or "")
            indicator = str(row.get("indicator") or "")
            value = _float_or_none(row.get("value"))
            if not quarter or value is None:
                continue
            by_quarter.setdefault(quarter, {})[f"google::{indicator}"] = value
    gam_path = paths.get("gam_program_support_csv")
    if gam_path is not None:
        grouped: dict[tuple[str, str], list[float]] = {}
        for row in _read_csv(gam_path):
            quarter = str(row.get("time_period") or "")
            module = str(row.get("module_target") or "")
            value = _float_or_none(row.get("value"))
            if not quarter or not module or value is None:
                continue
            grouped.setdefault((quarter, module), []).append(max(value, 0.0))
        for (quarter, module), values in grouped.items():
            by_quarter.setdefault(quarter, {})[f"gam::{module}::count"] = float(len(values))
            by_quarter.setdefault(quarter, {})[f"gam::{module}::log_sum"] = float(np.log1p(sum(values)))
    return by_quarter


def _feature_names_for_metric(metric_name: str, mode: str) -> list[str]:
    module = _metric_module(metric_name)
    google = [
        "google::retail_and_recreation_percent_change_from_baseline",
        "google::grocery_and_pharmacy_percent_change_from_baseline",
        "google::transit_stations_percent_change_from_baseline",
        "google::workplaces_percent_change_from_baseline",
        "google::residential_percent_change_from_baseline",
    ]
    gam = [f"gam::{module}::count", f"gam::{module}::log_sum"]
    if mode == "reporting_state":
        return google
    if mode == "capacity_state":
        return google + gam
    if mode == "combined_service_intensity_capacity":
        return google + gam
    return []


def _last_feature_quarter(train_rows: list[dict[str, Any]], feature_rows: dict[str, dict[str, float]]) -> str | None:
    train_quarters = [str(row.get("quarter") or "") for row in train_rows if row.get("quarter")]
    if not train_quarters:
        return None
    train_end = max(train_quarters, key=quarter_sort_key)
    eligible = [quarter for quarter in feature_rows if quarter_sort_key(quarter) <= quarter_sort_key(train_end)]
    return None if not eligible else max(eligible, key=quarter_sort_key)


def _raw_feature_vector(
    quarter: str,
    feature_rows: dict[str, dict[str, float]],
    feature_names: list[str],
    *,
    fallback_quarter: str | None,
) -> list[float]:
    source_quarter = quarter if quarter in feature_rows else fallback_quarter
    values = dict(feature_rows.get(str(source_quarter or ""), {}))
    return [float(values.get(name, 0.0)) for name in feature_names]


def _fit_linear_feature_model(
    train_rows: list[dict[str, Any]],
    metric_name: str,
    feature_rows: dict[str, dict[str, float]],
    *,
    mode: str,
) -> dict[str, Any]:
    metric_rows = [
        dict(row)
        for row in sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _finite_float(row.get(metric_name)) is not None
    ]
    feature_names = _feature_names_for_metric(metric_name, mode)
    if len(metric_rows) < 2:
        return {"status": "not_estimable", "reason": "too_few_metric_rows", "feature_names": feature_names}
    origin = quarter_ordinal(str(metric_rows[0].get("quarter") or ""))
    fallback_quarter = _last_feature_quarter(metric_rows, feature_rows)
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    for row in metric_rows:
        quarter = str(row.get("quarter") or "")
        target = _finite_float(row.get(metric_name))
        if target is None:
            continue
        t = float(quarter_ordinal(quarter) - origin)
        x_rows.append([1.0, t, *_raw_feature_vector(quarter, feature_rows, feature_names, fallback_quarter=fallback_quarter)])
        y_values.append(float(np.log1p(max(float(target), 0.0))))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.shape[0] < x.shape[1]:
        # This branch is intentionally conservative: no underdetermined feature regression.
        return {
            "status": "not_estimable",
            "reason": "underdetermined_feature_design",
            "feature_names": feature_names,
            "row_count": int(x.shape[0]),
            "parameter_count": int(x.shape[1]),
        }
    coefficients = np.linalg.pinv(x) @ y
    fitted = x @ coefficients
    residuals = y - fitted
    return {
        "status": "completed",
        "mode": mode,
        "metric_name": metric_name,
        "feature_names": feature_names,
        "coefficients": [float(value) for value in coefficients],
        "origin_quarter": str(metric_rows[0].get("quarter") or ""),
        "fallback_feature_quarter": fallback_quarter,
        "train_row_count": int(x.shape[0]),
        "train_rmse_log1p": float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else None,
        "contract": "ordinary least squares via pseudoinverse, train rows only, no validation-only targets",
    }


def _predict_feature_model(
    model: dict[str, Any],
    quarter: str,
    feature_rows: dict[str, dict[str, float]],
    *,
    policy: str,
) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    feature_names = [str(value) for value in list(model.get("feature_names") or [])]
    if policy == "forecast_safe":
        feature_quarter = str(model.get("fallback_feature_quarter") or "")
    elif policy == "contemporaneous_nowcast":
        feature_quarter = quarter
    else:
        feature_quarter = str(model.get("fallback_feature_quarter") or "")
    origin = str(model.get("origin_quarter") or quarter)
    t = float(quarter_ordinal(quarter) - quarter_ordinal(origin))
    x = np.asarray(
        [1.0, t, *_raw_feature_vector(feature_quarter, feature_rows, feature_names, fallback_quarter=str(model.get("fallback_feature_quarter") or ""))],
        dtype=np.float64,
    )
    coef = np.asarray(list(model.get("coefficients") or []), dtype=np.float64)
    if x.size != coef.size:
        return None
    return float(max(np.expm1(float(x @ coef)), 0.0))


def _feature_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    *,
    policy: str,
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_modes = {
        "diagnosed_plhiv": "reporting_state",
        "new_diagnosed_cases_period": "reporting_state",
        "alive_on_art": "capacity_state",
        "tested_for_viral_load": "capacity_state",
        "virally_suppressed": "capacity_state",
    }
    models = {
        metric: _fit_linear_feature_model(train_rows, metric, feature_rows, mode=mode)
        for metric, mode in metric_modes.items()
    }
    predictions: list[dict[str, Any]] = []
    carry_rows = _carry_forward_prediction(train_rows, holdout_rows)
    carry_by_quarter = {str(row.get("quarter") or ""): dict(row) for row in carry_rows}
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        prediction = {"quarter": quarter}
        fallback = carry_by_quarter.get(quarter, {})
        for metric in R71_METRICS:
            value = _predict_feature_model(models.get(metric, {}), quarter, feature_rows, policy=policy)
            prediction[metric] = float(fallback.get(metric) or 0.0) if value is None else value
        projected = project_cascade_stock_row(prediction)
        for metric_name, value in dict(projected.get("projected") or {}).items():
            prediction[metric_name] = value
        prediction["cascade_projection_changed"] = bool(projected.get("changed"))
        predictions.append(prediction)
    return predictions, {
        "family": family,
        "policy": policy,
        "metric_models": models,
        "contract": (
            "R71 feature branch fits diagnosis reporting and service-capacity OLS models using train rows only. "
            "forecast_safe freezes external features at train origin; contemporaneous_nowcast is diagnostic only."
        ),
    }


def _prediction_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("quarter") or ""): dict(row) for row in rows}


def _score_prediction_family(
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    family: str,
    train_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    predictions = _prediction_index(prediction_rows)
    scales = {
        metric: max([abs(float(row.get(metric) or 0.0)) for row in train_rows if _finite_float(row.get(metric)) is not None] or [float(np.finfo(np.float32).eps)])
        for metric in R71_METRICS
    }
    rows: list[dict[str, Any]] = []
    for target in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(target.get("quarter") or "")
        prediction = predictions.get(quarter, {})
        for metric in R71_METRICS:
            target_value = _finite_float(target.get(metric))
            predicted_value = _finite_float(prediction.get(metric))
            if target_value is None or predicted_value is None:
                continue
            rows.append(
                {
                    "family": family,
                    "quarter": quarter,
                    "year": quarter_year(quarter),
                    "metric_name": metric,
                    "target_value": float(target_value),
                    "predicted_value": float(predicted_value),
                    "scale": float(scales[metric]),
                    "normalized_absolute_error": float(abs(float(predicted_value) - float(target_value)) / float(scales[metric])),
                }
            )
    return rows


def _summary_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in score_rows:
        grouped.setdefault((str(row.get("family")), str(row.get("metric_name"))), []).append(row)
    rows: list[dict[str, Any]] = []
    for (family, metric), values in sorted(grouped.items()):
        errors = [float(row["normalized_absolute_error"]) for row in values]
        rows.append(
            {
                "family": family,
                "metric_name": metric,
                "entry_count": len(values),
                "mean_normalized_absolute_error": float(np.mean(np.asarray(errors, dtype=np.float64))) if errors else None,
                "p90_normalized_absolute_error": float(np.quantile(np.asarray(errors, dtype=np.float64), 0.9)) if errors else None,
            }
        )
    families = sorted({str(row.get("family")) for row in score_rows})
    for family in families:
        values = [float(row["normalized_absolute_error"]) for row in score_rows if row.get("family") == family]
        rows.append(
            {
                "family": family,
                "metric_name": "__overall__",
                "entry_count": len(values),
                "mean_normalized_absolute_error": float(np.mean(np.asarray(values, dtype=np.float64))) if values else None,
                "p90_normalized_absolute_error": float(np.quantile(np.asarray(values, dtype=np.float64), 0.9)) if values else None,
            }
        )
    return rows


def _gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {str(row.get("family")): row for row in summary_rows if row.get("metric_name") == "__overall__"}
    safe = _finite_float((overall.get("r71_forecast_safe_service_intensity_capacity") or {}).get("mean_normalized_absolute_error"))
    nowcast = _finite_float((overall.get("r71_contemporaneous_nowcast_service_intensity_capacity") or {}).get("mean_normalized_absolute_error"))
    carry = _finite_float((overall.get("carry_forward") or {}).get("mean_normalized_absolute_error"))
    r41 = _finite_float((overall.get("r41_research_champion_reference") or {}).get("mean_normalized_absolute_error"))
    promoted = safe is not None and carry is not None and safe < carry and (r41 is None or safe < r41)
    return {
        "status": "r71_forecast_safe_promoted" if promoted else "r71_diagnostic_only",
        "forecast_safe_mean_nae": safe,
        "contemporaneous_nowcast_mean_nae": nowcast,
        "carry_forward_mean_nae": carry,
        "r41_reference_mean_nae": r41,
        "forecast_safe_beats_carry_forward": None if safe is None or carry is None else bool(safe < carry),
        "forecast_safe_beats_r41": None if safe is None or r41 is None else bool(safe < r41),
        "nowcast_beats_carry_forward": None if nowcast is None or carry is None else bool(nowcast < carry),
        "contract": (
            "Only the forecast_safe branch can promote. The contemporaneous_nowcast branch is diagnostic because it may use "
            "same-quarter external support signals. No validation-only UNAIDS annual rows are used as training targets."
        ),
    }


def _split_years(rows: list[dict[str, Any]], *, start_year: int, end_year: int, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in rows if row.get("quarter")})
    return [
        year
        for year in years
        if int(start_year) <= year <= int(end_year)
        and len([prior for prior in years if prior < year]) >= int(min_train_years)
    ]


def run_r71_service_intensity_capacity_branch(
    *,
    run_id: str = R71_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    start_year: int = 2018,
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
    holdout_years = _split_years(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years)
    all_score_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for holdout_year in holdout_years:
        train_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) < holdout_year]
        holdout_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) == holdout_year]
        if not train_rows or not holdout_rows:
            continue
        carry = _carry_forward_prediction(train_rows, holdout_rows)
        r41, r41_model = _r41_monotone_growth_component_predictions(train_rows, holdout_rows)
        safe, safe_model = _feature_predictions(
            train_rows,
            holdout_rows,
            feature_rows,
            policy="forecast_safe",
            family="r71_forecast_safe_service_intensity_capacity",
        )
        nowcast, nowcast_model = _feature_predictions(
            train_rows,
            holdout_rows,
            feature_rows,
            policy="contemporaneous_nowcast",
            family="r71_contemporaneous_nowcast_service_intensity_capacity",
        )
        for family, predictions in (
            ("carry_forward", carry),
            ("r41_research_champion_reference", r41),
            ("r71_forecast_safe_service_intensity_capacity", safe),
            ("r71_contemporaneous_nowcast_service_intensity_capacity", nowcast),
        ):
            for row in _score_prediction_family(holdout_rows, predictions, family=family, train_rows=train_rows):
                row["holdout_year"] = holdout_year
                all_score_rows.append(row)
        model_rows.append(
            {
                "holdout_year": holdout_year,
                "train_row_count": len(train_rows),
                "holdout_row_count": len(holdout_rows),
                "r41_policy": r41_model.get("selected_policy"),
                "safe_model_status": {metric: model.get("status") for metric, model in dict(safe_model.get("metric_models") or {}).items()},
                "nowcast_model_status": {metric: model.get("status") for metric, model in dict(nowcast_model.get("metric_models") or {}).items()},
                "feature_contract": safe_model.get("contract"),
            }
        )
    summary = _summary_rows(all_score_rows)
    gate = _gate(summary)
    report_path = analysis_dir / "r71_service_intensity_capacity_branch_report.json"
    markdown_path = analysis_dir / "r71_service_intensity_capacity_branch_report.md"
    score_csv = analysis_dir / "r71_score_rows.csv"
    summary_csv = analysis_dir / "r71_summary_rows.csv"
    model_csv = analysis_dir / "r71_model_rows.csv"
    report = {
        "schema_version": R71_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "service_intensity_capacity_gate": gate,
        "holdout_years": holdout_years,
        "score_rows": all_score_rows,
        "summary_rows": summary,
        "model_rows": model_rows,
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
            "model_rows_csv": model_csv.as_posix(),
        },
    }
    _write_csv(score_csv, all_score_rows)
    _write_csv(summary_csv, summary)
    _write_csv(model_csv, model_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("service_intensity_capacity_gate") or {})
    lines = [
        "# Phase 3 R71 Service-Intensity + Capacity Branch",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Forecast-safe mean NAE: `{gate.get('forecast_safe_mean_nae')}`",
        f"- Nowcast mean NAE: `{gate.get('contemporaneous_nowcast_mean_nae')}`",
        f"- Carry-forward mean NAE: `{gate.get('carry_forward_mean_nae')}`",
        f"- R41 reference mean NAE: `{gate.get('r41_reference_mean_nae')}`",
        "",
        "## Summary",
        "",
        "| Family | Metric | Mean NAE | p90 NAE |",
        "|---|---|---:|---:|",
    ]
    for row in report.get("summary_rows") or []:
        lines.append(
            f"| `{row.get('family')}` | `{row.get('metric_name')}` | "
            f"{float(row.get('mean_normalized_absolute_error') or 0.0):.6f} | "
            f"{float(row.get('p90_normalized_absolute_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R71 service-intensity and capacity branch.")
    parser.add_argument("--run-id", default=R71_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    run_r71_service_intensity_capacity_branch(
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
