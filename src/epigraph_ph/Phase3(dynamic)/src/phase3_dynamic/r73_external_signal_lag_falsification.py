from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_ordinal, quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256, project_cascade_stock_row
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import (
    R71_METRICS,
    _compile_feature_rows,
    _default_evidence_root,
    _score_prediction_family,
    _summary_rows,
)
from .r72_service_feature_selector_gate import _family_predictions
from .runtime import ensure_dir, read_json, write_json


R73_SCHEMA_VERSION = "phase3_dynamic.r73_external_signal_lag_falsification.v1"
R73_RUN_ID = "p3d-r73-external-signal-lag-falsification-20260506-s00"
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


def _feature_family(feature_name: str) -> str:
    if feature_name.startswith("google::"):
        return "google_mobility"
    if feature_name.startswith("gam::diagnosis_reporting::"):
        return "gam_diagnosis_reporting"
    if feature_name.startswith("gam::art_retention::"):
        return "gam_art_retention"
    if feature_name.startswith("gam::vl_suppression_service::"):
        return "gam_vl_suppression_service"
    return "other"


def _feature_catalog(feature_rows: dict[str, dict[str, float]]) -> list[dict[str, str]]:
    names = sorted({name for values in feature_rows.values() for name in values})
    return [{"feature_family": _feature_family(name), "feature_name": name} for name in names if _feature_family(name) != "other"]


def _last_feature_quarter_before(
    feature_rows: dict[str, dict[str, float]],
    feature_name: str,
    quarter: str,
) -> str | None:
    eligible = [
        candidate
        for candidate, values in feature_rows.items()
        if feature_name in values and quarter_sort_key(candidate) <= quarter_sort_key(quarter)
    ]
    return None if not eligible else max(eligible, key=quarter_sort_key)


def _feature_value(
    feature_rows: dict[str, dict[str, float]],
    feature_name: str,
    quarter: str,
    *,
    policy: str,
    origin_quarter: str,
) -> float | None:
    if policy == "oracle_contemporaneous":
        source_quarter = quarter if feature_name in feature_rows.get(quarter, {}) else _last_feature_quarter_before(feature_rows, feature_name, quarter)
    else:
        source_quarter = _last_feature_quarter_before(feature_rows, feature_name, origin_quarter)
    if source_quarter is None:
        return None
    return _finite_float((feature_rows.get(source_quarter) or {}).get(feature_name))


def _fit_univariate_feature_model(
    train_rows: list[dict[str, Any]],
    metric_name: str,
    feature_rows: dict[str, dict[str, float]],
    feature_name: str,
) -> dict[str, Any]:
    metric_rows = [
        dict(row)
        for row in sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _finite_float(row.get(metric_name)) is not None
    ]
    if len(metric_rows) < 3:
        return {"status": "not_estimable", "reason": "too_few_rows"}
    origin_quarter = str(metric_rows[0].get("quarter") or "")
    last_train_quarter = str(metric_rows[-1].get("quarter") or "")
    x_rows: list[list[float]] = []
    y_values: list[float] = []
    for row in metric_rows:
        quarter = str(row.get("quarter") or "")
        value = _finite_float(row.get(metric_name))
        feature = _feature_value(feature_rows, feature_name, quarter, policy="oracle_contemporaneous", origin_quarter=last_train_quarter)
        if value is None or feature is None:
            continue
        x_rows.append([1.0, float(quarter_ordinal(quarter) - quarter_ordinal(origin_quarter)), float(feature)])
        y_values.append(float(np.log1p(max(float(value), 0.0))))
    if len(x_rows) < 3:
        return {"status": "not_estimable", "reason": "too_few_feature_rows"}
    x = np.asarray(x_rows, dtype=np.float64)
    if int(np.linalg.matrix_rank(x)) < x.shape[1]:
        return {"status": "not_estimable", "reason": "rank_deficient_feature_design"}
    y = np.asarray(y_values, dtype=np.float64)
    coefficients = np.linalg.pinv(x) @ y
    residuals = y - (x @ coefficients)
    return {
        "status": "completed",
        "metric_name": metric_name,
        "feature_name": feature_name,
        "feature_family": _feature_family(feature_name),
        "coefficients": [float(value) for value in coefficients],
        "origin_quarter": origin_quarter,
        "last_train_quarter": last_train_quarter,
        "train_row_count": int(x.shape[0]),
        "train_rmse_log1p": float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else None,
    }


def _predict_univariate_feature_model(
    model: dict[str, Any],
    quarter: str,
    feature_rows: dict[str, dict[str, float]],
    *,
    policy: str,
) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    feature = _feature_value(
        feature_rows,
        str(model.get("feature_name") or ""),
        quarter,
        policy=policy,
        origin_quarter=str(model.get("last_train_quarter") or quarter),
    )
    if feature is None:
        return None
    coefficients = np.asarray(list(model.get("coefficients") or []), dtype=np.float64)
    x = np.asarray(
        [
            1.0,
            float(quarter_ordinal(quarter) - quarter_ordinal(str(model.get("origin_quarter") or quarter))),
            float(feature),
        ],
        dtype=np.float64,
    )
    if coefficients.size != x.size:
        return None
    return float(max(np.expm1(float(x @ coefficients)), 0.0))


def _score_metric_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    metric_name: str,
    *,
    family: str,
) -> list[dict[str, Any]]:
    return [
        row
        for row in _score_prediction_family(holdout_rows, predictions, family=family, train_rows=train_rows)
        if row.get("metric_name") == metric_name
    ]


def _mean_metric_error(rows: list[dict[str, Any]]) -> float | None:
    values = [float(row["normalized_absolute_error"]) for row in rows if _finite_float(row.get("normalized_absolute_error")) is not None]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _inner_split_years(train_rows: list[dict[str, Any]], *, min_train_years: int) -> list[int]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    return [year for year in years if len([prior for prior in years if prior < year]) >= int(min_train_years)]


def _candidate_prediction_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    model: dict[str, Any],
    metric_name: str,
    *,
    policy: str,
    base_family: str = "r41_research_champion_reference",
) -> list[dict[str, Any]]:
    base_predictions = _family_predictions(train_rows, holdout_rows, feature_rows).get(base_family, [])
    rows: list[dict[str, Any]] = []
    for base in base_predictions:
        prediction = dict(base)
        quarter = str(prediction.get("quarter") or "")
        value = _predict_univariate_feature_model(model, quarter, feature_rows, policy=policy)
        if value is not None:
            prediction[metric_name] = value
        projected = project_cascade_stock_row(prediction)
        for projected_metric, projected_value in dict(projected.get("projected") or {}).items():
            prediction[projected_metric] = projected_value
        prediction["cascade_projection_changed"] = bool(projected.get("changed"))
        rows.append(prediction)
    return rows


def _select_feature_models(
    train_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    *,
    policy: str,
    min_train_years: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    catalog = _feature_catalog(feature_rows)
    candidate_errors: dict[tuple[str, str, str], list[float]] = {}
    base_errors: dict[tuple[str, str], list[float]] = {}
    for inner_year in _inner_split_years(train_rows, min_train_years=min_train_years):
        inner_train = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) < inner_year]
        inner_holdout = [dict(row) for row in train_rows if quarter_year(str(row.get("quarter") or "")) == inner_year]
        if not inner_train or not inner_holdout:
            continue
        base_predictions = _family_predictions(inner_train, inner_holdout, feature_rows)
        for family, predictions in base_predictions.items():
            for metric in R71_METRICS:
                error = _mean_metric_error(_score_metric_predictions(inner_train, inner_holdout, predictions, metric, family=family))
                if error is not None:
                    base_errors.setdefault((family, metric), []).append(error)
        for metric in R71_METRICS:
            for feature in catalog:
                model = _fit_univariate_feature_model(inner_train, metric, feature_rows, feature["feature_name"])
                if str(model.get("status") or "") != "completed":
                    continue
                predictions = _candidate_prediction_rows(
                    inner_train,
                    inner_holdout,
                    feature_rows,
                    model,
                    metric,
                    policy=policy,
                )
                error = _mean_metric_error(
                    _score_metric_predictions(inner_train, inner_holdout, predictions, metric, family=f"r73_{policy}_{feature['feature_family']}")
                )
                if error is not None:
                    candidate_errors.setdefault((metric, feature["feature_family"], feature["feature_name"]), []).append(error)
    selected: dict[str, dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for metric in R71_METRICS:
        base_scored = []
        for base_family in ("r41_research_champion_reference", "carry_forward"):
            values = base_errors.get((base_family, metric), [])
            if values:
                base_scored.append((base_family, float(np.mean(np.asarray(values, dtype=np.float64)))))
        base_scored.sort(key=lambda item: (item[1], item[0]))
        best_base_family, best_base_error = base_scored[0] if base_scored else ("r41_research_champion_reference", float("inf"))
        feature_scored = []
        for (candidate_metric, family, feature_name), values in candidate_errors.items():
            if candidate_metric != metric or not values:
                continue
            feature_scored.append((family, feature_name, float(np.mean(np.asarray(values, dtype=np.float64)))))
        feature_scored.sort(key=lambda item: (item[2], item[0], item[1]))
        if feature_scored and feature_scored[0][2] < best_base_error:
            family, feature_name, error = feature_scored[0]
            selected[metric] = {
                "policy": policy,
                "selection": "feature_model",
                "feature_family": family,
                "feature_name": feature_name,
                "inner_mean_nae": error,
                "best_base_family": best_base_family,
                "best_base_inner_mean_nae": best_base_error,
            }
        else:
            selected[metric] = {
                "policy": policy,
                "selection": "base_family",
                "base_family": best_base_family,
                "inner_mean_nae": best_base_error,
            }
        selection_rows.append({"metric_name": metric, **selected[metric]})
    return selected, selection_rows


def _selector_predictions(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    feature_rows: dict[str, dict[str, float]],
    selected: dict[str, dict[str, Any]],
    *,
    policy: str,
) -> list[dict[str, Any]]:
    base_by_family = {
        family: {str(row.get("quarter") or ""): dict(row) for row in predictions}
        for family, predictions in _family_predictions(train_rows, holdout_rows, feature_rows).items()
    }
    fitted_models = {
        metric: _fit_univariate_feature_model(train_rows, metric, feature_rows, str(spec.get("feature_name") or ""))
        for metric, spec in selected.items()
        if spec.get("selection") == "feature_model"
    }
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        prediction: dict[str, Any] = {"quarter": quarter}
        for metric in R71_METRICS:
            spec = dict(selected.get(metric) or {})
            if spec.get("selection") == "feature_model":
                value = _predict_univariate_feature_model(fitted_models.get(metric, {}), quarter, feature_rows, policy=policy)
                if value is None:
                    fallback = (base_by_family.get("r41_research_champion_reference") or {}).get(quarter, {})
                    value = _finite_float(fallback.get(metric))
            else:
                family = str(spec.get("base_family") or "r41_research_champion_reference")
                value = _finite_float((base_by_family.get(family) or {}).get(quarter, {}).get(metric))
            prediction[metric] = value
        projected = project_cascade_stock_row(prediction)
        for projected_metric, projected_value in dict(projected.get("projected") or {}).items():
            prediction[projected_metric] = projected_value
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


def _gate(summary_rows: list[dict[str, Any]], selection_rows: list[dict[str, Any]]) -> dict[str, Any]:
    overall = {str(row.get("family")): row for row in summary_rows if row.get("metric_name") == "__overall__"}
    forecast = _finite_float((overall.get("r73_forecast_origin_external_signal_selector") or {}).get("mean_normalized_absolute_error"))
    oracle = _finite_float((overall.get("r73_oracle_contemporaneous_external_signal_selector") or {}).get("mean_normalized_absolute_error"))
    carry = _finite_float((overall.get("carry_forward") or {}).get("mean_normalized_absolute_error"))
    r41 = _finite_float((overall.get("r41_research_champion_reference") or {}).get("mean_normalized_absolute_error"))
    forecast_feature_count = sum(1 for row in selection_rows if row.get("policy") == "forecast_origin" and row.get("selection") == "feature_model")
    oracle_feature_count = sum(1 for row in selection_rows if row.get("policy") == "oracle_contemporaneous" and row.get("selection") == "feature_model")
    strict = forecast is not None and carry is not None and r41 is not None and forecast < carry and forecast < r41
    return {
        "status": "r73_forecast_external_signal_promoted" if strict else "r73_diagnostic_only",
        "forecast_selector_mean_nae": forecast,
        "oracle_nowcast_selector_mean_nae": oracle,
        "carry_forward_mean_nae": carry,
        "r41_reference_mean_nae": r41,
        "forecast_beats_carry_forward": None if forecast is None or carry is None else bool(forecast < carry),
        "forecast_beats_r41": None if forecast is None or r41 is None else bool(forecast < r41),
        "oracle_beats_r41": None if oracle is None or r41 is None else bool(oracle < r41),
        "forecast_selected_feature_metric_count": forecast_feature_count,
        "oracle_selected_feature_metric_count": oracle_feature_count,
        "contract": (
            "R73 separates forecast-origin external signal use from oracle nowcasting. Only forecast-origin selected features "
            "can promote; oracle-contemporaneous selection is leakage-labeled diagnostic evidence only."
        ),
    }


def run_r73_external_signal_lag_falsification(
    *,
    run_id: str = R73_RUN_ID,
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
    selection_rows: list[dict[str, Any]] = []
    feature_catalog = _feature_catalog(feature_rows)
    for holdout_year in _split_years(observation_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years):
        train_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) < holdout_year]
        holdout_rows = [dict(row) for row in observation_rows if quarter_year(str(row.get("quarter") or "")) == holdout_year]
        base_predictions = _family_predictions(train_rows, holdout_rows, feature_rows)
        for family, rows in base_predictions.items():
            for score in _score_prediction_family(holdout_rows, rows, family=family, train_rows=train_rows):
                score["holdout_year"] = holdout_year
                all_score_rows.append(score)
        for policy, family_name in (
            ("forecast_origin", "r73_forecast_origin_external_signal_selector"),
            ("oracle_contemporaneous", "r73_oracle_contemporaneous_external_signal_selector"),
        ):
            selected, policy_selection_rows = _select_feature_models(
                train_rows,
                feature_rows,
                policy=policy,
                min_train_years=min_train_years,
            )
            predictions = _selector_predictions(train_rows, holdout_rows, feature_rows, selected, policy=policy)
            for score in _score_prediction_family(holdout_rows, predictions, family=family_name, train_rows=train_rows):
                score["holdout_year"] = holdout_year
                all_score_rows.append(score)
            for row in policy_selection_rows:
                selection_rows.append(
                    {
                        "holdout_year": holdout_year,
                        "train_row_count": len(train_rows),
                        "holdout_row_count": len(holdout_rows),
                        **row,
                    }
                )
    summary = _summary_rows(all_score_rows)
    gate = _gate(summary, selection_rows)
    report_path = analysis_dir / "r73_external_signal_lag_falsification_report.json"
    markdown_path = analysis_dir / "r73_external_signal_lag_falsification_report.md"
    score_csv = analysis_dir / "r73_score_rows.csv"
    summary_csv = analysis_dir / "r73_summary_rows.csv"
    selection_csv = analysis_dir / "r73_selection_rows.csv"
    catalog_csv = analysis_dir / "r73_feature_catalog.csv"
    report = {
        "schema_version": R73_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "external_signal_gate": gate,
        "feature_catalog_rows": feature_catalog,
        "score_rows": all_score_rows,
        "summary_rows": summary,
        "selection_rows": selection_rows,
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
            "selection_rows_csv": selection_csv.as_posix(),
            "feature_catalog_csv": catalog_csv.as_posix(),
        },
    }
    _write_csv(score_csv, all_score_rows)
    _write_csv(summary_csv, summary)
    _write_csv(selection_csv, selection_rows)
    _write_csv(catalog_csv, feature_catalog)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("external_signal_gate") or {})
    lines = [
        "# Phase 3 R73 External Signal Lag Falsification",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Forecast selector mean NAE: `{gate.get('forecast_selector_mean_nae')}`",
        f"- Oracle nowcast selector mean NAE: `{gate.get('oracle_nowcast_selector_mean_nae')}`",
        f"- Carry-forward mean NAE: `{gate.get('carry_forward_mean_nae')}`",
        f"- R41 reference mean NAE: `{gate.get('r41_reference_mean_nae')}`",
        f"- Forecast selected feature metrics: `{gate.get('forecast_selected_feature_metric_count')}`",
        f"- Oracle selected feature metrics: `{gate.get('oracle_selected_feature_metric_count')}`",
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
    parser = argparse.ArgumentParser(description="Run R73 external signal lag falsification.")
    parser.add_argument("--run-id", default=R73_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--min-train-years", type=int, default=5)
    args = parser.parse_args()
    run_r73_external_signal_lag_falsification(
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
