from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import quarter_sort_key, quarter_year
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
    _apply_official_annual_measurement_error_heads,
    _finite_float,
    _fit_official_annual_measurement_error_heads,
    _generated_at,
    _sha256,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r75_bulk_unaids_annual_challenge import (
    _annual_carry_forward_prediction,
    _annual_score_row,
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _rolling_annual_splits,
    _score_summary_by_fields,
)
from .data import sandbox_repo_root
from .runtime import ensure_dir, read_json, write_json


R76_SCHEMA_VERSION = "phase3_dynamic.r76_public_domain_annual_comparator.v1"
R76_RUN_ID = "p3d-r76-public-domain-annual-comparator-20260507-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
PUBLIC_COMPARATOR_FAMILIES: tuple[str, ...] = (
    "public_carry_forward",
    "public_log_linear_trend",
    "public_joint_local_level_mass_balance",
)
PUBLIC_SELECTED_FAMILY = "public_train_selected_annual_proxy"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
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


def _target_panel_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _merge_external_targets_into_observations([], target_rows)


def _carry_prediction_rows(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        row: dict[str, Any] = {"quarter": str(holdout.get("quarter") or "")}
        for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            row[metric] = _annual_carry_forward_prediction(train_rows, holdout, metric)
        rows.append(row)
    return rows


def _fit_log_linear_metric(train_rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    metric_rows = [
        dict(row)
        for row in sorted(train_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if _finite_float(row.get(metric_name)) is not None
    ]
    if len(metric_rows) < 2:
        return {"status": "not_estimable", "reason": "too_few_rows", "metric_name": metric_name}
    years = np.asarray([quarter_year(str(row.get("quarter") or "")) for row in metric_rows], dtype=np.float64)
    values = np.asarray([max(float(row.get(metric_name) or 0.0), 0.0) for row in metric_rows], dtype=np.float64)
    origin_year = float(years[0])
    x = np.column_stack([np.ones_like(years), years - origin_year])
    y = np.log1p(values)
    coefficients = np.linalg.pinv(x) @ y
    residuals = y - (x @ coefficients)
    return {
        "status": "completed",
        "metric_name": metric_name,
        "origin_year": int(origin_year),
        "coefficients": [float(value) for value in coefficients],
        "train_row_count": len(metric_rows),
        "train_rmse_log1p": float(np.sqrt(np.mean(np.square(residuals)))) if residuals.size else None,
        "contract": "public log-linear annual trend fit on train years only",
    }


def _predict_log_linear_metric(model: dict[str, Any], quarter: str) -> float | None:
    if str(model.get("status") or "") != "completed":
        return None
    coefficients = np.asarray(list(model.get("coefficients") or []), dtype=np.float64)
    if coefficients.size != 2:
        return None
    year = quarter_year(quarter)
    origin_year = int(model.get("origin_year") or year)
    x = np.asarray([1.0, float(year - origin_year)], dtype=np.float64)
    return float(max(np.expm1(float(x @ coefficients)), 0.0))


def _log_linear_prediction_rows(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = {metric: _fit_log_linear_metric(train_rows, metric) for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS}
    carry_rows = {
        str(row.get("quarter") or ""): dict(row)
        for row in _carry_prediction_rows(train_rows, holdout_rows)
    }
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        row: dict[str, Any] = {"quarter": quarter}
        for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            value = _predict_log_linear_metric(models.get(metric, {}), quarter)
            row[metric] = _finite_float(carry_rows.get(quarter, {}).get(metric)) if value is None else value
        rows.append(row)
    return rows


def _joint_local_level_prediction_rows(train_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    process = _fit_official_annual_measurement_error_heads(train_rows)
    base_rows = [{"quarter": str(row.get("quarter") or "")} for row in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))]
    predictions, head_rows = _apply_official_annual_measurement_error_heads(base_rows, holdout_rows, process)
    return predictions, {
        "status": str(process.get("status") or ""),
        "joint_conservation_status": str(process.get("joint_conservation_status") or ""),
        "annual_measurement_head_row_count": len(head_rows),
        "contract": str(process.get("contract") or ""),
    }


def _prediction_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    family: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if family == "public_carry_forward":
        return _carry_prediction_rows(train_rows, holdout_rows), {"family": family, "contract": "last public annual value carried forward"}
    if family == "public_log_linear_trend":
        return _log_linear_prediction_rows(train_rows, holdout_rows), {"family": family, "contract": "metric-wise public annual log-linear trend"}
    if family == "public_joint_local_level_mass_balance":
        rows, summary = _joint_local_level_prediction_rows(train_rows, holdout_rows)
        return rows, {"family": family, **summary}
    raise ValueError(f"Unknown R76 public comparator family: {family}")


def _score_prediction_rows(
    *,
    family: str,
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    horizon: int,
    train_end_year: int,
    holdout_years: list[int],
) -> list[dict[str, Any]]:
    predictions = {str(row.get("quarter") or ""): dict(row) for row in prediction_rows}
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        prediction = predictions.get(quarter, {})
        for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            target = _finite_float(holdout.get(metric))
            if target is None:
                continue
            rows.append(
                _annual_score_row(
                    family=family,
                    horizon=horizon,
                    train_end_year=train_end_year,
                    holdout_years=holdout_years,
                    holdout_row=holdout,
                    metric_name=metric,
                    candidate_value=_finite_float(prediction.get(metric)),
                    carry_value=_annual_carry_forward_prediction(train_rows, holdout, metric),
                    scale=_annual_target_scale(train_rows, metric),
                )
            )
    return rows


def _mean_error(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["candidate_norm_error"])
        for row in rows
        if _finite_float(row.get("candidate_norm_error")) is not None
    ]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _select_metric_families(
    train_rows: list[dict[str, Any]],
    *,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> dict[str, str]:
    years = sorted({quarter_year(str(row.get("quarter") or "")) for row in train_rows if row.get("quarter")})
    if len(years) <= int(min_train_years):
        return {metric: "public_carry_forward" for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS}
    score_by_metric_family: dict[tuple[str, str], list[float]] = defaultdict(list)
    for split in _rolling_annual_splits(
        train_rows,
        start_year=min(years),
        end_year=max(years),
        min_train_years=min_train_years,
        horizons=horizons,
    ):
        inner_train = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(split["train_end_year"])
        ]
        inner_holdout = [
            dict(row)
            for row in train_rows
            if quarter_year(str(row.get("quarter") or "")) in set(int(year) for year in split["holdout_years"])
        ]
        if not inner_train or not inner_holdout:
            continue
        for family in PUBLIC_COMPARATOR_FAMILIES:
            predictions, _summary = _prediction_rows(inner_train, inner_holdout, family)
            scores = _score_prediction_rows(
                family=family,
                train_rows=inner_train,
                holdout_rows=inner_holdout,
                prediction_rows=predictions,
                horizon=int(split["horizon_years"]),
                train_end_year=int(split["train_end_year"]),
                holdout_years=[int(year) for year in split["holdout_years"]],
            )
            for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                metric_scores = [row for row in scores if row.get("metric_name") == metric]
                mean = _mean_error(metric_scores)
                if mean is not None:
                    score_by_metric_family[(metric, family)].append(mean)
    selected: dict[str, str] = {}
    for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        candidates: list[tuple[str, float]] = []
        for family in PUBLIC_COMPARATOR_FAMILIES:
            values = score_by_metric_family.get((metric, family), [])
            if values:
                candidates.append((family, float(np.mean(np.asarray(values, dtype=np.float64)))))
        candidates.sort(key=lambda item: (item[1], PUBLIC_COMPARATOR_FAMILIES.index(item[0])))
        selected[metric] = candidates[0][0] if candidates else "public_carry_forward"
    return selected


def _selected_proxy_prediction_rows(
    train_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    selected: dict[str, str],
) -> list[dict[str, Any]]:
    by_family = {
        family: {str(row.get("quarter") or ""): dict(row) for row in _prediction_rows(train_rows, holdout_rows, family)[0]}
        for family in PUBLIC_COMPARATOR_FAMILIES
    }
    rows: list[dict[str, Any]] = []
    for holdout in sorted(holdout_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        quarter = str(holdout.get("quarter") or "")
        row: dict[str, Any] = {"quarter": quarter}
        for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            family = selected.get(metric, "public_carry_forward")
            row[metric] = _finite_float((by_family.get(family) or {}).get(quarter, {}).get(metric))
        rows.append(row)
    return rows


def _score_public_comparators(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        train_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not train_rows or not holdout_rows:
            continue
        for family in PUBLIC_COMPARATOR_FAMILIES:
            predictions, summary = _prediction_rows(train_rows, holdout_rows, family)
            score_rows.extend(
                _score_prediction_rows(
                    family=family,
                    train_rows=train_rows,
                    holdout_rows=holdout_rows,
                    prediction_rows=predictions,
                    horizon=horizon,
                    train_end_year=train_end_year,
                    holdout_years=holdout_years,
                )
            )
            manifest_rows.append(
                {
                    "candidate_family": family,
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "holdout_years": holdout_years,
                    **summary,
                }
            )
        selected = _select_metric_families(train_rows, min_train_years=min_train_years, horizons=horizons)
        selected_predictions = _selected_proxy_prediction_rows(train_rows, holdout_rows, selected)
        score_rows.extend(
            _score_prediction_rows(
                family=PUBLIC_SELECTED_FAMILY,
                train_rows=train_rows,
                holdout_rows=holdout_rows,
                prediction_rows=selected_predictions,
                horizon=horizon,
                train_end_year=train_end_year,
                holdout_years=holdout_years,
            )
        )
        selection_rows.append(
            {
                "candidate_family": PUBLIC_SELECTED_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "selected_metric_families": selected,
            }
        )
    return score_rows, manifest_rows, selection_rows


def _gate(family_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {str(row.get("candidate_family") or ""): dict(row) for row in family_rows}
    selected = by_family.get(PUBLIC_SELECTED_FAMILY, {})
    carry = by_family.get("public_carry_forward", {})
    selected_mean = _finite_float(selected.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(carry.get("candidate_mean_norm_error"))
    selected_coverage = _finite_float(selected.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(carry.get("candidate_interval_coverage"))
    blockers: list[str] = []
    if not target_rows:
        blockers.append("no_public_annual_targets")
    if not score_rows:
        blockers.append("no_public_comparator_scores")
    leakage_rows = [
        row
        for row in score_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("public_target_role_leakage")
    if selected_mean is None or carry_mean is None:
        blockers.append("selected_or_carry_not_evaluable")
    elif selected_mean >= carry_mean:
        blockers.append("selected_proxy_not_better_than_public_carry_forward")
    if selected_coverage is not None and carry_coverage is not None and selected_coverage < carry_coverage:
        blockers.append("selected_proxy_interval_coverage_worse_than_public_carry_forward")
    return {
        "status": "public_domain_annual_comparator_ready" if not blockers else "public_domain_annual_comparator_diagnostic_only",
        "blockers": blockers,
        "target_row_count": len(target_rows),
        "score_row_count": len(score_rows),
        "comparator_family_count": len({str(row.get("candidate_family") or "") for row in family_rows}),
        "selected_proxy_mean_norm_error": selected_mean,
        "public_carry_forward_mean_norm_error": carry_mean,
        "selected_minus_public_carry_forward_mean_norm_error": None
        if selected_mean is None or carry_mean is None
        else float(selected_mean - carry_mean),
        "selected_proxy_interval_coverage": selected_coverage,
        "public_carry_forward_interval_coverage": carry_coverage,
        "contract": (
            "R76 builds an open public-domain annual comparator from bulk UNAIDS all-ages annual targets. "
            "It is an AEM/Spectrum-style proxy benchmark only, not official AEM output. All families are fitted "
            "on train years only and scored on held-out annual incidence, AIDS deaths, and PLHIV."
        ),
    }


def run_r76_public_domain_annual_comparator(
    *,
    run_id: str = R76_RUN_ID,
    r69_report_path: Path | None = None,
    external_start_year: int = 2010,
    start_year: int = 2019,
    end_year: int = 2024,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    panel_rows = _target_panel_rows(target_rows)
    score_rows, manifest_rows, selection_rows = _score_public_comparators(
        panel_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    metric_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family",))
    gate = _gate(family_rows, score_rows, target_rows)
    report_path = analysis_dir / "r76_public_domain_annual_comparator_report.json"
    markdown_path = analysis_dir / "r76_public_domain_annual_comparator_report.md"
    score_csv = analysis_dir / "r76_score_rows.csv"
    family_csv = analysis_dir / "r76_family_rows.csv"
    metric_csv = analysis_dir / "r76_metric_rows.csv"
    horizon_csv = analysis_dir / "r76_horizon_rows.csv"
    selection_csv = analysis_dir / "r76_selection_rows.csv"
    manifest_csv = analysis_dir / "r76_manifest_rows.csv"
    target_csv = analysis_dir / "r76_public_target_rows.csv"
    report = {
        "schema_version": R76_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "public_domain_annual_gate": gate,
        "comparator_families": list(PUBLIC_COMPARATOR_FAMILIES) + [PUBLIC_SELECTED_FAMILY],
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "external_start_year": int(external_start_year),
        "start_year": int(start_year),
        "end_year": int(end_year),
        "target_rows": target_rows,
        "score_rows": score_rows,
        "family_rows": family_rows,
        "metric_rows": metric_rows,
        "horizon_rows": horizon_rows,
        "selection_rows": selection_rows,
        "manifest_rows": manifest_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "score_rows_csv": score_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "selection_rows_csv": selection_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
            "target_rows_csv": target_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(selection_csv, selection_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_csv(target_csv, target_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("public_domain_annual_gate") or {})
    lines = [
        "# Phase 3 R76 Public-Domain Annual Comparator",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Selected proxy mean normalized error: `{gate.get('selected_proxy_mean_norm_error')}`",
        f"- Public carry-forward mean normalized error: `{gate.get('public_carry_forward_mean_norm_error')}`",
        f"- Selected proxy interval coverage: `{gate.get('selected_proxy_interval_coverage')}`",
        f"- Public carry-forward interval coverage: `{gate.get('public_carry_forward_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Family Scores",
        "",
        "| Family | Entries | Mean error | p90 error | Interval coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("family_rows") or []:
        lines.append(
            f"| `{row.get('candidate_family')}` | {int(row.get('entry_count') or 0)} | "
            f"{float(row.get('candidate_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('candidate_p90_norm_error') or 0.0):.6f} | "
            f"{float(row.get('candidate_interval_coverage') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R76 public-domain annual comparator.")
    parser.add_argument("--run-id", default=R76_RUN_ID)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--external-start-year", type=int, default=2010)
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--min-train-years", type=int, default=5)
    parser.add_argument("--horizon", type=int, action="append", default=None)
    args = parser.parse_args()
    run_r76_public_domain_annual_comparator(
        run_id=str(args.run_id),
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        external_start_year=int(args.external_start_year),
        start_year=int(args.start_year),
        end_year=int(args.end_year),
        min_train_years=int(args.min_train_years),
        horizons=tuple(int(value) for value in (args.horizon or [1, 3, 5])),
    )


if __name__ == "__main__":
    _main()
