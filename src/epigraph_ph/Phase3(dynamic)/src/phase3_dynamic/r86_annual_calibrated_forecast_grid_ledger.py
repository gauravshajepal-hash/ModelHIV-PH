from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
    _apply_official_annual_measurement_error_heads,
    _finite_float,
    _fit_official_annual_measurement_error_heads,
    _generated_at,
    _sha256,
)
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .r71_service_intensity_capacity_branch import _default_evidence_root
from .r75_bulk_unaids_annual_challenge import (
    _annual_carry_forward_prediction,
    _annual_score_row,
    _annual_target_scale,
    _bulk_unaids_target_rows,
    _merge_external_targets_into_observations,
    _r69_annual_path,
    _rolling_annual_splits,
    _score_summary_by_fields,
    _write_csv,
)
from .r83_quarterly_emission_bridge_audit import _annualized_candidate_value
from .r85_annual_ledger_forecast_grid import _dynamic_forecast_grid_predictions, _forecast_grid_rows
from .runtime import ensure_dir, read_json, write_json


R86_SCHEMA_VERSION = "phase3_dynamic.r86_annual_calibrated_forecast_grid_ledger.v1"
R86_RUN_ID = "p3d-r86-annual-calibrated-forecast-grid-ledger-20260507-s00"
R86_FAMILY = "annual_calibrated_conserved_forecast_grid_ledger"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)

ANNUAL_TOTAL_TO_QUARTERLY_EMISSION: dict[str, str] = {
    "annual_new_infections": "incident_infections_period",
    "annual_aids_deaths": "aids_deaths_period",
}


def _distribute_annual_total_by_quarter_shape(
    year_rows: list[dict[str, Any]],
    *,
    annual_total: float | None,
    quarterly_metric: str,
) -> dict[str, float]:
    sorted_rows = sorted(year_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
    if annual_total is None or not sorted_rows:
        return {}
    values = [
        max(float(value), 0.0)
        for value in (_finite_float(row.get(quarterly_metric)) for row in sorted_rows)
        if value is not None
    ]
    if len(values) != len(sorted_rows):
        values = [0.0 for _ in sorted_rows]
    total = float(sum(values))
    if total <= float(1e-12):
        share = 1.0 / float(len(sorted_rows))
        shares = [share for _ in sorted_rows]
        shape_status = "uniform_no_positive_quarter_shape"
    else:
        shares = [float(value) / total for value in values]
        shape_status = "base_dynamic_quarter_shape"
    output = {
        str(row.get("quarter") or ""): float(max(float(annual_total), 0.0) * float(shares[index]))
        for index, row in enumerate(sorted_rows)
    }
    output["_shape_status"] = shape_status  # type: ignore[assignment]
    return output


def _annual_calibrated_forecast_grid_predictions(
    rows: list[dict[str, Any]],
    holdout_years: list[int],
    *,
    train_end_year: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base_predictions, base_summary = _dynamic_forecast_grid_predictions(
        rows,
        holdout_years,
        train_end_year=train_end_year,
    )
    raw_train_rows = [
        dict(row)
        for row in rows
        if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
    ]
    annual_head_process = _fit_official_annual_measurement_error_heads(raw_train_rows)
    q4_support_rows = [
        row
        for row in _forecast_grid_rows(rows, holdout_years=holdout_years, train_end_year=train_end_year)
        if str(row.get("quarter") or "").endswith("-Q4")
    ]
    q4_predictions = [
        row
        for row in sorted(base_predictions, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))
        if str(row.get("quarter") or "").endswith("-Q4")
        and quarter_year(str(row.get("quarter") or "")) in {int(year) for year in holdout_years}
    ]
    annual_prediction_rows, annual_head_rows = _apply_official_annual_measurement_error_heads(
        q4_predictions,
        q4_support_rows,
        annual_head_process,
    )
    annual_by_year = {
        quarter_year(str(row.get("quarter") or "")): dict(row)
        for row in annual_prediction_rows
        if str(row.get("quarter") or "").endswith("-Q4")
    }
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in base_predictions if row.get("quarter")}
    shape_rows: list[dict[str, Any]] = []
    for year in sorted({int(value) for value in holdout_years}):
        year_quarters = [
            by_quarter[quarter]
            for quarter in sorted(by_quarter, key=quarter_sort_key)
            if quarter_year(quarter) == int(year)
        ]
        annual_row = dict(annual_by_year.get(int(year)) or {})
        for annual_metric, quarterly_metric in ANNUAL_TOTAL_TO_QUARTERLY_EMISSION.items():
            annual_total = _finite_float(annual_row.get(annual_metric))
            distribution = _distribute_annual_total_by_quarter_shape(
                year_quarters,
                annual_total=None if annual_total is None else float(annual_total),
                quarterly_metric=quarterly_metric,
            )
            shape_status = str(distribution.pop("_shape_status", "not_distributed"))
            for quarter, value in distribution.items():
                by_quarter[str(quarter)][quarterly_metric] = float(value)
            shape_rows.append(
                {
                    "year": int(year),
                    "annual_metric": annual_metric,
                    "quarterly_metric": quarterly_metric,
                    "annual_total": annual_total,
                    "shape_status": shape_status,
                    "quarter_count": len(year_quarters),
                    "training_use": "train_origin_annual_head_plus_dynamic_quarter_shape",
                }
            )
        plhiv = _finite_float(annual_row.get("estimated_plhiv"))
        if plhiv is not None:
            q4 = f"{int(year)}-Q4"
            by_quarter.setdefault(q4, {"quarter": q4})["estimated_plhiv"] = float(plhiv)
    calibrated_rows = [by_quarter[quarter] for quarter in sorted(by_quarter, key=quarter_sort_key)]
    summary = {
        **base_summary,
        "annual_head_process_status": str(annual_head_process.get("status") or "not_estimable"),
        "annual_head_contract": str(annual_head_process.get("contract") or ""),
        "annual_head_summary": {
            "joint_conservation_status": annual_head_process.get("joint_conservation_status"),
            "head_statuses": {
                metric_name: dict((annual_head_process.get("heads") or {}).get(metric_name) or {}).get("status")
                for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
            },
            "mass_balance_status": dict(annual_head_process.get("joint_mass_balance_head") or {}).get("status"),
        },
        "calibration_contract": (
            "annual heads are fitted from train-origin annual validation rows only; holdout annual targets score the "
            "annualized quarterly ledger but are never copied into quarterly emissions"
        ),
    }
    return calibrated_rows, summary, annual_head_rows + shape_rows


def _score_annual_calibrated_forecast_grid(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    score_rows: list[dict[str, Any]] = []
    annualized_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    annual_head_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(
        q4_rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    ):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_q4_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_q4_rows:
            continue
        predictions, summary, split_head_rows = _annual_calibrated_forecast_grid_predictions(
            rows,
            holdout_years,
            train_end_year=train_end_year,
        )
        for row in split_head_rows:
            row = dict(row)
            row.update(
                {
                    "candidate_family": R86_FAMILY,
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "holdout_years": holdout_years,
                }
            )
            annual_head_rows.append(row)
        annualized_by_year_metric: dict[tuple[int, str], dict[str, Any]] = {}
        for year in holdout_years:
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                annualized = _annualized_candidate_value(predictions, year=int(year), annual_metric=metric_name)
                annualized.update(
                    {
                        "candidate_family": R86_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "training_use": "train_origin_annual_head_plus_dynamic_quarter_shape",
                    }
                )
                annualized_rows.append(annualized)
                coverage_rows.append(
                    {
                        "candidate_family": R86_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "year": int(year),
                        "annual_metric": metric_name,
                        "coverage_status": annualized.get("coverage_status"),
                        "present_quarter_count": annualized.get("present_quarter_count"),
                        "emitted_quarter_count": annualized.get("emitted_quarter_count"),
                        "missing_required_outputs": annualized.get("missing_required_outputs"),
                    }
                )
                annualized_by_year_metric[(int(year), metric_name)] = annualized
        for holdout_row in holdout_q4_rows:
            holdout_year = quarter_year(str(holdout_row.get("quarter") or ""))
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                target = _finite_float(holdout_row.get(metric_name))
                if target is None:
                    continue
                annualized = annualized_by_year_metric.get((int(holdout_year), metric_name), {})
                candidate_value = _finite_float(annualized.get("candidate_value"))
                carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, metric_name)
                score = _annual_score_row(
                    family=R86_FAMILY,
                    horizon=horizon,
                    train_end_year=train_end_year,
                    holdout_years=holdout_years,
                    holdout_row=holdout_row,
                    metric_name=metric_name,
                    candidate_value=None if candidate_value is None else float(candidate_value),
                    carry_value=None if carry_value is None else float(carry_value),
                    scale=_annual_target_scale(raw_train_rows, metric_name),
                )
                score["prediction_status"] = str(annualized.get("coverage_status") or "not_predicted")
                score["training_use"] = "train_origin_annual_head_plus_dynamic_quarter_shape"
                score_rows.append(score)
        manifest_rows.append(
            {
                "candidate_family": R86_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "annual_head_process_status": summary.get("annual_head_process_status"),
                "annual_head_summary": summary.get("annual_head_summary"),
                "forecast_grid_contract": summary.get("forecast_grid_contract"),
                "dynamic_mae": summary.get("mae"),
                "model_contract": summary.get("model_contract"),
                "hazard_semantics": summary.get("hazard_semantics"),
                "incidence_diagnostics": summary.get("incidence_diagnostics"),
                "contract": summary.get("calibration_contract"),
            }
        )
    return score_rows, annualized_rows, coverage_rows, annual_head_rows, manifest_rows


def _gate(*, score_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not target_rows:
        blockers.append("no_public_annual_target_rows")
    leakage_rows = [
        row
        for row in score_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("annual_validation_role_leakage")
    target_counts_by_metric: dict[str, int] = {}
    scored_counts_by_metric: dict[str, int] = {}
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        metric_rows = [row for row in score_rows if str(row.get("metric_name") or "") == metric_name]
        target_counts_by_metric[metric_name] = len(metric_rows)
        scored_counts_by_metric[metric_name] = sum(1 for row in metric_rows if _finite_float(row.get("candidate_norm_error")) is not None)
        if not metric_rows:
            blockers.append(f"{metric_name}_not_scored")
        elif scored_counts_by_metric[metric_name] < len(metric_rows):
            blockers.append(f"{metric_name}_incomplete_target_coverage")
    best = dict(family_rows[0]) if family_rows else {}
    candidate_mean = _finite_float(best.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(best.get("carry_forward_mean_norm_error"))
    candidate_coverage = _finite_float(best.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(best.get("carry_forward_interval_coverage"))
    if candidate_mean is None or carry_mean is None:
        blockers.append("candidate_or_carry_not_evaluable")
    elif candidate_mean >= carry_mean:
        blockers.append("annual_calibrated_ledger_not_better_than_carry_forward")
    if candidate_coverage is not None and carry_coverage is not None and candidate_coverage < carry_coverage:
        blockers.append("annual_calibrated_ledger_interval_coverage_worse_than_carry_forward")
    return {
        "status": "annual_calibrated_forecast_grid_ledger_pass" if not blockers else "annual_calibrated_forecast_grid_ledger_diagnostic_only",
        "blockers": blockers,
        "required_metric_count": len(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "scored_metric_count": sum(1 for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if scored_counts_by_metric.get(metric, 0) > 0),
        "target_counts_by_metric": target_counts_by_metric,
        "scored_counts_by_metric": scored_counts_by_metric,
        "candidate_mean_norm_error": candidate_mean,
        "carry_forward_mean_norm_error": carry_mean,
        "candidate_minus_carry_forward_mean_norm_error": None
        if candidate_mean is None or carry_mean is None
        else float(candidate_mean - carry_mean),
        "candidate_interval_coverage": candidate_coverage,
        "carry_forward_interval_coverage": carry_coverage,
        "contract": (
            "R86 is a scoped annual-calibrated ledger gate: it preserves complete quarterly emission support from R85, "
            "fits annual weak-measurement heads only on train-origin annual rows, distributes incidence/death annual totals "
            "by train-origin dynamic quarterly shape, and scores held-out annual targets as validation-only evidence."
        ),
    }


def run_r86_annual_calibrated_forecast_grid_ledger(
    *,
    run_id: str = R86_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    external_start_year: int = 2010,
    start_year: int = 2019,
    end_year: int = 2024,
    min_train_years: int = 5,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    root = Path(epigraph_root) if epigraph_root is not None else _default_evidence_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(root, source_run_id=source_run_id, preferred=baseline_source_run_id)
    observation_rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    annual_csv = _r69_annual_path(r69)
    target_rows = [] if annual_csv is None else _bulk_unaids_target_rows(annual_csv, external_start_year=external_start_year)
    rows = _merge_external_targets_into_observations(observation_rows, target_rows)
    score_rows, annualized_rows, coverage_rows, annual_head_rows, manifest_rows = _score_annual_calibrated_forecast_grid(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    metric_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family",))
    gate = _gate(score_rows=score_rows, family_rows=family_rows, target_rows=target_rows)
    report_path = analysis_dir / "r86_annual_calibrated_forecast_grid_ledger_report.json"
    markdown_path = analysis_dir / "r86_annual_calibrated_forecast_grid_ledger_report.md"
    score_csv = analysis_dir / "r86_score_rows.csv"
    annualized_csv = analysis_dir / "r86_annualized_rows.csv"
    coverage_csv = analysis_dir / "r86_emission_coverage_rows.csv"
    annual_head_csv = analysis_dir / "r86_annual_head_rows.csv"
    family_csv = analysis_dir / "r86_family_rows.csv"
    metric_csv = analysis_dir / "r86_metric_rows.csv"
    horizon_csv = analysis_dir / "r86_horizon_rows.csv"
    manifest_csv = analysis_dir / "r86_manifest_rows.csv"
    report = {
        "schema_version": R86_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "annual_calibrated_forecast_grid_ledger_gate": gate,
        "candidate_family": R86_FAMILY,
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "target_row_count": len(target_rows),
        "score_rows": score_rows,
        "annualized_rows": annualized_rows,
        "coverage_rows": coverage_rows,
        "annual_head_rows": annual_head_rows,
        "family_rows": family_rows,
        "metric_rows": metric_rows,
        "horizon_rows": horizon_rows,
        "manifest_rows": manifest_rows,
        "source_artifacts": {
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
            "annual_external_challenge_csv": None if annual_csv is None else annual_csv.as_posix(),
            "annual_external_challenge_csv_sha256": None if annual_csv is None or not annual_csv.exists() else _sha256(annual_csv),
            "epigraph_root": root.as_posix(),
            "source_run_id": source_run_id,
            "baseline_source_run_id": baseline_source_run_id,
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "score_rows_csv": score_csv.as_posix(),
            "annualized_rows_csv": annualized_csv.as_posix(),
            "coverage_rows_csv": coverage_csv.as_posix(),
            "annual_head_rows_csv": annual_head_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(annualized_csv, annualized_rows)
    _write_csv(coverage_csv, coverage_rows)
    _write_csv(annual_head_csv, annual_head_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("annual_calibrated_forecast_grid_ledger_gate") or {})
    lines = [
        "# Phase 3 R86 Annual-Calibrated Forecast Grid Ledger",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Candidate mean normalized error: `{gate.get('candidate_mean_norm_error')}`",
        f"- Carry-forward mean normalized error: `{gate.get('carry_forward_mean_norm_error')}`",
        f"- Candidate minus carry-forward: `{gate.get('candidate_minus_carry_forward_mean_norm_error')}`",
        f"- Candidate interval coverage: `{gate.get('candidate_interval_coverage')}`",
        f"- Carry-forward interval coverage: `{gate.get('carry_forward_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Metric Rows",
        "",
        "| Metric | Scored Entries | Candidate Mean Error | Carry Mean Error | Candidate Interval Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('scored_candidate_entry_count')}` | "
            f"`{row.get('candidate_mean_norm_error')}` | `{row.get('carry_forward_mean_norm_error')}` | "
            f"`{row.get('candidate_interval_coverage')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R86 annual-calibrated forecast grid ledger gate.")
    parser.add_argument("--run-id", default=R86_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r86_annual_calibrated_forecast_grid_ledger(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
