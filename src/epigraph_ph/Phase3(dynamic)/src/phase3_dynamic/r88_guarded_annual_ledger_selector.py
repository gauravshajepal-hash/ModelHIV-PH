from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, _finite_float, _generated_at, _sha256
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
from .r85_annual_ledger_forecast_grid import _dynamic_forecast_grid_predictions
from .r86_annual_calibrated_forecast_grid_ledger import _distribute_annual_total_by_quarter_shape
from .r87_train_backtested_emission_process_calibration import _train_backtest_pairs
from .runtime import ensure_dir, read_json, write_json


R88_SCHEMA_VERSION = "phase3_dynamic.r88_guarded_annual_ledger_selector.v1"
R88_RUN_ID = "p3d-r88-guarded-annual-ledger-selector-20260507-s00"
R88_FAMILY = "guarded_raw_process_or_carry_forward_annual_ledger"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)

ANNUAL_TO_QUARTERLY_METRIC: dict[str, str] = {
    "annual_new_infections": "incident_infections_period",
    "annual_aids_deaths": "aids_deaths_period",
}


def _internal_guarded_metric_selector(pairs: list[dict[str, Any]], *, metric_name: str) -> dict[str, Any]:
    metric_pairs = [
        dict(row)
        for row in sorted(pairs, key=lambda item: int(item.get("year") or 0))
        if str(row.get("metric_name") or "") == str(metric_name)
        and _finite_float(row.get("raw_value")) is not None
        and _finite_float(row.get("target_value")) is not None
    ]
    raw_errors: list[float] = []
    carry_errors: list[float] = []
    previous_target: float | None = None
    for row in metric_pairs:
        target = _finite_float(row.get("target_value"))
        raw = _finite_float(row.get("raw_value"))
        scale = max(float(_finite_float(row.get("scale")) or 0.0), float(np.finfo(np.float32).eps))
        if target is None or raw is None:
            continue
        if previous_target is not None:
            raw_errors.append(abs(float(raw) - float(target)) / scale)
            carry_errors.append(abs(float(previous_target) - float(target)) / scale)
        previous_target = float(target)
    raw_mean = None if not raw_errors else float(np.mean(np.asarray(raw_errors, dtype=np.float64)))
    carry_mean = None if not carry_errors else float(np.mean(np.asarray(carry_errors, dtype=np.float64)))
    if raw_mean is not None and carry_mean is not None and raw_mean < carry_mean:
        selected = "raw_quarterly_process"
    else:
        selected = "carry_forward_prior"
    return {
        "metric_name": metric_name,
        "selected_policy": selected,
        "raw_internal_mean_norm_error": raw_mean,
        "carry_internal_mean_norm_error": carry_mean,
        "internal_eval_count": len(raw_errors),
        "pair_count": len(metric_pairs),
        "contract": "policy selected by internal rolling-origin train error; carry-forward prior is allowed only as a conservative guardrail",
    }


def _apply_guarded_selector(
    predictions: list[dict[str, Any]],
    *,
    holdout_q4_rows: list[dict[str, Any]],
    raw_train_rows: list[dict[str, Any]],
    selectors: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_quarter = {str(row.get("quarter") or ""): dict(row) for row in predictions if row.get("quarter")}
    selector_rows: list[dict[str, Any]] = []
    for holdout_row in sorted(holdout_q4_rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or ""))):
        year = quarter_year(str(holdout_row.get("quarter") or ""))
        year_rows = [
            by_quarter[quarter]
            for quarter in sorted(by_quarter, key=quarter_sort_key)
            if quarter_year(quarter) == int(year)
        ]
        for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
            raw_annual = _annualized_candidate_value(list(by_quarter.values()), year=int(year), annual_metric=metric_name)
            raw_value = _finite_float(raw_annual.get("candidate_value"))
            policy = str((selectors.get(metric_name) or {}).get("selected_policy") or "carry_forward_prior")
            if policy == "raw_quarterly_process":
                selected_value = raw_value
            else:
                selected_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, metric_name)
            selector_rows.append(
                {
                    "year": int(year),
                    "metric_name": metric_name,
                    "selected_policy": policy,
                    "raw_annual_value": raw_value,
                    "selected_annual_value": selected_value,
                    "training_use": "internal_guarded_selector_no_holdout_target_use",
                }
            )
            if selected_value is None:
                continue
            if metric_name in ANNUAL_TO_QUARTERLY_METRIC:
                quarterly_metric = ANNUAL_TO_QUARTERLY_METRIC[metric_name]
                distribution = _distribute_annual_total_by_quarter_shape(
                    year_rows,
                    annual_total=float(selected_value),
                    quarterly_metric=quarterly_metric,
                )
                distribution.pop("_shape_status", None)
                for quarter, value in distribution.items():
                    by_quarter[str(quarter)][quarterly_metric] = float(value)
            elif metric_name == "estimated_plhiv":
                q4 = f"{int(year)}-Q4"
                by_quarter.setdefault(q4, {"quarter": q4})["estimated_plhiv"] = float(selected_value)
    return [by_quarter[quarter] for quarter in sorted(by_quarter, key=quarter_sort_key)], selector_rows


def _guarded_forecast_grid_predictions(
    rows: list[dict[str, Any]],
    holdout_q4_rows: list[dict[str, Any]],
    *,
    holdout_years: list[int],
    train_end_year: int,
    min_train_years: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    base_predictions, base_summary = _dynamic_forecast_grid_predictions(rows, holdout_years, train_end_year=train_end_year)
    pairs = _train_backtest_pairs(rows, train_end_year=train_end_year, min_train_years=min_train_years)
    selectors = {
        metric_name: _internal_guarded_metric_selector(pairs, metric_name=metric_name)
        for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
    }
    raw_train_rows = [dict(row) for row in rows if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)]
    guarded_rows, selector_rows = _apply_guarded_selector(
        base_predictions,
        holdout_q4_rows=holdout_q4_rows,
        raw_train_rows=raw_train_rows,
        selectors=selectors,
    )
    policy_rows = [dict(row, train_end_year=int(train_end_year)) for row in selectors.values()]
    summary = {
        **base_summary,
        "selected_policies": {metric: selector.get("selected_policy") for metric, selector in selectors.items()},
        "guarded_selector_contract": (
            "R88 keeps raw quarterly ledger emissions only for annual heads that beat carry-forward inside train-window "
            "rolling-origin checks; otherwise the metric uses a carry-forward prior distributed through the same quarterly "
            "emission shape. Held-out annual targets are scoring-only."
        ),
    }
    return guarded_rows, summary, policy_rows + selector_rows


def _score_guarded_selector(
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
    selector_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_q4_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_q4_rows:
            continue
        predictions, summary, split_selector_rows = _guarded_forecast_grid_predictions(
            rows,
            holdout_q4_rows,
            holdout_years=holdout_years,
            train_end_year=train_end_year,
            min_train_years=min_train_years,
        )
        for row in split_selector_rows:
            row = dict(row)
            row.update(
                {
                    "candidate_family": R88_FAMILY,
                    "horizon_years": horizon,
                    "train_end_year": train_end_year,
                    "holdout_years": holdout_years,
                }
            )
            selector_rows.append(row)
        annualized_by_year_metric: dict[tuple[int, str], dict[str, Any]] = {}
        for year in holdout_years:
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                annualized = _annualized_candidate_value(predictions, year=int(year), annual_metric=metric_name)
                annualized.update(
                    {
                        "candidate_family": R88_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "training_use": "internal_guarded_selector_no_holdout_target_use",
                    }
                )
                annualized_rows.append(annualized)
                coverage_rows.append(
                    {
                        "candidate_family": R88_FAMILY,
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
                    family=R88_FAMILY,
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
                score["training_use"] = "internal_guarded_selector_no_holdout_target_use"
                score_rows.append(score)
        manifest_rows.append(
            {
                "candidate_family": R88_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "selected_policies": summary.get("selected_policies"),
                "forecast_grid_contract": summary.get("forecast_grid_contract"),
                "dynamic_mae": summary.get("mae"),
                "model_contract": summary.get("model_contract"),
                "hazard_semantics": summary.get("hazard_semantics"),
                "incidence_diagnostics": summary.get("incidence_diagnostics"),
                "contract": summary.get("guarded_selector_contract"),
            }
        )
    return score_rows, annualized_rows, coverage_rows, selector_rows, manifest_rows


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
        blockers.append("guarded_annual_ledger_selector_not_better_than_carry_forward")
    if candidate_coverage is not None and carry_coverage is not None and candidate_coverage < carry_coverage:
        blockers.append("guarded_annual_ledger_selector_interval_coverage_worse_than_carry_forward")
    return {
        "status": "guarded_annual_ledger_selector_pass" if not blockers else "guarded_annual_ledger_selector_diagnostic_only",
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
            "R88 is a guarded annual-ledger selector. It is allowed to use the raw quarterly process only when internal "
            "train rolling-origin evidence beats carry-forward for that metric; otherwise it preserves a carry-forward "
            "prior and scores held-out annual targets as validation-only."
        ),
    }


def run_r88_guarded_annual_ledger_selector(
    *,
    run_id: str = R88_RUN_ID,
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
    score_rows, annualized_rows, coverage_rows, selector_rows, manifest_rows = _score_guarded_selector(
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
    report_path = analysis_dir / "r88_guarded_annual_ledger_selector_report.json"
    markdown_path = analysis_dir / "r88_guarded_annual_ledger_selector_report.md"
    score_csv = analysis_dir / "r88_score_rows.csv"
    annualized_csv = analysis_dir / "r88_annualized_rows.csv"
    coverage_csv = analysis_dir / "r88_emission_coverage_rows.csv"
    selector_csv = analysis_dir / "r88_selector_rows.csv"
    family_csv = analysis_dir / "r88_family_rows.csv"
    metric_csv = analysis_dir / "r88_metric_rows.csv"
    horizon_csv = analysis_dir / "r88_horizon_rows.csv"
    manifest_csv = analysis_dir / "r88_manifest_rows.csv"
    report = {
        "schema_version": R88_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "guarded_annual_ledger_selector_gate": gate,
        "candidate_family": R88_FAMILY,
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "target_row_count": len(target_rows),
        "score_rows": score_rows,
        "annualized_rows": annualized_rows,
        "coverage_rows": coverage_rows,
        "selector_rows": selector_rows,
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
            "selector_rows_csv": selector_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(annualized_csv, annualized_rows)
    _write_csv(coverage_csv, coverage_rows)
    _write_csv(selector_csv, selector_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("guarded_annual_ledger_selector_gate") or {})
    lines = [
        "# Phase 3 R88 Guarded Annual-Ledger Selector",
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
    parser = argparse.ArgumentParser(description="Run R88 guarded annual-ledger selector gate.")
    parser.add_argument("--run-id", default=R88_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r88_guarded_annual_ledger_selector(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
