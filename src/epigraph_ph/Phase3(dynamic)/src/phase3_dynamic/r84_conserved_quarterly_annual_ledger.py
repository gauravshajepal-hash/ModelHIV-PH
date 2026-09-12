from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import build_blocked_time_dataset, build_observation_rows, sandbox_repo_root
from .incidence import IncidenceFlowConfig
from .metrics import quarter_year
from .model import DynamicBaselineConfig, ObservationModelConfig, forecast_dynamic_baseline
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
from .runtime import ensure_dir, read_json, write_json


R84_SCHEMA_VERSION = "phase3_dynamic.r84_conserved_quarterly_annual_ledger.v1"
R84_RUN_ID = "p3d-r84-conserved-quarterly-annual-ledger-20260507-s00"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
R84_FAMILY = "conserved_dynamic_quarterly_ledger"


def _dynamic_ledger_predictions(rows: list[dict[str, Any]], holdout_years: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = build_blocked_time_dataset(rows, holdout_years)
    result = forecast_dynamic_baseline(
        dataset,
        DynamicBaselineConfig(),
        incidence_cfg=IncidenceFlowConfig(),
        observation_cfg=ObservationModelConfig(),
    )
    return [dict(row) for row in list(result.get("prediction_rows") or [])], {
        "mae": _finite_float(result.get("mae")),
        "model_contract": dict(result.get("model_contract") or {}),
        "hazard_semantics": dict(result.get("hazard_semantics") or {}),
        "incidence_diagnostics": dict((result.get("incidence_paths") or {}).get("diagnostics") or {}),
    }


def _score_conserved_ledger(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    score_rows: list[dict[str, Any]] = []
    annualized_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        holdout_q4_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_q4_rows:
            continue
        predictions, summary = _dynamic_ledger_predictions(rows, holdout_years)
        prediction_keys = sorted({str(key) for row in predictions for key in row.keys() if key != "quarter"})
        annualized_by_year_metric: dict[tuple[int, str], dict[str, Any]] = {}
        for year in holdout_years:
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                annualized = _annualized_candidate_value(predictions, year=int(year), annual_metric=metric_name)
                annualized.update(
                    {
                        "candidate_family": R84_FAMILY,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "training_use": "train_origin_dynamic_hazard_plus_incidence_ledger",
                    }
                )
                annualized_rows.append(annualized)
                coverage_rows.append(
                    {
                        "candidate_family": R84_FAMILY,
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
                    family=R84_FAMILY,
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
                score["training_use"] = "train_origin_dynamic_hazard_plus_incidence_ledger"
                score_rows.append(score)
        manifest_rows.append(
            {
                "candidate_family": R84_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "emitted_prediction_keys": prediction_keys,
                "dynamic_mae": summary.get("mae"),
                "model_contract": summary.get("model_contract"),
                "hazard_semantics": summary.get("hazard_semantics"),
                "incidence_diagnostics": summary.get("incidence_diagnostics"),
                "contract": "conserved dynamic simulator emissions scored against annual validation targets; no annual weak-measurement head is applied",
            }
        )
    return score_rows, annualized_rows, coverage_rows, manifest_rows


def _gate(
    *,
    score_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
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
    scored_metrics = {
        str(row.get("metric_name") or "")
        for row in score_rows
        if _finite_float(row.get("candidate_norm_error")) is not None
    }
    missing = [metric for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if metric not in scored_metrics]
    blockers.extend([f"{metric}_not_scored" for metric in missing])
    target_counts_by_metric: dict[str, int] = {}
    scored_counts_by_metric: dict[str, int] = {}
    for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
        metric_score_rows = [row for row in score_rows if str(row.get("metric_name") or "") == metric_name]
        target_counts_by_metric[metric_name] = len(metric_score_rows)
        scored_counts_by_metric[metric_name] = sum(
            1 for row in metric_score_rows if _finite_float(row.get("candidate_norm_error")) is not None
        )
        if metric_score_rows and scored_counts_by_metric[metric_name] < len(metric_score_rows):
            blockers.append(f"{metric_name}_incomplete_target_coverage")
    best = dict(family_rows[0]) if family_rows else {}
    candidate_mean = _finite_float(best.get("candidate_mean_norm_error"))
    carry_mean = _finite_float(best.get("carry_forward_mean_norm_error"))
    candidate_coverage = _finite_float(best.get("candidate_interval_coverage"))
    carry_coverage = _finite_float(best.get("carry_forward_interval_coverage"))
    if candidate_mean is None or carry_mean is None:
        blockers.append("candidate_or_carry_not_evaluable")
    elif candidate_mean >= carry_mean:
        blockers.append("conserved_ledger_not_better_than_carry_forward")
    if candidate_coverage is not None and carry_coverage is not None and candidate_coverage < carry_coverage:
        blockers.append("conserved_ledger_interval_coverage_worse_than_carry_forward")
    return {
        "status": "conserved_quarterly_annual_ledger_pass" if not blockers else "conserved_quarterly_annual_ledger_diagnostic_only",
        "blockers": blockers,
        "scored_metric_count": len(scored_metrics),
        "required_metric_count": len(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
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
            "R84 tests whether the mechanistic dynamic simulator itself can emit annual incidence, AIDS deaths, "
            "and total PLHIV through conserved quarterly state accounting. Passing requires explicit emissions, "
            "validation-only annual targets, improvement over carry-forward, and no annual weak-measurement head."
        ),
    }


def run_r84_conserved_quarterly_annual_ledger(
    *,
    run_id: str = R84_RUN_ID,
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
    score_rows, annualized_rows, coverage_rows, manifest_rows = _score_conserved_ledger(
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
    report_path = analysis_dir / "r84_conserved_quarterly_annual_ledger_report.json"
    markdown_path = analysis_dir / "r84_conserved_quarterly_annual_ledger_report.md"
    score_csv = analysis_dir / "r84_score_rows.csv"
    annualized_csv = analysis_dir / "r84_annualized_rows.csv"
    coverage_csv = analysis_dir / "r84_emission_coverage_rows.csv"
    family_csv = analysis_dir / "r84_family_rows.csv"
    metric_csv = analysis_dir / "r84_metric_rows.csv"
    horizon_csv = analysis_dir / "r84_horizon_rows.csv"
    manifest_csv = analysis_dir / "r84_manifest_rows.csv"
    report = {
        "schema_version": R84_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "conserved_quarterly_annual_ledger_gate": gate,
        "candidate_family": R84_FAMILY,
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "target_row_count": len(target_rows),
        "score_rows": score_rows,
        "annualized_rows": annualized_rows,
        "coverage_rows": coverage_rows,
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
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
            "manifest_rows_csv": manifest_csv.as_posix(),
        },
    }
    _write_csv(score_csv, score_rows)
    _write_csv(annualized_csv, annualized_rows)
    _write_csv(coverage_csv, coverage_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_csv(manifest_csv, manifest_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("conserved_quarterly_annual_ledger_gate") or {})
    lines = [
        "# Phase 3 R84 Conserved Quarterly Annual Ledger",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Scored annual metrics: `{gate.get('scored_metric_count')}` / `{gate.get('required_metric_count')}`",
        f"- Candidate mean normalized error: `{gate.get('candidate_mean_norm_error')}`",
        f"- Carry-forward mean normalized error: `{gate.get('carry_forward_mean_norm_error')}`",
        f"- Candidate minus carry-forward: `{gate.get('candidate_minus_carry_forward_mean_norm_error')}`",
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
    parser = argparse.ArgumentParser(description="Run R84 conserved quarterly annual ledger gate.")
    parser.add_argument("--run-id", default=R84_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r84_conserved_quarterly_annual_ledger(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
