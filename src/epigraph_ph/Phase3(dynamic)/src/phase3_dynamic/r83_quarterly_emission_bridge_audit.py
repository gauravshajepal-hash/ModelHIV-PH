from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS,
    _candidate_predictions,
    _finite_float,
    _generated_at,
    _sha256,
    _strip_official_annual_validation_targets,
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
from .r82_quarterly_annual_bridge_gate import BRIDGE_REQUIREMENTS
from .runtime import ensure_dir, read_json, write_json


R83_SCHEMA_VERSION = "phase3_dynamic.r83_quarterly_emission_bridge_audit.v1"
R83_RUN_ID = "p3d-r83-quarterly-emission-bridge-audit-20260507-s00"
R83_DEFAULT_CANDIDATE_FAMILIES: tuple[str, ...] = ("multi_horizon_weighted_process",)
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)


def _prediction_year_rows(prediction_rows: list[dict[str, Any]], year: int) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in prediction_rows
        if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) == int(year)
    ]
    return sorted(rows, key=lambda item: quarter_sort_key(str(item.get("quarter") or "")))


def _annualized_candidate_value(
    prediction_rows: list[dict[str, Any]],
    *,
    year: int,
    annual_metric: str,
) -> dict[str, Any]:
    requirement = dict(BRIDGE_REQUIREMENTS.get(str(annual_metric)) or {})
    required_outputs = tuple(str(value) for value in requirement.get("required_quarterly_outputs") or ())
    year_rows = _prediction_year_rows(prediction_rows, int(year))
    if not required_outputs:
        return {
            "year": int(year),
            "annual_metric": str(annual_metric),
            "candidate_value": None,
            "coverage_status": "blocked_no_bridge_rule",
            "required_quarterly_outputs": [],
            "present_quarter_count": len(year_rows),
            "emitted_quarter_count": 0,
            "missing_required_outputs": [],
        }

    missing_outputs = sorted(
        {
            output
            for output in required_outputs
            if not any(_finite_float(row.get(output)) is not None for row in year_rows)
        }
    )
    if missing_outputs:
        return {
            "year": int(year),
            "annual_metric": str(annual_metric),
            "candidate_value": None,
            "coverage_status": "blocked_missing_quarterly_outputs",
            "required_quarterly_outputs": list(required_outputs),
            "present_quarter_count": len(year_rows),
            "emitted_quarter_count": 0,
            "missing_required_outputs": missing_outputs,
        }

    if str(requirement.get("aggregation_operator") or "") == "sum_four_quarters":
        output = required_outputs[0]
        values = [_finite_float(row.get(output)) for row in year_rows]
        finite_values = [float(value) for value in values if value is not None]
        if len(year_rows) != 4 or len(finite_values) != 4:
            return {
                "year": int(year),
                "annual_metric": str(annual_metric),
                "candidate_value": None,
                "coverage_status": "blocked_incomplete_four_quarter_emission",
                "required_quarterly_outputs": list(required_outputs),
                "present_quarter_count": len(year_rows),
                "emitted_quarter_count": len(finite_values),
                "missing_required_outputs": [],
            }
        return {
            "year": int(year),
            "annual_metric": str(annual_metric),
            "candidate_value": float(sum(finite_values)),
            "coverage_status": "annualized_from_quarterly_emission",
            "required_quarterly_outputs": list(required_outputs),
            "present_quarter_count": len(year_rows),
            "emitted_quarter_count": len(finite_values),
            "missing_required_outputs": [],
        }

    if str(requirement.get("aggregation_operator") or "") == "q4_stock":
        output = required_outputs[0]
        q4_rows = [row for row in year_rows if str(row.get("quarter") or "").endswith("-Q4")]
        value = _finite_float(q4_rows[-1].get(output)) if q4_rows else None
        if value is None:
            return {
                "year": int(year),
                "annual_metric": str(annual_metric),
                "candidate_value": None,
                "coverage_status": "blocked_missing_q4_stock_emission",
                "required_quarterly_outputs": list(required_outputs),
                "present_quarter_count": len(year_rows),
                "emitted_quarter_count": 0,
                "missing_required_outputs": [],
            }
        return {
            "year": int(year),
            "annual_metric": str(annual_metric),
            "candidate_value": float(value),
            "coverage_status": "annualized_from_q4_stock_emission",
            "required_quarterly_outputs": list(required_outputs),
            "present_quarter_count": len(year_rows),
            "emitted_quarter_count": 1,
            "missing_required_outputs": [],
        }

    return {
        "year": int(year),
        "annual_metric": str(annual_metric),
        "candidate_value": None,
        "coverage_status": "blocked_unknown_aggregation_operator",
        "required_quarterly_outputs": list(required_outputs),
        "present_quarter_count": len(year_rows),
        "emitted_quarter_count": 0,
        "missing_required_outputs": [],
    }


def _emitted_prediction_keys(prediction_rows: list[dict[str, Any]]) -> list[str]:
    ignored = {"quarter", "metric_provenance", "cascade_projection_changed"}
    return sorted(
        {
            str(key)
            for row in prediction_rows
            for key in row.keys()
            if str(key) not in ignored and _finite_float(row.get(key)) is not None
        }
    )


def _score_family_emissions_against_annual_targets(
    rows: list[dict[str, Any]],
    *,
    family: str,
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
        train_rows = [_strip_official_annual_validation_targets(dict(row)) for row in raw_train_rows]
        holdout_rows = [dict(row) for row in rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        holdout_q4_rows = [dict(row) for row in q4_rows if quarter_year(str(row.get("quarter") or "")) in set(holdout_years)]
        if not raw_train_rows or not holdout_rows or not holdout_q4_rows:
            continue
        candidate_predictions, candidate_summary = _candidate_predictions(train_rows, holdout_rows, family=family)
        emitted_keys = _emitted_prediction_keys(candidate_predictions)
        annualized_by_year_metric: dict[tuple[int, str], dict[str, Any]] = {}
        for year in holdout_years:
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                row = _annualized_candidate_value(candidate_predictions, year=int(year), annual_metric=metric_name)
                row.update(
                    {
                        "candidate_family": family,
                        "horizon_years": horizon,
                        "train_end_year": train_end_year,
                        "holdout_years": holdout_years,
                        "training_use": "quarterly_candidate_emission_only_no_annual_head",
                    }
                )
                annualized_rows.append(row)
                coverage_rows.append(
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        in {
                            "candidate_family",
                            "horizon_years",
                            "train_end_year",
                            "holdout_years",
                            "year",
                            "annual_metric",
                            "coverage_status",
                            "required_quarterly_outputs",
                            "present_quarter_count",
                            "emitted_quarter_count",
                            "missing_required_outputs",
                        }
                    }
                )
                annualized_by_year_metric[(int(year), metric_name)] = row
        for holdout_row in holdout_q4_rows:
            holdout_year = quarter_year(str(holdout_row.get("quarter") or ""))
            for metric_name in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS:
                target = _finite_float(holdout_row.get(metric_name))
                if target is None:
                    continue
                annualized = annualized_by_year_metric.get((int(holdout_year), metric_name), {})
                scale = _annual_target_scale(raw_train_rows, metric_name)
                candidate_value = _finite_float(annualized.get("candidate_value"))
                carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, metric_name)
                score = _annual_score_row(
                    family=family,
                    horizon=horizon,
                    train_end_year=train_end_year,
                    holdout_years=holdout_years,
                    holdout_row=holdout_row,
                    metric_name=metric_name,
                    candidate_value=None if candidate_value is None else float(candidate_value),
                    carry_value=None if carry_value is None else float(carry_value),
                    scale=scale,
                )
                score["prediction_status"] = str(annualized.get("coverage_status") or "not_predicted")
                score["training_use"] = "quarterly_candidate_emission_only_no_annual_head"
                score_rows.append(score)
        manifest_rows.append(
            {
                "candidate_family": family,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                "candidate_summary_family": str(candidate_summary.get("family") or ""),
                "emitted_prediction_keys": emitted_keys,
                "contract": "actual candidate prediction rows inspected; annual weak-measurement heads are deliberately not applied",
            }
        )
    return score_rows, annualized_rows, coverage_rows, manifest_rows


def _gate(
    *,
    score_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
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
    scored_by_metric: dict[str, int] = defaultdict(int)
    for row in score_rows:
        if _finite_float(row.get("candidate_norm_error")) is not None:
            scored_by_metric[str(row.get("metric_name") or "")] += 1
    missing_metrics = [metric for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if scored_by_metric.get(metric, 0) <= 0]
    blockers.extend([f"{metric}_not_emitted_by_quarterly_candidate" for metric in missing_metrics])
    coverage_status_counts: dict[str, int] = defaultdict(int)
    for row in coverage_rows:
        coverage_status_counts[str(row.get("coverage_status") or "")] += 1
    evaluable_family_rows = [
        row
        for row in family_rows
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
        and _finite_float(row.get("carry_forward_mean_norm_error")) is not None
    ]
    evaluable_family_rows.sort(key=lambda row: (float(row["candidate_mean_norm_error"]), str(row.get("candidate_family") or "")))
    best = evaluable_family_rows[0] if evaluable_family_rows else {}
    best_mean = _finite_float(best.get("candidate_mean_norm_error"))
    best_carry = _finite_float(best.get("carry_forward_mean_norm_error"))
    if not score_rows:
        blockers.append("no_quarterly_emission_score_rows")
    if best_mean is not None and best_carry is not None and best_mean >= best_carry:
        blockers.append("best_quarterly_emission_bridge_not_better_than_carry_forward")
    scored_metric_count = len([metric for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS if scored_by_metric.get(metric, 0) > 0])
    if not blockers and scored_metric_count == len(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS):
        status = "quarterly_emission_bridge_ready"
    elif scored_metric_count > 0 and "annual_validation_role_leakage" not in blockers:
        status = "quarterly_emission_bridge_partial_diagnostic"
    else:
        status = "quarterly_emission_bridge_blocked"
    return {
        "status": status,
        "blockers": blockers,
        "required_metric_count": len(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "scored_metric_count": scored_metric_count,
        "scored_counts_by_metric": dict(sorted(scored_by_metric.items())),
        "coverage_status_counts": dict(sorted(coverage_status_counts.items())),
        "best_candidate_family": best.get("candidate_family"),
        "best_candidate_mean_norm_error": best_mean,
        "best_carry_forward_mean_norm_error": best_carry,
        "best_candidate_minus_carry_forward_mean_norm_error": None
        if best_mean is None or best_carry is None
        else float(best_mean - best_carry),
        "contract": (
            "R83 scores only annual quantities directly reconstructable from quarterly candidate prediction rows. "
            "It never applies the R75 annual weak-measurement head, never treats diagnosed stock as total PLHIV, "
            "and never treats aggregate attrition as AIDS deaths. Missing emissions block annual-mechanistic claims."
        ),
    }


def run_r83_quarterly_emission_bridge_audit(
    *,
    run_id: str = R83_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    r69_report_path: Path | None = None,
    candidate_families: tuple[str, ...] = R83_DEFAULT_CANDIDATE_FAMILIES,
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
    score_rows: list[dict[str, Any]] = []
    annualized_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for family in candidate_families:
        family_scores, family_annualized, family_coverage, family_manifest = _score_family_emissions_against_annual_targets(
            rows,
            family=str(family),
            start_year=start_year,
            end_year=end_year,
            min_train_years=min_train_years,
            horizons=horizons,
        )
        score_rows.extend(family_scores)
        annualized_rows.extend(family_annualized)
        coverage_rows.extend(family_coverage)
        manifest_rows.extend(family_manifest)
    metric_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "metric_name"))
    horizon_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family", "horizon_years"))
    family_rows = _score_summary_by_fields(score_rows, group_fields=("candidate_family",))
    gate = _gate(score_rows=score_rows, coverage_rows=coverage_rows, target_rows=target_rows, family_rows=family_rows)
    report_path = analysis_dir / "r83_quarterly_emission_bridge_audit_report.json"
    markdown_path = analysis_dir / "r83_quarterly_emission_bridge_audit_report.md"
    score_csv = analysis_dir / "r83_score_rows.csv"
    annualized_csv = analysis_dir / "r83_annualized_rows.csv"
    coverage_csv = analysis_dir / "r83_emission_coverage_rows.csv"
    family_csv = analysis_dir / "r83_family_rows.csv"
    metric_csv = analysis_dir / "r83_metric_rows.csv"
    horizon_csv = analysis_dir / "r83_horizon_rows.csv"
    manifest_csv = analysis_dir / "r83_manifest_rows.csv"
    report = {
        "schema_version": R83_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "quarterly_emission_bridge_gate": gate,
        "candidate_families": list(candidate_families),
        "metric_scope": list(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS),
        "horizons": list(horizons),
        "external_start_year": int(external_start_year),
        "start_year": int(start_year),
        "end_year": int(end_year),
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
    gate = dict(report.get("quarterly_emission_bridge_gate") or {})
    lines = [
        "# Phase 3 R83 Quarterly Emission Bridge Audit",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Scored annual metrics: `{gate.get('scored_metric_count')}` / `{gate.get('required_metric_count')}`",
        f"- Best family: `{gate.get('best_candidate_family')}`",
        f"- Best candidate mean normalized error: `{gate.get('best_candidate_mean_norm_error')}`",
        f"- Best carry-forward mean normalized error: `{gate.get('best_carry_forward_mean_norm_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Metric Coverage",
        "",
        "| Candidate Family | Annual Metric | Scored Entries | Mean Normalized Error | Carry-Forward Mean Normalized Error |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("metric_rows") or []:
        lines.append(
            f"| `{row.get('candidate_family')}` | `{row.get('metric_name')}` | "
            f"`{row.get('scored_candidate_entry_count')}` | `{row.get('candidate_mean_norm_error')}` | "
            f"`{row.get('carry_forward_mean_norm_error')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R83 quarterly emission bridge audit.")
    parser.add_argument("--run-id", default=R83_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument(
        "--candidate-family",
        action="append",
        default=None,
        help="Candidate family to audit. Repeat to audit more than one; defaults to the locked R11-28 quarterly reference.",
    )
    args = parser.parse_args()
    run_r83_quarterly_emission_bridge_audit(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        candidate_families=tuple(args.candidate_family) if args.candidate_family else R83_DEFAULT_CANDIDATE_FAMILIES,
    )


if __name__ == "__main__":
    _main()
