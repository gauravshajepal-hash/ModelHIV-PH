from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from .data import build_observation_rows, sandbox_repo_root
from .metrics import quarter_sort_key, quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
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
from .runtime import ensure_dir, read_json, write_json


R89_SCHEMA_VERSION = "phase3_dynamic.r89_incidence_mortality_mechanism_support_gate.v1"
R89_RUN_ID = "p3d-r89-incidence-mortality-mechanism-support-gate-20260507-s00"
R89_FAMILY = "reported_death_bridge_and_incidence_support_gate"
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)


def _metric_support_summary(rows: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    years: Counter[int] = Counter()
    roles: Counter[str] = Counter()
    allowed: Counter[str] = Counter()
    semantics: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    count = 0
    for row in rows:
        value = _finite_float(row.get(metric_name))
        if value is None:
            continue
        count += 1
        quarter = str(row.get("quarter") or "")
        if "-Q" in quarter:
            years[quarter_year(quarter)] += 1
        provenance = dict((row.get("metric_provenance") or {}).get(metric_name) or {})
        roles[str(provenance.get("observation_role") or "")] += 1
        allowed[str(provenance.get("allowed_use") or "")] += 1
        semantics[str(provenance.get("measurement_semantics") or "")] += 1
        sources[str(provenance.get("source_id") or "")] += 1
    return {
        "metric_name": metric_name,
        "count": count,
        "year_count": len(years),
        "years": dict(sorted(years.items())),
        "observation_roles": dict(roles.most_common()),
        "allowed_use": dict(allowed.most_common()),
        "measurement_semantics": dict(semantics.most_common()),
        "top_sources": dict(sources.most_common(10)),
    }


def _reported_death_annual_proxy(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    by_year: dict[int, list[float]] = defaultdict(list)
    provenance_by_year: dict[int, Counter[str]] = defaultdict(Counter)
    for row in rows:
        value = _finite_float(row.get("deaths_reported_period"))
        if value is None:
            continue
        quarter = str(row.get("quarter") or "")
        if "-Q" not in quarter:
            continue
        year = quarter_year(quarter)
        by_year[int(year)].append(max(float(value), 0.0))
        provenance = dict((row.get("metric_provenance") or {}).get("deaths_reported_period") or {})
        provenance_by_year[int(year)][str(provenance.get("source_id") or "")] += 1
    output: dict[int, dict[str, Any]] = {}
    for year, values in sorted(by_year.items()):
        quarter_count = len(values)
        annualized = None if quarter_count <= 0 else float(sum(values) * 4.0 / float(quarter_count))
        output[int(year)] = {
            "year": int(year),
            "observed_quarter_count": quarter_count,
            "reported_deaths_sum": float(sum(values)),
            "annualized_reported_deaths": annualized,
            "source_counts": dict(provenance_by_year[int(year)].most_common()),
            "contract": "annualized direct reported-death proxy = sum(observed direct reported death quarters) * 4 / observed quarter count",
        }
    return output


def _train_mortality_bridge_model(
    *,
    rows: list[dict[str, Any]],
    train_end_year: int,
    reported_proxy_by_year: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    q4_rows = [
        dict(row)
        for row in rows
        if str(row.get("quarter") or "").endswith("-Q4")
        and quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
    ]
    pairs: list[dict[str, Any]] = []
    for row in q4_rows:
        year = quarter_year(str(row.get("quarter") or ""))
        aids_deaths = _finite_float(row.get("annual_aids_deaths"))
        proxy = _finite_float((reported_proxy_by_year.get(int(year)) or {}).get("annualized_reported_deaths"))
        if aids_deaths is None or proxy is None or proxy <= 0.0:
            continue
        pairs.append(
            {
                "year": int(year),
                "annualized_reported_deaths": float(proxy),
                "annual_aids_deaths": float(aids_deaths),
                "aids_to_reported_ratio": float(aids_deaths) / float(proxy),
            }
        )
    if not pairs:
        return {
            "status": "not_estimable",
            "reason": "no_train_reported_death_to_external_aids_death_pairs",
            "pair_count": 0,
        }
    ratios = [float(row["aids_to_reported_ratio"]) for row in pairs]
    last_reported_year = max(int(year) for year in reported_proxy_by_year if int(year) <= int(train_end_year))
    last_reported_proxy = float(reported_proxy_by_year[last_reported_year]["annualized_reported_deaths"])
    return {
        "status": "completed",
        "pair_count": len(pairs),
        "first_pair_year": int(pairs[0]["year"]),
        "last_pair_year": int(pairs[-1]["year"]),
        "median_aids_to_reported_ratio": float(median(ratios)),
        "last_reported_proxy_year": int(last_reported_year),
        "last_reported_proxy": last_reported_proxy,
        "pairs": pairs,
        "contract": "AIDS-death forecast = last train direct reported-death annual proxy * median train ratio(external AIDS deaths / direct reported deaths)",
    }


def _mortality_bridge_score_rows(
    rows: list[dict[str, Any]],
    *,
    start_year: int,
    end_year: int,
    min_train_years: int,
    horizons: tuple[int, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    q4_rows = [dict(row) for row in rows if str(row.get("quarter") or "").endswith("-Q4")]
    reported_proxy_by_year = _reported_death_annual_proxy(rows)
    score_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for split in _rolling_annual_splits(q4_rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizons=horizons):
        train_end_year = int(split["train_end_year"])
        horizon = int(split["horizon_years"])
        holdout_years = [int(year) for year in split["holdout_years"]]
        raw_train_rows = [dict(row) for row in rows if row.get("quarter") and quarter_year(str(row.get("quarter") or "")) <= train_end_year]
        model = _train_mortality_bridge_model(
            rows=rows,
            train_end_year=train_end_year,
            reported_proxy_by_year=reported_proxy_by_year,
        )
        model_rows.append(
            {
                "candidate_family": R89_FAMILY,
                "horizon_years": horizon,
                "train_end_year": train_end_year,
                "holdout_years": holdout_years,
                **{key: value for key, value in model.items() if key != "pairs"},
            }
        )
        ratio = _finite_float(model.get("median_aids_to_reported_ratio"))
        last_proxy = _finite_float(model.get("last_reported_proxy"))
        candidate_value = None if ratio is None or last_proxy is None else float(ratio) * float(last_proxy)
        for holdout_row in q4_rows:
            holdout_year = quarter_year(str(holdout_row.get("quarter") or ""))
            if int(holdout_year) not in set(holdout_years):
                continue
            if _finite_float(holdout_row.get("annual_aids_deaths")) is None:
                continue
            carry_value = _annual_carry_forward_prediction(raw_train_rows, holdout_row, "annual_aids_deaths")
            score = _annual_score_row(
                family=R89_FAMILY,
                horizon=horizon,
                train_end_year=train_end_year,
                holdout_years=holdout_years,
                holdout_row=holdout_row,
                metric_name="annual_aids_deaths",
                candidate_value=None if candidate_value is None else float(candidate_value),
                carry_value=None if carry_value is None else float(carry_value),
                scale=_annual_target_scale(raw_train_rows, "annual_aids_deaths"),
            )
            score["training_use"] = "train_origin_reported_death_bridge_only"
            score["prediction_status"] = "reported_death_bridge_scored" if candidate_value is not None else "reported_death_bridge_not_estimable"
            score_rows.append(score)
    return score_rows, model_rows


def _gate(
    *,
    support_rows: list[dict[str, Any]],
    mortality_family_rows: list[dict[str, Any]],
    mortality_score_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[str] = []
    if not target_rows:
        blockers.append("no_public_annual_target_rows")
    support_by_metric = {str(row.get("metric_name") or ""): dict(row) for row in support_rows}
    direct_incidence_count = int((support_by_metric.get("incident_infections_period") or {}).get("count") or 0)
    weak_annual_incidence_count = int((support_by_metric.get("annual_new_infections") or {}).get("count") or 0)
    direct_death_count = int((support_by_metric.get("deaths_reported_period") or {}).get("count") or 0)
    if direct_incidence_count <= 0:
        blockers.append("direct_incidence_process_support_absent")
    if direct_death_count <= 0:
        blockers.append("direct_reported_death_support_absent")
    best_mortality = dict(mortality_family_rows[0]) if mortality_family_rows else {}
    mortality_candidate_mean = _finite_float(best_mortality.get("candidate_mean_norm_error"))
    mortality_carry_mean = _finite_float(best_mortality.get("carry_forward_mean_norm_error"))
    mortality_candidate_coverage = _finite_float(best_mortality.get("candidate_interval_coverage"))
    mortality_carry_coverage = _finite_float(best_mortality.get("carry_forward_interval_coverage"))
    if mortality_candidate_mean is None or mortality_carry_mean is None:
        blockers.append("reported_death_bridge_not_evaluable")
    elif mortality_candidate_mean >= mortality_carry_mean:
        blockers.append("reported_death_bridge_not_better_than_carry_forward")
    if mortality_candidate_coverage is not None and mortality_carry_coverage is not None and mortality_candidate_coverage < mortality_carry_coverage:
        blockers.append("reported_death_bridge_interval_coverage_worse_than_carry_forward")
    leakage_rows = [
        row
        for row in mortality_score_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("annual_validation_role_leakage")
    return {
        "status": "incidence_mortality_mechanism_support_ready" if not blockers else "incidence_mortality_mechanism_support_diagnostic_only",
        "blockers": blockers,
        "direct_incidence_process_support_count": direct_incidence_count,
        "weak_annual_incidence_validation_count": weak_annual_incidence_count,
        "direct_reported_death_support_count": direct_death_count,
        "reported_death_bridge_candidate_mean_norm_error": mortality_candidate_mean,
        "reported_death_bridge_carry_forward_mean_norm_error": mortality_carry_mean,
        "reported_death_bridge_candidate_minus_carry_forward": None
        if mortality_candidate_mean is None or mortality_carry_mean is None
        else float(mortality_candidate_mean - mortality_carry_mean),
        "reported_death_bridge_candidate_interval_coverage": mortality_candidate_coverage,
        "reported_death_bridge_carry_forward_interval_coverage": mortality_carry_coverage,
        "contract": (
            "R89 is an admissibility gate for true incidence/death process claims. Incidence requires direct or process-level "
            "incidence support; mortality requires the direct reported-death bridge to beat carry-forward against validation-only "
            "annual AIDS-death targets. Failure blocks raw mechanism claims without invalidating R86/R88 annual readout wins."
        ),
    }


def run_r89_incidence_mortality_mechanism_support_gate(
    *,
    run_id: str = R89_RUN_ID,
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
    support_rows = [
        _metric_support_summary(rows, "incident_infections_period"),
        _metric_support_summary(rows, "annual_new_infections"),
        _metric_support_summary(rows, "deaths_reported_period"),
        _metric_support_summary(rows, "annual_aids_deaths"),
    ]
    reported_proxy_by_year = _reported_death_annual_proxy(rows)
    reported_proxy_rows = list(reported_proxy_by_year.values())
    mortality_score_rows, mortality_model_rows = _mortality_bridge_score_rows(
        rows,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
        horizons=horizons,
    )
    mortality_metric_rows = _score_summary_by_fields(mortality_score_rows, group_fields=("candidate_family", "metric_name"))
    mortality_horizon_rows = _score_summary_by_fields(mortality_score_rows, group_fields=("candidate_family", "horizon_years"))
    mortality_family_rows = _score_summary_by_fields(mortality_score_rows, group_fields=("candidate_family",))
    gate = _gate(
        support_rows=support_rows,
        mortality_family_rows=mortality_family_rows,
        mortality_score_rows=mortality_score_rows,
        target_rows=target_rows,
    )
    report_path = analysis_dir / "r89_incidence_mortality_mechanism_support_gate_report.json"
    markdown_path = analysis_dir / "r89_incidence_mortality_mechanism_support_gate_report.md"
    support_csv = analysis_dir / "r89_support_rows.csv"
    reported_proxy_csv = analysis_dir / "r89_reported_death_proxy_rows.csv"
    mortality_score_csv = analysis_dir / "r89_mortality_score_rows.csv"
    mortality_model_csv = analysis_dir / "r89_mortality_model_rows.csv"
    mortality_metric_csv = analysis_dir / "r89_mortality_metric_rows.csv"
    mortality_horizon_csv = analysis_dir / "r89_mortality_horizon_rows.csv"
    mortality_family_csv = analysis_dir / "r89_mortality_family_rows.csv"
    report = {
        "schema_version": R89_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "incidence_mortality_mechanism_support_gate": gate,
        "candidate_family": R89_FAMILY,
        "support_rows": support_rows,
        "reported_death_proxy_rows": reported_proxy_rows,
        "mortality_score_rows": mortality_score_rows,
        "mortality_model_rows": mortality_model_rows,
        "mortality_metric_rows": mortality_metric_rows,
        "mortality_horizon_rows": mortality_horizon_rows,
        "mortality_family_rows": mortality_family_rows,
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
            "support_rows_csv": support_csv.as_posix(),
            "reported_death_proxy_rows_csv": reported_proxy_csv.as_posix(),
            "mortality_score_rows_csv": mortality_score_csv.as_posix(),
            "mortality_model_rows_csv": mortality_model_csv.as_posix(),
            "mortality_metric_rows_csv": mortality_metric_csv.as_posix(),
            "mortality_horizon_rows_csv": mortality_horizon_csv.as_posix(),
            "mortality_family_rows_csv": mortality_family_csv.as_posix(),
        },
    }
    _write_csv(support_csv, support_rows)
    _write_csv(reported_proxy_csv, reported_proxy_rows)
    _write_csv(mortality_score_csv, mortality_score_rows)
    _write_csv(mortality_model_csv, mortality_model_rows)
    _write_csv(mortality_metric_csv, mortality_metric_rows)
    _write_csv(mortality_horizon_csv, mortality_horizon_rows)
    _write_csv(mortality_family_csv, mortality_family_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("incidence_mortality_mechanism_support_gate") or {})
    lines = [
        "# Phase 3 R89 Incidence/Mortality Mechanism Support Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        f"- Direct incidence support count: `{gate.get('direct_incidence_process_support_count')}`",
        f"- Weak annual incidence validation count: `{gate.get('weak_annual_incidence_validation_count')}`",
        f"- Direct reported death support count: `{gate.get('direct_reported_death_support_count')}`",
        f"- Mortality bridge mean error: `{gate.get('reported_death_bridge_candidate_mean_norm_error')}`",
        f"- Mortality carry-forward mean error: `{gate.get('reported_death_bridge_carry_forward_mean_norm_error')}`",
        "",
        "## Support Rows",
        "",
        "| Metric | Count | Year Count | Roles | Allowed Use |",
        "|---|---:|---:|---|---|",
    ]
    for row in report.get("support_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | `{row.get('count')}` | `{row.get('year_count')}` | "
            f"`{row.get('observation_roles')}` | `{row.get('allowed_use')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R89 incidence/mortality mechanism support gate.")
    parser.add_argument("--run-id", default=R89_RUN_ID)
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--r69-report-path", default=None)
    args = parser.parse_args()
    run_r89_incidence_mortality_mechanism_support_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
    )


if __name__ == "__main__":
    _main()
