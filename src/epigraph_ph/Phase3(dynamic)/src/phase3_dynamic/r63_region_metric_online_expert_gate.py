from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .metrics import quarter_sort_key
from .r48_subnational_proxy_champion_contract import (
    COUNT_METRICS,
    _finite_float,
    _load_r44_rows,
    _projection_cascade,
    _rows_by_period_region,
)
from .r54_national_total_regional_adapter import (
    R48_DEFAULT_REPORT,
    R49_DEFAULT_REPORT,
    R51_DEFAULT_REPORT,
    _base_prediction_rows,
    _load_report,
)
from .r55_adapter_split_stability_gate import (
    _comparison_rows as _split_comparison_rows,
    _gate as _split_stability_gate,
)
from .r60_regional_experiment_queue import (
    CARRY_FORWARD_FAMILY,
    REFERENCE_FAMILY,
    R54_DEFAULT_REPORT,
    R59_DEFAULT_REPORT,
    R60_DEFAULT_REPORT,
    _candidate_mean_gate,
    _prediction_rows_for_spec,
)
from .r61_split_risk_regional_router import _r60_nonregression
from .r62_leakage_expert_student_gate import (
    _read_report,
    _rename_family,
    _score_candidate,
    _write_csv,
)
from .runtime import ensure_dir, write_json
from .r11_sparse_state_space import _generated_at


R63_SCHEMA_VERSION = "phase3_dynamic.r63_region_metric_online_expert_gate.v1"
R63_RUN_ID = "p3d-r63-region-metric-online-expert-gate-20260503-s00"
R63_ORACLE_FAMILY = "region_metric_leakage_oracle_not_promotable"
R63_STUDENT_FAMILY = "region_metric_online_expert_student"


def _target_value(rows_by_key: dict[tuple[str, str], dict[str, Any]], *, period: str, region: str, metric: str) -> float | None:
    return _finite_float((rows_by_key.get((period, region)) or {}).get(metric))


def _candidate_error_rows(
    source_rows: list[dict[str, Any]],
    rows_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        family = str(row.get("candidate_family") or "")
        period = str(row.get("holdout_period") or "")
        region = str(row.get("region") or "")
        if not family or not period or not region:
            continue
        for metric in COUNT_METRICS:
            predicted = _finite_float(row.get(metric))
            target = _target_value(rows_by_key, period=period, region=region, metric=metric)
            if predicted is None or target is None:
                continue
            absolute_error = abs(float(predicted) - float(target))
            scale = max(abs(float(target)), 1.0)
            rows.append(
                {
                    "candidate_family": family,
                    "holdout_period": period,
                    "region": region,
                    "metric_name": metric,
                    "predicted_value": float(predicted),
                    "target_value": float(target),
                    "absolute_error": float(absolute_error),
                    "normalized_absolute_error": float(absolute_error / scale),
                }
            )
    return rows


def _error_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    return {
        (
            str(row.get("candidate_family") or ""),
            str(row.get("holdout_period") or ""),
            str(row.get("region") or ""),
            str(row.get("metric_name") or ""),
        ): dict(row)
        for row in rows
    }


def _region_metric_oracle_prediction_rows(
    source_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    *,
    candidate_family: str = R63_ORACLE_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in source_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in error_rows:
        key = (str(row.get("holdout_period") or ""), str(row.get("region") or ""), str(row.get("metric_name") or ""))
        grouped.setdefault(key, []).append(dict(row))
    output: dict[tuple[str, str], dict[str, Any]] = {}
    selection_rows: list[dict[str, Any]] = []
    for (period, region, metric), candidates in sorted(grouped.items(), key=lambda item: (quarter_sort_key(item[0][0]), item[0][1], item[0][2])):
        candidates.sort(
            key=lambda row: (
                float(row.get("normalized_absolute_error") or float("inf")),
                float(row.get("absolute_error") or float("inf")),
                str(row.get("candidate_family") or ""),
            )
        )
        selected = candidates[0]
        source_family = str(selected.get("candidate_family") or "")
        source = indexed.get((source_family, period, region)) or {}
        value = _finite_float(source.get(metric))
        if value is None:
            continue
        output.setdefault(
            (period, region),
            {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
                "leakage_status": "same_region_metric_holdout_oracle_not_promotable",
            },
        )
        output[(period, region)][metric] = float(value)
        output[(period, region)][f"{metric}_source_family"] = source_family
        selection_rows.append(
            {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
                "metric_name": metric,
                "selected_source_family": source_family,
                "selected_normalized_absolute_error": selected.get("normalized_absolute_error"),
                "selected_absolute_error": selected.get("absolute_error"),
                "candidate_count": len(candidates),
                "leakage_status": "same_region_metric_holdout_oracle_not_promotable",
            }
        )
    return [_projection_cascade(row) for row in output.values()], selection_rows


def _region_metric_student_prediction_rows(
    source_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    *,
    default_family: str,
    candidate_family: str = R63_STUDENT_FAMILY,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {
        (str(row.get("candidate_family") or ""), str(row.get("holdout_period") or ""), str(row.get("region") or "")): dict(row)
        for row in source_rows
        if row.get("candidate_family") and row.get("holdout_period") and row.get("region")
    }
    error_by_key = _error_index(error_rows)
    periods = sorted({period for _family, period, _region in indexed}, key=quarter_sort_key)
    families = sorted({family for family, _period, _region in indexed})
    output_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for period in periods:
        regions = sorted({region for family, candidate_period, region in indexed if candidate_period == period and family == default_family})
        if not regions:
            regions = sorted({region for _family, candidate_period, region in indexed if candidate_period == period})
        for region in regions:
            output: dict[str, Any] = {
                "candidate_family": candidate_family,
                "holdout_period": period,
                "region": region,
                "leakage_status": "online_student_uses_prior_region_metric_errors_only",
            }
            for metric in COUNT_METRICS:
                scored: list[tuple[float, float, int, str]] = []
                for family in families:
                    prior_errors = [
                        float((error_by_key.get((family, prior_period, region, metric)) or {}).get("normalized_absolute_error"))
                        for prior_period in periods
                        if quarter_sort_key(prior_period) < quarter_sort_key(period)
                        and _finite_float((error_by_key.get((family, prior_period, region, metric)) or {}).get("normalized_absolute_error")) is not None
                    ]
                    if not prior_errors:
                        continue
                    scored.append(
                        (
                            float(np.mean(np.asarray(prior_errors, dtype=np.float64))),
                            float(np.max(np.asarray(prior_errors, dtype=np.float64))),
                            -len(prior_errors),
                            family,
                        )
                    )
                if scored:
                    scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                    selected_family = scored[0][3]
                    reason = "minimum_prior_region_metric_error"
                    prior_count = -int(scored[0][2])
                else:
                    selected_family = default_family
                    reason = "default_no_prior_region_metric_error"
                    prior_count = 0
                source = indexed.get((selected_family, period, region)) or {}
                value = _finite_float(source.get(metric))
                if value is None:
                    source = indexed.get((default_family, period, region)) or {}
                    value = _finite_float(source.get(metric))
                    selected_family = default_family
                    reason = f"{reason}_fallback_default"
                if value is None:
                    continue
                output[metric] = float(value)
                output[f"{metric}_source_family"] = selected_family
                selection_rows.append(
                    {
                        "candidate_family": candidate_family,
                        "holdout_period": period,
                        "region": region,
                        "metric_name": metric,
                        "selected_source_family": selected_family,
                        "selection_reason": reason,
                        "prior_region_metric_error_count": prior_count,
                        "leakage_training_status": (
                            "no_prior_error_default_only"
                            if prior_count == 0
                            else "uses_prior_region_metric_errors_only"
                        ),
                    }
                )
            output_rows.append(_projection_cascade(output))
    return output_rows, selection_rows


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("online_expert_gate") or {})
    lines = [
        "# Phase 3 R63 Region-Metric Online Expert Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Oracle regional / mass / share: `{gate.get('oracle_mean_regional_normalized_absolute_error')}` / `{gate.get('oracle_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('oracle_mean_regional_share_half_l1_error')}`",
        f"- Student regional / mass / share: `{gate.get('student_mean_regional_normalized_absolute_error')}` / `{gate.get('student_mean_aggregate_mass_normalized_absolute_error')}` / `{gate.get('student_mean_regional_share_half_l1_error')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Contract",
        "",
        str(gate.get("contract") or ""),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r63_region_metric_online_expert_gate(
    *,
    run_id: str = R63_RUN_ID,
    r54_report_path: Path | None = None,
    r59_report_path: Path | None = None,
    r60_report_path: Path | None = None,
    r48_report_path: Path | None = None,
    r49_report_path: Path | None = None,
    r51_report_path: Path | None = None,
) -> dict[str, Any]:
    r54_path = Path(r54_report_path) if r54_report_path is not None else R54_DEFAULT_REPORT
    r59_path = Path(r59_report_path) if r59_report_path is not None else R59_DEFAULT_REPORT
    r60_path = Path(r60_report_path) if r60_report_path is not None else R60_DEFAULT_REPORT
    r54 = _read_report(r54_path)
    r59 = _read_report(r59_path)
    r60 = _read_report(r60_path)
    r48_path = Path(r48_report_path or r54.get("r48_report_path") or R48_DEFAULT_REPORT)
    r49_path = Path(r49_report_path or r54.get("r49_report_path") or R49_DEFAULT_REPORT)
    r51_path = Path(r51_report_path or r54.get("r51_report_path") or R51_DEFAULT_REPORT)
    r48 = _load_report(r48_path)
    r49 = _load_report(r49_path)
    r51 = _load_report(r51_path)
    r44_path = Path(str(r54.get("r44_report_path") or r48.get("r44_report_path") or r51.get("r44_report_path") or ""))
    regional_rows, _r44_report = _load_r44_rows(r44_path)
    rows_by_key = _rows_by_period_region(regional_rows)
    r54_promoted = str((r54.get("adapter_gate") or {}).get("promoted_candidate_family") or REFERENCE_FAMILY)
    source_rows = (
        _base_prediction_rows(r48, r49, r51)
        + [dict(row) for row in list(r54.get("adapter_prediction_rows") or [])]
        + [dict(row) for row in list(r59.get("prediction_rows") or [])]
    )
    r60_gate = dict(r60.get("queue_gate") or {})
    r60_best_id = str(r60_gate.get("best_experiment_id") or "")
    r60_best_spec = next((dict(spec) for spec in list(r60.get("experiment_specs") or []) if str(spec.get("experiment_id") or "") == r60_best_id), {})
    r60_best_rows, _weights = _prediction_rows_for_spec(
        r60_best_spec,
        source_rows=source_rows,
        rows_by_key=rows_by_key,
        combined_rows=[dict(row) for row in list(r54.get("combined_candidate_table") or [])],
        r54_promoted_family=r54_promoted,
        r59_report=r59,
    )
    r60_best_family = str(r60_gate.get("best_candidate_family") or (r60_best_rows[0].get("candidate_family") if r60_best_rows else "") or r54_promoted)
    source_rows = source_rows + _rename_family(r60_best_rows, r60_best_family)
    error_rows = _candidate_error_rows(source_rows, rows_by_key)
    oracle_rows, oracle_selection_rows = _region_metric_oracle_prediction_rows(source_rows, error_rows)
    student_rows, student_selection_rows = _region_metric_student_prediction_rows(
        source_rows,
        error_rows,
        default_family=r60_best_family,
    )
    carry = next((dict(row) for row in list(r54.get("combined_candidate_table") or []) if str(row.get("candidate_family") or "") == CARRY_FORWARD_FAMILY), {})
    reference = next((dict(row) for row in list(r54.get("combined_candidate_table") or []) if str(row.get("candidate_family") or "") == REFERENCE_FAMILY), {})
    candidate_score_rows: list[dict[str, Any]] = []
    split_comparison_rows: list[dict[str, Any]] = []
    for family, rows, candidate_type in [
        (R63_ORACLE_FAMILY, oracle_rows, "same_split_region_metric_oracle"),
        (R63_STUDENT_FAMILY, student_rows, "blocked_time_region_metric_student"),
    ]:
        split_rows, coherence_rows, score = _score_candidate(rows_by_key, rows)
        comparison = _split_comparison_rows(
            regional_score_rows=[dict(row) for row in list(r54.get("split_metric_score_rows") or [])] + split_rows,
            coherence_rows=[dict(row) for row in list(r54.get("coherence_rows") or [])] + coherence_rows,
            promoted_family=family,
        )
        split_gate = _split_stability_gate(comparison)
        split_comparison_rows.extend(comparison)
        if family == R63_ORACLE_FAMILY:
            mean_pass, mean_blockers = False, ["same_split_region_metric_oracle_not_promotable"]
            r60_pass, r60_blockers = False, ["same_split_region_metric_oracle_not_promotable"]
        else:
            mean_pass, mean_blockers = _candidate_mean_gate(
                score,
                carry=carry,
                reference=reference,
                r54_gate=dict(r54.get("adapter_gate") or {}),
                r59_gate=dict(r59.get("ensemble_gate") or {}),
            )
            r60_pass, r60_blockers = _r60_nonregression(score, r60_gate)
        candidate_score_rows.append(
            {
                "candidate_family": family,
                "candidate_type": candidate_type,
                "mean_regional_normalized_absolute_error": score.get("mean_regional_normalized_absolute_error"),
                "mean_aggregate_mass_normalized_absolute_error": score.get("mean_aggregate_mass_normalized_absolute_error"),
                "mean_regional_share_half_l1_error": score.get("mean_regional_share_half_l1_error"),
                "worst_regional_normalized_absolute_error": score.get("worst_regional_normalized_absolute_error"),
                "mean_gate_status": "pass" if mean_pass else "fail",
                "mean_gate_blockers": mean_blockers,
                "r60_nonregression_status": "pass" if r60_pass else "fail",
                "r60_nonregression_blockers": r60_blockers,
                "split_gate_status": split_gate.get("status"),
                "split_gate_failure_counts": split_gate.get("failure_counts"),
                "split_metric_count": split_gate.get("split_metric_count"),
            }
        )
    oracle_score = next(row for row in candidate_score_rows if row["candidate_family"] == R63_ORACLE_FAMILY)
    student_score = next(row for row in candidate_score_rows if row["candidate_family"] == R63_STUDENT_FAMILY)
    blockers: list[str] = []
    strict = (
        student_score.get("mean_gate_status") == "pass"
        and student_score.get("r60_nonregression_status") == "pass"
        and student_score.get("split_gate_status") == "strict_split_stable_adapter_promoted"
    )
    mean_safe = student_score.get("mean_gate_status") == "pass" and student_score.get("r60_nonregression_status") == "pass"
    if not strict:
        blockers.append("region_metric_student_not_strict_split_stable")
    if not mean_safe:
        blockers.append("region_metric_student_does_not_preserve_r60_mean_contract")
    status = (
        "region_metric_online_expert_student_promoted"
        if strict
        else ("region_metric_online_expert_student_mean_promoted_split_limited" if mean_safe else "region_metric_online_expert_diagnostic_only")
    )
    gate = {
        "status": status,
        "blockers": blockers,
        "oracle_mean_regional_normalized_absolute_error": oracle_score.get("mean_regional_normalized_absolute_error"),
        "oracle_mean_aggregate_mass_normalized_absolute_error": oracle_score.get("mean_aggregate_mass_normalized_absolute_error"),
        "oracle_mean_regional_share_half_l1_error": oracle_score.get("mean_regional_share_half_l1_error"),
        "student_mean_regional_normalized_absolute_error": student_score.get("mean_regional_normalized_absolute_error"),
        "student_mean_aggregate_mass_normalized_absolute_error": student_score.get("mean_aggregate_mass_normalized_absolute_error"),
        "student_mean_regional_share_half_l1_error": student_score.get("mean_regional_share_half_l1_error"),
        "student_split_gate_status": student_score.get("split_gate_status"),
        "r60_best_candidate_family": r60_best_family,
        "contract": (
            "R63 is an online expert-advice test. The region-metric oracle uses same-holdout targets and is never "
            "promotable. The student may choose region-metric experts only from prior observed region-metric losses; "
            "promotion requires the R60 mean contract and the R55 split-stability gate."
        ),
    }
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r63_region_metric_online_expert_gate_report.json"
    md_path = analysis_dir / "r63_region_metric_online_expert_gate_report.md"
    score_csv = analysis_dir / "r63_candidate_score_rows.csv"
    oracle_csv = analysis_dir / "r63_oracle_selection_rows.csv"
    student_csv = analysis_dir / "r63_student_selection_rows.csv"
    errors_csv = analysis_dir / "r63_candidate_error_rows.csv"
    comparison_csv = analysis_dir / "r63_split_comparison_rows.csv"
    report = {
        "schema_version": R63_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": status,
        "blockers": blockers,
        "verdict": (
            "R63 found a strict region-metric online expert student."
            if strict
            else (
                "R63 found a mean-promoted online expert student, but split-stability remains limited."
                if mean_safe
                else "R63 confirms strong region-metric oracle signal, but the online student does not preserve the R60 contract."
            )
        ),
        "r44_report_path": r44_path.as_posix(),
        "r48_report_path": r48_path.as_posix(),
        "r49_report_path": r49_path.as_posix(),
        "r51_report_path": r51_path.as_posix(),
        "r54_report_path": r54_path.as_posix(),
        "r59_report_path": r59_path.as_posix(),
        "r60_report_path": r60_path.as_posix(),
        "online_expert_gate": gate,
        "candidate_score_rows": candidate_score_rows,
        "oracle_selection_rows": oracle_selection_rows,
        "student_selection_rows": student_selection_rows,
        "split_comparison_rows": split_comparison_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "candidate_score_csv": score_csv.as_posix(),
            "oracle_selection_csv": oracle_csv.as_posix(),
            "student_selection_csv": student_csv.as_posix(),
            "candidate_error_csv": errors_csv.as_posix(),
            "split_comparison_csv": comparison_csv.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(score_csv, candidate_score_rows)
    _write_csv(oracle_csv, oracle_selection_rows)
    _write_csv(student_csv, student_selection_rows)
    _write_csv(errors_csv, error_rows)
    _write_csv(comparison_csv, split_comparison_rows)
    _write_markdown(md_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R63 region-metric online expert gate.")
    parser.add_argument("--run-id", default=R63_RUN_ID)
    parser.add_argument("--r54-report-path", default=None)
    parser.add_argument("--r59-report-path", default=None)
    parser.add_argument("--r60-report-path", default=None)
    parser.add_argument("--r48-report-path", default=None)
    parser.add_argument("--r49-report-path", default=None)
    parser.add_argument("--r51-report-path", default=None)
    args = parser.parse_args()
    run_r63_region_metric_online_expert_gate(
        run_id=str(args.run_id),
        r54_report_path=None if args.r54_report_path is None else Path(args.r54_report_path),
        r59_report_path=None if args.r59_report_path is None else Path(args.r59_report_path),
        r60_report_path=None if args.r60_report_path is None else Path(args.r60_report_path),
        r48_report_path=None if args.r48_report_path is None else Path(args.r48_report_path),
        r49_report_path=None if args.r49_report_path is None else Path(args.r49_report_path),
        r51_report_path=None if args.r51_report_path is None else Path(args.r51_report_path),
    )


if __name__ == "__main__":
    _main()
