from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .metrics import quarter_year
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    R10_COMPARABLE_METRICS,
    _apply_back_half_rate_process,
    _finite_float,
    _fit_back_half_rate_process,
    _generated_at,
    _project_prediction_row,
    _r34_art_ratio_prediction,
    _r34_metric_policy_prediction,
)
from .r28_r10_contract_lineage_audit import _collect_r10_records
from .r29_strict_ledger_matched_r10_gate import (
    R29_RUN_ID,
    _is_program_lineage,
    _reference_by_scope_horizon,
    _strict_reference_rows,
    run_r29_strict_ledger_matched_r10_gate,
)
from .runtime import ensure_dir, read_json, write_json


R37_SCHEMA_VERSION = "phase3_dynamic.r37_fixed_policy_strict_scan.v1"
R37_RUN_ID = "p3d-r37-fixed-policy-strict-scan-20260502-s00"
R37_DIAGNOSED_POLICIES: tuple[str, ...] = ("carry_forward", "positive_velocity", "median_velocity")
R37_FLOW_POLICIES: tuple[str, ...] = ("carry_forward", "positive_velocity", "median_velocity")
R37_ART_POLICIES: tuple[str, ...] = (
    "direct_positive_velocity",
    "direct_median_velocity",
    "diagnosed_ratio_positive_velocity__rate_logit_velocity",
    "diagnosed_ratio_positive_velocity__rate_logit_trend",
    "diagnosed_ratio_median_velocity__rate_logit_velocity",
    "diagnosed_ratio_median_velocity__rate_logit_trend",
)


def _candidate_specs() -> list[dict[str, str]]:
    specs: list[dict[str, str]] = []
    for diagnosed_policy in R37_DIAGNOSED_POLICIES:
        for flow_policy in R37_FLOW_POLICIES:
            for art_policy in R37_ART_POLICIES:
                candidate_id = (
                    f"diag-{diagnosed_policy}"
                    f"__flow-{flow_policy}"
                    f"__art-{art_policy}"
                )
                specs.append(
                    {
                        "candidate_id": candidate_id,
                        "diagnosed_policy": diagnosed_policy,
                        "flow_policy": flow_policy,
                        "art_policy": art_policy,
                    }
                )
    return specs


def _source_rows_by_quarter(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("quarter") or ""): dict(row)
        for row in rows
        if row.get("quarter")
    }


def _record_scope(record: dict[str, Any], scope: str) -> bool:
    if bool(record.get("unknown_provenance")):
        return False
    if scope == "all_mapped":
        return True
    if scope == "program_mapped":
        return _is_program_lineage(record)
    raise ValueError(f"Unknown R37 scope: {scope}")


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record.get("unknown_provenance")):
            continue
        grouped[(int(record.get("train_end_year") or 0), int(record.get("horizon_years") or 0))].append(dict(record))
    return grouped


def _predict_candidate_row(
    *,
    train_rows: list[dict[str, Any]],
    holdout_row: dict[str, Any],
    back_half_process: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    quarter = str(holdout_row.get("quarter") or "")
    diagnosed = _r34_metric_policy_prediction(
        train_rows,
        "diagnosed_plhiv",
        quarter,
        str(candidate["diagnosed_policy"]),
    )
    flow = _r34_metric_policy_prediction(
        train_rows,
        "new_diagnosed_cases_period",
        quarter,
        str(candidate["flow_policy"]),
    )
    art = None
    if diagnosed is not None:
        art = _r34_art_ratio_prediction(
            train_rows,
            quarter,
            str(candidate["art_policy"]),
            output_diagnosed_value=float(diagnosed),
        )
    row = {
        "quarter": quarter,
        "diagnosed_plhiv": diagnosed,
        "alive_on_art": art,
        "new_diagnosed_cases_period": flow,
    }
    return _apply_back_half_rate_process(_project_prediction_row(row), holdout_row, back_half_process)


def _prediction_rows_by_candidate(
    *,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    candidates: list[dict[str, str]],
) -> dict[str, dict[tuple[int, int, str], dict[str, Any]]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    output: dict[str, dict[tuple[int, int, str], dict[str, Any]]] = {
        str(candidate["candidate_id"]): {}
        for candidate in candidates
    }
    for (train_end_year, horizon), grouped_records in sorted(_group_records(records).items()):
        train_rows = [
            dict(row)
            for row in source_rows
            if quarter_year(str(row.get("quarter") or "")) <= int(train_end_year)
        ]
        holdout_quarters = sorted(
            {str(record.get("quarter") or "") for record in grouped_records if record.get("quarter")}
        )
        holdout_rows = [
            dict(source_by_quarter[quarter])
            for quarter in holdout_quarters
            if quarter in source_by_quarter
        ]
        if not train_rows or not holdout_rows:
            continue
        back_half_process = _fit_back_half_rate_process(train_rows)
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            for holdout_row in holdout_rows:
                quarter = str(holdout_row.get("quarter") or "")
                if not quarter:
                    continue
                output[candidate_id][(int(train_end_year), int(horizon), quarter)] = _predict_candidate_row(
                    train_rows=train_rows,
                    holdout_row=holdout_row,
                    back_half_process=back_half_process,
                    candidate=candidate,
                )
    return output


def _score_candidate_records(
    *,
    candidate: dict[str, str],
    records: list[dict[str, Any]],
    prediction_rows: dict[tuple[int, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        metric_name = str(record.get("metric_name") or "")
        if metric_name not in R10_COMPARABLE_METRICS:
            continue
        key = (
            int(record.get("train_end_year") or 0),
            int(record.get("horizon_years") or 0),
            str(record.get("quarter") or ""),
        )
        prediction = prediction_rows.get(key, {})
        value = _finite_float(prediction.get(metric_name))
        target = _finite_float(record.get("target_value"))
        scale = _finite_float(record.get("scale"))
        if value is None or target is None or scale is None:
            continue
        scale = max(float(scale), float(np.finfo(np.float32).eps))
        error = abs(float(value) - float(target)) / scale
        rows.append(
            {
                **{
                    key_name: record.get(key_name)
                    for key_name in (
                        "horizon_years",
                        "train_end_year",
                        "target_year",
                        "quarter",
                        "metric_name",
                        "source_lineage",
                        "unknown_provenance",
                        "target_value",
                        "scale",
                        "r10_norm_error",
                        "carry_forward_norm_error",
                    )
                },
                **candidate,
                "candidate_value": float(value),
                "candidate_norm_error": float(error),
                "candidate_minus_r10_norm_error": float(error - float(record["r10_norm_error"])),
                "candidate_minus_carry_forward_norm_error": float(error - float(record["carry_forward_norm_error"])),
            }
        )
    return rows


def _mean_error(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(row[key])
        for row in rows
        if _finite_float(row.get(key)) is not None
    ]
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _candidate_gate_rows(
    *,
    scored_rows_by_candidate: dict[str, list[dict[str, Any]]],
    strict_reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference = _reference_by_scope_horizon(strict_reference_rows)
    rows: list[dict[str, Any]] = []
    required = (
        ("all_mapped", 1),
        ("all_mapped", 3),
        ("all_mapped", 5),
        ("program_mapped", 3),
        ("program_mapped", 5),
    )
    for candidate_id, candidate_rows in sorted(scored_rows_by_candidate.items()):
        first = candidate_rows[0] if candidate_rows else {}
        for scope, horizon in required:
            scoped = [
                row
                for row in candidate_rows
                if int(row.get("horizon_years") or 0) == int(horizon)
                and _record_scope(row, scope)
            ]
            ref = reference.get((scope, int(horizon)), {})
            candidate_mean = _mean_error(scoped, "candidate_norm_error")
            matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
            carry = _mean_error(scoped, "carry_forward_norm_error")
            blockers: list[str] = []
            if not scoped:
                blockers.append("no_scored_rows")
            if candidate_mean is None or matched_r10 is None:
                blockers.append("candidate_or_strict_r10_missing")
            elif candidate_mean >= matched_r10:
                blockers.append("candidate_not_better_than_strict_mapped_r10")
            if candidate_mean is None or carry is None:
                blockers.append("candidate_or_carry_missing")
            elif candidate_mean >= carry:
                blockers.append("candidate_not_better_than_carry_forward")
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "diagnosed_policy": first.get("diagnosed_policy"),
                    "flow_policy": first.get("flow_policy"),
                    "art_policy": first.get("art_policy"),
                    "scope": scope,
                    "horizon_years": int(horizon),
                    "entry_count": len(scoped),
                    "candidate_mean_mae": candidate_mean,
                    "strict_matched_r10_mean_mae": matched_r10,
                    "candidate_minus_strict_r10": None if candidate_mean is None or matched_r10 is None else float(candidate_mean - matched_r10),
                    "carry_forward_mean_mae": carry,
                    "candidate_minus_carry_forward": None if candidate_mean is None or carry is None else float(candidate_mean - carry),
                    "status": "pass" if not blockers else "fail",
                    "blockers": blockers,
                }
            )
    return rows


def _candidate_summary_rows(gate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        grouped[str(row.get("candidate_id") or "")].append(row)
    output: list[dict[str, Any]] = []
    for candidate_id, rows in sorted(grouped.items()):
        first = rows[0] if rows else {}
        deltas = [
            float(row["candidate_minus_strict_r10"])
            for row in rows
            if _finite_float(row.get("candidate_minus_strict_r10")) is not None
        ]
        statuses = [str(row.get("status") or "") for row in rows]
        output.append(
            {
                "candidate_id": candidate_id,
                "diagnosed_policy": first.get("diagnosed_policy"),
                "flow_policy": first.get("flow_policy"),
                "art_policy": first.get("art_policy"),
                "pass_count": sum(1 for status in statuses if status == "pass"),
                "required_count": len(rows),
                "max_candidate_minus_strict_r10": None if not deltas else float(max(deltas)),
                "mean_candidate_minus_strict_r10": None if not deltas else float(np.mean(np.asarray(deltas, dtype=np.float64))),
                "all_required_pass": bool(rows and all(status == "pass" for status in statuses)),
            }
        )
    output.sort(
        key=lambda row: (
            -int(row.get("pass_count") or 0),
            float(row.get("max_candidate_minus_strict_r10") if row.get("max_candidate_minus_strict_r10") is not None else float("inf")),
            float(row.get("mean_candidate_minus_strict_r10") if row.get("mean_candidate_minus_strict_r10") is not None else float("inf")),
            str(row.get("candidate_id") or ""),
        )
    )
    return output


def _metric_anatomy_rows(scored_rows_by_candidate: dict[str, list[dict[str, Any]]], *, candidate_id: str) -> list[dict[str, Any]]:
    rows = scored_rows_by_candidate.get(candidate_id, [])
    output: list[dict[str, Any]] = []
    for scope in ("all_mapped", "program_mapped"):
        for horizon in (1, 3, 5):
            for metric_name in R10_COMPARABLE_METRICS:
                scoped = [
                    row
                    for row in rows
                    if int(row.get("horizon_years") or 0) == int(horizon)
                    and str(row.get("metric_name") or "") == metric_name
                    and _record_scope(row, scope)
                ]
                if not scoped:
                    continue
                candidate_mean = _mean_error(scoped, "candidate_norm_error")
                r10_mean = _mean_error(scoped, "r10_norm_error")
                carry_mean = _mean_error(scoped, "carry_forward_norm_error")
                output.append(
                    {
                        "candidate_id": candidate_id,
                        "scope": scope,
                        "horizon_years": int(horizon),
                        "metric_name": metric_name,
                        "entry_count": len(scoped),
                        "candidate_mean_mae": candidate_mean,
                        "strict_matched_r10_mean_mae": r10_mean,
                        "carry_forward_mean_mae": carry_mean,
                        "candidate_minus_strict_r10": None if candidate_mean is None or r10_mean is None else float(candidate_mean - r10_mean),
                    }
                )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "blockers"})
    if any("blockers" in row for row in rows):
        fieldnames.append("blockers")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            if "blockers" in output:
                output["blockers"] = ";".join(str(item) for item in list(output.get("blockers") or []))
            writer.writerow(output)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R37 Fixed-Policy Strict Scan",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Top Fixed Candidates",
        "",
        "| Candidate | Passes | Max Delta R10 | Mean Delta R10 |",
        "|---|---:|---:|---:|",
    ]
    for row in list(report.get("candidate_summary_rows") or [])[:10]:
        lines.append(
            f"| `{row.get('candidate_id')}` | {row.get('pass_count')}/{row.get('required_count')} | "
            f"{_format_float(row.get('max_candidate_minus_strict_r10'))} | "
            f"{_format_float(row.get('mean_candidate_minus_strict_r10'))} |"
        )
    lines.extend(
        [
            "",
            "## Gate Rows For Best Candidate",
            "",
            "| Scope | Horizon | Candidate | Strict R10 | Delta R10 | Carry | Status | Blockers |",
            "|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    best_candidate = str(report.get("best_candidate_id") or "")
    for row in list(report.get("candidate_gate_rows") or []):
        if str(row.get("candidate_id") or "") != best_candidate:
            continue
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | {_format_float(row.get('candidate_mean_mae'))} | "
            f"{_format_float(row.get('strict_matched_r10_mean_mae'))} | {_format_float(row.get('candidate_minus_strict_r10'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | `{row.get('status')}` | "
            f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R37 is exact over strict mapped R10 records, but it is a scan over fixed policies scored on those records.",
            "- Therefore `scan_passed=true` is evidence of a viable policy family, not publication-grade promotion.",
            "- A promoted branch must convert the winning family into a train-origin selector and rerun annual/R29 gates.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    top = rows[:12]
    labels = [str(row.get("candidate_id") or "")[:45] for row in top]
    max_delta = np.asarray(
        [
            float(row.get("max_candidate_minus_strict_r10"))
            if _finite_float(row.get("max_candidate_minus_strict_r10")) is not None
            else np.nan
            for row in top
        ],
        dtype=np.float64,
    )
    pass_count = np.asarray([float(row.get("pass_count") or 0) for row in top], dtype=np.float64)
    x = np.arange(len(top), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    fig.suptitle("R37 fixed-policy strict scan", fontsize=14, fontweight="bold")
    axes[0].bar(x, pass_count, color="#3d5a80")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    axes[0].set_ylabel("passed required routes")
    axes[0].set_title("Route pass count")
    axes[1].bar(x, max_delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in max_delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    axes[1].set_ylabel("max candidate minus strict R10")
    axes[1].set_title("Worst strict R10 margin")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r37_fixed_policy_strict_scan(
    *,
    run_id: str = R37_RUN_ID,
    epigraph_root: Path | None = None,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    horizons: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    source_rows = build_observation_rows(
        epigraph_root=root,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    phase3_root = sandbox_repo_root()
    r29_path = phase3_root / "artifacts" / "runs" / R29_RUN_ID / "analysis" / "r29_strict_ledger_matched_r10_gate.json"
    if r29_path.exists():
        r29_report = read_json(r29_path, default={})
    else:
        r29_report = run_r29_strict_ledger_matched_r10_gate(epigraph_root=root)
    strict_rows = list(dict(r29_report).get("strict_reference_rows") or [])
    records, reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    mapped_records = [record for record in records if not bool(record.get("unknown_provenance"))]
    if not strict_rows:
        strict_rows = _strict_reference_rows(mapped_records, horizons)
    candidates = _candidate_specs()
    prediction_rows = _prediction_rows_by_candidate(source_rows=source_rows, records=mapped_records, candidates=candidates)
    scored_rows_by_candidate = {
        str(candidate["candidate_id"]): _score_candidate_records(
            candidate=candidate,
            records=mapped_records,
            prediction_rows=prediction_rows.get(str(candidate["candidate_id"]), {}),
        )
        for candidate in candidates
    }
    gate_rows = _candidate_gate_rows(
        scored_rows_by_candidate=scored_rows_by_candidate,
        strict_reference_rows=strict_rows,
    )
    summary_rows = _candidate_summary_rows(gate_rows)
    best_candidate = str(summary_rows[0].get("candidate_id") or "") if summary_rows else ""
    metric_rows = _metric_anatomy_rows(scored_rows_by_candidate, candidate_id=best_candidate)
    scan_passed = bool(summary_rows and bool(summary_rows[0].get("all_required_pass")))
    verdict = (
        "R37 found at least one fixed component-policy combination that beats strict mapped R10 and carry-forward on every required route."
        if scan_passed
        else "R37 found no fixed component-policy combination that beats strict mapped R10 and carry-forward on every required route."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R37_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "horizons": list(horizons),
        "candidate_count": len(candidates),
        "promotion_eligible": False,
        "scan_passed": scan_passed,
        "best_candidate_id": best_candidate,
        "verdict": verdict,
        "reference_rows": reference_rows,
        "strict_reference_rows": strict_rows,
        "candidate_summary_rows": summary_rows,
        "candidate_gate_rows": gate_rows,
        "metric_anatomy_rows": metric_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r37_fixed_policy_strict_scan.json"
    md_path = analysis_dir / "r37_fixed_policy_strict_scan.md"
    summary_csv = analysis_dir / "r37_candidate_summary_rows.csv"
    gate_csv = analysis_dir / "r37_candidate_gate_rows.csv"
    metric_csv = analysis_dir / "r37_metric_anatomy_rows.csv"
    dashboard_path = analysis_dir / "r37_fixed_policy_strict_scan_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "summary_csv": summary_csv.as_posix(),
        "candidate_gate_csv": gate_csv.as_posix(),
        "metric_anatomy_csv": metric_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(summary_csv, summary_rows)
    _write_csv(gate_csv, gate_rows)
    _write_csv(metric_csv, metric_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, summary_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R37 fixed-policy strict scan.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R37_RUN_ID)
    args = parser.parse_args()
    run_r37_fixed_policy_strict_scan(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
