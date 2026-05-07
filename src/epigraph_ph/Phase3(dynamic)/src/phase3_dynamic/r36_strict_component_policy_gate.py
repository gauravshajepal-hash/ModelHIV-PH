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
from .r11_sparse_state_space import R10_COMPARABLE_METRICS, _candidate_predictions, _finite_float, _generated_at
from .r28_r10_contract_lineage_audit import _collect_r10_records
from .r29_strict_ledger_matched_r10_gate import (
    R29_RUN_ID,
    _is_program_lineage,
    _reference_by_scope_horizon,
    _strict_reference_rows,
    run_r29_strict_ledger_matched_r10_gate,
)
from .runtime import ensure_dir, read_json, write_json


R36_SCHEMA_VERSION = "phase3_dynamic.r36_strict_component_policy_gate.v1"
R36_RUN_ID = "p3d-r36-strict-component-policy-gate-20260502-s00"
R36_FAMILY = "r36_component_policy_cascade_process"


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
    raise ValueError(f"Unknown R36 scope: {scope}")


def _group_records(records: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if bool(record.get("unknown_provenance")):
            continue
        grouped[(int(record.get("train_end_year") or 0), int(record.get("horizon_years") or 0))].append(dict(record))
    return grouped


def _prediction_rows(
    *,
    source_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[tuple[int, int, str], dict[str, Any]]:
    source_by_quarter = _source_rows_by_quarter(source_rows)
    output: dict[tuple[int, int, str], dict[str, Any]] = {}
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
        predictions, _summary = _candidate_predictions(train_rows, holdout_rows, family=R36_FAMILY)
        for row in predictions:
            quarter = str(row.get("quarter") or "")
            if quarter:
                output[(int(train_end_year), int(horizon), quarter)] = dict(row)
    return output


def _score_records(
    *,
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
                "candidate_family": R36_FAMILY,
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
    scored_rows: list[dict[str, Any]],
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
    for scope, horizon in required:
        scoped = [
            row
            for row in scored_rows
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
                "candidate_family": R36_FAMILY,
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


def _metric_anatomy_rows(scored_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope in ("all_mapped", "program_mapped"):
        for horizon in (1, 3, 5):
            for metric_name in R10_COMPARABLE_METRICS:
                scoped = [
                    row
                    for row in scored_rows
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
                        "candidate_family": R36_FAMILY,
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
        "# Phase 3 R36 Strict Component-Policy Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate Rows",
        "",
        "| Scope | Horizon | Entries | Candidate | Strict R10 | Delta R10 | Carry | Status | Blockers |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("candidate_gate_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | {row.get('entry_count')} | "
            f"{_format_float(row.get('candidate_mean_mae'))} | {_format_float(row.get('strict_matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('candidate_minus_strict_r10'))} | {_format_float(row.get('carry_forward_mean_mae'))} | "
            f"`{row.get('status')}` | `{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R36 is exact over the strict mapped R10 records, not an aggregate replay.",
            "- Training uses all available training rows up to each blocked origin; scoring uses the strict mapped target rows.",
            "- It has no R10 teacher: all candidate values come from train-origin component policies and conditional-rate projections.",
            "- Annual official gate is still separate and must pass before any publication-grade champion claim.",
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
    labels = [f"{row.get('scope')} h{row.get('horizon_years')}" for row in rows]
    deltas = np.asarray([float(row.get("candidate_minus_strict_r10") or np.nan) for row in rows], dtype=np.float64)
    x = np.arange(len(rows), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("R36 exact strict gate: candidate minus strict mapped R10", fontsize=14, fontweight="bold")
    ax.bar(x, deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in deltas])
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("normalized MAE delta")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r36_strict_component_policy_gate(
    *,
    run_id: str = R36_RUN_ID,
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
    predictions = _prediction_rows(source_rows=source_rows, records=mapped_records)
    scored_rows = _score_records(records=mapped_records, prediction_rows=predictions)
    gate_rows = _candidate_gate_rows(scored_rows=scored_rows, strict_reference_rows=strict_rows)
    metric_rows = _metric_anatomy_rows(scored_rows)
    passed = bool(gate_rows and all(str(row.get("status") or "") == "pass" for row in gate_rows))
    verdict = (
        "R36 exact strict component-policy gate passes every required mapped R10/carry-forward route."
        if passed
        else "R36 exact strict component-policy gate fails at least one mapped R10/carry-forward route."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R36_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "candidate_family": R36_FAMILY,
        "horizons": list(horizons),
        "promotion_eligible": passed,
        "verdict": verdict,
        "reference_rows": reference_rows,
        "strict_reference_rows": strict_rows,
        "candidate_gate_rows": gate_rows,
        "metric_anatomy_rows": metric_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r36_strict_component_policy_gate.json"
    md_path = analysis_dir / "r36_strict_component_policy_gate.md"
    gate_csv = analysis_dir / "r36_candidate_gate_rows.csv"
    metric_csv = analysis_dir / "r36_metric_anatomy_rows.csv"
    dashboard_path = analysis_dir / "r36_strict_component_policy_gate_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "candidate_gate_csv": gate_csv.as_posix(),
        "metric_anatomy_csv": metric_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(gate_csv, gate_rows)
    _write_csv(metric_csv, metric_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, gate_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R36 strict component-policy gate.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R36_RUN_ID)
    args = parser.parse_args()
    run_r36_strict_component_policy_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
