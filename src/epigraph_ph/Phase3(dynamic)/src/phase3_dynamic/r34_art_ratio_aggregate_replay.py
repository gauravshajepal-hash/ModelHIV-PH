from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import R10_COMPARABLE_METRICS, _finite_float, _generated_at
from .runtime import ensure_dir, read_json, write_json


R34_AGGREGATE_SCHEMA_VERSION = "phase3_dynamic.r34_art_ratio_aggregate_replay.v1"
R34_AGGREGATE_RUN_ID = "p3d-r34-art-ratio-aggregate-replay-20260502-s00"
R34_AGGREGATE_CANDIDATE_ID = "r18_r31_flow_plus_r34_art_ratio_aggregate"


def _r32_report_path(phase3_root: Path) -> Path:
    return (
        phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r32-r18-flow-composite-replay-20260502-s00"
        / "analysis"
        / "r32_strict_composite_replay.json"
    )


def _r33_report_path(phase3_root: Path) -> Path:
    return (
        phase3_root
        / "artifacts"
        / "runs"
        / "p3d-r33-h5-backhalf-process-audit-20260502-s00"
        / "analysis"
        / "r33_h5_backhalf_process_audit.json"
    )


def _reference_lookup(strict_reference_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.get("scope") or ""), int(row.get("horizon_years") or 0)): dict(row)
        for row in strict_reference_rows
    }


def _r32_metric_lookup(r32_report: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in list(r32_report.get("metric_anatomy_rows") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("candidate_id") or "") != "r18_plus_r31_positive_flow_aggregate":
            continue
        key = (str(row.get("scope") or ""), int(row.get("horizon_years") or 0), str(row.get("metric_name") or ""))
        lookup[key] = dict(row)
    return lookup


def _r33_art_lookup(r33_report: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    lookup: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in list(r33_report.get("best_art_policy_rows") or []):
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") != "pass":
            continue
        key = (str(row.get("scope") or ""), int(row.get("horizon_years") or 0), "alive_on_art")
        lookup[key] = {
            "candidate_mean_mae": row.get("policy_mean_mae"),
            "entry_count": row.get("entry_count"),
            "source": "r33_train_only_art_ratio_policy",
            "policy_id": row.get("policy_id"),
            "strict_matched_r10_mean_mae": row.get("strict_r10_mean_mae"),
            "carry_forward_mean_mae": row.get("carry_forward_mean_mae"),
        }
    return lookup


def _aggregate_candidate_rows(
    *,
    r32_lookup: dict[tuple[str, int, str], dict[str, Any]],
    r33_art_lookup: dict[tuple[str, int, str], dict[str, Any]],
    strict_reference_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference = _reference_lookup(strict_reference_rows)
    required = (
        ("all_mapped", 1),
        ("all_mapped", 3),
        ("all_mapped", 5),
        ("program_mapped", 3),
        ("program_mapped", 5),
    )
    gate_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    for scope, horizon in required:
        weighted_error = 0.0
        entry_count = 0
        missing_metrics: list[str] = []
        for metric_name in R10_COMPARABLE_METRICS:
            key = (scope, int(horizon), metric_name)
            metric = r33_art_lookup.get(key) if metric_name == "alive_on_art" else None
            metric = metric or r32_lookup.get(key)
            if metric is None:
                missing_metrics.append(metric_name)
                continue
            metric_value = _finite_float(metric.get("candidate_mean_mae"))
            if metric_value is None:
                missing_metrics.append(metric_name)
                continue
            metric_entry_count = max(int(metric.get("entry_count") or 0), 1)
            weighted_error += float(metric_value) * float(metric_entry_count)
            entry_count += int(metric_entry_count)
            metric_rows.append(
                {
                    "candidate_id": R34_AGGREGATE_CANDIDATE_ID,
                    "scope": scope,
                    "horizon_years": int(horizon),
                    "metric_name": metric_name,
                    "candidate_mean_mae": float(metric_value),
                    "entry_count": int(metric_entry_count),
                    "source": str(metric.get("source") or ""),
                    "policy_id": metric.get("policy_id"),
                }
            )
        candidate_mean = None if entry_count == 0 else float(weighted_error / float(entry_count))
        ref = reference.get((scope, int(horizon)), {})
        matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
        carry = _finite_float(ref.get("carry_forward_mean_mae"))
        blockers: list[str] = []
        if missing_metrics:
            blockers.append("missing_aggregate_metrics_" + "_".join(missing_metrics))
        if candidate_mean is None or matched_r10 is None:
            blockers.append("candidate_or_strict_r10_missing")
        elif candidate_mean >= matched_r10:
            blockers.append("candidate_not_better_than_strict_mapped_r10")
        if candidate_mean is None or carry is None:
            blockers.append("candidate_or_carry_missing")
        elif candidate_mean >= carry:
            blockers.append("candidate_not_better_than_carry_forward")
        gate_rows.append(
            {
                "candidate_id": R34_AGGREGATE_CANDIDATE_ID,
                "scope": scope,
                "horizon_years": int(horizon),
                "entry_count": int(entry_count),
                "candidate_mean_mae": candidate_mean,
                "strict_matched_r10_mean_mae": matched_r10,
                "candidate_minus_strict_r10": None if candidate_mean is None or matched_r10 is None else float(candidate_mean - matched_r10),
                "carry_forward_mean_mae": carry,
                "candidate_minus_carry_forward": None if candidate_mean is None or carry is None else float(candidate_mean - carry),
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
    return gate_rows, metric_rows


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
        "# Phase 3 R34 ART-Ratio Aggregate Replay",
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
            "- This is an aggregate non-promotional replay.",
            "- It combines R32's R18 plus R31 diagnosis-flow aggregate with R33's strict train-only ART-ratio aggregate.",
            "- It answers whether the ART-ratio repair is large enough to close the remaining strict h5 R10 gap before spending compute on exact refits.",
            "- A publication claim still requires exact R34 family replay through the R13/R29/annual gates.",
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
    fig.suptitle("R34 aggregate replay: candidate minus strict mapped R10", fontsize=14, fontweight="bold")
    ax.bar(x, deltas, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in deltas])
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("normalized MAE delta")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r34_art_ratio_aggregate_replay(*, run_id: str = R34_AGGREGATE_RUN_ID) -> dict[str, Any]:
    phase3_root = sandbox_repo_root()
    r32_path = _r32_report_path(phase3_root)
    r33_path = _r33_report_path(phase3_root)
    r32_report = read_json(r32_path, default={})
    r33_report = read_json(r33_path, default={})
    if not isinstance(r32_report, dict) or not r32_report:
        raise FileNotFoundError(f"Missing R32 report: {r32_path}")
    if not isinstance(r33_report, dict) or not r33_report:
        raise FileNotFoundError(f"Missing R33 report: {r33_path}")
    candidate_rows, metric_rows = _aggregate_candidate_rows(
        r32_lookup=_r32_metric_lookup(r32_report),
        r33_art_lookup=_r33_art_lookup(r33_report),
        strict_reference_rows=list(r32_report.get("strict_reference_rows") or []),
    )
    promoted = bool(candidate_rows and all(str(row.get("status") or "") == "pass" for row in candidate_rows))
    verdict = (
        "R34 aggregate replay closes the strict mapped R10 gap on every required route; exact replay is now worth optimizing."
        if promoted
        else "R34 aggregate replay still fails at least one strict mapped R10 route; exact replay is not yet justified as a champion."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R34_AGGREGATE_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "candidate_id": R34_AGGREGATE_CANDIDATE_ID,
        "replay_mode": "aggregate_nonpromotional",
        "promotion_eligible": False,
        "aggregate_gate_passed": promoted,
        "verdict": verdict,
        "r32_report_path": r32_path.as_posix(),
        "r33_report_path": r33_path.as_posix(),
        "candidate_gate_rows": candidate_rows,
        "metric_anatomy_rows": metric_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r34_art_ratio_aggregate_replay.json"
    md_path = analysis_dir / "r34_art_ratio_aggregate_replay.md"
    gate_csv = analysis_dir / "r34_candidate_gate_rows.csv"
    metric_csv = analysis_dir / "r34_metric_anatomy_rows.csv"
    dashboard_path = analysis_dir / "r34_art_ratio_aggregate_replay_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "candidate_gate_csv": gate_csv.as_posix(),
        "metric_anatomy_csv": metric_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(gate_csv, candidate_rows)
    _write_csv(metric_csv, metric_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, candidate_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R34 ART-ratio aggregate replay.")
    parser.add_argument("--run-id", default=R34_AGGREGATE_RUN_ID)
    args = parser.parse_args()
    run_r34_art_ratio_aggregate_replay(run_id=str(args.run_id))


if __name__ == "__main__":
    _main()
