from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import _finite_float, _generated_at
from .r28_r10_contract_lineage_audit import (
    _collect_r10_records,
    _r24_results_path,
)
from .runtime import ensure_dir, read_json, write_json


R29_SCHEMA_VERSION = "phase3_dynamic.r29_strict_ledger_matched_r10_gate.v1"
R29_RUN_ID = "p3d-r29-strict-ledger-matched-r10-gate-20260502-s00"


def _is_program_lineage(record: dict[str, Any]) -> bool:
    lineage = str(record.get("source_lineage") or "")
    return lineage.startswith("official_doh_archive|program_observed_harp|")


def _score_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    r10_errors = [
        float(record["r10_norm_error"])
        for record in records
        if _finite_float(record.get("r10_norm_error")) is not None
    ]
    carry_errors = [
        float(record["carry_forward_norm_error"])
        for record in records
        if _finite_float(record.get("carry_forward_norm_error")) is not None
    ]
    if not r10_errors or not carry_errors:
        return {
            "entry_count": len(records),
            "matched_r10_mean_mae": None,
            "carry_forward_mean_mae": None,
            "matched_r10_worst_mae": None,
            "carry_forward_worst_mae": None,
        }
    r10_array = np.asarray(r10_errors, dtype=np.float64)
    carry_array = np.asarray(carry_errors, dtype=np.float64)
    return {
        "entry_count": len(records),
        "matched_r10_mean_mae": float(np.mean(r10_array)),
        "carry_forward_mean_mae": float(np.mean(carry_array)),
        "matched_r10_worst_mae": float(np.max(r10_array)),
        "carry_forward_worst_mae": float(np.max(carry_array)),
        "matched_r10_better_share": float(
            np.mean(
                np.asarray(
                    [
                        1.0 if float(record["r10_norm_error"]) < float(record["carry_forward_norm_error"]) else 0.0
                        for record in records
                    ],
                    dtype=np.float64,
                )
            )
        ),
    }


def _strict_reference_rows(records: list[dict[str, Any]], horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        horizon_records = [record for record in records if int(record.get("horizon_years") or 0) == int(horizon)]
        mapped = [record for record in horizon_records if not bool(record.get("unknown_provenance"))]
        program = [record for record in mapped if _is_program_lineage(record)]
        for scope, scoped_records in (("all_mapped", mapped), ("program_mapped", program)):
            score = _score_records(scoped_records)
            rows.append(
                {
                    "scope": scope,
                    "horizon_years": int(horizon),
                    "total_replay_entry_count": len(horizon_records),
                    "mapped_entry_count": len(mapped),
                    "scored_entry_count": int(score["entry_count"]),
                    "mapped_share": 0.0 if not horizon_records else float(len(mapped) / len(horizon_records)),
                    **{key: value for key, value in score.items() if key != "entry_count"},
                }
            )
    return rows


def _reference_by_scope_horizon(reference_rows: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(row.get("scope") or ""), int(row.get("horizon_years") or 0)): dict(row)
        for row in reference_rows
    }


def _best_program_rows(r24: dict[str, Any]) -> dict[int, dict[str, Any]]:
    best: dict[int, dict[str, Any]] = {}
    for row in list(r24.get("program_route_rows") or []):
        if not isinstance(row, dict):
            continue
        horizon = int(row.get("horizon_years") or 0)
        value = _finite_float(row.get("r10_comparable_candidate_mean_mae"))
        if value is None:
            continue
        previous = best.get(horizon)
        previous_value = None if previous is None else _finite_float(previous.get("r10_comparable_candidate_mean_mae"))
        if previous is None or previous_value is None or value < previous_value:
            best[horizon] = dict(row)
    return best


def _candidate_gate_rows(
    *,
    reference_rows: list[dict[str, Any]],
    r24: dict[str, Any],
) -> list[dict[str, Any]]:
    reference = _reference_by_scope_horizon(reference_rows)
    rows: list[dict[str, Any]] = []
    for source_row in list(r24.get("all_support_r19_rows") or []):
        if not isinstance(source_row, dict):
            continue
        horizon = int(source_row.get("horizon_years") or 0)
        ref = reference.get(("all_mapped", horizon), {})
        candidate = _finite_float(source_row.get("r10_comparable_candidate_mean_mae"))
        matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
        carry = _finite_float(source_row.get("carry_forward_mean_mae"))
        blockers: list[str] = []
        if candidate is None or matched_r10 is None:
            blockers.append("candidate_or_strict_r10_missing")
        elif candidate >= matched_r10:
            blockers.append("candidate_not_better_than_strict_mapped_r10")
        if candidate is None or carry is None:
            blockers.append("candidate_or_carry_missing")
        elif candidate >= carry:
            blockers.append("candidate_not_better_than_carry_forward")
        rows.append(
            {
                "candidate_scope": "full_sentinel_r19",
                "reference_scope": "all_mapped",
                "horizon_years": horizon,
                "candidate_family": str(source_row.get("family") or "r19_joint_service_cascade_process"),
                "candidate_mean_mae": candidate,
                "strict_matched_r10_mean_mae": matched_r10,
                "candidate_minus_strict_r10": None if candidate is None or matched_r10 is None else float(candidate - matched_r10),
                "carry_forward_mean_mae": carry,
                "mapped_reference_entry_count": ref.get("scored_entry_count"),
                "mapped_share": ref.get("mapped_share"),
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
    for horizon, source_row in sorted(_best_program_rows(r24).items()):
        ref = reference.get(("program_mapped", horizon), {})
        candidate = _finite_float(source_row.get("r10_comparable_candidate_mean_mae"))
        matched_r10 = _finite_float(ref.get("matched_r10_mean_mae"))
        carry = _finite_float(source_row.get("carry_forward_mean_mae"))
        blockers = []
        if candidate is None or matched_r10 is None:
            blockers.append("candidate_or_strict_r10_missing")
        elif candidate >= matched_r10:
            blockers.append("candidate_not_better_than_strict_program_r10")
        if candidate is None or carry is None:
            blockers.append("candidate_or_carry_missing")
        elif candidate >= carry:
            blockers.append("candidate_not_better_than_carry_forward")
        rows.append(
            {
                "candidate_scope": "program_best_phase3",
                "reference_scope": "program_mapped",
                "horizon_years": int(horizon),
                "candidate_family": str(source_row.get("family") or ""),
                "candidate_mean_mae": candidate,
                "strict_matched_r10_mean_mae": matched_r10,
                "candidate_minus_strict_r10": None if candidate is None or matched_r10 is None else float(candidate - matched_r10),
                "carry_forward_mean_mae": carry,
                "mapped_reference_entry_count": ref.get("scored_entry_count"),
                "mapped_share": ref.get("mapped_share"),
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
            }
        )
    return rows


def _official_annual_gate(r24: dict[str, Any]) -> dict[str, Any]:
    gate = dict(r24.get("official_annual_gate") or {})
    return {
        "available": bool(gate),
        "status": str(gate.get("annual_status") or "not_available"),
        "decision": str(gate.get("decision") or ""),
        "candidate_family": str(gate.get("candidate_family") or ""),
        "candidate_mean_mae": _finite_float(gate.get("candidate_mean_mae")),
        "carry_forward_mean_mae": _finite_float(gate.get("carry_forward_mean_mae")),
        "annual_max_conservation_residual": _finite_float(gate.get("annual_max_conservation_residual")),
        "stock_cone_violation_count": int(gate.get("stock_cone_violation_count") or 0),
    }


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
        "# Phase 3 R29 Strict-Ledger Matched R10 Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Candidate Gate Rows",
        "",
        "| Candidate scope | Ref scope | Horizon | Candidate | Strict R10 | Delta | Carry | Status | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("candidate_gate_rows") or []):
        lines.append(
            f"| {row.get('candidate_scope')} | {row.get('reference_scope')} | {row.get('horizon_years')} | "
            f"{_format_float(row.get('candidate_mean_mae'))} | "
            f"{_format_float(row.get('strict_matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('candidate_minus_strict_r10'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | `{row.get('status')}` | "
            f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Strict R10 References",
            "",
            "| Scope | Horizon | Entries | Mapped share | Strict R10 | Carry | R10 better share |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in list(report.get("strict_reference_rows") or []):
        lines.append(
            f"| {row.get('scope')} | {row.get('horizon_years')} | {row.get('scored_entry_count')} | "
            f"{_format_float(row.get('mapped_share'))} | "
            f"{_format_float(row.get('matched_r10_mean_mae'))} | "
            f"{_format_float(row.get('carry_forward_mean_mae'))} | "
            f"{_format_float(row.get('matched_r10_better_share'))} |"
        )
    annual = dict(report.get("official_annual_gate") or {})
    lines.extend(
        [
            "",
            "## Annual Gate",
            "",
            f"- Status: `{annual.get('status')}`",
            f"- Decision: `{annual.get('decision')}`",
            f"- Candidate MAE: `{_format_float(annual.get('candidate_mean_mae'))}`",
            f"- Carry-forward MAE: `{_format_float(annual.get('carry_forward_mean_mae'))}`",
            f"- Conservation residual: `{_format_float(annual.get('annual_max_conservation_residual'))}`",
            "",
            "## Contract",
            "",
            "- R29 is a strict-ledger rescore of the matched-R10 gate, not a new model.",
            "- `all_mapped` excludes R10 target rows that cannot be mapped to active ObservationRoleLedger provenance.",
            "- `program_mapped` additionally restricts the R10 reference to DOH HARP program lineages.",
            "- Promotion requires the Phase 3 candidate to beat carry-forward, strict mapped R10, and the official annual gate.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, report: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    rows = list(report.get("candidate_gate_rows") or [])
    labels = [f"{row.get('candidate_scope')} h{row.get('horizon_years')}" for row in rows]
    candidate = np.asarray([float(row.get("candidate_mean_mae") or np.nan) for row in rows], dtype=np.float64)
    r10 = np.asarray([float(row.get("strict_matched_r10_mean_mae") or np.nan) for row in rows], dtype=np.float64)
    carry = np.asarray([float(row.get("carry_forward_mean_mae") or np.nan) for row in rows], dtype=np.float64)
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.25
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("R29 Strict-Ledger Matched R10 Gate", fontsize=15, fontweight="bold")
    axes[0].bar(x - width, carry, width=width, color="#c9b879", label="carry-forward")
    axes[0].bar(x, candidate, width=width, color="#20639b", label="Phase3 candidate")
    axes[0].bar(x + width, r10, width=width, color="#111827", label="strict matched R10")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("normalized MAE")
    axes[0].legend(loc="upper left")
    delta = candidate - r10
    axes[1].bar(x, delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("candidate minus strict R10")
    axes[1].set_title("Negative values pass the strict mapped R10 gate")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r29_strict_ledger_matched_r10_gate(
    *,
    run_id: str = R29_RUN_ID,
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
    r10_records, _reference_rows = _collect_r10_records(epigraph_root=root, horizons=horizons, source_rows=source_rows)
    strict_reference_rows = _strict_reference_rows(r10_records, horizons)
    r24 = read_json(_r24_results_path(phase3_root), default={})
    if not isinstance(r24, dict) or not r24:
        raise FileNotFoundError(f"Missing R24 fairness audit artifact: {_r24_results_path(phase3_root)}")
    candidate_gate_rows = _candidate_gate_rows(reference_rows=strict_reference_rows, r24=r24)
    annual_gate = _official_annual_gate(r24)
    blockers = sorted({blocker for row in candidate_gate_rows for blocker in list(row.get("blockers") or [])})
    if str(annual_gate.get("status") or "") != "pass":
        blockers.append("official_annual_gate_not_passed")
    promotion_eligible = bool(candidate_gate_rows) and not blockers
    if promotion_eligible:
        verdict = "R29 strict-ledger gate passes: Phase 3 beats carry-forward, strict mapped R10, and the annual gate."
    else:
        verdict = "R29 strict-ledger gate fails. Even after excluding unmapped R10 rows, Phase 3 does not beat strict mapped R10 on every required route."
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R29_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "promotion_eligible": promotion_eligible,
        "blockers": blockers,
        "verdict": verdict,
        "strict_reference_rows": strict_reference_rows,
        "candidate_gate_rows": candidate_gate_rows,
        "official_annual_gate": annual_gate,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r29_strict_ledger_matched_r10_gate.json"
    md_path = analysis_dir / "r29_strict_ledger_matched_r10_gate.md"
    dashboard_path = analysis_dir / "r29_strict_ledger_matched_r10_gate_dashboard.png"
    reference_csv_path = analysis_dir / "r29_strict_reference_rows.csv"
    gate_csv_path = analysis_dir / "r29_candidate_gate_rows.csv"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
        "strict_reference_csv": reference_csv_path.as_posix(),
        "candidate_gate_csv": gate_csv_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(reference_csv_path, strict_reference_rows)
    _write_csv(gate_csv_path, candidate_gate_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R29 strict-ledger matched R10 gate.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R29_RUN_ID)
    args = parser.parse_args()
    run_r29_strict_ledger_matched_r10_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
