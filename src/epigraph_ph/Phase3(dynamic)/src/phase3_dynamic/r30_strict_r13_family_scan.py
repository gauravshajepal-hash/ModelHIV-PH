from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .r11_sparse_state_space import _finite_float, _generated_at
from .r28_r10_contract_lineage_audit import _r13_results_path
from .r29_strict_ledger_matched_r10_gate import (
    R29_RUN_ID,
    R29_SCHEMA_VERSION,
    run_r29_strict_ledger_matched_r10_gate,
)
from .runtime import ensure_dir, read_json, write_json
from .data import default_epigraph_root, sandbox_repo_root


R30_SCHEMA_VERSION = "phase3_dynamic.r30_strict_r13_family_scan.v1"
R30_RUN_ID = "p3d-r30-strict-r13-family-scan-20260502-s00"


def _strict_reference_lookup(r29_report: dict[str, Any]) -> dict[tuple[str, int], float]:
    lookup: dict[tuple[str, int], float] = {}
    for row in list(r29_report.get("strict_reference_rows") or []):
        if not isinstance(row, dict):
            continue
        value = _finite_float(row.get("matched_r10_mean_mae"))
        if value is None:
            continue
        lookup[(str(row.get("scope") or ""), int(row.get("horizon_years") or 0))] = value
    return lookup


def _reference_scope_for_row_scope(row_scope: str) -> str:
    return "program_mapped" if row_scope == "program" else "all_mapped"


def _scan_r13_results(
    *,
    r13_report: dict[str, Any],
    strict_reference: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in list(r13_report.get("results") or []):
        if not isinstance(result, dict):
            continue
        row_scope = str(result.get("row_scope") or "")
        reference_scope = _reference_scope_for_row_scope(row_scope)
        horizon_deltas: list[float] = []
        horizon_rows: list[dict[str, Any]] = []
        for horizon_row in list(result.get("horizon_rows") or []):
            if not isinstance(horizon_row, dict):
                continue
            horizon = int(horizon_row.get("horizon_years") or 0)
            reference = strict_reference.get((reference_scope, horizon))
            candidate = _finite_float(horizon_row.get("r10_comparable_candidate_mean_mae"))
            if candidate is None or reference is None:
                continue
            delta = float(candidate - reference)
            horizon_deltas.append(delta)
            horizon_rows.append(
                {
                    "horizon_years": horizon,
                    "candidate_mean_mae": candidate,
                    "strict_matched_r10_mean_mae": reference,
                    "candidate_minus_strict_r10": delta,
                    "status": "pass" if delta < 0.0 else "fail",
                }
            )
        if not horizon_rows:
            continue
        required_long_horizons = {3, 5} if row_scope == "program" else {1, 3, 5}
        covered_horizons = {int(row["horizon_years"]) for row in horizon_rows}
        missing_required = sorted(required_long_horizons.difference(covered_horizons))
        blockers = []
        if missing_required:
            blockers.append("missing_required_horizons_" + "_".join(f"h{h}" for h in missing_required))
        if any(delta >= 0.0 for delta in horizon_deltas):
            blockers.append("strict_r10_gate_failed")
        rows.append(
            {
                "experiment_id": str(result.get("experiment_id") or ""),
                "family": str(result.get("family") or ""),
                "row_scope": row_scope,
                "reference_scope": reference_scope,
                "decision": str(result.get("decision") or ""),
                "horizon_count": len(horizon_rows),
                "max_candidate_minus_strict_r10": float(np.max(np.asarray(horizon_deltas, dtype=np.float64))),
                "mean_candidate_minus_strict_r10": float(np.mean(np.asarray(horizon_deltas, dtype=np.float64))),
                "covered_horizons": sorted(covered_horizons),
                "missing_required_horizons": missing_required,
                "status": "pass" if not blockers else "fail",
                "blockers": blockers,
                "horizon_rows": horizon_rows,
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("row_scope") or ""),
            float(row.get("max_candidate_minus_strict_r10") or 0.0),
            float(row.get("mean_candidate_minus_strict_r10") or 0.0),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat_rows.append(
            {
                "experiment_id": row.get("experiment_id"),
                "family": row.get("family"),
                "row_scope": row.get("row_scope"),
                "reference_scope": row.get("reference_scope"),
                "decision": row.get("decision"),
                "horizon_count": row.get("horizon_count"),
                "covered_horizons": ",".join(f"h{h}" for h in list(row.get("covered_horizons") or [])),
                "missing_required_horizons": ",".join(f"h{h}" for h in list(row.get("missing_required_horizons") or [])),
                "max_candidate_minus_strict_r10": row.get("max_candidate_minus_strict_r10"),
                "mean_candidate_minus_strict_r10": row.get("mean_candidate_minus_strict_r10"),
                "status": row.get("status"),
                "blockers": ";".join(str(item) for item in list(row.get("blockers") or [])),
            }
        )
    if not flat_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    rows = list(report.get("scan_rows") or [])
    rows_by_scope = {
        scope: [row for row in rows if str(row.get("row_scope") or "") == scope]
        for scope in ("all", "program")
    }
    lines = [
        "# Phase 3 R30 Strict R13 Family Scan",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Best Full-Sentinel Rows",
        "",
        "| Scope | Experiment | Family | Horizons | Max delta | Mean delta | Status | Blockers |",
        "|---|---|---|---|---:|---:|---|---|",
    ]
    for scope in ("all", "program"):
        if scope == "program":
            lines.extend(
                [
                    "",
                    "## Best Program Rows",
                    "",
                    "| Scope | Experiment | Family | Horizons | Max delta | Mean delta | Status | Blockers |",
                    "|---|---|---|---|---:|---:|---|---|",
                ]
            )
        for row in rows_by_scope.get(scope, [])[:12]:
            lines.append(
                f"| {row.get('row_scope')} | `{row.get('experiment_id')}` | `{row.get('family')}` | "
                f"{','.join(f'h{h}' for h in list(row.get('covered_horizons') or []))} | "
                f"{_format_float(row.get('max_candidate_minus_strict_r10'))} | "
                f"{_format_float(row.get('mean_candidate_minus_strict_r10'))} | `{row.get('status')}` | "
                f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
            )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R30 does not run a new model. It rescans already-run R13 families against the R29 strict-ledger references.",
            "- Program rows must cover h3 and h5; full/all rows must cover h1, h3, and h5.",
            "- A family passes only if every required horizon beats the strict mapped R10 reference.",
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
    top_rows = rows[:15]
    labels = [f"{row.get('experiment_id')}\\n{row.get('row_scope')}" for row in top_rows]
    max_delta = np.asarray([float(row.get("max_candidate_minus_strict_r10") or np.nan) for row in top_rows], dtype=np.float64)
    x = np.arange(len(top_rows), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    fig.suptitle("R30 Existing R13 Families vs Strict Mapped R10", fontsize=14, fontweight="bold")
    ax.bar(x, max_delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in max_delta])
    ax.axhline(0.0, color="#111827", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("max candidate minus strict R10")
    ax.set_title("Negative values beat strict R10 on covered horizons")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r30_strict_r13_family_scan(
    *,
    run_id: str = R30_RUN_ID,
    epigraph_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    r29_report_path = phase3_root / "artifacts" / "runs" / R29_RUN_ID / "analysis" / "r29_strict_ledger_matched_r10_gate.json"
    if r29_report_path.exists():
        r29_report = read_json(r29_report_path, default={})
    else:
        r29_report = run_r29_strict_ledger_matched_r10_gate(run_id=R29_RUN_ID, epigraph_root=root)
    if not isinstance(r29_report, dict):
        raise ValueError("Invalid R29 strict-ledger report.")
    r13_path = _r13_results_path(phase3_root)
    r13_report = read_json(r13_path, default={})
    if not isinstance(r13_report, dict):
        raise FileNotFoundError(f"Missing R13 report: {r13_path}")
    scan_rows = _scan_r13_results(
        r13_report=r13_report,
        strict_reference=_strict_reference_lookup(r29_report),
    )
    passing_rows = [row for row in scan_rows if str(row.get("status") or "") == "pass"]
    blockers: list[str] = []
    if not passing_rows:
        blockers.append("no_existing_r13_family_beats_strict_mapped_r10_on_required_horizons")
    verdict = (
        "R30 found at least one existing R13 family that beats strict mapped R10 on its required horizons."
        if passing_rows
        else "R30 found no already-run R13 family that beats strict mapped R10 on required long-horizon routes."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R30_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "r29_schema_version": R29_SCHEMA_VERSION,
        "r29_report_path": r29_report_path.as_posix(),
        "r13_report_path": r13_path.as_posix(),
        "promotion_eligible": bool(passing_rows),
        "blockers": blockers,
        "verdict": verdict,
        "passing_rows": passing_rows,
        "scan_rows": scan_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r30_strict_r13_family_scan.json"
    md_path = analysis_dir / "r30_strict_r13_family_scan.md"
    csv_path = analysis_dir / "r30_strict_r13_family_scan_rows.csv"
    dashboard_path = analysis_dir / "r30_strict_r13_family_scan_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "csv": csv_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, scan_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, scan_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R30 strict R13 family scan.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--run-id", default=R30_RUN_ID)
    args = parser.parse_args()
    run_r30_strict_r13_family_scan(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
    )


if __name__ == "__main__":
    _main()
