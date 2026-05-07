from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .data import build_observation_rows, default_epigraph_root, sandbox_repo_root
from .observation_ledger import resolve_active_source_run_id, resolve_baseline_source_run_id
from .r11_sparse_state_space import (
    OFFICIAL_ANNUAL_CHALLENGE_METRICS,
    R10_COMPARABLE_METRICS,
    R11_MULTI_HORIZON_YEARS,
    _artifact_paths,
    _build_r10_horizon_replay_report,
    _build_r12_official_annual_challenge_gate_report,
    _finite_float,
    _generated_at,
    _read_path_payload,
    _r10_reference,
)
from .r13_priority_experiments import _r13_annual_spec_report, _r13_model_spec_report
from .runtime import ensure_dir, write_json


R34_SCHEMA_VERSION = "phase3_dynamic.r34_art_ratio_branch_gate.v1"
R34_RUN_ID = "p3d-r34-art-ratio-branch-gate-20260502-s00"
R34_FAMILY = "r34_art_ratio_guarded_process"


def _r34_specs() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "R34-001",
            "priority": 1,
            "layer": "annual_official_measurement",
            "title": "Conserved official annual gate on R34",
            "family": "official_annual_challenge_gate",
            "candidate_family": R34_FAMILY,
            "row_scope": "official_annual_q4",
            "metrics": tuple(OFFICIAL_ANNUAL_CHALLENGE_METRICS),
            "horizons": (1,),
            "r10_required": False,
            "hypothesis": "R34 may be promoted only if the ART-ratio branch preserves the conserved annual incidence/deaths/PLHIV challenge gate.",
        },
        {
            "experiment_id": "R34-002",
            "priority": 2,
            "layer": "strict_r10_scope",
            "title": "Full R10-comparable blocked trajectory gate",
            "family": R34_FAMILY,
            "row_scope": "all",
            "metrics": tuple(R10_COMPARABLE_METRICS),
            "horizons": tuple(R11_MULTI_HORIZON_YEARS),
            "r10_required": True,
            "hypothesis": "Replacing only ART with the train-selected diagnosed-ratio process should not weaken the full diagnosis/ART/flow trajectory against matched R10.",
        },
        {
            "experiment_id": "R34-003",
            "priority": 3,
            "layer": "program_route",
            "title": "Program h3/h5 ART plus back-half route",
            "family": R34_FAMILY,
            "row_scope": "program",
            "metrics": (
                "alive_on_art",
                "tested_for_viral_load",
                "virally_suppressed",
                "new_diagnosed_cases_period",
            ),
            "horizons": (3, 5),
            "r10_required": True,
            "hypothesis": "R34 should close the ART part of the h3/h5 program failure without pretending that VL/suppression rates are better than conditional carry-forward.",
        },
    ]


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


def _flat_row(result: dict[str, Any]) -> dict[str, Any]:
    candidate = _finite_float(result.get("candidate_mean_mae"))
    carry = _finite_float(result.get("carry_forward_mean_mae"))
    r10_candidate = _finite_float(result.get("r10_comparable_candidate_mean_mae"))
    r10_reference = _finite_float(result.get("r10_reference_mae"))
    return {
        "experiment_id": str(result.get("experiment_id") or ""),
        "title": str(result.get("title") or ""),
        "layer": str(result.get("layer") or ""),
        "family": str(result.get("family") or ""),
        "candidate_family": str(result.get("candidate_family") or ""),
        "row_scope": str(result.get("row_scope") or ""),
        "candidate_mean_mae": candidate,
        "carry_forward_mean_mae": carry,
        "candidate_minus_carry_forward": None if candidate is None or carry is None else float(candidate - carry),
        "r10_comparable_candidate_mean_mae": r10_candidate,
        "r10_reference_mae": r10_reference,
        "candidate_minus_r10": None if r10_candidate is None or r10_reference is None else float(r10_candidate - r10_reference),
        "carry_gate": str(result.get("carry_gate") or ""),
        "r10_gate": str(result.get("r10_gate") or ""),
        "decision": str(result.get("decision") or ""),
        "blockers": list(result.get("blockers") or []),
    }


def _format_float(value: Any) -> str:
    finite = _finite_float(value)
    if finite is None:
        return "NA"
    return f"{finite:.6f}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Phase 3 R34 ART-Ratio Branch Gate",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate Rows",
        "",
        "| Experiment | Scope | Candidate | Carry | Delta Carry | R10 | Delta R10 | Decision | Blockers |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in list(report.get("flat_rows") or []):
        lines.append(
            f"| `{row.get('experiment_id')}` | {row.get('row_scope')} | "
            f"{_format_float(row.get('candidate_mean_mae'))} | {_format_float(row.get('carry_forward_mean_mae'))} | "
            f"{_format_float(row.get('candidate_minus_carry_forward'))} | {_format_float(row.get('r10_reference_mae'))} | "
            f"{_format_float(row.get('candidate_minus_r10'))} | `{row.get('decision')}` | "
            f"`{';'.join(str(item) for item in list(row.get('blockers') or []))}` |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "- R34 is promotion-eligible only if all three gates pass: official annual, full R10-comparable, and program h3/h5.",
            "- The family mutates only `alive_on_art` relative to R19, then reprojects VL/suppression through train-fitted conditional rates.",
            "- Diagnosis-flow and diagnosed stock remain governed by R19, so any remaining failure localizes outside the ART-ratio process.",
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
    labels = [str(row.get("experiment_id") or "") for row in rows]
    carry_delta = np.asarray(
        [
            float(row.get("candidate_minus_carry_forward"))
            if _finite_float(row.get("candidate_minus_carry_forward")) is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    r10_delta = np.asarray(
        [
            float(row.get("candidate_minus_r10"))
            if _finite_float(row.get("candidate_minus_r10")) is not None
            else np.nan
            for row in rows
        ],
        dtype=np.float64,
    )
    x = np.arange(len(rows), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("R34 ART-ratio branch gate", fontsize=14, fontweight="bold")
    axes[0].bar(x, carry_delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in carry_delta])
    axes[0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")
    axes[0].set_title("Candidate minus carry-forward")
    axes[0].set_ylabel("normalized MAE delta")
    axes[1].bar(x, r10_delta, color=["#2f6b4f" if value < 0.0 else "#9a3f3f" for value in r10_delta])
    axes[1].axhline(0.0, color="#111827", linewidth=1.0)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")
    axes[1].set_title("Candidate minus matched R10")
    axes[1].set_ylabel("normalized MAE delta")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r34_art_ratio_branch_gate(
    *,
    run_id: str = R34_RUN_ID,
    source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
    epigraph_root: Path | None = None,
    start_year: int = 2010,
    end_year: int = 2025,
    min_train_years: int = 5,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    phase3_root = sandbox_repo_root()
    source_run_id = resolve_active_source_run_id(root, source_run_id)
    baseline_source_run_id = resolve_baseline_source_run_id(
        root,
        source_run_id=source_run_id,
        preferred=baseline_source_run_id,
    )
    rows = build_observation_rows(root, source_run_id, baseline_source_run_id=baseline_source_run_id)
    validation_rows = build_observation_rows(
        root,
        source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        include_validation_only=True,
    )
    reports = {name: _read_path_payload(path_text) for name, path_text in _artifact_paths(phase3_root).items()}
    r10_horizon_replay = _build_r10_horizon_replay_report(
        root=root,
        horizons=R11_MULTI_HORIZON_YEARS,
        start_year=start_year,
        end_year=end_year,
        min_train_years=min_train_years,
    )
    r10_reference_mae = _r10_reference(reports.get("incidence_full_gate")) or _r10_reference(reports.get("u_to_d_coupling"))
    prediction_cache: dict[tuple[str, str, int, tuple[int, ...]], tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    annual_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for spec in _r34_specs():
        if str(spec.get("family") or "") == "official_annual_challenge_gate":
            result = _r13_annual_spec_report(
                spec=spec,
                validation_rows=validation_rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                annual_cache=annual_cache,
            )
        else:
            result = _r13_model_spec_report(
                spec=spec,
                rows=rows,
                start_year=start_year,
                end_year=end_year,
                min_train_years=min_train_years,
                r10_horizon_replay=r10_horizon_replay,
                prediction_cache=prediction_cache,
            )
        result["r10_scalar_reference_mae"] = r10_reference_mae
        results.append(result)
    flat_rows = [_flat_row(row) for row in results]
    blockers = [
        f"{row.get('experiment_id')}_{';'.join(str(item) for item in list(row.get('blockers') or [])) or 'gate_failed'}"
        for row in flat_rows
        if str(row.get("decision") or "") != "promote_for_next_wave"
    ]
    verdict = (
        "R34 passes the compact branch gate and is eligible for the next full R13/R29 replay."
        if not blockers
        else "R34 is not yet a champion: at least one official annual, full R10-scope, or program h3/h5 gate failed."
    )
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / run_id / "analysis")
    report = {
        "schema_version": R34_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "source_run_id": source_run_id,
        "baseline_source_run_id": baseline_source_run_id,
        "candidate_family": R34_FAMILY,
        "start_year": int(start_year),
        "end_year": int(end_year),
        "min_train_years": int(min_train_years),
        "promotion_eligible": not blockers,
        "blockers": blockers,
        "verdict": verdict,
        "results": results,
        "flat_rows": flat_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r34_art_ratio_branch_gate.json"
    md_path = analysis_dir / "r34_art_ratio_branch_gate.md"
    csv_path = analysis_dir / "r34_art_ratio_branch_gate.csv"
    dashboard_path = analysis_dir / "r34_art_ratio_branch_gate_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "csv": csv_path.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(csv_path, flat_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, flat_rows)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R34 ART-ratio compact branch gate.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--source-run-id", default=None)
    parser.add_argument("--baseline-source-run-id", default=None)
    parser.add_argument("--run-id", default=R34_RUN_ID)
    args = parser.parse_args()
    run_r34_art_ratio_branch_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        source_run_id=args.source_run_id,
        baseline_source_run_id=args.baseline_source_run_id,
    )


if __name__ == "__main__":
    _main()
