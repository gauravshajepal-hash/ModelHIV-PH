from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r42_r41_champion_hardening import R42_RUN_ID
from .r46_phase2_lineage_driver_gate import R46_RUN_ID
from .r53_publication_claim_registry import R53_RUN_ID
from .r60_regional_experiment_queue import R60_RUN_ID
from .r64_leakage_support_gap_prioritizer import R64_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R65_SCHEMA_VERSION = "phase3_dynamic.r65_transmission_model_readiness_gate.v1"
R65_RUN_ID = "p3d-r65-transmission-model-readiness-gate-20260503-s00"
R42_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R42_RUN_ID
    / "analysis"
    / "r42_r41_champion_hardening_report.json"
)
R46_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R46_RUN_ID
    / "analysis"
    / "r46_phase2_lineage_driver_gate_report.json"
)
R53_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R53_RUN_ID
    / "analysis"
    / "r53_publication_claim_registry_report.json"
)
R60_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R60_RUN_ID
    / "analysis"
    / "r60_regional_experiment_queue_report.json"
)
R64_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R64_RUN_ID
    / "analysis"
    / "r64_leakage_support_gap_prioritizer_report.json"
)


def _read_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _module_status_rows(
    *,
    r42: dict[str, Any],
    r46: dict[str, Any],
    r53: dict[str, Any],
    r60: dict[str, Any],
    r64: dict[str, Any],
) -> list[dict[str, Any]]:
    registry = dict(r53.get("registry_gate") or {})
    determinant_gate = dict(r46.get("lineage_gate") or {})
    r60_gate = dict(r60.get("queue_gate") or {})
    r64_gate = dict(r64.get("support_gap_gate") or {})
    national_ready = (
        str(r42.get("strict_gate_status") or "") == "pass"
        and str(r42.get("promotion_claim") or "") == "freeze_as_national_research_champion"
    )
    strict_determinants = int(determinant_gate.get("strict_phase3_prior_count") or 0)
    sensitivity_determinants = int(determinant_gate.get("sensitivity_only_driver_count") or 0)
    regional_promoted = bool(registry.get("regional_adapter_promoted") or registry.get("regional_readout_promoted"))
    regional_strict = str(registry.get("regional_r60_experiment_queue_status") or "") == "promoted"
    top_stream = str(r64_gate.get("top_metric_stream") or "")
    rows = [
        {
            "module_id": "national_forecast_readout",
            "readiness_status": "ready" if national_ready else "blocked",
            "allowed_use": "national forecast/readout champion" if national_ready else "not claimable",
            "evidence_basis": "R42 strict mapped gate",
            "primary_blocker": "" if national_ready else "R42 strict gate not passed",
            "next_experiment": "maintain source-family ablation and external annual challenge refresh",
        },
        {
            "module_id": "national_full_transmission_scenario",
            "readiness_status": "sensitivity_only" if national_ready and sensitivity_determinants > 0 and strict_determinants == 0 else ("ready" if strict_determinants > 0 else "blocked"),
            "allowed_use": "national scenario sensitivity, not fitted determinant-prior champion" if strict_determinants == 0 else "national determinant-prior transmission scenario",
            "evidence_basis": "R42 national champion plus R46 determinant lineage gate",
            "primary_blocker": "no strict source-stable time-validated Phase2 determinant prior" if strict_determinants == 0 else "",
            "next_experiment": "rerun Phase2 source-family re-estimation and blocked-time determinant recovery",
        },
        {
            "module_id": "regional_cascade_readout",
            "readiness_status": "mean_ready_split_limited" if regional_promoted and not regional_strict else ("ready" if regional_strict else "blocked"),
            "allowed_use": "regional readout/proxy claim with split-limited caveat" if regional_promoted else "not claimable",
            "evidence_basis": "R53 registry, R60 regional queue",
            "primary_blocker": "R60 remains split-limited" if regional_promoted and not regional_strict else "",
            "next_experiment": "onboard additional historical regional HASP periods; do not add free regional equations",
        },
        {
            "module_id": "regional_full_transmission_model",
            "readiness_status": "blocked",
            "allowed_use": "not claimable; use regional cascade diagnostics only",
            "evidence_basis": "R60/R63/R64 regional gates",
            "primary_blocker": f"regional oracle gap dominated by {top_stream or 'unknown'} and no blocked-time leakage student passes",
            "next_experiment": "identify regional VL/suppression and ART-retention support before fitting regional incidence/transmission effects",
        },
        {
            "module_id": "leakage_expert_overlay",
            "readiness_status": "diagnostic_only",
            "allowed_use": "oracle ceiling, failure anatomy, evidence acquisition priority",
            "evidence_basis": "R62/R63/R64",
            "primary_blocker": "same-holdout oracle is target leakage; blocked-time students fail",
            "next_experiment": "use R64 priorities to acquire support, then rerun online student",
        },
    ]
    return rows


def _state_equation_rows() -> list[dict[str, Any]]:
    return [
        {
            "state_or_transition": "S_eff -> I",
            "equation": "I_{g,t} = S_eff_{g,t} * (1 - exp(-lambda_{g,t} * Delta_t))",
            "current_status": "national_partial_regional_blocked",
            "identification_requirement": "population denominator, incidence validation, determinant-stable exposure pressure",
        },
        {
            "state_or_transition": "I -> U_early -> U_late",
            "equation": "U_early,U_late advance through diagnosis-delay/backlog hazards with late-diagnosis emissions",
            "current_status": "national_partial_regional_blocked",
            "identification_requirement": "CD4/AHD/late-presenter support by time and geography",
        },
        {
            "state_or_transition": "U -> D",
            "equation": "h_{U->D,g,t} = f(reporting intensity, testing reach, backlog, shock state)",
            "current_status": "national_supported_regional_split_limited",
            "identification_requirement": "monthly diagnosis/reporting support by region",
        },
        {
            "state_or_transition": "D -> A",
            "equation": "h_{D->A,g,t,l} uses linkage delay l and ART initiation capacity",
            "current_status": "national_supported_regional_split_limited",
            "identification_requirement": "ART initiation/linkage support by region and period",
        },
        {
            "state_or_transition": "A -> interrupted/reengaged/removal",
            "equation": "A exits through mortality/removal, interruption/LTFU, transfer-out; reentry is sensitivity unless observed",
            "current_status": "national_partial_regional_blocked",
            "identification_requirement": "ART active stock, retention, interruption, death, transfer-out support",
        },
        {
            "state_or_transition": "A -> VL_tested -> suppressed",
            "equation": "VL testing and suppression are service states/observation processes, not endpoint-only corrections",
            "current_status": "highest_regional_blocker",
            "identification_requirement": "regional VL testing numerator, lab capacity, suppression conditional on VL testing",
        },
    ]


def _experiment_queue_rows(r64: dict[str, Any]) -> list[dict[str, Any]]:
    top_region_metric = list(r64.get("region_metric_priority_rows") or [])[:10]
    rows = [
        {
            "priority": 1,
            "experiment_id": "T65-01",
            "experiment": "onboard_historical_regional_vl_suppression_support",
            "why": "R64 shows VL/suppression service stream dominates the leakage gap",
            "promotion_gate": "rerun R60/R63; require no regression in regional/mass/share and split improvement",
        },
        {
            "priority": 2,
            "experiment_id": "T65-02",
            "experiment": "regional_art_retention_support_layer",
            "why": "ART stock/retention is second-largest R64 stream gap",
            "promotion_gate": "alive_on_art split failures must reduce without worsening VL/suppression",
        },
        {
            "priority": 3,
            "experiment_id": "T65-03",
            "experiment": "blocked_time_phase2_determinant_recovery",
            "why": "R46 has zero strict determinant priors; full transmission scenario remains sensitivity-only",
            "promotion_gate": "at least one direct determinant edge survives source-family re-estimation and blocked-time validation",
        },
        {
            "priority": 4,
            "experiment_id": "T65-04",
            "experiment": "regional_incidence_proxy_holdout_gate_after_service_support",
            "why": "Regional incidence/transmission effects should not be fitted until service observation drift is identified",
            "promotion_gate": "annual incidence remains validation/weak measurement; no diagnosis-flow incidence truth",
        },
    ]
    for index, item in enumerate(top_region_metric, start=5):
        rows.append(
            {
                "priority": index,
                "experiment_id": f"T65-{index:02d}",
                "experiment": f"targeted_support_gap_{item.get('region')}_{item.get('metric_name')}",
                "why": f"R64 region-metric gap sum={item.get('sum_positive_student_gap_vs_oracle')}",
                "promotion_gate": str(item.get("top_evidence_recommendation") or ""),
            }
        )
    return rows


def _gate(module_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_module = {str(row.get("module_id") or ""): dict(row) for row in module_rows}
    national_ready = str((by_module.get("national_forecast_readout") or {}).get("readiness_status")) == "ready"
    regional_transmission_ready = str((by_module.get("regional_full_transmission_model") or {}).get("readiness_status")) == "ready"
    blockers: list[str] = []
    if not national_ready:
        blockers.append("national_forecast_readout_not_ready")
    if not regional_transmission_ready:
        blockers.append("regional_full_transmission_not_ready")
    return {
        "status": "full_transmission_model_ready" if not blockers else "full_transmission_model_not_yet_ready",
        "blockers": blockers,
        "national_forecast_readout_ready": national_ready,
        "regional_full_transmission_ready": regional_transmission_ready,
        "contract": (
            "R65 is a readiness gate, not a forecast. It prevents claiming a full national/regional transmission model "
            "until the evidence supports incidence, diagnosis delay, care transitions, service states, geography, and "
            "Phase2 determinant priors under blocked time."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("readiness_gate") or {})
    lines = [
        "# Phase 3 R65 Transmission Model Readiness Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- National readout ready: `{gate.get('national_forecast_readout_ready')}`",
        f"- Regional full transmission ready: `{gate.get('regional_full_transmission_ready')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Module Status",
        "",
        "| Module | Status | Allowed Use | Blocker |",
        "|---|---|---|---|",
    ]
    for row in list(report.get("module_status_rows") or []):
        lines.append(
            f"| `{row.get('module_id')}` | `{row.get('readiness_status')}` | {row.get('allowed_use')} | {row.get('primary_blocker')} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r65_transmission_model_readiness_gate(
    *,
    run_id: str = R65_RUN_ID,
    r42_report_path: Path | None = None,
    r46_report_path: Path | None = None,
    r53_report_path: Path | None = None,
    r60_report_path: Path | None = None,
    r64_report_path: Path | None = None,
) -> dict[str, Any]:
    r42_path = Path(r42_report_path) if r42_report_path is not None else R42_DEFAULT_REPORT
    r46_path = Path(r46_report_path) if r46_report_path is not None else R46_DEFAULT_REPORT
    r53_path = Path(r53_report_path) if r53_report_path is not None else R53_DEFAULT_REPORT
    r60_path = Path(r60_report_path) if r60_report_path is not None else R60_DEFAULT_REPORT
    r64_path = Path(r64_report_path) if r64_report_path is not None else R64_DEFAULT_REPORT
    r42 = _read_report(r42_path)
    r46 = _read_report(r46_path)
    r53 = _read_report(r53_path)
    r60 = _read_report(r60_path)
    r64 = _read_report(r64_path)
    module_rows = _module_status_rows(r42=r42, r46=r46, r53=r53, r60=r60, r64=r64)
    state_rows = _state_equation_rows()
    experiment_rows = _experiment_queue_rows(r64)
    gate = _gate(module_rows)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r65_transmission_model_readiness_gate_report.json"
    md_path = analysis_dir / "r65_transmission_model_readiness_gate_report.md"
    module_csv = analysis_dir / "r65_module_status_rows.csv"
    state_csv = analysis_dir / "r65_state_equation_rows.csv"
    experiments_csv = analysis_dir / "r65_next_experiment_queue.csv"
    report = {
        "schema_version": R65_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": (
            "R65 says a full national/regional transmission model is ready."
            if gate["status"] == "full_transmission_model_ready"
            else "R65 blocks broad full-transmission claims: national readout is ready, but regional full-transmission modeling needs service-state support and strict determinant priors."
        ),
        "source_artifacts": {
            "r42": {"path": r42_path.as_posix(), "sha256": _sha256(r42_path) if r42_path.exists() else None},
            "r46": {"path": r46_path.as_posix(), "sha256": _sha256(r46_path) if r46_path.exists() else None},
            "r53": {"path": r53_path.as_posix(), "sha256": _sha256(r53_path) if r53_path.exists() else None},
            "r60": {"path": r60_path.as_posix(), "sha256": _sha256(r60_path) if r60_path.exists() else None},
            "r64": {"path": r64_path.as_posix(), "sha256": _sha256(r64_path) if r64_path.exists() else None},
        },
        "readiness_gate": gate,
        "module_status_rows": module_rows,
        "state_equation_rows": state_rows,
        "next_experiment_queue": experiment_rows,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "module_status_csv": module_csv.as_posix(),
            "state_equations_csv": state_csv.as_posix(),
            "next_experiment_queue_csv": experiments_csv.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(module_csv, module_rows)
    _write_csv(state_csv, state_rows)
    _write_csv(experiments_csv, experiment_rows)
    _write_markdown(md_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R65 transmission model readiness gate.")
    parser.add_argument("--run-id", default=R65_RUN_ID)
    parser.add_argument("--r42-report-path", default=None)
    parser.add_argument("--r46-report-path", default=None)
    parser.add_argument("--r53-report-path", default=None)
    parser.add_argument("--r60-report-path", default=None)
    parser.add_argument("--r64-report-path", default=None)
    args = parser.parse_args()
    run_r65_transmission_model_readiness_gate(
        run_id=str(args.run_id),
        r42_report_path=None if args.r42_report_path is None else Path(args.r42_report_path),
        r46_report_path=None if args.r46_report_path is None else Path(args.r46_report_path),
        r53_report_path=None if args.r53_report_path is None else Path(args.r53_report_path),
        r60_report_path=None if args.r60_report_path is None else Path(args.r60_report_path),
        r64_report_path=None if args.r64_report_path is None else Path(args.r64_report_path),
    )


if __name__ == "__main__":
    _main()
