from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS, _finite_float, _generated_at, _sha256
from .r53_publication_claim_registry import R86_DEFAULT_REPORT, R88_DEFAULT_REPORT, R89_DEFAULT_REPORT
from .r75_bulk_unaids_annual_challenge import _write_csv
from .runtime import ensure_dir, read_json, write_json


R90_SCHEMA_VERSION = "phase3_dynamic.r90_claim_grade_gate.v1"
R90_RUN_ID = "p3d-r90-claim-grade-gate-20260509-s00"

REQUIRED_ANNUAL_METRICS = tuple(OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS)
NONREGRESSION_TOLERANCE = 1e-12


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _gate_for_report(report: dict[str, Any], gate_key: str) -> dict[str, Any]:
    return dict(report.get(gate_key) or {})


def _gate_requirement(
    *,
    candidate_id: str,
    requirement_id: str,
    passed: bool,
    detail: str,
    candidate_value: float | None = None,
    reference_value: float | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "requirement_id": requirement_id,
        "passed": bool(passed),
        "candidate_value": candidate_value,
        "reference_value": reference_value,
        "detail": detail,
    }


def _append_requirement(
    rows: list[dict[str, Any]],
    blockers: list[str],
    *,
    candidate_id: str,
    requirement_id: str,
    passed: bool,
    detail: str,
    candidate_value: float | None = None,
    reference_value: float | None = None,
) -> None:
    rows.append(
        _gate_requirement(
            candidate_id=candidate_id,
            requirement_id=requirement_id,
            passed=passed,
            detail=detail,
            candidate_value=candidate_value,
            reference_value=reference_value,
        )
    )
    if not passed:
        blockers.append(requirement_id)


def _role_leakage_rows(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaked: list[dict[str, Any]] = []
    for row in score_rows:
        observation_role = str(row.get("observation_role") or "")
        allowed_use = str(row.get("allowed_use") or "")
        if observation_role != "validation_only" or allowed_use != "validation_only":
            leaked.append(
                {
                    "metric_name": row.get("metric_name"),
                    "quarter": row.get("quarter"),
                    "year": row.get("year"),
                    "horizon_years": row.get("horizon_years"),
                    "train_end_year": row.get("train_end_year"),
                    "observation_role": observation_role,
                    "allowed_use": allowed_use,
                    "source_id": row.get("source_id"),
                    "support_partition": row.get("support_partition"),
                }
            )
    return leaked


def _complete_support(gate: dict[str, Any]) -> tuple[bool, list[str]]:
    scored = dict(gate.get("scored_counts_by_metric") or {})
    targets = dict(gate.get("target_counts_by_metric") or {})
    missing: list[str] = []
    for metric_name in REQUIRED_ANNUAL_METRICS:
        scored_count = int(scored.get(metric_name) or 0)
        target_count = int(targets.get(metric_name) or 0)
        if target_count <= 0 or scored_count != target_count:
            missing.append(f"{metric_name}:{scored_count}/{target_count}")
    return not missing, missing


def _comparison_pass(candidate: Any, reference: Any, *, direction: str, tolerance: float = NONREGRESSION_TOLERANCE) -> bool:
    candidate_value = _finite_float(candidate)
    reference_value = _finite_float(reference)
    if candidate_value is None or reference_value is None:
        return False
    if direction == "lower_or_equal":
        return candidate_value <= reference_value + tolerance
    if direction == "strictly_lower":
        return candidate_value < reference_value - tolerance
    if direction == "higher_or_equal":
        return candidate_value + tolerance >= reference_value
    raise ValueError(f"unknown comparison direction: {direction}")


def _nonregression_rows(
    *,
    candidate_id: str,
    rows: list[dict[str, Any]],
    group_field: str,
    blockers: list[str],
) -> list[dict[str, Any]]:
    requirement_rows: list[dict[str, Any]] = []
    for row in rows:
        group_value = str(row.get(group_field) or "unknown")
        mean_req = f"{group_field}:{group_value}:mean_nonregression"
        p90_req = f"{group_field}:{group_value}:p90_nonregression"
        coverage_req = f"{group_field}:{group_value}:interval_coverage_nonregression"
        _append_requirement(
            requirement_rows,
            blockers,
            candidate_id=candidate_id,
            requirement_id=mean_req,
            passed=_comparison_pass(row.get("candidate_mean_norm_error"), row.get("carry_forward_mean_norm_error"), direction="lower_or_equal"),
            detail="candidate mean normalized error must not exceed carry-forward on the same scored support",
            candidate_value=_finite_float(row.get("candidate_mean_norm_error")),
            reference_value=_finite_float(row.get("carry_forward_mean_norm_error")),
        )
        _append_requirement(
            requirement_rows,
            blockers,
            candidate_id=candidate_id,
            requirement_id=p90_req,
            passed=_comparison_pass(row.get("candidate_p90_norm_error"), row.get("carry_forward_p90_norm_error"), direction="lower_or_equal"),
            detail="candidate p90 normalized error must not exceed carry-forward on the same scored support",
            candidate_value=_finite_float(row.get("candidate_p90_norm_error")),
            reference_value=_finite_float(row.get("carry_forward_p90_norm_error")),
        )
        _append_requirement(
            requirement_rows,
            blockers,
            candidate_id=candidate_id,
            requirement_id=coverage_req,
            passed=_comparison_pass(row.get("candidate_interval_coverage"), row.get("carry_forward_interval_coverage"), direction="higher_or_equal"),
            detail="candidate interval coverage must not be worse than carry-forward on the same scored support",
            candidate_value=_finite_float(row.get("candidate_interval_coverage")),
            reference_value=_finite_float(row.get("carry_forward_interval_coverage")),
        )
    return requirement_rows


def _annual_candidate_claim_grade(
    *,
    candidate_id: str,
    report: dict[str, Any],
    path: Path,
    gate_key: str,
    expected_status: str,
    claim_scope: str,
    allowed_claim_if_ready: str,
    claim_limit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    gate = _gate_for_report(report, gate_key)
    status = str(report.get("status") or gate.get("status") or "")
    blockers: list[str] = []
    requirement_rows: list[dict[str, Any]] = []
    missing_artifact = not path.exists()
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="artifact_present",
        passed=not missing_artifact,
        detail="source experiment artifact must be present and hashable",
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="primary_gate_pass",
        passed=status == expected_status,
        detail=f"source gate must equal {expected_status}",
    )
    leakage_rows = _role_leakage_rows(list(report.get("score_rows") or []))
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="no_validation_role_leakage",
        passed=not leakage_rows,
        detail="all annual challenge score rows must be validation_only / validation_only",
    )
    support_ok, support_missing = _complete_support(gate)
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="complete_required_annual_support",
        passed=support_ok,
        detail="required annual heads must score every target row: " + (";".join(support_missing) if support_missing else "complete"),
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="overall_mean_improves_carry_forward",
        passed=_comparison_pass(gate.get("candidate_mean_norm_error"), gate.get("carry_forward_mean_norm_error"), direction="strictly_lower"),
        detail="overall mean normalized error must strictly beat carry-forward",
        candidate_value=_finite_float(gate.get("candidate_mean_norm_error")),
        reference_value=_finite_float(gate.get("carry_forward_mean_norm_error")),
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="overall_interval_coverage_nonregression",
        passed=_comparison_pass(gate.get("candidate_interval_coverage"), gate.get("carry_forward_interval_coverage"), direction="higher_or_equal"),
        detail="overall interval coverage must not be worse than carry-forward",
        candidate_value=_finite_float(gate.get("candidate_interval_coverage")),
        reference_value=_finite_float(gate.get("carry_forward_interval_coverage")),
    )
    requirement_rows.extend(
        _nonregression_rows(
            candidate_id=candidate_id,
            rows=list(report.get("metric_rows") or []),
            group_field="metric_name",
            blockers=blockers,
        )
    )
    requirement_rows.extend(
        _nonregression_rows(
            candidate_id=candidate_id,
            rows=list(report.get("horizon_rows") or []),
            group_field="horizon_years",
            blockers=blockers,
        )
    )
    claim_status = "claim_grade_ready" if not blockers else "blocked"
    claim_row = {
        "candidate_id": candidate_id,
        "claim_scope": claim_scope,
        "claim_status": claim_status,
        "source_status": status,
        "model_family": report.get("candidate_family"),
        "blockers": blockers,
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": allowed_claim_if_ready if claim_status == "claim_grade_ready" else "Claim-grade publication use is blocked until every R90 requirement passes.",
        "claim_limit": claim_limit,
        "key_metrics": {
            "candidate_mean_norm_error": gate.get("candidate_mean_norm_error"),
            "carry_forward_mean_norm_error": gate.get("carry_forward_mean_norm_error"),
            "candidate_interval_coverage": gate.get("candidate_interval_coverage"),
            "carry_forward_interval_coverage": gate.get("carry_forward_interval_coverage"),
            "scored_counts_by_metric": gate.get("scored_counts_by_metric"),
            "target_counts_by_metric": gate.get("target_counts_by_metric"),
        },
        "contract": "R90 is a claim-grade adjudicator over locked source artifacts; it does not refit models or change scores.",
    }
    return claim_row, requirement_rows, leakage_rows


def _mechanism_claim_grade(r89: dict[str, Any], path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_id = "r89_incidence_mortality_mechanism_support"
    gate = _gate_for_report(r89, "incidence_mortality_mechanism_support_gate")
    status = str(r89.get("status") or gate.get("status") or "")
    source_blockers = list(gate.get("blockers") or [])
    blockers: list[str] = []
    requirement_rows: list[dict[str, Any]] = []
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="artifact_present",
        passed=path.exists(),
        detail="source mechanism-support artifact must be present and hashable",
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="mechanism_support_gate_ready",
        passed=status == "incidence_mortality_mechanism_support_ready",
        detail="R89 must be ready before incidence/death mechanism claims are allowed",
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="direct_incidence_process_support_present",
        passed=int(gate.get("direct_incidence_process_support_count") or 0) > 0,
        detail="direct process evidence for incidence must be present; annual incidence estimates alone are weak validation evidence",
        candidate_value=float(gate.get("direct_incidence_process_support_count") or 0),
        reference_value=1.0,
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="reported_death_bridge_beats_carry_forward",
        passed=_comparison_pass(
            gate.get("reported_death_bridge_candidate_mean_norm_error"),
            gate.get("reported_death_bridge_carry_forward_mean_norm_error"),
            direction="strictly_lower",
        ),
        detail="reported-death bridge must strictly beat carry-forward before AIDS-death mechanism support is claim-grade",
        candidate_value=_finite_float(gate.get("reported_death_bridge_candidate_mean_norm_error")),
        reference_value=_finite_float(gate.get("reported_death_bridge_carry_forward_mean_norm_error")),
    )
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="reported_death_bridge_interval_coverage_nonregression",
        passed=_comparison_pass(
            gate.get("reported_death_bridge_candidate_interval_coverage"),
            gate.get("reported_death_bridge_carry_forward_interval_coverage"),
            direction="higher_or_equal",
        ),
        detail="reported-death bridge interval coverage must not be worse than carry-forward",
        candidate_value=_finite_float(gate.get("reported_death_bridge_candidate_interval_coverage")),
        reference_value=_finite_float(gate.get("reported_death_bridge_carry_forward_interval_coverage")),
    )
    leakage_rows = _role_leakage_rows(list(r89.get("mortality_score_rows") or []))
    _append_requirement(
        requirement_rows,
        blockers,
        candidate_id=candidate_id,
        requirement_id="no_validation_role_leakage",
        passed=not leakage_rows,
        detail="mechanism-support annual mortality score rows must remain validation_only / validation_only",
    )
    for source_blocker in source_blockers:
        if source_blocker not in blockers:
            blockers.append(str(source_blocker))
    claim_status = "mechanism_claim_ready" if not blockers else "mechanism_claim_blocked"
    claim_row = {
        "candidate_id": candidate_id,
        "claim_scope": "raw_incidence_and_mortality_process_admissibility",
        "claim_status": claim_status,
        "source_status": status,
        "model_family": r89.get("candidate_family"),
        "blockers": blockers,
        "evidence_artifact": path.as_posix(),
        "evidence_artifact_sha256": _sha256(path) if path.exists() else None,
        "allowed_claim": (
            "Incidence and mortality process support is claim-grade enough to attempt a raw mechanism claim."
            if claim_status == "mechanism_claim_ready"
            else "Raw incidence/death mechanism claims are blocked; R86/R88 can only be cited as scoped annual/readout wins."
        ),
        "claim_limit": (
            "R89/R90 mechanism support is an admissibility condition, not a standalone forecast champion. "
            "A mechanism paper still needs a fitted mechanism branch that passes blocked-time evaluation."
        ),
        "key_metrics": gate,
        "contract": "Mechanism claims require direct process support and reported-death bridge non-regression, not only annual validation-head wins.",
    }
    return claim_row, requirement_rows, leakage_rows


def _claim_grade_gate(claim_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("candidate_id") or ""): dict(row) for row in claim_rows}
    r86_ready = str((by_id.get("r86_annual_calibrated_ledger") or {}).get("claim_status")) == "claim_grade_ready"
    r88_ready = str((by_id.get("r88_guarded_annual_ledger") or {}).get("claim_status")) == "claim_grade_ready"
    mechanism_ready = str((by_id.get("r89_incidence_mortality_mechanism_support") or {}).get("claim_status")) == "mechanism_claim_ready"
    blockers: list[str] = []
    if not r86_ready:
        blockers.append("r86_claim_grade_not_ready")
    if not r88_ready:
        blockers.append("r88_claim_grade_not_ready")
    if not mechanism_ready:
        blockers.append("incidence_mortality_mechanism_claim_blocked")
    if r86_ready and r88_ready and mechanism_ready:
        status = "claim_grade_annual_readout_and_mechanisms_ready"
    elif r86_ready and r88_ready:
        status = "claim_grade_annual_readout_ready_mechanisms_blocked"
    else:
        status = "claim_grade_blocked"
    return {
        "status": status,
        "blockers": blockers,
        "r86_annual_calibrated_ledger_ready": r86_ready,
        "r88_guarded_annual_ledger_ready": r88_ready,
        "incidence_mortality_mechanism_ready": mechanism_ready,
        "allowed_claims": [
            row.get("allowed_claim")
            for row in claim_rows
            if str(row.get("claim_status") or "") in {"claim_grade_ready", "mechanism_claim_ready"}
        ],
        "blocked_claims": [
            {
                "candidate_id": row.get("candidate_id"),
                "claim_status": row.get("claim_status"),
                "blockers": row.get("blockers"),
                "allowed_claim": row.get("allowed_claim"),
            }
            for row in claim_rows
            if str(row.get("claim_status") or "") not in {"claim_grade_ready", "mechanism_claim_ready"}
        ],
        "contract": (
            "R90 separates publication-grade annual/readout wins from raw incidence/death mechanism claims. "
            "It requires validation-only scoring rows, complete annual support, per-metric and per-horizon non-regression, "
            "interval coverage non-regression, and R89 mechanism-support readiness."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("claim_grade_gate") or {})
    lines = [
        "# Phase 3 R90 Claim-Grade Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- R86 annual calibrated ledger ready: `{gate.get('r86_annual_calibrated_ledger_ready')}`",
        f"- R88 guarded annual ledger ready: `{gate.get('r88_guarded_annual_ledger_ready')}`",
        f"- Incidence/mortality mechanism ready: `{gate.get('incidence_mortality_mechanism_ready')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Claim Rows",
        "",
        "| Candidate | Scope | Status | Allowed Claim | Limit |",
        "|---|---|---|---|---|",
    ]
    for row in list(report.get("claim_rows") or []):
        lines.append(
            f"| `{row.get('candidate_id')}` | `{row.get('claim_scope')}` | `{row.get('claim_status')}` | "
            f"{row.get('allowed_claim')} | {row.get('claim_limit')} |"
        )
    lines.extend(["", "## Failed Requirements", ""])
    failed = [row for row in list(report.get("requirement_rows") or []) if not bool(row.get("passed"))]
    if not failed:
        lines.append("No failed requirements.")
    else:
        lines.extend(["| Candidate | Requirement | Detail | Candidate | Reference |", "|---|---|---|---:|---:|"])
        for row in failed:
            lines.append(
                f"| `{row.get('candidate_id')}` | `{row.get('requirement_id')}` | {row.get('detail')} | "
                f"{row.get('candidate_value')} | {row.get('reference_value')} |"
            )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r90_claim_grade_gate(
    *,
    run_id: str = R90_RUN_ID,
    r86_report_path: Path | None = None,
    r88_report_path: Path | None = None,
    r89_report_path: Path | None = None,
) -> dict[str, Any]:
    r86_path = Path(r86_report_path) if r86_report_path is not None else R86_DEFAULT_REPORT
    r88_path = Path(r88_report_path) if r88_report_path is not None else R88_DEFAULT_REPORT
    r89_path = Path(r89_report_path) if r89_report_path is not None else R89_DEFAULT_REPORT
    r86 = _load_report(r86_path)
    r88 = _load_report(r88_path)
    r89 = _load_report(r89_path)
    claim_rows: list[dict[str, Any]] = []
    requirement_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    claim, reqs, leaks = _annual_candidate_claim_grade(
        candidate_id="r86_annual_calibrated_ledger",
        report=r86,
        path=r86_path,
        gate_key="annual_calibrated_forecast_grid_ledger_gate",
        expected_status="annual_calibrated_forecast_grid_ledger_pass",
        claim_scope="complete_annual_ledger_readout_with_train_origin_calibration",
        allowed_claim_if_ready=(
            "R86 is claim-grade as a scoped annual/readout ledger win: it beats carry-forward on validation-only annual "
            "incidence, AIDS deaths, and PLHIV targets with complete support and no per-metric or per-horizon regression."
        ),
        claim_limit=(
            "Annual/readout claim only. R86 does not identify direct quarterly incidence or AIDS-death mechanisms and does "
            "not justify broad official-model replacement claims."
        ),
    )
    claim_rows.append(claim)
    requirement_rows.extend(reqs)
    leakage_rows.extend(leaks)
    claim, reqs, leaks = _annual_candidate_claim_grade(
        candidate_id="r88_guarded_annual_ledger",
        report=r88,
        path=r88_path,
        gate_key="guarded_annual_ledger_selector_gate",
        expected_status="guarded_annual_ledger_selector_pass",
        claim_scope="guarded_annual_readout_selector_with_carry_forward_fallback",
        allowed_claim_if_ready=(
            "R88 is claim-grade as a conservative guarded annual/readout selector: raw quarterly process channels are used "
            "only where train-window evidence beats carry-forward, otherwise the carry-forward prior is retained."
        ),
        claim_limit=(
            "Guarded annual/readout claim only. Incidence and death channels that fall back to carry-forward are not evidence "
            "of identified raw mechanisms."
        ),
    )
    claim_rows.append(claim)
    requirement_rows.extend(reqs)
    leakage_rows.extend(leaks)
    claim, reqs, leaks = _mechanism_claim_grade(r89, r89_path)
    claim_rows.append(claim)
    requirement_rows.extend(reqs)
    leakage_rows.extend(leaks)
    gate = _claim_grade_gate(claim_rows)
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    json_path = analysis_dir / "r90_claim_grade_gate_report.json"
    md_path = analysis_dir / "r90_claim_grade_gate_report.md"
    claim_csv_path = analysis_dir / "r90_claim_rows.csv"
    req_csv_path = analysis_dir / "r90_requirement_rows.csv"
    leakage_csv_path = analysis_dir / "r90_leakage_rows.csv"
    source_artifacts = {
        "r86_report": r86_path.as_posix(),
        "r86_report_sha256": _sha256(r86_path) if r86_path.exists() else None,
        "r88_report": r88_path.as_posix(),
        "r88_report_sha256": _sha256(r88_path) if r88_path.exists() else None,
        "r89_report": r89_path.as_posix(),
        "r89_report_sha256": _sha256(r89_path) if r89_path.exists() else None,
    }
    report = {
        "schema_version": R90_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "claim_grade_gate": gate,
        "claim_rows": claim_rows,
        "requirement_rows": requirement_rows,
        "leakage_rows": leakage_rows,
        "source_artifacts": source_artifacts,
        "artifact_paths": {
            "json": json_path.as_posix(),
            "markdown": md_path.as_posix(),
            "claim_rows_csv": claim_csv_path.as_posix(),
            "requirement_rows_csv": req_csv_path.as_posix(),
            "leakage_rows_csv": leakage_csv_path.as_posix(),
        },
    }
    write_json(json_path, report)
    _write_csv(claim_csv_path, claim_rows)
    _write_csv(req_csv_path, requirement_rows)
    _write_csv(leakage_csv_path, leakage_rows)
    _write_markdown(md_path, report)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R90 claim-grade gate over locked R86/R88/R89 artifacts.")
    parser.add_argument("--run-id", default=R90_RUN_ID)
    parser.add_argument("--r86-report-path", default=None)
    parser.add_argument("--r88-report-path", default=None)
    parser.add_argument("--r89-report-path", default=None)
    args = parser.parse_args()
    run_r90_claim_grade_gate(
        run_id=str(args.run_id),
        r86_report_path=None if args.r86_report_path is None else Path(args.r86_report_path),
        r88_report_path=None if args.r88_report_path is None else Path(args.r88_report_path),
        r89_report_path=None if args.r89_report_path is None else Path(args.r89_report_path),
    )


if __name__ == "__main__":
    _main()
