from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Any

from .data import default_epigraph_root, sandbox_repo_root
from .r11_sparse_state_space import _generated_at
from .runtime import ensure_dir, write_json


R45_SCHEMA_VERSION = "phase3_dynamic.r45_phase01215_determinant_readiness_audit.v1"
R45_RUN_ID = "p3d-r45-phase01215-determinant-readiness-20260502-s00"

REQUIRED_CAPABILITIES: tuple[dict[str, Any], ...] = (
    {
        "capability_id": "phase0_numeric_and_literature_candidate_extraction",
        "phase": "phase0",
        "relative_path": "src/epigraph_ph/phase0/pipeline.py",
        "evidence_tokens": (
            "structured_numeric_sources",
            "canonical_parameter_candidates",
            "phase3_target_contract",
        ),
        "scientific_role": "determinant inventory and citation-backed candidate extraction",
    },
    {
        "capability_id": "phase0_official_determinant_bridge",
        "phase": "phase0",
        "relative_path": "src/epigraph_ph/phase0/official_determinant_bridge.py",
        "evidence_tokens": ("World Bank", "WDI", "Google", "PhilHealth", "mobility"),
        "scientific_role": "official source family bridge for macro, service, mobility, and demographic determinants",
    },
    {
        "capability_id": "phase0_structured_official_numeric_sources",
        "phase": "phase0",
        "relative_path": "src/epigraph_ph/phase0/structured_numeric_sources.py",
        "evidence_tokens": ("PSA", "WDI", "GOOGLE_MOBILITY", "PhilHealth", "mobility"),
        "scientific_role": "official numeric determinant source adapters for PSA, WDI, Google Mobility, and PhilHealth rows",
    },
    {
        "capability_id": "phase1_observability_and_measurement_noise",
        "phase": "phase1",
        "relative_path": "src/epigraph_ph/phase1/pipeline.py",
        "evidence_tokens": ("observability", "measurement_role", "observation_operator", "density_tensor"),
        "scientific_role": "separate observed support from context-only determinant rows",
    },
    {
        "capability_id": "phase15_multiscale_hierarchical_latent_states",
        "phase": "phase15",
        "relative_path": "src/epigraph_ph/phase15/v2_engine.py",
        "evidence_tokens": ("hierarchical", "province", "region", "national", "observation"),
        "scientific_role": "mixed-frequency province/region/national latent state construction",
    },
    {
        "capability_id": "phase15_province_factor_graph",
        "phase": "phase15",
        "relative_path": "src/epigraph_ph/phase15/province_factor_graph.py",
        "evidence_tokens": ("province", "factor", "graph"),
        "scientific_role": "subnational determinant support graph before Phase 2 lag discovery",
    },
    {
        "capability_id": "phase2_lagged_latent_temporal_graph",
        "phase": "phase2",
        "relative_path": "src/epigraph_ph/phase2/latent_temporal_graph.py",
        "evidence_tokens": ("lag", "direct_surface", "hidden_driver_rows", "temporal"),
        "scientific_role": "lagged determinant graph and hidden/direct surface discovery",
    },
    {
        "capability_id": "phase2_structural_payload",
        "phase": "phase2",
        "relative_path": "src/epigraph_ph/phase2/structural_payload.py",
        "evidence_tokens": ("hidden_mode", "direct", "phase2_structural_payload", "tensor"),
        "scientific_role": "Phase 3 structural payload for direct covariates and hidden sidecars",
    },
    {
        "capability_id": "phase2_source_family_reestimation_ablation",
        "phase": "phase2",
        "relative_path": "src/epigraph_ph/phase2/source_reestimate_ablation.py",
        "evidence_tokens": ("source", "ablation", "reestimate", "family"),
        "scientific_role": "source-family stability gate before determinant promotion",
    },
    {
        "capability_id": "phase2_edge_falsification",
        "phase": "phase2",
        "relative_path": "src/epigraph_ph/phase2/edge_falsification.py",
        "evidence_tokens": ("source_ablation", "ablation", "time_window", "falsification"),
        "scientific_role": "blocked-time, placebo, and source ablation falsification of lagged edges",
    },
)

SCENARIO_DRIVER_MODULES: tuple[dict[str, str], ...] = (
    {
        "phase3_module": "incidence",
        "phase2_driver_family": "kp_mobility_network_exposure",
        "allowed_effect": "modulate S_eff -> incidence -> U pressure after source-stability gates",
        "blocked_until": "direct incidence validation and source-stable determinant bundles",
    },
    {
        "phase3_module": "U_to_D",
        "phase2_driver_family": "testing_sex_ed_stigma_late_diagnosis",
        "allowed_effect": "modulate diagnosis hazard and reporting/backlog state, not incidence truth",
        "blocked_until": "late-diagnosis/AHD/CD4 support and diagnosis-flow stock consistency",
    },
    {
        "phase3_module": "D_to_A",
        "phase2_driver_family": "linkage_access_art_initiation_capacity",
        "allowed_effect": "modulate linkage delay and ART initiation capacity",
        "blocked_until": "diagnosed-stock and ART-stock cone remains valid in blocked time",
    },
    {
        "phase3_module": "ART_retention",
        "phase2_driver_family": "service_continuity_ltfu_reengagement_access",
        "allowed_effect": "modulate ART interruption/removal/re-entry as sensitivity unless true cohort evidence exists",
        "blocked_until": "ART retention/interruption evidence or explicit sensitivity-only label",
    },
    {
        "phase3_module": "VL_suppression",
        "phase2_driver_family": "lab_capacity_service_quality_suppression_capacity",
        "allowed_effect": "modulate VL testing and suppression conditional-rate channels",
        "blocked_until": "conditional-rate support passes last-observed carry-forward comparison",
    },
)


def _file_sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _capability_status(epigraph_root: Path, capability: dict[str, Any]) -> dict[str, Any]:
    path = epigraph_root / str(capability["relative_path"])
    text = _read_text(path)
    tokens = tuple(str(token) for token in capability.get("evidence_tokens") or ())
    token_hits = {token: int(text.count(token)) for token in tokens}
    missing_tokens = [token for token, count in token_hits.items() if count == 0]
    line_count = len(text.splitlines()) if text else 0
    status = "present" if path.exists() and not missing_tokens else "incomplete"
    if not path.exists():
        status = "missing"
    return {
        "capability_id": capability["capability_id"],
        "phase": capability["phase"],
        "relative_path": capability["relative_path"],
        "absolute_path": path.as_posix(),
        "status": status,
        "exists": path.exists(),
        "line_count": line_count,
        "sha256": _file_sha256(path),
        "token_hits": token_hits,
        "missing_tokens": missing_tokens,
        "scientific_role": capability["scientific_role"],
    }


def _safe_rglob(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    matches: dict[str, Path] = {}
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache"}
    for pattern in patterns:
        for path in root.rglob(pattern):
            if ignored_parts.intersection(path.parts):
                continue
            if path.is_file():
                matches[path.resolve().as_posix()] = path
    return [matches[key] for key in sorted(matches)]


def _artifact_inventory(epigraph_root: Path) -> dict[str, Any]:
    patterns = (
        "*phase2_structural_payload*.json",
        "*structural_payload*.json",
        "*edge_falsification*.json",
        "*source_reestimate_ablation*.json",
        "*determinant_robustness*.json",
        "*phase15*.npz",
        "*latent*.npz",
    )
    artifacts = _safe_rglob(epigraph_root / "artifacts", patterns)
    artifacts.extend(_safe_rglob(epigraph_root / "src" / "epigraph_ph", patterns))
    unique: dict[str, Path] = {path.resolve().as_posix(): path for path in artifacts}
    rows = [
        {
            "path": path.as_posix(),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
        for path in [unique[key] for key in sorted(unique)]
    ]
    structural_payloads = [row for row in rows if "structural_payload" in str(row["name"])]
    falsification_artifacts = [
        row
        for row in rows
        if any(token in str(row["name"]) for token in ("edge_falsification", "source_reestimate", "determinant_robustness"))
    ]
    return {
        "artifact_count": len(rows),
        "structural_payload_count": len(structural_payloads),
        "falsification_artifact_count": len(falsification_artifacts),
        "artifacts": rows[:200],
        "truncated": len(rows) > 200,
    }


def _scenario_readiness(
    capability_rows: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    missing_or_incomplete = [
        row["capability_id"]
        for row in capability_rows
        if row.get("status") != "present"
    ]
    blockers: list[str] = []
    if missing_or_incomplete:
        blockers.append("required_phase_capabilities_missing_or_incomplete")
    if int(inventory.get("structural_payload_count") or 0) == 0:
        blockers.append("missing_phase2_structural_payload_artifact")
    if int(inventory.get("falsification_artifact_count") or 0) == 0:
        blockers.append("missing_phase2_source_or_edge_falsification_artifacts")
    status = "ready_for_phase3_scenario_interface" if not blockers else "blocked_pending_phase2_payload_and_falsification"
    return {
        "status": status,
        "blockers": blockers,
        "missing_or_incomplete_capabilities": missing_or_incomplete,
        "contract": (
            "Phase 2 determinants may become Phase 3 scenario drivers only after the active structural payload "
            "exists and source-family/time-window falsification artifacts show stable bundles. Candidate volume "
            "can be large in Phase 0, but promoted Phase 3 degrees of freedom must stay sparse and gated."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    readiness = dict(report.get("scenario_readiness") or {})
    lines = [
        "# Phase 0/1/15/2 Determinant Readiness Audit For Phase 3 Scenarios",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Scenario Readiness",
        "",
        f"- Status: `{readiness.get('status')}`",
        f"- Blockers: `{', '.join(readiness.get('blockers') or []) or 'none'}`",
        "",
        "## Capability Audit",
        "",
        "| Capability | Phase | Status | Missing tokens |",
        "|---|---|---|---|",
    ]
    for row in list(report.get("capability_rows") or []):
        lines.append(
            f"| {row.get('capability_id')} | {row.get('phase')} | {row.get('status')} | "
            f"{', '.join(row.get('missing_tokens') or []) or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Scenario Driver Contract",
            "",
            "| Phase 3 module | Phase 2 driver family | Allowed effect | Blocked until |",
            "|---|---|---|---|",
        ]
    )
    for row in list(report.get("scenario_driver_modules") or []):
        lines.append(
            f"| {row.get('phase3_module')} | {row.get('phase2_driver_family')} | "
            f"{row.get('allowed_effect')} | {row.get('blocked_until')} |"
        )
    lines.extend(
        [
            "",
            "## Cross-Domain Transfer Note",
            "",
            "This contract borrows from astronomy survey pipelines: keep a large weak-signal candidate catalog, "
            "then promote only sources that survive provenance, selection-bias, and held-out recovery checks. "
            "For Phase 3, that means many Phase 0 determinants are allowed as candidates, but only "
            "source-stable Phase 2 bundles can modulate semi-Markov hazards.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, capability_rows: list[dict[str, Any]], readiness: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    phases = sorted({str(row.get("phase") or "") for row in capability_rows})
    present = []
    incomplete = []
    missing = []
    for phase in phases:
        rows = [row for row in capability_rows if row.get("phase") == phase]
        present.append(sum(1 for row in rows if row.get("status") == "present"))
        incomplete.append(sum(1 for row in rows if row.get("status") == "incomplete"))
        missing.append(sum(1 for row in rows if row.get("status") == "missing"))
    y = np.arange(len(phases), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    fig.suptitle("R45 Phase 0/1/15/2 determinant readiness", fontsize=14, fontweight="bold")
    ax.barh(y, present, color="#2f6b4f", label="present")
    ax.barh(y, incomplete, left=present, color="#c49a3a", label="incomplete")
    left_missing = np.asarray(present, dtype=np.float64) + np.asarray(incomplete, dtype=np.float64)
    ax.barh(y, missing, left=left_missing, color="#9b3a34", label="missing")
    ax.set_yticks(y)
    ax.set_yticklabels(phases)
    ax.set_xlabel("capabilities")
    ax.set_title(str(readiness.get("status") or ""))
    ax.legend()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r45_phase01215_determinant_readiness_audit(
    *,
    run_id: str = R45_RUN_ID,
    epigraph_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    capability_rows = [_capability_status(root, capability) for capability in REQUIRED_CAPABILITIES]
    inventory = _artifact_inventory(root)
    readiness = _scenario_readiness(capability_rows, inventory)
    verdict = (
        "R45 found the code-level Phase 0/1/15/2 machinery and active payload/falsification artifacts needed for determinant scenario work."
        if readiness["status"] == "ready_for_phase3_scenario_interface"
        else "R45 found determinant machinery, but Phase 3 scenario use remains blocked until active Phase 2 structural payload and falsification artifacts are available for the current evidence universe."
    )
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    report = {
        "schema_version": R45_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": readiness["status"],
        "blockers": readiness["blockers"],
        "verdict": verdict,
        "epigraph_root": root.as_posix(),
        "capability_rows": capability_rows,
        "artifact_inventory": inventory,
        "scenario_readiness": readiness,
        "scenario_driver_modules": list(SCENARIO_DRIVER_MODULES),
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r45_phase01215_determinant_readiness_audit.json"
    md_path = analysis_dir / "r45_phase01215_determinant_readiness_audit.md"
    capabilities_csv = analysis_dir / "r45_capability_rows.csv"
    scenario_csv = analysis_dir / "r45_scenario_driver_modules.csv"
    dashboard_path = analysis_dir / "r45_determinant_readiness_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "capability_csv": capabilities_csv.as_posix(),
        "scenario_driver_csv": scenario_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(capabilities_csv, capability_rows)
    _write_csv(scenario_csv, list(SCENARIO_DRIVER_MODULES))
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, capability_rows, readiness)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R45 Phase 0/1/15/2 determinant readiness audit.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--run-id", default=R45_RUN_ID)
    args = parser.parse_args()
    run_r45_phase01215_determinant_readiness_audit(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
    )


if __name__ == "__main__":
    _main()
