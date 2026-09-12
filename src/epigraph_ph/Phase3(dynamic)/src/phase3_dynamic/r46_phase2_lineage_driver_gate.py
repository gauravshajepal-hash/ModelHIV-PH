from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

from .data import default_epigraph_root, sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .runtime import ensure_dir, read_json, write_json


R46_SCHEMA_VERSION = "phase3_dynamic.r46_phase2_lineage_driver_gate.v1"
R46_RUN_ID = "p3d-r46-phase2-lineage-driver-gate-20260502-s00"
DEFAULT_PHASE2_LINEAGE_RUN_ID = "phase0-2-incidence-official-augmented-20260429-s01"

BLOCK_TO_PHASE3_MODULES: dict[str, tuple[str, ...]] = {
    "testing_prevention_reach": ("incidence", "U_to_D"),
    "care_access_continuity": ("D_to_A", "ART_retention", "VL_suppression"),
    "suppression_capacity": ("VL_suppression",),
    "mobility_exposure_pressure": ("incidence",),
    "structural_barrier_pressure": ("incidence", "U_to_D", "D_to_A", "ART_retention"),
}


def _default_phase2_lineage_run_dir(epigraph_root: Path) -> Path:
    return Path(epigraph_root) / "artifacts" / "runs" / DEFAULT_PHASE2_LINEAGE_RUN_ID


def _load_report(path: Path) -> dict[str, Any]:
    return dict(read_json(path, default={}) or {}) if path.exists() else {}


def _lineage_paths(lineage_run_dir: Path) -> dict[str, Path]:
    phase2_dir = Path(lineage_run_dir) / "phase2"
    return {
        "structural_payload": phase2_dir / "phase2_structural_payload.json",
        "determinant_robustness": phase2_dir / "determinant_robustness" / "phase2_determinant_robustness_report.json",
        "determinant_robustness_broad": phase2_dir
        / "determinant_robustness_broad"
        / "phase2_determinant_robustness_report.json",
        "edge_falsification": phase2_dir / "falsification_with_reestimate" / "phase2_edge_falsification_report.json",
        "edge_falsification_broad": phase2_dir
        / "falsification_with_broad_reestimate"
        / "phase2_edge_falsification_report.json",
        "phase0_evidence_ledger": Path(lineage_run_dir)
        / "phase0"
        / "evidence_ledger"
        / "phase3_determinant_evidence_ledger.jsonl",
    }


def _select_robustness_report(paths: dict[str, Path]) -> tuple[str, Path | None, dict[str, Any]]:
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for label in ("determinant_robustness_broad", "determinant_robustness"):
        path = paths[label]
        report = _load_report(path)
        if report:
            candidates.append((label, path, report))
    if not candidates:
        return "missing", None, {}
    candidates.sort(
        key=lambda item: int((item[2].get("completed_source_family_count") or 0)),
        reverse=True,
    )
    return candidates[0]


def _edge_driver_status(edge: dict[str, Any]) -> str:
    if bool(edge.get("phase3_default_allowed")) or bool(edge.get("phase3_prior_eligible")):
        return "strict_phase3_prior"
    if str(edge.get("edge_kind") or "") == "hidden":
        return "hidden_diagnostic_only"
    if (
        str(edge.get("edge_kind") or "") == "direct"
        and bool(edge.get("support_ablation_passed"))
        and bool(edge.get("source_reestimated_passed"))
        and int(edge.get("sign_conflict_family_count") or 0) == 0
    ):
        if bool(edge.get("blocked_time_passed")):
            return "source_stable_direct_driver"
        return "source_stable_but_time_blocked_sensitivity_only"
    return "blocked"


def _module_targets_for_edge(edge: dict[str, Any]) -> list[str]:
    modules: set[str] = set()
    for block in (str(edge.get("source") or ""), str(edge.get("target") or "")):
        modules.update(BLOCK_TO_PHASE3_MODULES.get(block, ()))
    return sorted(modules)


def _driver_rows(robustness_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for edge in list(robustness_report.get("edge_rows") or []):
        source_families = sorted(
            {
                str(row.get("excluded_source_family") or "")
                for row in list(edge.get("evaluation_rows") or [])
                if row.get("excluded_source_family")
            }
        )
        status = _edge_driver_status(dict(edge))
        rows.append(
            {
                "edge_key": edge.get("edge_key"),
                "edge_kind": edge.get("edge_kind"),
                "source": edge.get("source"),
                "target": edge.get("target"),
                "lag": edge.get("lag"),
                "baseline_weight": edge.get("baseline_weight"),
                "driver_status": status,
                "phase3_modules": _module_targets_for_edge(dict(edge)),
                "support_ablation_passed": bool(edge.get("support_ablation_passed")),
                "source_reestimated_passed": bool(edge.get("source_reestimated_passed")),
                "blocked_time_passed": bool(edge.get("blocked_time_passed")),
                "survival_fraction": edge.get("survival_fraction"),
                "evaluated_family_count": int(edge.get("evaluated_family_count") or 0),
                "survived_family_count": int(edge.get("survived_family_count") or 0),
                "sign_conflict_family_count": int(edge.get("sign_conflict_family_count") or 0),
                "source_families": source_families,
                "scientific_status": edge.get("scientific_status"),
                "promotion_blockers": list(edge.get("promotion_blockers") or []),
            }
        )
    return rows


def _bundle_rows(robustness_report: dict[str, Any], driver_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bundle = {
        str(row.get("bundle_key") or ""): dict(row)
        for row in list(robustness_report.get("bundle_rows") or [])
        if row.get("bundle_key")
    }
    status_by_pair: dict[tuple[str, str, str], Counter[str]] = {}
    for row in driver_rows:
        key = (str(row.get("edge_kind") or ""), str(row.get("source") or ""), str(row.get("target") or ""))
        status_by_pair.setdefault(key, Counter())[str(row.get("driver_status") or "blocked")] += 1
    output: list[dict[str, Any]] = []
    for key, row in sorted(by_bundle.items()):
        pair = (str(row.get("edge_kind") or ""), str(row.get("source") or ""), str(row.get("target") or ""))
        status_counts = dict(sorted(status_by_pair.get(pair, Counter()).items()))
        strict_count = int(status_counts.get("strict_phase3_prior", 0))
        sensitivity_count = int(status_counts.get("source_stable_but_time_blocked_sensitivity_only", 0))
        output.append(
            {
                "bundle_key": key,
                "edge_kind": row.get("edge_kind"),
                "source": row.get("source"),
                "target": row.get("target"),
                "lags": list(row.get("lags") or []),
                "phase3_modules": sorted(
                    set(BLOCK_TO_PHASE3_MODULES.get(str(row.get("source") or ""), ()))
                    | set(BLOCK_TO_PHASE3_MODULES.get(str(row.get("target") or ""), ()))
                ),
                "edge_count": int(row.get("edge_count") or 0),
                "strict_phase3_prior_count": strict_count,
                "sensitivity_only_count": sensitivity_count,
                "driver_status_counts": status_counts,
                "min_survival_fraction": row.get("min_survival_fraction"),
                "max_survival_fraction": row.get("max_survival_fraction"),
                "phase3_default_allowed_count": int(row.get("phase3_default_allowed_count") or 0),
            }
        )
    return output


def _lineage_gate(driver_rows: list[dict[str, Any]], robustness_report: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(str(row.get("driver_status") or "blocked") for row in driver_rows)
    strict_count = int(status_counts.get("strict_phase3_prior", 0))
    source_stable_count = int(status_counts.get("source_stable_direct_driver", 0))
    sensitivity_count = int(status_counts.get("source_stable_but_time_blocked_sensitivity_only", 0))
    hidden_count = int(status_counts.get("hidden_diagnostic_only", 0))
    blockers: list[str] = []
    if strict_count == 0:
        blockers.append("no_direct_edge_survives_strict_phase3_prior_gate")
    if strict_count == 0 and source_stable_count == 0:
        blockers.append("no_source_stable_time_validated_direct_driver")
    if int(robustness_report.get("completed_source_family_count") or 0) == 0:
        blockers.append("no_completed_source_family_reestimation")
    status = "strict_determinant_priors_ready" if not blockers else "sensitivity_only_determinant_scenarios"
    if not driver_rows:
        status = "blocked_missing_driver_rows"
    return {
        "status": status,
        "blockers": blockers,
        "driver_status_counts": dict(sorted(status_counts.items())),
        "strict_phase3_prior_count": strict_count,
        "source_stable_direct_driver_count": source_stable_count,
        "sensitivity_only_driver_count": sensitivity_count,
        "hidden_diagnostic_count": hidden_count,
        "completed_source_family_count": int(robustness_report.get("completed_source_family_count") or 0),
        "completed_source_families": list(robustness_report.get("completed_source_families") or []),
        "contract": (
            "Strict Phase 3 priors require direct edges that pass support ablation, true source-family "
            "re-estimation, and blocked-time falsification. Source-stable but time-blocked edges may only be "
            "used as labeled sensitivity/scenario sliders, not as fitted champion priors."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = {}
        for key, value in row.items():
            normalized[key] = "|".join(str(item) for item in value) if isinstance(value, list) else value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("lineage_gate") or {})
    lines = [
        "# Phase 3 R46 Phase 2 Lineage Driver Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Verdict",
        "",
        str(report.get("verdict") or ""),
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Strict Phase 3 prior edges: `{gate.get('strict_phase3_prior_count')}`",
        f"- Source-stable time-valid direct drivers: `{gate.get('source_stable_direct_driver_count')}`",
        f"- Sensitivity-only drivers: `{gate.get('sensitivity_only_driver_count')}`",
        f"- Hidden diagnostic edges: `{gate.get('hidden_diagnostic_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Driver Bundles",
        "",
        "| Bundle | Modules | Strict priors | Sensitivity-only | Status counts |",
        "|---|---|---:|---:|---|",
    ]
    for row in list(report.get("bundle_rows") or []):
        status_counts = ", ".join(f"{key}={value}" for key, value in dict(row.get("driver_status_counts") or {}).items())
        lines.append(
            f"| {row.get('bundle_key')} | {', '.join(row.get('phase3_modules') or [])} | "
            f"{row.get('strict_phase3_prior_count')} | {row.get('sensitivity_only_count')} | {status_counts} |"
        )
    lines.extend(
        [
            "",
            "## Contract",
            "",
            str(gate.get("contract") or ""),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_dashboard(path: Path, driver_rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        path.write_text("matplotlib unavailable", encoding="utf-8")
        return
    status_counts = Counter(str(row.get("driver_status") or "blocked") for row in driver_rows)
    labels = list(status_counts.keys())
    values = np.asarray([status_counts[label] for label in labels], dtype=np.float64)
    module_counts: Counter[str] = Counter()
    for row in driver_rows:
        for module in list(row.get("phase3_modules") or []):
            module_counts[str(module)] += 1
    module_labels = list(module_counts.keys())
    module_values = np.asarray([module_counts[label] for label in module_labels], dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle("R46 Phase 2 driver gate", fontsize=14, fontweight="bold")
    axes[0].barh(np.arange(len(labels)), values, color="#5c6f83")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("edge count")
    axes[0].set_title(str(gate.get("status") or ""))
    axes[1].barh(np.arange(len(module_labels)), module_values, color="#8a6f2a")
    axes[1].set_yticks(np.arange(len(module_labels)))
    axes[1].set_yticklabels(module_labels)
    axes[1].set_xlabel("edge-module mentions")
    axes[1].set_title("Affected Phase 3 modules")
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r46_phase2_lineage_driver_gate(
    *,
    run_id: str = R46_RUN_ID,
    epigraph_root: Path | None = None,
    phase2_lineage_run_dir: Path | None = None,
    determinant_robustness_path: Path | None = None,
    edge_falsification_path: Path | None = None,
) -> dict[str, Any]:
    root = Path(epigraph_root) if epigraph_root else default_epigraph_root()
    lineage_dir = Path(phase2_lineage_run_dir) if phase2_lineage_run_dir else _default_phase2_lineage_run_dir(root)
    paths = _lineage_paths(lineage_dir)
    if determinant_robustness_path is not None:
        selected_label = "explicit_determinant_robustness_path"
        selected_path = Path(determinant_robustness_path)
        robustness_report = _load_report(selected_path)
    else:
        selected_label, selected_path, robustness_report = _select_robustness_report(paths)
    structural_payload = _load_report(paths["structural_payload"])
    edge_falsification = (
        _load_report(Path(edge_falsification_path))
        if edge_falsification_path is not None
        else _load_report(paths["edge_falsification_broad"]) or _load_report(paths["edge_falsification"])
    )
    driver_rows = _driver_rows(robustness_report)
    bundle_rows = _bundle_rows(robustness_report, driver_rows)
    gate = _lineage_gate(driver_rows, robustness_report)
    verdict = (
        "R46 found strict Phase 2 determinant priors ready for Phase 3 scenario/fitted-driver experiments."
        if gate["status"] == "strict_determinant_priors_ready"
        else "R46 found source-stable exploratory determinant edges, but zero strict direct priors. Phase 2 can drive labeled sensitivity scenarios, not fitted champion priors, until blocked-time edge recovery improves."
    )
    phase3_root = sandbox_repo_root()
    analysis_dir = ensure_dir(phase3_root / "artifacts" / "runs" / str(run_id) / "analysis")
    report = {
        "schema_version": R46_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "status": gate["status"],
        "blockers": gate["blockers"],
        "verdict": verdict,
        "epigraph_root": root.as_posix(),
        "phase2_lineage_run_dir": lineage_dir.as_posix(),
        "selected_robustness_report_label": selected_label,
        "selected_robustness_report_path": None if selected_path is None else selected_path.as_posix(),
        "selected_edge_falsification_report_path": None if edge_falsification_path is None else Path(edge_falsification_path).as_posix(),
        "lineage_artifact_paths": {
            key: value.as_posix()
            for key, value in paths.items()
        },
        "lineage_artifact_sha256": {
            key: (_sha256(value) if value.exists() and value.is_file() else None)
            for key, value in paths.items()
        },
        "structural_payload_counts": {
            "direct_temporal_edge_rows": len(list(structural_payload.get("direct_temporal_edge_rows") or [])),
            "hidden_driver_rows": len(list(structural_payload.get("hidden_driver_rows") or [])),
            "multiscale_factor_support_rows": len(list(structural_payload.get("multiscale_factor_support_rows") or [])),
        },
        "edge_falsification_counts": {
            "direct_edge_count": edge_falsification.get("direct_edge_count"),
            "hidden_edge_count": edge_falsification.get("hidden_edge_count"),
            "prior_eligible_direct_edge_count": edge_falsification.get("prior_eligible_direct_edge_count"),
            "blocked_edge_count": edge_falsification.get("blocked_edge_count"),
        },
        "lineage_gate": gate,
        "driver_rows": driver_rows,
        "bundle_rows": bundle_rows,
        "artifact_paths": {},
    }
    json_path = analysis_dir / "r46_phase2_lineage_driver_gate_report.json"
    md_path = analysis_dir / "r46_phase2_lineage_driver_gate_report.md"
    driver_csv = analysis_dir / "r46_driver_rows.csv"
    bundle_csv = analysis_dir / "r46_bundle_rows.csv"
    dashboard_path = analysis_dir / "r46_phase2_lineage_driver_gate_dashboard.png"
    report["artifact_paths"] = {
        "json": json_path.as_posix(),
        "markdown": md_path.as_posix(),
        "driver_csv": driver_csv.as_posix(),
        "bundle_csv": bundle_csv.as_posix(),
        "dashboard_png": dashboard_path.as_posix(),
    }
    write_json(json_path, report)
    _write_csv(driver_csv, driver_rows)
    _write_csv(bundle_csv, bundle_rows)
    _write_markdown(md_path, report)
    _write_dashboard(dashboard_path, driver_rows, gate)
    write_json(json_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase3 R46 Phase 2 lineage driver gate.")
    parser.add_argument("--epigraph-root", default=None)
    parser.add_argument("--phase2-lineage-run-dir", default=None)
    parser.add_argument("--determinant-robustness-path", default=None)
    parser.add_argument("--edge-falsification-path", default=None)
    parser.add_argument("--run-id", default=R46_RUN_ID)
    args = parser.parse_args()
    run_r46_phase2_lineage_driver_gate(
        run_id=str(args.run_id),
        epigraph_root=None if args.epigraph_root is None else Path(args.epigraph_root),
        phase2_lineage_run_dir=None
        if args.phase2_lineage_run_dir is None
        else Path(args.phase2_lineage_run_dir),
        determinant_robustness_path=None
        if args.determinant_robustness_path is None
        else Path(args.determinant_robustness_path),
        edge_falsification_path=None if args.edge_falsification_path is None else Path(args.edge_falsification_path),
    )


if __name__ == "__main__":
    _main()
