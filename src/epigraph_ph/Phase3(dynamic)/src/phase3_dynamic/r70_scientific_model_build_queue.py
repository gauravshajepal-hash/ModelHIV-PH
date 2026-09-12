from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .data import sandbox_repo_root
from .r11_sparse_state_space import _generated_at, _sha256
from .r67_transmission_model_family_queue import R67_RUN_ID
from .r69_bulk_signal_feature_compiler import R69_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R70_SCHEMA_VERSION = "phase3_dynamic.r70_scientific_model_build_queue.v1"
R70_RUN_ID = "p3d-r70-scientific-model-build-queue-20260506-s00"
R67_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R67_RUN_ID
    / "analysis"
    / "r67_transmission_model_family_queue_report.json"
)
R69_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R69_RUN_ID
    / "analysis"
    / "r69_bulk_signal_feature_compiler_report.json"
)
DEFAULT_STEP_COUNT = 240


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _readiness_index(r69_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row.get("family_id") or ""): dict(row) for row in r69_report.get("family_readiness_rows") or []}


def _family_rows(r67_report: dict[str, Any], r69_report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _readiness_index(r69_report)
    rows: list[dict[str, Any]] = []
    for row in r67_report.get("model_family_rows") or []:
        family_id = str(row.get("family_id") or "")
        feature = readiness.get(family_id, {})
        rows.append(
            {
                **dict(row),
                "feature_readiness_status": feature.get("readiness_status", "not_in_r69"),
                "feature_next_action": feature.get("next_action", ""),
                "module_signal_counts": feature.get("module_signal_counts", "{}"),
            }
        )
    return rows


def _priority_base(family: dict[str, Any]) -> int:
    status = str(family.get("feature_readiness_status") or "")
    if status == "feature_table_ready" and str(family.get("readiness_status") or "").startswith("ready"):
        return 0
    if status == "feature_table_ready":
        return 10
    if "source_stable" in str(family.get("readiness_status") or ""):
        return 40
    if "diagnostic_only" in str(family.get("readiness_status") or ""):
        return 70
    return 50


def _module_list(family: dict[str, Any]) -> list[str]:
    modules = [module for module in str(family.get("module_targets") or "").split("|") if module]
    preferred = [
        "diagnosis_reporting",
        "art_retention",
        "vl_suppression_service",
        "incidence_validation",
        "incidence_pressure",
        "kp_overlay",
        "regional_shrinkage",
        "prep_persistence",
        "mortality_reporting",
        "annual_challenge",
    ]
    ordered = [module for module in preferred if module in modules]
    ordered.extend(module for module in modules if module not in ordered)
    return ordered[:5]


def _experiment_modes(family: dict[str, Any]) -> list[str]:
    family_id = str(family.get("family_id") or "")
    if "hidden_service_intensity" in family_id:
        return ["fit_reporting_state", "source_ablation", "shock_holdout", "stock_cone_gate"]
    if "service_capacity_queue" in family_id:
        return ["fit_capacity_state", "capacity_ablation", "conditional_rate_gate", "stock_cone_gate"]
    if "semi_markov" in family_id:
        return ["fit_sojourn_kernel", "linkage_delay_scan", "competing_exit_ablation", "posterior_coverage"]
    if "backcalculation" in family_id:
        return ["fit_delay_kernel", "late_diagnosis_emission_gate", "annual_incidence_weak_measurement", "placebo_kernel"]
    if "kp_metapopulation" in family_id:
        return ["source_stability_gate", "kp_bundle_sensitivity", "hierarchical_shrinkage_probe", "annual_validation_only_gate"]
    return ["blocked_replay", "source_ablation", "placebo_gate", "posterior_coverage"]


def _promotion_gate(mode: str, module: str, horizon: str, geography: str) -> str:
    base = "beat locked carry-forward/R10/R41-or-R60 reference under blocked time"
    if "ablation" in mode or "source" in mode:
        return f"{base}; source-family ablation must not reverse sign or lose {module} improvement"
    if "placebo" in mode:
        return f"{base}; placebo transition must not improve similarly"
    if "stock_cone" in mode or module in {"art_retention", "vl_suppression_service"}:
        return f"{base}; preserve D>=A>=VL>=S stock cone and conditional-rate gates"
    if "annual" in mode or horizon == "5y":
        return f"{base}; annual validation remains weak/external, not quarterly training truth"
    if geography != "national":
        return f"{base}; national-total coherence and split stability required"
    return base


def _queue_rows(family_rows: list[dict[str, Any]], *, max_steps: int = DEFAULT_STEP_COUNT) -> list[dict[str, Any]]:
    geographies = ["national", "region_hierarchical", "province_auxiliary"]
    horizons = ["1y", "3y", "5y"]
    rows: list[dict[str, Any]] = []
    for family in family_rows:
        family_id = str(family.get("family_id") or "")
        modes = _experiment_modes(family)
        modules = _module_list(family)
        for geography in geographies:
            for horizon in horizons:
                for module in modules:
                    for mode in modes:
                        rows.append(
                            {
                                "family_id": family_id,
                                "experiment_mode": mode,
                                "module_target": module,
                                "geography_scope": geography,
                                "horizon": horizon,
                                "feature_readiness_status": family.get("feature_readiness_status"),
                                "family_readiness_status": family.get("readiness_status"),
                                "allowed_use": _allowed_use_for_step(family, geography, module),
                                "promotion_gate": _promotion_gate(mode, module, horizon, geography),
                                "overfit_guard": family.get("overfit_guard"),
                                "leakage_policy": family.get("leakage_policy"),
                                "source_domain_transfer": family.get("source_domain"),
                                "equation": family.get("equation"),
                                "priority_score": _priority_base(family)
                                + _mode_penalty(mode)
                                + _geography_penalty(geography)
                                + _horizon_penalty(horizon)
                                + _module_penalty(module),
                            }
                        )
    rows.sort(
        key=lambda row: (
            int(row.get("priority_score") or 0),
            str(row.get("family_id") or ""),
            str(row.get("module_target") or ""),
            str(row.get("geography_scope") or ""),
            str(row.get("horizon") or ""),
            str(row.get("experiment_mode") or ""),
        )
    )
    trimmed = rows[:max_steps]
    for index, row in enumerate(trimmed, start=1):
        row["step_id"] = f"R70-{index:03d}"
    return trimmed


def _allowed_use_for_step(family: dict[str, Any], geography: str, module: str) -> str:
    family_status = str(family.get("readiness_status") or "")
    if "source_stable" in family_status or module in {"incidence_pressure", "kp_overlay"}:
        return "sensitivity_or_falsification_until_R46_source_stable"
    if geography == "province_auxiliary":
        return "auxiliary_diagnostic_not_validation_claim"
    return "bounded_candidate_experiment"


def _mode_penalty(mode: str) -> int:
    if mode.startswith("fit_"):
        return 0
    if "gate" in mode:
        return 2
    if "ablation" in mode:
        return 4
    return 6


def _geography_penalty(geography: str) -> int:
    return {"national": 0, "region_hierarchical": 4, "province_auxiliary": 12}.get(geography, 20)


def _horizon_penalty(horizon: str) -> int:
    return {"1y": 0, "3y": 2, "5y": 4}.get(horizon, 6)


def _module_penalty(module: str) -> int:
    return {
        "diagnosis_reporting": 0,
        "art_retention": 1,
        "vl_suppression_service": 2,
        "incidence_validation": 3,
        "mortality_reporting": 4,
        "regional_shrinkage": 5,
        "prep_persistence": 6,
        "kp_overlay": 8,
        "incidence_pressure": 9,
    }.get(module, 10)


def _gate(queue_rows: list[dict[str, Any]]) -> dict[str, Any]:
    bounded = sum(1 for row in queue_rows if row.get("allowed_use") == "bounded_candidate_experiment")
    sensitivity = sum(1 for row in queue_rows if "sensitivity" in str(row.get("allowed_use") or ""))
    auxiliary = sum(1 for row in queue_rows if "auxiliary" in str(row.get("allowed_use") or ""))
    return {
        "status": "scientific_model_build_queue_ready" if queue_rows else "scientific_model_build_queue_empty",
        "queued_step_count": len(queue_rows),
        "bounded_candidate_step_count": bounded,
        "sensitivity_or_falsification_step_count": sensitivity,
        "auxiliary_diagnostic_step_count": auxiliary,
        "contract": (
            "R70 is the next multi-hundred-step build queue. It is intentionally staged: fit small bounded branches, "
            "then source-ablate, then placebo-test, then expand geography/horizon. No step may promote unless its "
            "promotion_gate passes against locked references."
        ),
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("build_queue_gate") or {})
    lines = [
        "# Phase 3 R70 Scientific Model Build Queue",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Queued steps: `{gate.get('queued_step_count')}`",
        f"- Bounded candidate steps: `{gate.get('bounded_candidate_step_count')}`",
        f"- Sensitivity/falsification steps: `{gate.get('sensitivity_or_falsification_step_count')}`",
        "",
        "## First 20 Steps",
        "",
        "| Step | Family | Module | Geography | Horizon | Mode |",
        "|---|---|---|---|---|---|",
    ]
    for row in list(report.get("queue_rows") or [])[:20]:
        lines.append(
            f"| `{row.get('step_id')}` | `{row.get('family_id')}` | `{row.get('module_target')}` | "
            f"`{row.get('geography_scope')}` | `{row.get('horizon')}` | `{row.get('experiment_mode')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_r70_scientific_model_build_queue(
    *,
    run_id: str = R70_RUN_ID,
    r67_report_path: Path | None = None,
    r69_report_path: Path | None = None,
    max_steps: int = DEFAULT_STEP_COUNT,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r67_path = R67_DEFAULT_REPORT if r67_report_path is None else Path(r67_report_path)
    r69_path = R69_DEFAULT_REPORT if r69_report_path is None else Path(r69_report_path)
    r67 = dict(read_json(r67_path, default={}) or {}) if r67_path.exists() else {}
    r69 = dict(read_json(r69_path, default={}) or {}) if r69_path.exists() else {}
    families = _family_rows(r67, r69)
    queue_rows = _queue_rows(families, max_steps=max_steps)
    gate = _gate(queue_rows)
    report_path = analysis_dir / "r70_scientific_model_build_queue_report.json"
    markdown_path = analysis_dir / "r70_scientific_model_build_queue_report.md"
    queue_csv = analysis_dir / "r70_queue_rows.csv"
    report = {
        "schema_version": R70_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "build_queue_gate": gate,
        "queue_rows": queue_rows,
        "source_artifacts": {
            "r67": {"path": r67_path.as_posix(), "sha256": _sha256(r67_path) if r67_path.exists() else None},
            "r69": {"path": r69_path.as_posix(), "sha256": _sha256(r69_path) if r69_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "queue_csv": queue_csv.as_posix(),
        },
    }
    _write_csv(queue_csv, queue_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R70 scientific model build queue.")
    parser.add_argument("--run-id", default=R70_RUN_ID)
    parser.add_argument("--r67-report-path", default=None)
    parser.add_argument("--r69-report-path", default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_STEP_COUNT)
    args = parser.parse_args()
    run_r70_scientific_model_build_queue(
        run_id=str(args.run_id),
        r67_report_path=None if args.r67_report_path is None else Path(args.r67_report_path),
        r69_report_path=None if args.r69_report_path is None else Path(args.r69_report_path),
        max_steps=int(args.max_steps),
    )


if __name__ == "__main__":
    _main()
