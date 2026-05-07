from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
from .r80_public_annual_projection_head import R80_RUN_ID
from .runtime import ensure_dir, read_json, write_json


R81_SCHEMA_VERSION = "phase3_dynamic.r81_phase2_knob_admissibility_gate.v1"
R81_RUN_ID = "p3d-r81-phase2-knob-admissibility-gate-20260507-s00"
R46_CURRENT_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / "p3d-r47-phase2-current-lineage-driver-gate-20260503-s01"
    / "analysis"
    / "r46_phase2_lineage_driver_gate_report.json"
)
R80_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R80_RUN_ID
    / "analysis"
    / "r80_public_annual_projection_head_report.json"
)

MODULE_TO_PRIMARY_OUTPUTS: dict[str, tuple[str, ...]] = {
    "incidence": ("annual_new_infections", "incident_infections_period"),
    "U_to_D": ("new_diagnosed_cases_period", "diagnosed_plhiv"),
    "D_to_A": ("alive_on_art", "diagnosed_plhiv"),
    "ART_retention": ("alive_on_art",),
    "VL_suppression": ("tested_for_viral_load", "virally_suppressed"),
}
BLOCK_DIRECTION_HYPOTHESES: dict[str, dict[str, str]] = {
    "testing_prevention_reach": {
        "incidence": "lower_pressure_if_block_improves",
        "U_to_D": "higher_transition_if_block_improves",
    },
    "care_access_continuity": {
        "D_to_A": "higher_transition_if_block_improves",
        "ART_retention": "lower_leakage_if_block_improves",
        "VL_suppression": "higher_transition_if_block_improves",
    },
    "suppression_capacity": {
        "VL_suppression": "higher_transition_if_block_improves",
    },
    "mobility_exposure_pressure": {
        "incidence": "higher_pressure_if_block_increases",
    },
    "structural_barrier_pressure": {
        "incidence": "higher_pressure_if_block_increases",
        "U_to_D": "lower_transition_if_block_increases",
        "D_to_A": "lower_transition_if_block_increases",
        "ART_retention": "higher_leakage_if_block_increases",
        "VL_suppression": "lower_transition_if_block_increases",
    },
}


def _driver_kind(row: dict[str, Any]) -> str:
    status = str(row.get("driver_status") or "")
    edge_kind = str(row.get("edge_kind") or "")
    if status == "strict_phase3_prior":
        return "strict_quantitative_prior"
    if status == "source_stable_direct_driver":
        return "time_validated_sensitivity_knob"
    if status == "source_stable_but_time_blocked_sensitivity_only" and edge_kind == "direct":
        return "directional_sensitivity_knob"
    if edge_kind == "hidden":
        return "hidden_shock_diagnostic"
    return "blocked"


def _edge_direction(row: dict[str, Any]) -> str:
    weight = _finite_float(row.get("baseline_weight"))
    if weight is None:
        return "unknown"
    return "positive_source_to_target" if float(weight) >= 0.0 else "negative_source_to_target"


def _module_direction_hypotheses(row: dict[str, Any]) -> list[dict[str, str]]:
    source = str(row.get("source") or "")
    target = str(row.get("target") or "")
    modules = [str(module) for module in list(row.get("phase3_modules") or [])]
    output: list[dict[str, str]] = []
    for module in modules:
        source_direction = BLOCK_DIRECTION_HYPOTHESES.get(source, {}).get(module, "not_semantically_mapped")
        target_direction = BLOCK_DIRECTION_HYPOTHESES.get(target, {}).get(module, "not_semantically_mapped")
        output.append(
            {
                "module": module,
                "primary_outputs": "|".join(MODULE_TO_PRIMARY_OUTPUTS.get(module, ())),
                "source_block_hypothesis": source_direction,
                "target_block_hypothesis": target_direction,
            }
        )
    return output


def _knob_rows(r46: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in list(r46.get("driver_rows") or []):
        driver = dict(raw)
        kind = _driver_kind(driver)
        modules = [str(module) for module in list(driver.get("phase3_modules") or [])]
        status = str(driver.get("driver_status") or "")
        quantitative_allowed = kind in {"strict_quantitative_prior", "time_validated_sensitivity_knob"}
        scenario_allowed = kind in {
            "strict_quantitative_prior",
            "time_validated_sensitivity_knob",
            "directional_sensitivity_knob",
        }
        rows.append(
            {
                "edge_key": str(driver.get("edge_key") or ""),
                "edge_kind": str(driver.get("edge_kind") or ""),
                "source": str(driver.get("source") or ""),
                "target": str(driver.get("target") or ""),
                "lag": int(driver.get("lag") or 0),
                "baseline_weight": _finite_float(driver.get("baseline_weight")),
                "edge_direction": _edge_direction(driver),
                "driver_status": status,
                "knob_kind": kind,
                "phase3_modules": modules,
                "primary_outputs": sorted({output for module in modules for output in MODULE_TO_PRIMARY_OUTPUTS.get(module, ())}),
                "quantitative_modulation_allowed": bool(quantitative_allowed),
                "directional_scenario_allowed": bool(scenario_allowed),
                "intervention_claim_allowed": bool(kind == "strict_quantitative_prior"),
                "allowed_use": _allowed_use_for_kind(kind),
                "blocked_time_passed": bool(driver.get("blocked_time_passed")),
                "source_reestimated_passed": bool(driver.get("source_reestimated_passed")),
                "support_ablation_passed": bool(driver.get("support_ablation_passed")),
                "evaluated_family_count": int(driver.get("evaluated_family_count") or 0),
                "survived_family_count": int(driver.get("survived_family_count") or 0),
                "sign_conflict_family_count": int(driver.get("sign_conflict_family_count") or 0),
                "module_direction_hypotheses": _module_direction_hypotheses(driver),
                "promotion_blockers": list(driver.get("promotion_blockers") or []),
            }
        )
    return rows


def _allowed_use_for_kind(kind: str) -> str:
    if kind == "strict_quantitative_prior":
        return "candidate_model_prior_after_blocked_time_gate"
    if kind == "time_validated_sensitivity_knob":
        return "quantitative_sensitivity_with_noncausal_caveat"
    if kind == "directional_sensitivity_knob":
        return "directional_sensitivity_only_no_numeric_effect_size"
    if kind == "hidden_shock_diagnostic":
        return "shared_latent_shock_diagnostic_not_intervention_knob"
    return "blocked"


def _module_rows(knob_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: dict[str, Counter[str]] = {}
    output_counter: dict[str, Counter[str]] = {}
    for row in knob_rows:
        kind = str(row.get("knob_kind") or "blocked")
        for module in list(row.get("phase3_modules") or []):
            module_key = str(module)
            counter.setdefault(module_key, Counter())[kind] += 1
            for output in list(row.get("primary_outputs") or []):
                output_counter.setdefault(module_key, Counter())[str(output)] += 1
    rows: list[dict[str, Any]] = []
    for module in sorted(counter):
        counts = dict(sorted(counter[module].items()))
        rows.append(
            {
                "module": module,
                "status_counts": counts,
                "strict_quantitative_prior_count": int(counts.get("strict_quantitative_prior", 0)),
                "directional_sensitivity_knob_count": int(counts.get("directional_sensitivity_knob", 0)),
                "hidden_shock_diagnostic_count": int(counts.get("hidden_shock_diagnostic", 0)),
                "primary_outputs": sorted(output_counter.get(module, Counter()).keys()),
            }
        )
    return rows


def _r80_projection_scope(r80: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in list(r80.get("projection_rows") or [])]
    return {
        "projection_row_count": len(rows),
        "projection_years": sorted({int(row.get("year") or 0) for row in rows if row.get("year")}),
        "projection_metrics": sorted({str(row.get("metric_name") or "") for row in rows if row.get("metric_name")}),
        "projection_gate_status": str(dict(r80.get("public_annual_projection_gate") or {}).get("status") or ""),
    }


def _gate(knob_rows: list[dict[str, Any]], r46: dict[str, Any], r80: dict[str, Any]) -> dict[str, Any]:
    strict_count = sum(1 for row in knob_rows if str(row.get("knob_kind") or "") == "strict_quantitative_prior")
    quantitative_count = sum(1 for row in knob_rows if bool(row.get("quantitative_modulation_allowed")))
    directional_count = sum(1 for row in knob_rows if str(row.get("knob_kind") or "") == "directional_sensitivity_knob")
    hidden_count = sum(1 for row in knob_rows if str(row.get("knob_kind") or "") == "hidden_shock_diagnostic")
    r46_status = str(dict(r46.get("lineage_gate") or {}).get("status") or "")
    r80_status = str(dict(r80.get("public_annual_projection_gate") or {}).get("status") or "")
    blockers: list[str] = []
    if not knob_rows:
        blockers.append("no_phase2_driver_rows")
    if strict_count == 0:
        blockers.append("no_strict_phase2_quantitative_priors")
    if r46_status not in {"strict_determinant_priors_ready", "sensitivity_only_determinant_scenarios"}:
        blockers.append("phase2_lineage_gate_not_ready")
    if r80_status != "public_annual_projection_head_ready":
        blockers.append("public_annual_projection_head_not_ready")
    status = (
        "phase2_quantitative_knobs_ready"
        if strict_count > 0 and not blockers
        else "phase2_directional_sensitivity_knobs_only"
        if directional_count > 0 and "phase2_lineage_gate_not_ready" not in blockers
        else "phase2_knobs_blocked"
    )
    return {
        "status": status,
        "blockers": blockers,
        "strict_quantitative_prior_count": int(strict_count),
        "quantitative_modulation_allowed_count": int(quantitative_count),
        "directional_sensitivity_knob_count": int(directional_count),
        "hidden_shock_diagnostic_count": int(hidden_count),
        "r46_lineage_gate_status": r46_status,
        "r80_projection_gate_status": r80_status,
        "contract": (
            "R81 decides whether Phase 2 outputs can modulate model outputs. Strict numeric knobs require "
            "Phase 2 direct edges that pass support ablation, source-family re-estimation, and blocked-time "
            "falsification. Source-stable but time-blocked edges may only label directional sensitivity scenarios; "
            "hidden edges remain latent-shock diagnostics and cannot be intervention knobs."
        ),
    }


def run_r81_phase2_knob_admissibility_gate(
    *,
    run_id: str = R81_RUN_ID,
    r46_report_path: Path | None = None,
    r80_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r46_path = R46_CURRENT_DEFAULT_REPORT if r46_report_path is None else Path(r46_report_path)
    r80_path = R80_DEFAULT_REPORT if r80_report_path is None else Path(r80_report_path)
    r46 = dict(read_json(r46_path, default={}) or {}) if r46_path.exists() else {}
    r80 = dict(read_json(r80_path, default={}) or {}) if r80_path.exists() else {}
    knobs = _knob_rows(r46)
    modules = _module_rows(knobs)
    gate = _gate(knobs, r46, r80)
    report_path = analysis_dir / "r81_phase2_knob_admissibility_gate_report.json"
    markdown_path = analysis_dir / "r81_phase2_knob_admissibility_gate_report.md"
    knob_csv = analysis_dir / "r81_knob_rows.csv"
    module_csv = analysis_dir / "r81_module_rows.csv"
    dashboard_path = analysis_dir / "r81_phase2_knob_admissibility_dashboard.png"
    report = {
        "schema_version": R81_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "phase2_knob_gate": gate,
        "knob_rows": knobs,
        "module_rows": modules,
        "r80_projection_scope": _r80_projection_scope(r80),
        "source_artifacts": {
            "r46": {"path": r46_path.as_posix(), "sha256": _sha256(r46_path) if r46_path.exists() else None},
            "r80": {"path": r80_path.as_posix(), "sha256": _sha256(r80_path) if r80_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "knob_rows_csv": knob_csv.as_posix(),
            "module_rows_csv": module_csv.as_posix(),
            "dashboard_png": dashboard_path.as_posix(),
        },
    }
    _write_csv(knob_csv, knobs)
    _write_csv(module_csv, modules)
    _write_dashboard(dashboard_path, knobs, modules, gate)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, list):
                normalized[key] = "|".join(str(item) for item in value)
            elif isinstance(value, dict):
                normalized[key] = str(value)
            else:
                normalized[key] = value
        normalized_rows.append(normalized)
    fieldnames = sorted({key for row in normalized_rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)


def _write_dashboard(path: Path, knob_rows: list[dict[str, Any]], module_rows: list[dict[str, Any]], gate: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    ensure_dir(path.parent)
    status_counts = Counter(str(row.get("knob_kind") or "blocked") for row in knob_rows)
    labels = list(status_counts)
    values = [int(status_counts[label]) for label in labels]
    modules = [str(row.get("module") or "") for row in module_rows]
    directional = [int(row.get("directional_sensitivity_knob_count") or 0) for row in module_rows]
    hidden = [int(row.get("hidden_shock_diagnostic_count") or 0) for row in module_rows]
    strict = [int(row.get("strict_quantitative_prior_count") or 0) for row in module_rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(np.arange(len(labels)), values, color="#51606f")
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    axes[0].set_xlabel("edge count")
    axes[0].set_title(str(gate.get("status") or ""))
    x = np.arange(len(modules))
    axes[1].bar(x, directional, label="directional", color="#b7791f")
    axes[1].bar(x, hidden, bottom=directional, label="hidden diagnostic", color="#718096")
    axes[1].bar(x, strict, bottom=np.asarray(directional) + np.asarray(hidden), label="strict", color="#2c7a7b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(modules, rotation=40, ha="right")
    axes[1].set_ylabel("edge-module mentions")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Module coverage")
    fig.suptitle("R81 Phase 2 knob admissibility", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("phase2_knob_gate") or {})
    lines = [
        "# Phase 3 R81 Phase 2 Knob Admissibility Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Strict quantitative priors: `{gate.get('strict_quantitative_prior_count')}`",
        f"- Quantitative modulation allowed edges: `{gate.get('quantitative_modulation_allowed_count')}`",
        f"- Directional sensitivity knobs: `{gate.get('directional_sensitivity_knob_count')}`",
        f"- Hidden shock diagnostics: `{gate.get('hidden_shock_diagnostic_count')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Allowed Knobs",
        "",
        "| Edge | Kind | Modules | Outputs | Allowed Use |",
        "|---|---|---|---|---|",
    ]
    for row in report.get("knob_rows") or []:
        if str(row.get("knob_kind") or "") == "blocked":
            continue
        lines.append(
            f"| `{row.get('edge_key')}` | `{row.get('knob_kind')}` | "
            f"`{', '.join(row.get('phase3_modules') or [])}` | "
            f"`{', '.join(row.get('primary_outputs') or [])}` | `{row.get('allowed_use')}` |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R81 Phase 2 knob admissibility gate.")
    parser.add_argument("--run-id", default=R81_RUN_ID)
    parser.add_argument("--r46-report-path", default=None)
    parser.add_argument("--r80-report-path", default=None)
    args = parser.parse_args()
    run_r81_phase2_knob_admissibility_gate(
        run_id=str(args.run_id),
        r46_report_path=None if args.r46_report_path is None else Path(args.r46_report_path),
        r80_report_path=None if args.r80_report_path is None else Path(args.r80_report_path),
    )


if __name__ == "__main__":
    _main()
