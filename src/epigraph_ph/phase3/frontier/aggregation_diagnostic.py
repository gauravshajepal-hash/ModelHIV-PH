from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.geography import infer_region_code
from epigraph_ph.runtime import ROOT_DIR, read_json, write_json

from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .numeric_policy import numerical_guard_entry
from .registry import TRANSITION_NAMES
from .sources import TransitionResearchInputs, load_transition_research_inputs


HOOK_TO_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "diagnosis_transitions": ("U_to_D",),
    "linkage_transitions": ("D_to_A", "L_to_A"),
    "suppression_transitions": ("A_to_V",),
    "retention_attrition_transitions": ("A_to_L", "L_to_A"),
    "subgroup_allocation_priors": (),
}

BEST_TARGET_TO_TRANSITION: dict[str, str] = {
    "diagnosed_stock": "U_to_D",
    "testing_coverage": "U_to_D",
    "art_stock": "D_to_A",
    "alive_on_art": "D_to_A",
    "documented_suppression": "A_to_V",
    "viral_load_tested_stock": "A_to_V",
    "viral_load_tested_among_art": "A_to_V",
    "suppressed_among_art": "A_to_V",
}

STATE_NAMES: tuple[str, ...] = ("U", "D", "A", "V", "L")
INCIDENCE_MODULE = "incidence"
REPORTING_MODULES: tuple[str, ...] = (INCIDENCE_MODULE,) + TRANSITION_NAMES


@dataclass(slots=True)
class ProvincialEvidence:
    run_id: str
    run_dir: Path
    fit_artifact: dict[str, Any]
    benchmark_gate_report: dict[str, Any]
    determinant_modifiers: dict[str, Any]
    province_weights: dict[str, float]
    province_weight_time: str | None
    overlap_count: int


def _geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.any(array <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _discover_latest_provincial_evidence(retained_factor_ids: set[str]) -> ProvincialEvidence:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[int, int, float, str, Path, dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        phase3_dir = run_dir / "phase3"
        fit_path = phase3_dir / "fit_artifact.json"
        benchmark_path = phase3_dir / "benchmark_gate_report.json"
        determinant_path = phase3_dir / "determinant_modifiers.json"
        state_rows_path = phase3_dir / "state_estimates_rows.json"
        if not (fit_path.exists() and benchmark_path.exists() and determinant_path.exists() and state_rows_path.exists()):
            continue
        fit_artifact = read_json(fit_path, default={})
        province_axis = list(((fit_artifact.get("axis_catalogs") or {}).get("province") or []))
        province_count = len([province for province in province_axis if str(province) not in {"", "Philippines", "unknown"}])
        if province_count <= 1:
            continue
        determinant_modifiers = read_json(determinant_path, default={})
        selected_rows = list(determinant_modifiers.get("selected_determinant_modifiers", []) or [])
        if not selected_rows:
            continue
        overlap_count = int(sum(1 for row in selected_rows if str(row.get("factor_id") or "") in retained_factor_ids))
        if overlap_count <= 0:
            continue
        benchmark_gate_report = read_json(benchmark_path, default={})
        latest_mtime = max(
            fit_path.stat().st_mtime,
            benchmark_path.stat().st_mtime,
            determinant_path.stat().st_mtime,
            state_rows_path.stat().st_mtime,
        )
        candidates.append(
            (
                int(overlap_count),
                int(province_count),
                float(latest_mtime),
                str(run_dir.name),
                run_dir,
                fit_artifact,
                benchmark_gate_report,
                determinant_modifiers,
            )
        )
    if not candidates:
        raise FileNotFoundError("No provincial Phase 3 evidence run with fit_artifact, determinant_modifiers, benchmark_gate_report, and state_estimates_rows could be found.")
    overlap_count, province_count, _mtime, run_id, run_dir, fit_artifact, benchmark_gate_report, determinant_modifiers = max(
        candidates,
        key=lambda item: (int(item[0]), float(item[2]), int(item[1]), str(item[3])),
    )
    province_axis = [str(value) for value in list(((fit_artifact.get("axis_catalogs") or {}).get("province") or []))]
    province_weights, province_weight_time = _load_province_weights(run_dir / "phase3" / "state_estimates_rows.json", province_axis)
    return ProvincialEvidence(
        run_id=run_id,
        run_dir=run_dir,
        fit_artifact=fit_artifact,
        benchmark_gate_report=benchmark_gate_report,
        determinant_modifiers=determinant_modifiers,
        province_weights=province_weights,
        province_weight_time=province_weight_time,
        overlap_count=int(overlap_count),
    )


def _load_province_weights(state_rows_path: Path, province_axis: list[str]) -> tuple[dict[str, float], str | None]:
    rows = list(read_json(state_rows_path, default=[]))
    totals_by_time: dict[str, dict[str, float]] = {}
    tracked_provinces = [province for province in province_axis if province not in {"", "Philippines", "unknown"}]
    tracked_set = set(tracked_provinces)
    for row in rows:
        province = str(row.get("province") or "")
        state = str(row.get("state") or "")
        time = str(row.get("time") or "")
        if province not in tracked_set or state not in STATE_NAMES or not time:
            continue
        totals_by_time.setdefault(time, {})
        totals_by_time[time][province] = float(totals_by_time[time].get(province, 0.0)) + float(row.get("value") or 0.0)
    if not totals_by_time:
        weight = 1.0 / float(len(tracked_provinces)) if tracked_provinces else 0.0
        return ({province: weight for province in tracked_provinces}, None)
    latest_time = max(totals_by_time)
    latest_totals = {province: max(float(value), 0.0) for province, value in totals_by_time[latest_time].items()}
    total_mass = float(sum(latest_totals.values()))
    if total_mass <= np.finfo(np.float64).eps:
        weight = 1.0 / float(len(tracked_provinces)) if tracked_provinces else 0.0
        return ({province: weight for province in tracked_provinces}, latest_time)
    return ({province: float(latest_totals.get(province, 0.0)) / total_mass for province in tracked_provinces}, latest_time)


def _transition_share_lookup(inputs: TransitionResearchInputs) -> dict[str, dict[str, float]]:
    lookup: dict[str, dict[str, float]] = {factor_id: {transition: 0.0 for transition in TRANSITION_NAMES} for factor_id in inputs.retained_factor_lookup}
    for factor_id, factor_row in inputs.retained_factor_lookup.items():
        hooks = [str(value) for value in list(factor_row.get("transition_hooks", []))]
        raw_scores = {transition: 0.0 for transition in TRANSITION_NAMES}
        for hook in hooks:
            mapped = HOOK_TO_TRANSITIONS.get(hook, ())
            contribution = 1.0 / float(len(mapped)) if mapped else 0.0
            for transition in mapped:
                raw_scores[transition] += contribution
        best_target_transition = BEST_TARGET_TO_TRANSITION.get(str(factor_row.get("best_target") or ""))
        if best_target_transition is not None:
            raw_scores[best_target_transition] += 1.0
        total = float(sum(raw_scores.values()))
        if total > 0.0:
            lookup[factor_id] = {transition: float(raw_scores[transition]) / total for transition in TRANSITION_NAMES}
    return lookup


def _selected_modifier_score_lookup(evidence: ProvincialEvidence) -> dict[str, float]:
    selected_rows = [dict(row) for row in list(evidence.determinant_modifiers.get("selected_determinant_modifiers", []) or [])]
    if not selected_rows:
        return {}
    max_score = max(float(row.get("diagnostic_score") or 0.0) for row in selected_rows)
    denominator = max(max_score, np.finfo(np.float64).eps)
    return {
        str(row.get("factor_id") or ""): float(row.get("diagnostic_score") or 0.0) / denominator
        for row in selected_rows
        if str(row.get("factor_id") or "")
    }


def _province_indices(inputs: TransitionResearchInputs, evidence: ProvincialEvidence) -> list[int]:
    weighted_provinces = set(evidence.province_weights)
    return [
        idx
        for idx, province in enumerate(inputs.province_axis)
        if str(province) in weighted_provinces and str(province) not in {"", "Philippines", "unknown"}
    ]


def _analysis_month_indices(inputs: TransitionResearchInputs) -> list[int]:
    analysis_years = set(int(year) for year in inputs.analysis_years)
    return [
        month_idx
        for month_idx, month_label in enumerate(inputs.month_axis)
        if int(str(month_label).split("-", maxsplit=1)[0]) in analysis_years
    ]


def _region_groups(inputs: TransitionResearchInputs, evidence: ProvincialEvidence, province_indices: list[int]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for province_idx in province_indices:
        province = str(inputs.province_axis[province_idx])
        region = infer_region_code(province) or "region_unknown"
        groups.setdefault(region, []).append(province_idx)
    return groups


def _weighted_region_energy(
    province_values: np.ndarray,
    *,
    province_indices: list[int],
    province_weights: dict[str, float],
    province_axis: list[str],
    region_groups: dict[str, list[int]],
) -> tuple[float, dict[str, float]]:
    eps = np.finfo(np.float64).eps
    group_energy = 0.0
    region_weight_map: dict[str, float] = {}
    global_index_lookup = {province_idx: local_idx for local_idx, province_idx in enumerate(province_indices)}
    for region, group_indices in region_groups.items():
        region_weight = float(sum(float(province_weights.get(str(province_axis[idx]), 0.0)) for idx in group_indices))
        region_weight_map[region] = region_weight
        if region_weight <= eps:
            continue
        local_indices = [global_index_lookup[idx] for idx in group_indices if idx in global_index_lookup]
        if not local_indices:
            continue
        local_weights = np.asarray(
            [float(province_weights.get(str(province_axis[idx]), 0.0)) / region_weight for idx in group_indices if idx in global_index_lookup],
            dtype=np.float64,
        )
        region_series = np.sum(local_weights[:, None] * province_values[local_indices, :], axis=0)
        group_energy += region_weight * float(np.mean(np.square(region_series)))
    return float(group_energy), region_weight_map


def _correlation(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_std = float(np.std(lhs))
    rhs_std = float(np.std(rhs))
    eps = np.finfo(np.float64).eps
    if lhs_std <= eps or rhs_std <= eps:
        return 1.0 if float(np.mean(np.abs(lhs - rhs))) <= eps else 0.0
    return float(np.corrcoef(lhs, rhs)[0, 1])


def _factor_rows(inputs: TransitionResearchInputs, evidence: ProvincialEvidence) -> dict[str, Any]:
    eps = np.finfo(np.float64).eps
    month_indices = _analysis_month_indices(inputs)
    province_indices = _province_indices(inputs, evidence)
    if not month_indices or not province_indices:
        raise ValueError("Aggregation-loss diagnostic requires overlapping analysis months and weighted province rows.")
    province_weights = np.asarray(
        [float(evidence.province_weights.get(str(inputs.province_axis[idx]), 0.0)) for idx in province_indices],
        dtype=np.float64,
    )
    if float(np.sum(province_weights)) <= eps:
        province_weights = np.full((len(province_indices),), 1.0 / float(len(province_indices)), dtype=np.float64)
    else:
        province_weights = province_weights / float(np.sum(province_weights))
    region_groups = _region_groups(inputs, evidence, province_indices)
    selected_modifier_score = _selected_modifier_score_lookup(evidence)
    transition_share_lookup = _transition_share_lookup(inputs)
    rows: list[dict[str, Any]] = []
    for factor_id in sorted(inputs.retained_factor_lookup):
        factor_idx = inputs.factor_index[factor_id]
        province_values = np.asarray(inputs.province_tensor[province_indices, :, factor_idx], dtype=np.float64)[:, month_indices]
        national_values = np.asarray(inputs.national_tensor[0, :, factor_idx], dtype=np.float64)[month_indices]
        weighted_aggregate = np.sum(province_weights[:, None] * province_values, axis=0)
        local_energy = float(np.mean(np.sum(province_weights[:, None] * np.square(province_values), axis=0)))
        national_energy = float(np.mean(np.square(weighted_aggregate)))
        region_energy, region_weight_map = _weighted_region_energy(
            province_values,
            province_indices=province_indices,
            province_weights=evidence.province_weights,
            province_axis=inputs.province_axis,
            region_groups=region_groups,
        )
        if local_energy <= eps:
            aggregation_loss = 0.0
            national_retention = 0.0
            region_retention = 0.0
            region_gain = 0.0
        else:
            national_retention = float(np.clip(national_energy / local_energy, 0.0, 1.0))
            region_retention = float(np.clip(region_energy / local_energy, 0.0, 1.0))
            aggregation_loss = float(np.clip(1.0 - national_retention, 0.0, 1.0))
            region_gain = float(np.clip(region_retention - national_retention, 0.0, 1.0))
        modifier_strength = float(selected_modifier_score.get(factor_id, 0.0))
        hierarchical_priority = _geometric_mean([aggregation_loss, modifier_strength])
        best_target = str((inputs.retained_factor_lookup.get(factor_id) or {}).get("best_target") or "")
        block_name = str((inputs.retained_factor_lookup.get(factor_id) or {}).get("block_name") or "")
        factor_name = str((inputs.retained_factor_lookup.get(factor_id) or {}).get("factor_name") or factor_id)
        transition_shares = transition_share_lookup.get(factor_id, {transition: 0.0 for transition in TRANSITION_NAMES})
        transition_priority = {
            transition: _geometric_mean([hierarchical_priority, float(transition_shares.get(transition, 0.0))])
            for transition in TRANSITION_NAMES
        }
        row = {
            "factor_id": factor_id,
            "factor_name": factor_name,
            "block_name": block_name,
            "best_target": best_target,
            "transition_hooks": list((inputs.retained_factor_lookup.get(factor_id) or {}).get("transition_hooks", [])),
            "selected_in_provincial_autoresearch": factor_id in selected_modifier_score,
            "provincial_modifier_strength": round(modifier_strength, 6),
            "aggregation_loss": round(aggregation_loss, 6),
            "national_retention": round(national_retention, 6),
            "region_retention": round(region_retention, 6),
            "region_gain": round(region_gain, 6),
            "provided_national_alignment_correlation": round(_correlation(weighted_aggregate, national_values), 6),
            "provided_national_alignment_rmse": round(float(np.sqrt(np.mean(np.square(weighted_aggregate - national_values)))), 6),
            "hierarchical_priority_score": round(hierarchical_priority, 6),
            "transition_priority": {transition: round(float(score), 6) for transition, score in transition_priority.items()},
            "transition_share": {transition: round(float(transition_shares.get(transition, 0.0)), 6) for transition in TRANSITION_NAMES},
            "region_weight_map": {region: round(float(weight), 6) for region, weight in region_weight_map.items() if float(weight) > 0.0},
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            -float(row["hierarchical_priority_score"]),
            -float(row["aggregation_loss"]),
            str(row["factor_id"]),
        )
    )
    return {
        "rows": rows,
        "month_indices": month_indices,
        "province_indices": province_indices,
    }


def _module_seed_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    module_factor_rankings: dict[str, list[dict[str, Any]]] = {}
    module_bundle_rankings: dict[str, list[dict[str, Any]]] = {}
    for module_name in REPORTING_MODULES:
        factor_rows: list[dict[str, Any]] = []
        for row in rows:
            if module_name == INCIDENCE_MODULE:
                module_score = float(row["hierarchical_priority_score"])
            else:
                module_score = float((row.get("transition_priority") or {}).get(module_name, 0.0))
            factor_rows.append(
                {
                    "factor_id": str(row["factor_id"]),
                    "factor_name": str(row["factor_name"]),
                    "block_name": str(row["block_name"]),
                    "best_target": str(row["best_target"]),
                    "module_score": round(module_score, 6),
                    "aggregation_loss": float(row["aggregation_loss"]),
                    "region_gain": float(row["region_gain"]),
                    "provincial_modifier_strength": float(row["provincial_modifier_strength"]),
                }
            )
        factor_rows.sort(key=lambda row: (-float(row["module_score"]), -float(row["aggregation_loss"]), str(row["factor_id"])))
        module_factor_rankings[module_name] = factor_rows

        grouped: dict[str, list[dict[str, Any]]] = {}
        for factor_row in factor_rows:
            if float(factor_row["module_score"]) <= 0.0:
                continue
            grouped.setdefault(str(factor_row["block_name"]) or "unlabeled_block", []).append(factor_row)
        bundle_rows: list[dict[str, Any]] = []
        for bundle_name, members in grouped.items():
            member_scores = [float(member["module_score"]) for member in members]
            bundle_rows.append(
                {
                    "bundle_name": bundle_name,
                    "bundle_score": round(float(np.mean(member_scores)), 6),
                    "member_count": int(len(members)),
                    "member_factor_ids": [str(member["factor_id"]) for member in members],
                    "member_factor_names": [str(member["factor_name"]) for member in members],
                }
            )
        bundle_rows.sort(key=lambda row: (-float(row["bundle_score"]), -int(row["member_count"]), str(row["bundle_name"])))
        module_bundle_rankings[module_name] = bundle_rows
    return {
        "module_factor_rankings": module_factor_rankings,
        "module_bundle_rankings": module_bundle_rankings,
    }


def _top_factor_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    top_rows = rows[:15]
    labels = [str(row["factor_id"]) for row in reversed(top_rows)]
    values = [float(row["hierarchical_priority_score"]) for row in reversed(top_rows)]
    fig, ax = plt.subplots()
    ax.barh(labels, values)
    ax.set_title("AN-03A Aggregation-Loss Priority")
    ax.set_xlabel("Hierarchical priority score")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _module_heatmap(rows: list[dict[str, Any]], output_path: Path) -> None:
    top_rows = rows[:15]
    labels = [str(row["factor_id"]) for row in top_rows]
    matrix = np.asarray(
        [
            [
                float(row["aggregation_loss"]),
                float(row["region_gain"]),
                float(row["provincial_modifier_strength"]),
                float((row.get("transition_priority") or {}).get("U_to_D", 0.0)),
                float((row.get("transition_priority") or {}).get("D_to_A", 0.0)),
                float((row.get("transition_priority") or {}).get("A_to_V", 0.0)),
                float((row.get("transition_priority") or {}).get("A_to_L", 0.0)),
                float((row.get("transition_priority") or {}).get("L_to_A", 0.0)),
            ]
            for row in top_rows
        ],
        dtype=np.float32,
    )
    fig, ax = plt.subplots()
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title("AN-03A Module-Specific Aggregation Loss")
    ax.set_xticks(
        np.arange(8),
        labels=["agg_loss", "region_gain", "prov_evidence", "U->D", "D->A", "A->V", "A->L", "L->A"],
    )
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_an03a(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    evidence = _discover_latest_provincial_evidence(set(inputs.retained_factor_lookup))
    factor_payload = _factor_rows(inputs, evidence)
    rows = factor_payload["rows"]
    module_seed_report = _module_seed_report(rows)
    write_json(ctx.experiment_dir / "aggregation_loss_diagnostic.json", rows)
    write_json(
        ctx.experiment_dir / "provincial_evidence_summary.json",
        {
            "provincial_evidence_run_id": evidence.run_id,
            "province_weight_time": evidence.province_weight_time,
            "provincial_overlap_count": int(evidence.overlap_count),
            "benchmark_gate_report": evidence.benchmark_gate_report,
            "selected_determinant_modifiers": evidence.determinant_modifiers.get("selected_determinant_modifiers", []),
            "coupling_covariate_names": evidence.determinant_modifiers.get("coupling_covariate_names", []),
        },
    )
    write_json(ctx.experiment_dir / "module_bundle_seed_report.json", module_seed_report)
    _top_factor_chart(rows, ctx.experiment_dir / "aggregation_loss_top_factors.png")
    _module_heatmap(rows, ctx.experiment_dir / "aggregation_loss_module_heatmap.png")
    coverage_summary = {
        "factor_count": len(rows),
        "analysis_years": list(inputs.analysis_years),
        "analysis_month_count": int(len(factor_payload["month_indices"])),
        "province_count": int(len(factor_payload["province_indices"])),
        "region_count": int(len({infer_region_code(str(inputs.province_axis[idx])) or "region_unknown" for idx in factor_payload["province_indices"]})),
        "provincial_evidence_run_id": evidence.run_id,
        "provincial_overlap_count": int(evidence.overlap_count),
        "selected_modifier_count": int(len(list(evidence.determinant_modifiers.get("selected_determinant_modifiers", []) or []))),
    }
    top_factor = rows[0] if rows else {}
    top_incidence_bundle = next(iter(module_seed_report["module_bundle_rankings"].get(INCIDENCE_MODULE, [])), {})
    decision = {
        "completed": True,
        "keep": True,
        "reason": "Aggregation-loss diagnostic generated from province-level Phase 2 tensors and kept provincial autoresearch evidence.",
        "top_factor_id": str(top_factor.get("factor_id") or ""),
        "top_factor_priority_score": float(top_factor.get("hierarchical_priority_score") or 0.0),
        "provincial_evidence_run_id": evidence.run_id,
        "provincial_overlap_count": int(evidence.overlap_count),
        "provincial_model_mae": float(((evidence.benchmark_gate_report.get("primary_gates") or [{}])[0].get("value") or {}).get("model_mae") or 0.0),
        "provincial_naive_mae": float(((evidence.benchmark_gate_report.get("primary_gates") or [{}])[0].get("value") or {}).get("naive_mae") or 0.0),
        "top_incidence_bundle": str(top_incidence_bundle.get("bundle_name") or ""),
        "top_incidence_bundle_score": float(top_incidence_bundle.get("bundle_score") or 0.0),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "phase15_dir": str(ctx.phase15_dir),
            "phase2_dir": str(ctx.phase2_dir),
            "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
            "provincial_evidence_run_id": evidence.run_id,
            "provincial_overlap_count": int(evidence.overlap_count),
            "expected_outputs": list(ctx.experiment.expected_outputs),
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            {
                "name": "analysis_month_count",
                "value": int(len(factor_payload["month_indices"])),
                "role": "aggregation_window_size",
                "source_type": "estimated",
                "estimation_data": "phase15 multiscale month axis filtered to supported analysis years",
                "estimation_method": "count retained month stamps whose year belongs to the transition research analysis set",
                "uncertainty": "none",
                "why_needed": "Defines the time window over which province-to-national aggregation loss is measured.",
            },
            {
                "name": "weighted_province_count",
                "value": int(len(factor_payload["province_indices"])),
                "role": "spatial_aggregation_cardinality",
                "source_type": "estimated",
                "estimation_data": "province axis intersected with provincial Phase 3 state estimates",
                "estimation_method": "count provinces with non-national, non-unknown rows and usable latest-burden weights",
                "uncertainty": "depends on provincial evidence run support",
                "why_needed": "Defines the exact province set contributing to the aggregation-loss operator.",
            },
            {
                "name": "selected_modifier_count",
                "value": int(len(list(evidence.determinant_modifiers.get("selected_determinant_modifiers", []) or []))),
                "role": "provincial_evidence_cardinality",
                "source_type": "estimated",
                "estimation_data": "kept provincial determinant_modifiers.selected_determinant_modifiers",
                "estimation_method": "count promoted provincial Phase 2 determinants retained as evidence anchors",
                "uncertainty": "none",
                "why_needed": "Restricts the hierarchical priority score to factors that survived provincial autoresearch evidence selection.",
            },
            {
                "name": "provincial_overlap_count",
                "value": int(evidence.overlap_count),
                "role": "cross_run_factor_overlap",
                "source_type": "estimated",
                "estimation_data": "intersection between current retained factor ids and promoted provincial determinant factor ids",
                "estimation_method": "count shared factor ids across the national source run and the selected provincial evidence run",
                "uncertainty": "none",
                "why_needed": "Ensures the diagnostic is anchored to provincial evidence that actually overlaps the active retained Phase 2 factor universe.",
            },
            numerical_guard_entry(
                name="float64_machine_epsilon",
                role="division_guard",
                why_needed="Prevents undefined retention ratios when a factor has vanishing province-level energy.",
            ),
        ],
    )
    return {
        "rows": rows,
        "module_bundle_seed_report": module_seed_report,
        "provincial_evidence_run_id": evidence.run_id,
        "artifacts": artifacts,
        "decision": decision,
    }
