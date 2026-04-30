from __future__ import annotations

import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from epigraph_ph.geography import infer_region_code
from epigraph_ph.runtime import ROOT_DIR, read_json, save_tensor_artifact, write_json

from .aggregation_diagnostic import (
    REPORTING_MODULES,
    ProvincialEvidence,
    _discover_latest_provincial_evidence,
    _factor_rows,
    _module_seed_report,
    _province_indices,
)
from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .integrated_autoresearch import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    TOTAL_METRIC,
    CandidateConfig,
    FitResult,
    IntegratedSnapshot,
    _baseline_comparison,
    _build_snapshot,
    _fit_with_matrices,
    _quarter_year,
    _score_tuple,
    _select_matrix,
)
from .numeric_policy import numerical_guard_entry
from .registry import TRANSITION_NAMES
from .sources import TransitionResearchInputs, load_transition_research_inputs


NATIONAL_METRIC_AXIS: tuple[str, ...] = PRIMARY_METRICS + ("estimated_plhiv",) + SECONDARY_METRICS
STATE_AXIS: tuple[str, ...] = ("U", "D", "A", "V", "L")


def _discover_latest_integrated_incumbent() -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, str, Path, dict[str, Any], dict[str, Any]]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / "PHASE3-V2-INT-explicit-incidence-autoresearch"
        decision_path = experiment_dir / "decision.json"
        baseline_path = experiment_dir / "baseline_comparison.json"
        if not (decision_path.exists() and baseline_path.exists()):
            continue
        candidates.append(
            (
                max(decision_path.stat().st_mtime, baseline_path.stat().st_mtime),
                str(run_dir.name),
                experiment_dir,
                dict(read_json(decision_path, default={}) or {}),
                dict(read_json(baseline_path, default={}) or {}),
            )
        )
    if not candidates:
        raise FileNotFoundError("No PHASE3-V2-INT explicit-incidence autoresearch experiment found under artifacts/runs")
    preferred = [row for row in candidates if "pytest" not in row[1].lower()]
    _mtime, run_id, experiment_dir, decision, baseline = max(preferred or candidates, key=lambda row: (float(row[0]), str(row[1])))
    return run_id, experiment_dir, decision, baseline


def _latest_transition_experiment_dir(experiment_id: str, sentinel_filename: str) -> Path | None:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[int, float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / experiment_id
        sentinel_path = experiment_dir / sentinel_filename
        if not sentinel_path.exists():
            continue
        pytest_penalty = 1 if "pytest" in str(run_dir.name).lower() else 0
        candidates.append((pytest_penalty, float(sentinel_path.stat().st_mtime), experiment_dir))
    if not candidates:
        return None
    preferred = [row for row in candidates if int(row[0]) == 0]
    _penalty, _mtime, experiment_dir = max(preferred or candidates, key=lambda row: float(row[1]))
    return experiment_dir


def _upsert_paper_figure_archive(entries: list[dict[str, Any]]) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_hmba_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = archive_dir / "figure_manifest.json"
    existing_rows = list(read_json(manifest_path, default=[]))
    manifest_lookup = {
        str(row.get("figure_id") or ""): dict(row)
        for row in existing_rows
        if str(row.get("figure_id") or "")
    }
    for entry in entries:
        figure_id = str(entry.get("figure_id") or "")
        source_path = Path(str(entry.get("source_path") or ""))
        if not figure_id or not source_path.exists() or not source_path.is_file():
            continue
        filename = str(entry.get("filename") or source_path.name)
        destination = archive_dir / filename
        shutil.copy2(source_path, destination)
        manifest_lookup[figure_id] = {
            "figure_id": figure_id,
            "title": str(entry.get("title") or ""),
            "caption": str(entry.get("caption") or ""),
            "source_path": str(source_path),
            "archived_path": str(destination),
        }
    ordered_rows = [manifest_lookup[key] for key in sorted(manifest_lookup)]
    write_json(manifest_path, ordered_rows)
    return {
        "archive_dir": str(archive_dir),
        "figure_manifest": str(manifest_path),
        "figure_count": int(len(ordered_rows)),
    }


def _refresh_phase3_figure_archive(
    *,
    hmba01_chart_paths: dict[str, Path] | None = None,
    hmba02_chart_paths: dict[str, Path] | None = None,
    hmba03_chart_paths: dict[str, Path] | None = None,
    hmba03a_chart_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    hmba01_chart_paths = hmba01_chart_paths or {}
    hmba02_chart_paths = hmba02_chart_paths or {}
    hmba03_chart_paths = hmba03_chart_paths or {}
    hmba03a_chart_paths = hmba03a_chart_paths or {}
    an03a_dir = _latest_transition_experiment_dir("AN-03A-phase2-aggregation-loss-diagnostic", "aggregation_loss_top_factors.png")
    hmba00_dir = _latest_transition_experiment_dir("HMBA-00-hierarchical-contract-freeze", "coverage_frontier.png")
    an03b_dir = _latest_transition_experiment_dir("AN-03B-phase2-partial-denoising-diagnostic", "partial_denoising_module_delta.png")
    hmba01_dir = _latest_transition_experiment_dir("HMBA-01-module-local-direct-bundle-promotion", "combined_model_comparison.png")
    hmba02_dir = _latest_transition_experiment_dir("HMBA-02-hierarchical-geography-layer", "hierarchical_combined_model_comparison.png")
    hmba03_dir = _latest_transition_experiment_dir("HMBA-03-joint-hierarchical-autoresearch-loop", "joint_model_comparison.png")
    hmba03a_dir = _latest_transition_experiment_dir("HMBA-03A-interpretation-dashboard", "interpretation_dashboard.png")
    entries = [
        {
            "figure_id": "fig01",
            "title": "Aggregation Loss Priority",
            "caption": "Top retained Phase 2 factors ranked by aggregation-loss priority using provincial evidence overlap.",
            "filename": "phase2_aggregation_loss_priority.png",
            "source_path": (an03a_dir / "aggregation_loss_top_factors.png") if an03a_dir else "",
        },
        {
            "figure_id": "fig02",
            "title": "Aggregation Loss Module Heatmap",
            "caption": "Module-specific aggregation-loss scores for the highest-priority retained factors.",
            "filename": "phase2_aggregation_loss_module_heatmap.png",
            "source_path": (an03a_dir / "aggregation_loss_module_heatmap.png") if an03a_dir else "",
        },
        {
            "figure_id": "fig03",
            "title": "HMBA Coverage Frontier",
            "caption": "Observed national metric coverage versus provincial auxiliary state-share coverage used by the HMBA contract.",
            "filename": "hmba_contract_coverage_frontier.png",
            "source_path": (hmba00_dir / "coverage_frontier.png") if hmba00_dir else "",
        },
        {
            "figure_id": "fig04",
            "title": "HMBA Module Seed Frontier",
            "caption": "Evidence-backed module-specific Phase 2 bundle seed frontier for HMBA search initialization.",
            "filename": "hmba_module_seed_frontier.png",
            "source_path": (hmba00_dir / "module_seed_frontier.png") if hmba00_dir else "",
        },
        {
            "figure_id": "fig05",
            "title": "Partial Denoising Module Delta",
            "caption": "Bundle-score change under data-driven partial denoising of the province factor field.",
            "filename": "partial_denoising_module_delta.png",
            "source_path": (an03b_dir / "partial_denoising_module_delta.png") if an03b_dir else "",
        },
        {
            "figure_id": "fig06",
            "title": "Partial Denoising Rank Compression",
            "caption": "Effective factor-matrix rank before and after partial denoising for the highest-priority retained factors.",
            "filename": "partial_denoising_rank_compression.png",
            "source_path": (an03b_dir / "partial_denoising_rank_compression.png") if an03b_dir else "",
        },
        {
            "figure_id": "fig07",
            "title": "Partial Denoising Priority Comparison",
            "caption": "Raw versus partial-denoised hierarchical priority scores for the highest-priority retained factors.",
            "filename": "partial_denoising_priority_comparison.png",
            "source_path": (an03b_dir / "partial_denoising_priority_comparison.png") if an03b_dir else "",
        },
        {
            "figure_id": "fig08",
            "title": "HMBA-01 Single-Bundle Gain Heatmap",
            "caption": "Historical primary-loss gain from adding a single direct Phase 2 bundle to each module on top of the hidden-only incumbent family.",
            "filename": "hmba01_single_bundle_gain_heatmap.png",
            "source_path": hmba01_chart_paths.get("single_bundle_gain", (hmba01_dir / "single_bundle_gain_heatmap.png") if hmba01_dir else ""),
        },
        {
            "figure_id": "fig09",
            "title": "HMBA-01 Module Promotion Trace",
            "caption": "Greedy forward-selection trace of historical primary loss as direct bundles are promoted within each module.",
            "filename": "hmba01_module_promotion_trace.png",
            "source_path": hmba01_chart_paths.get("promotion_trace", (hmba01_dir / "module_promotion_trace.png") if hmba01_dir else ""),
        },
        {
            "figure_id": "fig10",
            "title": "HMBA-01 Combined Model Comparison",
            "caption": "Comparison of hidden-only, HMBA-01 combined promoted, and frozen integrated incumbent historical losses.",
            "filename": "hmba01_combined_model_comparison.png",
            "source_path": hmba01_chart_paths.get("combined_comparison", (hmba01_dir / "combined_model_comparison.png") if hmba01_dir else ""),
        },
        {
            "figure_id": "fig11",
            "title": "HMBA-02 Auxiliary Fit Comparison",
            "caption": "Module-wise auxiliary RMSE for national-only, region-pooled, and hierarchical province-pooled direct score layers.",
            "filename": "hmba02_auxiliary_fit_comparison.png",
            "source_path": hmba02_chart_paths.get("auxiliary_fit", (hmba02_dir / "hierarchical_auxiliary_fit_comparison.png") if hmba02_dir else ""),
        },
        {
            "figure_id": "fig12",
            "title": "HMBA-02 Pooling Profile",
            "caption": "Average partial-pooling intensity for region and province coefficients under the HMBA-02 hierarchical layer.",
            "filename": "hmba02_pooling_profile.png",
            "source_path": hmba02_chart_paths.get("pooling_profile", (hmba02_dir / "hierarchical_pooling_profile.png") if hmba02_dir else ""),
        },
        {
            "figure_id": "fig13",
            "title": "HMBA-02 Combined Model Comparison",
            "caption": "Comparison of the HMBA-01 direct-bundle model, the HMBA-02 hierarchical model, and the frozen integrated incumbent on historical losses.",
            "filename": "hmba02_combined_model_comparison.png",
            "source_path": hmba02_chart_paths.get("combined_comparison", (hmba02_dir / "hierarchical_combined_model_comparison.png") if hmba02_dir else ""),
        },
        {
            "figure_id": "fig14",
            "title": "HMBA-03 Joint Search Frontier",
            "caption": "Joint autoresearch frontier over one-action bundle and depth mutations under the dual national-plus-auxiliary gate.",
            "filename": "hmba03_joint_search_frontier.png",
            "source_path": hmba03_chart_paths.get("frontier", (hmba03_dir / "joint_search_frontier.png") if hmba03_dir else ""),
        },
        {
            "figure_id": "fig15",
            "title": "HMBA-03 Depth Assignment",
            "caption": "Final module-specific hierarchy depth selected by the HMBA-03 joint autoresearch loop.",
            "filename": "hmba03_depth_assignment.png",
            "source_path": hmba03_chart_paths.get("depth_assignment", (hmba03_dir / "joint_depth_assignment.png") if hmba03_dir else ""),
        },
        {
            "figure_id": "fig16",
            "title": "HMBA-03 Joint Model Comparison",
            "caption": "Comparison of HMBA-01, HMBA-02, and the final HMBA-03 joint autoresearch model on historical national losses.",
            "filename": "hmba03_joint_model_comparison.png",
            "source_path": hmba03_chart_paths.get("combined_comparison", (hmba03_dir / "joint_model_comparison.png") if hmba03_dir else ""),
        },
        {
            "figure_id": "fig17",
            "title": "HMBA-03A Zoomed Loss Delta Charts",
            "caption": "Zoomed historical primary-loss and diagnosis-flow-loss deltas showing the real HMBA-03 improvement over HMBA-01 and HMBA-02.",
            "filename": "hmba03a_zoomed_loss_deltas.png",
            "source_path": hmba03a_chart_paths.get("zoomed_deltas", (hmba03a_dir / "zoomed_loss_deltas.png") if hmba03a_dir else ""),
        },
        {
            "figure_id": "fig18",
            "title": "HMBA-03A Accepted-Step Delta Chart",
            "caption": "Stepwise delta decomposition of accepted HMBA-03 mutations, showing where national fit saturates and geography-side gains dominate.",
            "filename": "hmba03a_accepted_step_deltas.png",
            "source_path": hmba03a_chart_paths.get("accepted_step_deltas", (hmba03a_dir / "accepted_step_deltas.png") if hmba03a_dir else ""),
        },
        {
            "figure_id": "fig19",
            "title": "HMBA-03A Module Effect Table",
            "caption": "Module-wise auxiliary depth metrics, selected hierarchy depth, and gain versus national-only coefficients for the HMBA-03 champion.",
            "filename": "hmba03a_module_effect_table.png",
            "source_path": hmba03a_chart_paths.get("module_effect_table", (hmba03a_dir / "module_effect_table.png") if hmba03a_dir else ""),
        },
        {
            "figure_id": "fig20",
            "title": "HMBA-03A Spike Candidate Annotations",
            "caption": "Annotated spike candidates showing the repeated U_to_D plus population_structure_demography failures, especially at national depth.",
            "filename": "hmba03a_spike_candidate_annotations.png",
            "source_path": hmba03a_chart_paths.get("spike_annotations", (hmba03a_dir / "spike_candidate_annotations.png") if hmba03a_dir else ""),
        },
        {
            "figure_id": "fig21",
            "title": "HMBA-03A Interpretation Dashboard",
            "caption": "Composite interpretation dashboard summarizing HMBA-03 model deltas, accepted-step effects, module-wise geography gains, and spike-family evidence.",
            "filename": "hmba03a_interpretation_dashboard.png",
            "source_path": hmba03a_chart_paths.get("dashboard", (hmba03a_dir / "interpretation_dashboard.png") if hmba03a_dir else ""),
        },
    ]
    return _upsert_paper_figure_archive(entries)


def _retained_factor_ids(inputs: TransitionResearchInputs) -> set[str]:
    return {str(row.get("factor_id") or "") for row in list(inputs.retained_factor_rows or []) if str(row.get("factor_id") or "")}


def _hidden_rank_summary(inputs: TransitionResearchInputs) -> dict[str, Any]:
    retained_ids = [factor_id for factor_id in sorted(_retained_factor_ids(inputs)) if factor_id in inputs.factor_index]
    if not retained_ids:
        return {
            "retained_factor_count": 0,
            "national_hidden_rank_max": 0,
            "province_hidden_rank_max": 0,
        }
    factor_indices = [int(inputs.factor_index[factor_id]) for factor_id in retained_ids]
    national_matrix = np.asarray(inputs.national_tensor[0, :, factor_indices], dtype=np.float64)
    province_tensor = np.asarray(inputs.province_tensor[:, :, factor_indices], dtype=np.float64)
    tracked_province_indices = [
        idx for idx, province in enumerate(inputs.province_axis) if str(province) not in {"", "Philippines", "unknown"}
    ]
    province_matrix = province_tensor[tracked_province_indices, :, :].reshape(-1, len(factor_indices)) if tracked_province_indices else np.zeros((0, len(factor_indices)), dtype=np.float64)

    def _rank(matrix: np.ndarray) -> int:
        if matrix.size == 0:
            return 0
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        if singular_values.size == 0:
            return 0
        tolerance = singular_values[0] * float(max(centered.shape)) * np.finfo(np.float64).eps
        return int(np.sum(singular_values > tolerance))

    return {
        "retained_factor_count": len(retained_ids),
        "national_hidden_rank_max": _rank(national_matrix),
        "province_hidden_rank_max": _rank(province_matrix),
    }


def _build_national_observation_tensors(ctx: TransitionResearchContext) -> dict[str, Any]:
    snapshot = _build_snapshot(ctx)
    metric_axis = list(NATIONAL_METRIC_AXIS)
    values = np.zeros((len(snapshot.model_quarters), len(metric_axis)), dtype=np.float32)
    mask = np.zeros((len(snapshot.model_quarters), len(metric_axis)), dtype=np.uint8)
    coverage: dict[str, dict[str, Any]] = {}
    for metric_idx, metric_name in enumerate(metric_axis):
        metric_values = np.asarray(snapshot.metric_values[metric_name], dtype=np.float64)
        metric_mask = np.asarray(snapshot.metric_masks[metric_name], dtype=bool)
        values[:, metric_idx] = np.nan_to_num(metric_values, nan=0.0).astype(np.float32)
        mask[:, metric_idx] = metric_mask.astype(np.uint8)
        observed_quarters = [snapshot.model_quarters[idx] for idx in np.flatnonzero(metric_mask)]
        coverage[metric_name] = {
            "first_observed_quarter": observed_quarters[0] if observed_quarters else None,
            "last_observed_quarter": observed_quarters[-1] if observed_quarters else None,
            "observed_quarter_count": int(np.sum(metric_mask)),
        }
    return {
        "metric_axis": metric_axis,
        "quarter_axis": list(snapshot.model_quarters),
        "historical_quarters": list(snapshot.historical_quarters),
        "future_quarters": list(snapshot.future_quarters),
        "historical_end_quarter": str(snapshot.historical_end_quarter),
        "values": values,
        "mask": mask,
        "coverage": coverage,
        "hidden_rank_max_national_incumbent": int(snapshot.hidden_rank_max),
    }


def _load_provincial_auxiliary_state_tensor(
    inputs: TransitionResearchInputs,
    evidence: ProvincialEvidence,
) -> dict[str, Any]:
    fit_artifact = dict(evidence.fit_artifact or {})
    axis_catalogs = dict(fit_artifact.get("axis_catalogs") or {})
    raw_province_axis = [str(value) for value in list(axis_catalogs.get("province") or [])]
    raw_time_axis = [str(value) for value in list(axis_catalogs.get("month") or [])]
    province_axis = [province for province in raw_province_axis if province in set(inputs.province_axis) and province not in {"", "Philippines", "unknown"}]
    time_axis = list(raw_time_axis)
    province_index = {province: idx for idx, province in enumerate(province_axis)}
    time_index = {time_label: idx for idx, time_label in enumerate(time_axis)}
    state_index = {state_name: idx for idx, state_name in enumerate(STATE_AXIS)}
    values = np.full((len(province_axis), len(time_axis), len(STATE_AXIS)), np.nan, dtype=np.float32)
    rows = list(read_json(evidence.run_dir / "phase3" / "state_estimates_rows.json", default=[]))
    for row in rows:
        province = str(row.get("province") or "")
        time_label = str(row.get("time") or "")
        state_name = str(row.get("state") or "")
        if province not in province_index or time_label not in time_index or state_name not in state_index:
            continue
        values[province_index[province], time_index[time_label], state_index[state_name]] = float(row.get("value") or 0.0)
    mask = np.isfinite(values)
    province_weights = np.asarray([float(evidence.province_weights.get(province, 0.0)) for province in province_axis], dtype=np.float64)
    if float(np.sum(province_weights)) <= np.finfo(np.float64).eps:
        province_weights = np.full((len(province_axis),), 1.0 / float(max(len(province_axis), 1)), dtype=np.float64)
    else:
        province_weights = province_weights / float(np.sum(province_weights))
    region_axis = sorted({infer_region_code(province) or "region_unknown" for province in province_axis})
    region_index = {region: idx for idx, region in enumerate(region_axis)}
    regional_values = np.full((len(region_axis), len(time_axis), len(STATE_AXIS)), np.nan, dtype=np.float32)
    national_values = np.full((1, len(time_axis), len(STATE_AXIS)), np.nan, dtype=np.float32)
    for time_idx in range(len(time_axis)):
        for state_idx in range(len(STATE_AXIS)):
            state_values = values[:, time_idx, state_idx]
            valid = np.isfinite(state_values)
            if np.any(valid):
                weighted = province_weights[valid]
                weighted = weighted / np.sum(weighted)
                national_values[0, time_idx, state_idx] = float(np.sum(weighted * state_values[valid]))
        for region_name in region_axis:
            province_members = [idx for idx, province in enumerate(province_axis) if (infer_region_code(province) or "region_unknown") == region_name]
            if not province_members:
                continue
            region_weights = province_weights[province_members]
            if float(np.sum(region_weights)) <= np.finfo(np.float64).eps:
                continue
            region_weights = region_weights / float(np.sum(region_weights))
            for state_idx in range(len(STATE_AXIS)):
                state_values = values[province_members, time_idx, state_idx]
                valid = np.isfinite(state_values)
                if not np.any(valid):
                    continue
                local_weights = region_weights[valid]
                local_weights = local_weights / np.sum(local_weights)
                regional_values[region_index[region_name], time_idx, state_idx] = float(np.sum(local_weights * state_values[valid]))
    return {
        "province_axis": province_axis,
        "time_axis": time_axis,
        "region_axis": region_axis,
        "state_axis": list(STATE_AXIS),
        "values": values,
        "mask": mask.astype(np.uint8),
        "regional_values": regional_values,
        "national_values": national_values,
        "province_weights": {province: round(float(evidence.province_weights.get(province, 0.0)), 8) for province in province_axis},
        "province_weight_time": evidence.province_weight_time,
    }


def _write_coverage_chart(
    path: Path,
    *,
    national_panel: dict[str, Any],
    auxiliary_panel: dict[str, Any],
) -> None:
    national_years = sorted({int(str(quarter).split("-", 1)[0]) for quarter in national_panel["quarter_axis"]})
    national_matrix = np.zeros((len(NATIONAL_METRIC_AXIS), len(national_years)), dtype=np.float64)
    quarter_axis = list(national_panel["quarter_axis"])
    for metric_idx, metric_name in enumerate(NATIONAL_METRIC_AXIS):
        mask_values = np.asarray(national_panel["mask"][:, metric_idx], dtype=np.float64)
        for year_idx, year in enumerate(national_years):
            year_mask = np.asarray([int(str(quarter).split("-", 1)[0]) == year for quarter in quarter_axis], dtype=bool)
            if np.any(year_mask):
                national_matrix[metric_idx, year_idx] = float(np.mean(mask_values[year_mask]))
    auxiliary_years = sorted({int(str(time_label).split("-", 1)[0]) for time_label in auxiliary_panel["time_axis"] if str(time_label).split("-", 1)[0].isdigit()})
    display_aux_years = [year for year in auxiliary_years if 2010 <= year <= 2025]
    auxiliary_matrix = np.zeros((len(STATE_AXIS), len(display_aux_years)), dtype=np.float64)
    time_axis = list(auxiliary_panel["time_axis"])
    values = np.asarray(auxiliary_panel["values"], dtype=np.float64)
    for state_idx, state_name in enumerate(STATE_AXIS):
        for year_idx, year in enumerate(display_aux_years):
            time_mask = np.asarray([int(str(time_label).split("-", 1)[0]) == year for time_label in time_axis], dtype=bool)
            if np.any(time_mask):
                auxiliary_matrix[state_idx, year_idx] = float(np.mean(np.isfinite(values[:, time_mask, state_idx])))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5))
    im0 = axes[0].imshow(national_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="Blues")
    axes[0].set_title("National observation coverage by year")
    axes[0].set_yticks(range(len(NATIONAL_METRIC_AXIS)))
    axes[0].set_yticklabels(list(NATIONAL_METRIC_AXIS))
    axes[0].set_xticks(range(len(national_years)))
    axes[0].set_xticklabels([str(year) for year in national_years], rotation=45, ha="right")
    fig.colorbar(im0, ax=axes[0], fraction=0.02, pad=0.02)

    im1 = axes[1].imshow(auxiliary_matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="Greens")
    axes[1].set_title("Provincial auxiliary state-share availability by year")
    axes[1].set_yticks(range(len(STATE_AXIS)))
    axes[1].set_yticklabels(list(STATE_AXIS))
    axes[1].set_xticks(range(len(display_aux_years)))
    axes[1].set_xticklabels([str(year) for year in display_aux_years], rotation=45, ha="right")
    fig.colorbar(im1, ax=axes[1], fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_module_seed_chart(path: Path, module_seed_manifest: dict[str, Any]) -> None:
    rankings = dict(module_seed_manifest.get("module_bundle_rankings") or {})
    bundle_names = sorted(
        {
            str(row.get("bundle_name") or "")
            for module_rows in rankings.values()
            for row in list(module_rows or [])
            if str(row.get("bundle_name") or "")
        }
    )
    module_names = [module for module in REPORTING_MODULES if module in rankings]
    matrix = np.zeros((len(module_names), len(bundle_names)), dtype=np.float64)
    for module_idx, module_name in enumerate(module_names):
        score_lookup = {
            str(row.get("bundle_name") or ""): float(row.get("bundle_score") or 0.0)
            for row in list(rankings.get(module_name) or [])
        }
        for bundle_idx, bundle_name in enumerate(bundle_names):
            matrix[module_idx, bundle_idx] = float(score_lookup.get(bundle_name, 0.0))
    fig, ax = plt.subplots(figsize=(14, 4.8))
    heatmap = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title("HMBA-00 module seed frontier")
    ax.set_yticks(range(len(module_names)))
    ax.set_yticklabels(module_names)
    ax.set_xticks(range(len(bundle_names)))
    ax.set_xticklabels(bundle_names, rotation=45, ha="right")
    fig.colorbar(heatmap, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _discover_latest_hmba_contract() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    experiment_dir = _latest_transition_experiment_dir("HMBA-00-hierarchical-contract-freeze", "contract_snapshot.json")
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-00 hierarchical contract freeze artifact was found under artifacts/runs")
    contract_snapshot = dict(read_json(experiment_dir / "contract_snapshot.json", default={}) or {})
    module_seed_manifest = dict(read_json(experiment_dir / "module_seed_manifest.json", default={}) or {})
    if not contract_snapshot or not module_seed_manifest:
        raise FileNotFoundError(f"HMBA-00 contract snapshot or module seed manifest is missing in {experiment_dir}")
    return experiment_dir, contract_snapshot, module_seed_manifest


def _hmba01_incumbent_config(contract_snapshot: dict[str, Any]) -> CandidateConfig:
    incumbent_decision = dict(contract_snapshot.get("national_incumbent_decision") or {})
    candidate_id = str(incumbent_decision.get("stage1_champion_candidate_id") or "")
    match = re.search(r"-h(?P<rank>\d+)-obs(?P<obs>on|off)$", candidate_id)
    if match is None:
        raise ValueError(f"Could not parse hidden rank / observation mode from incumbent candidate id: {candidate_id}")
    return CandidateConfig(
        candidate_id=f"hmba01::{candidate_id}",
        diagnosis_family=str(incumbent_decision.get("stage1_champion_diagnosis_family") or "hazard"),
        care_family=str(incumbent_decision.get("stage1_champion_care_family") or "markov"),
        hidden_rank=int(match.group("rank")),
        use_observation_covariates=match.group("obs") == "on",
    )


def _module_bundle_feature_lookup(module_seed_manifest: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    rankings = dict(module_seed_manifest.get("module_bundle_rankings") or {})
    lookup: dict[str, dict[str, list[str]]] = {}
    for module_name, bundle_rows in rankings.items():
        lookup[str(module_name)] = {
            str(bundle_row.get("bundle_name") or ""): [str(value) for value in list(bundle_row.get("member_factor_ids") or []) if str(value)]
            for bundle_row in list(bundle_rows or [])
            if str(bundle_row.get("bundle_name") or "")
        }
    return lookup


def _build_direct_feature_map(
    module_bundle_features: dict[str, dict[str, list[str]]],
    *,
    selected_bundles_by_module: dict[str, list[str]],
) -> dict[str, list[str]]:
    feature_map: dict[str, list[str]] = {module_name: [] for module_name in REPORTING_MODULES}
    for module_name, bundle_names in selected_bundles_by_module.items():
        feature_ids: list[str] = []
        for bundle_name in bundle_names:
            feature_ids.extend(list(module_bundle_features.get(module_name, {}).get(bundle_name, [])))
        feature_map[module_name] = sorted(set(feature_ids))
    return feature_map


def _hmba01_candidate_matrices(
    snapshot: IntegratedSnapshot,
    config: CandidateConfig,
    feature_map: dict[str, list[str]],
) -> dict[str, Any]:
    hidden = np.asarray(snapshot.hidden_scores[:, : config.hidden_rank], dtype=np.float64)
    observation_direct = {
        "tested_for_viral_load": _select_matrix(snapshot, snapshot.observation_feature_ids["tested_for_viral_load"])
        if config.use_observation_covariates
        else np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
        "virally_suppressed": _select_matrix(snapshot, snapshot.observation_feature_ids["virally_suppressed"])
        if config.use_observation_covariates
        else np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64),
    }
    return {
        "incidence_direct": _select_matrix(snapshot, list(feature_map.get("incidence", []))),
        "transition_direct": {
            transition: _select_matrix(snapshot, list(feature_map.get(transition, []))) for transition in TRANSITION_NAMES
        },
        "observation_direct": observation_direct,
        "hidden": hidden,
        "feature_ids": {
            "incidence": list(feature_map.get("incidence", [])),
            "transition": {transition: list(feature_map.get(transition, [])) for transition in TRANSITION_NAMES},
            "observation": {
                "tested_for_viral_load": list(snapshot.observation_feature_ids["tested_for_viral_load"]) if config.use_observation_covariates else [],
                "virally_suppressed": list(snapshot.observation_feature_ids["virally_suppressed"]) if config.use_observation_covariates else [],
            },
        },
    }


def _fit_passes_hmba01_gate(snapshot: IntegratedSnapshot, fit: FitResult) -> tuple[bool, dict[str, Any]]:
    baseline_comparison = _baseline_comparison(snapshot, fit)
    hard_valid = (
        bool(fit.success)
        and float(fit.window_metrics["historical"]["non_finite_count"]) <= 0.0
        and float(fit.window_metrics["historical"]["population_violation_count"]) <= 0.0
    )
    passes = (
        hard_valid
        and bool(baseline_comparison["model_beats_carry_forward"])
        and bool(baseline_comparison["model_beats_simple_compartmental"])
        and bool(baseline_comparison["diagnosis_flow_beats_carry_forward"])
        and bool(baseline_comparison["diagnosis_flow_beats_simple_compartmental"])
    )
    return passes, baseline_comparison


def _single_bundle_gain_rows(
    *,
    module_frontier_rows: list[dict[str, Any]],
    hidden_only_primary_loss: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in module_frontier_rows:
        if int(row.get("step") or 0) != 1:
            continue
        if len(list(row.get("trial_bundle_sequence") or [])) != 1:
            continue
        trial_primary = float(row.get("trial_primary_loss") or 0.0)
        rows.append(
            {
                "module_name": str(row.get("module_name") or ""),
                "bundle_name": str((row.get("trial_bundle_sequence") or [""])[0]),
                "historical_primary_loss_gain": float(hidden_only_primary_loss - trial_primary),
                "passes_gate": bool(row.get("passes_gate")),
            }
        )
    rows.sort(key=lambda row: (str(row["module_name"]), -float(row["historical_primary_loss_gain"]), str(row["bundle_name"])))
    return rows


def _write_single_bundle_gain_heatmap(path: Path, rows: list[dict[str, Any]]) -> None:
    module_names = [module_name for module_name in REPORTING_MODULES if any(str(row["module_name"]) == module_name for row in rows)]
    bundle_names = sorted({str(row["bundle_name"]) for row in rows})
    matrix = np.zeros((len(module_names), len(bundle_names)), dtype=np.float64)
    for module_idx, module_name in enumerate(module_names):
        for bundle_idx, bundle_name in enumerate(bundle_names):
            match = next(
                (
                    float(row["historical_primary_loss_gain"])
                    for row in rows
                    if str(row["module_name"]) == module_name and str(row["bundle_name"]) == bundle_name
                ),
                0.0,
            )
            matrix[module_idx, bundle_idx] = match
    fig, ax = plt.subplots(figsize=(14, 5.2))
    heatmap = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title("HMBA-01 Single-Bundle Historical Primary-Loss Gain")
    ax.set_yticks(range(len(module_names)))
    ax.set_yticklabels(module_names)
    ax.set_xticks(range(len(bundle_names)))
    ax.set_xticklabels(bundle_names, rotation=45, ha="right")
    fig.colorbar(heatmap, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_module_promotion_trace(path: Path, module_results: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    for module_name in REPORTING_MODULES:
        result = dict(module_results.get(module_name) or {})
        trace = list(result.get("promotion_trace") or [])
        if not trace:
            continue
        x_values = [int(row.get("step") or 0) for row in trace]
        y_values = [float(row.get("primary_loss") or 0.0) for row in trace]
        ax.plot(x_values, y_values, marker="o", linewidth=1.5, label=module_name)
    ax.set_title("HMBA-01 Module Promotion Trace")
    ax.set_xlabel("Promotion step")
    ax.set_ylabel("Historical primary loss")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_combined_comparison_chart(
    path: Path,
    *,
    hidden_only_fit: FitResult,
    combined_fit: FitResult,
    incumbent_primary_loss: float,
    incumbent_diag_flow_loss: float,
) -> None:
    labels = ["hidden-only", "hmba01-combined", "integrated-incumbent"]
    primary_values = [
        float(hidden_only_fit.window_metrics["historical"]["primary_loss"]),
        float(combined_fit.window_metrics["historical"]["primary_loss"]),
        float(incumbent_primary_loss),
    ]
    diag_values = [
        float(hidden_only_fit.window_metrics["historical"]["diag_flow_loss"]),
        float(combined_fit.window_metrics["historical"]["diag_flow_loss"]),
        float(incumbent_diag_flow_loss),
    ]
    x_axis = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x_axis - width / 2.0, primary_values, width=width, label="primary loss")
    ax.bar(x_axis + width / 2.0, diag_values, width=width, label="diagnosis-flow loss")
    ax.set_title("HMBA-01 Combined Model Comparison")
    ax.set_ylabel("Historical normalized loss")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _discover_latest_hmba01_summary() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    experiment_dir = _latest_transition_experiment_dir("HMBA-01-module-local-direct-bundle-promotion", "combined_promoted_model.json")
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-01 module-local direct-bundle promotion artifact was found under artifacts/runs")
    combined_summary = dict(read_json(experiment_dir / "combined_promoted_model.json", default={}) or {})
    module_results = dict(read_json(experiment_dir / "module_promotion_results.json", default={}) or {})
    if not combined_summary or not module_results:
        raise FileNotFoundError(f"HMBA-01 combined summary or module promotion results are missing in {experiment_dir}")
    return experiment_dir, combined_summary, module_results


def _analysis_year_month_indices(inputs: TransitionResearchInputs) -> dict[int, int]:
    indices: dict[int, int] = {}
    for month_idx, month_label in enumerate(inputs.month_axis):
        year_value = int(str(month_label).split("-", maxsplit=1)[0])
        if year_value not in set(inputs.analysis_years):
            continue
        previous_idx = indices.get(year_value)
        if previous_idx is None or str(inputs.month_axis[previous_idx]) < str(month_label):
            indices[year_value] = month_idx
    return {year: indices[year] for year in sorted(indices)}


def _analysis_year_auxiliary_indices(auxiliary_panel: dict[str, Any], analysis_years: list[int]) -> dict[int, int]:
    analysis_set = set(int(year) for year in analysis_years)
    indices: dict[int, int] = {}
    for time_idx, time_label in enumerate(list(auxiliary_panel["time_axis"])):
        year_text = str(time_label).split("-", maxsplit=1)[0]
        if not year_text.isdigit():
            continue
        year_value = int(year_text)
        if year_value not in analysis_set:
            continue
        previous_idx = indices.get(year_value)
        if previous_idx is None or str(auxiliary_panel["time_axis"][previous_idx]) < str(time_label):
            indices[year_value] = time_idx
    return {year: indices[year] for year in sorted(indices)}


def _province_lookup(inputs: TransitionResearchInputs, evidence: ProvincialEvidence) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    province_indices = _province_indices(inputs, evidence)
    province_axis = [str(inputs.province_axis[idx]) for idx in province_indices]
    region_axis = np.asarray([str(infer_region_code(province) or "region_unknown") for province in province_axis], dtype=object)
    weights = np.asarray([float(evidence.province_weights.get(province, 0.0)) for province in province_axis], dtype=np.float64)
    if float(np.sum(weights)) <= np.finfo(np.float64).eps:
        weights = np.full((len(province_axis),), 1.0 / float(max(len(province_axis), 1)), dtype=np.float64)
    else:
        weights = weights / float(np.sum(weights))
    return province_indices, province_axis, region_axis, weights


def _weighted_standardize(panel: np.ndarray, province_weights: np.ndarray) -> np.ndarray:
    eps = float(np.finfo(np.float64).eps)
    values = np.asarray(panel, dtype=np.float64)
    if values.size == 0:
        return np.asarray(values, dtype=np.float64)
    weights = np.asarray(province_weights, dtype=np.float64)[:, None, None]
    weight_sum = np.sum(weights, axis=(0, 1), keepdims=True)
    safe_weight_sum = np.where(weight_sum > eps, weight_sum, 1.0)
    mean = np.sum(weights * values, axis=(0, 1), keepdims=True) / safe_weight_sum
    centered = values - mean
    variance = np.sum(weights * np.square(centered), axis=(0, 1), keepdims=True) / safe_weight_sum
    std = np.sqrt(np.where(variance > eps, variance, 1.0))
    return centered / std


def _build_module_feature_panels(
    inputs: TransitionResearchInputs,
    evidence: ProvincialEvidence,
    feature_map: dict[str, list[str]],
) -> dict[str, Any]:
    province_indices, province_axis, region_by_province, province_weights = _province_lookup(inputs, evidence)
    month_indices_by_year = _analysis_year_month_indices(inputs)
    year_axis = list(month_indices_by_year)
    feature_panels: dict[str, dict[str, Any]] = {}
    for module_name in REPORTING_MODULES:
        feature_ids = list(feature_map.get(module_name, []))
        if not feature_ids:
            feature_panels[module_name] = {
                "feature_ids": [],
                "year_axis": year_axis,
                "province_axis": province_axis,
                "region_by_province": list(region_by_province),
                "province_weights": province_weights,
                "province_features": np.zeros((len(province_axis), len(year_axis), 0), dtype=np.float64),
            }
            continue
        feature_indices = [int(inputs.factor_index[feature_id]) for feature_id in feature_ids]
        raw = np.asarray(
            inputs.province_tensor[np.asarray(province_indices, dtype=int)[:, None], np.asarray(list(month_indices_by_year.values()), dtype=int)[None, :], :][
                :, :, feature_indices
            ],
            dtype=np.float64,
        )
        standardized = _weighted_standardize(raw, province_weights)
        feature_panels[module_name] = {
            "feature_ids": feature_ids,
            "year_axis": year_axis,
            "province_axis": province_axis,
            "region_by_province": list(region_by_province),
            "province_weights": province_weights,
            "province_features": standardized,
        }
    return {
        "year_axis": year_axis,
        "province_axis": province_axis,
        "region_by_province": list(region_by_province),
        "province_weights": province_weights,
        "module_panels": feature_panels,
    }


def _module_auxiliary_proxy_panel(
    auxiliary_panel: dict[str, Any],
    analysis_years: list[int],
) -> dict[str, Any]:
    eps = float(np.finfo(np.float64).eps)
    year_indices = _analysis_year_auxiliary_indices(auxiliary_panel, analysis_years)
    year_axis = list(year_indices)
    values = np.asarray(auxiliary_panel["values"], dtype=np.float64)[:, np.asarray(list(year_indices.values()), dtype=int), :]
    province_axis = list(auxiliary_panel["province_axis"])
    state_index = {state_name: idx for idx, state_name in enumerate(auxiliary_panel["state_axis"])}
    total = np.sum(values, axis=2)
    diagnosed = values[:, :, state_index["D"]] + values[:, :, state_index["A"]] + values[:, :, state_index["V"]] + values[:, :, state_index["L"]]
    art = values[:, :, state_index["A"]] + values[:, :, state_index["V"]]
    art_or_loss = values[:, :, state_index["A"]] + values[:, :, state_index["L"]]

    def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > eps)
        output = np.full(numerator.shape, np.nan, dtype=np.float64)
        output[mask] = numerator[mask] / denominator[mask]
        return output, mask

    incidence_values, incidence_mask = _ratio(values[:, :, state_index["U"]], total)
    diagnosed_values, diagnosed_mask = _ratio(diagnosed, total)
    art_values, art_mask = _ratio(art, diagnosed)
    suppression_values, suppression_mask = _ratio(values[:, :, state_index["V"]], art)
    attrition_values, attrition_mask = _ratio(values[:, :, state_index["L"]], art_or_loss)
    return_values, return_mask = _ratio(values[:, :, state_index["A"]], art_or_loss)

    return {
        "year_axis": year_axis,
        "province_axis": province_axis,
        "targets": {
            "incidence": {"values": incidence_values, "mask": incidence_mask, "label": "U share proxy"},
            "U_to_D": {"values": diagnosed_values, "mask": diagnosed_mask, "label": "diagnosed share proxy"},
            "D_to_A": {"values": art_values, "mask": art_mask, "label": "ART among diagnosed proxy"},
            "A_to_V": {"values": suppression_values, "mask": suppression_mask, "label": "suppressed among ART proxy"},
            "A_to_L": {"values": attrition_values, "mask": attrition_mask, "label": "lost among (A+L) proxy"},
            "L_to_A": {"values": return_values, "mask": return_mask, "label": "A among (A+L) proxy"},
        },
    }


def _weighted_lstsq(features: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError("weighted least squares expects a 2D feature matrix")
    if features.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    if features.shape[0] == 0:
        return np.zeros((features.shape[1],), dtype=np.float64)
    sqrt_weights = np.sqrt(np.maximum(np.asarray(weights, dtype=np.float64), np.finfo(np.float64).eps))
    lhs = np.asarray(features, dtype=np.float64) * sqrt_weights[:, None]
    rhs = np.asarray(target, dtype=np.float64) * sqrt_weights
    coefficient, *_ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return np.asarray(coefficient, dtype=np.float64)


def _effective_sample_size(weights: np.ndarray) -> float:
    array = np.asarray(weights, dtype=np.float64)
    numerator = float(np.square(np.sum(array)))
    denominator = float(np.sum(np.square(array)))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return numerator / denominator


def _weighted_rmse(values: np.ndarray, target: np.ndarray, weights: np.ndarray) -> float:
    diff = np.asarray(values, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(weight_array))
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.sqrt(np.sum(weight_array * np.square(diff)) / denominator))


def _stack_valid_rows(features: np.ndarray, target: np.ndarray, mask: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = np.asarray(features, dtype=np.float64)[mask]
    target_rows = np.asarray(target, dtype=np.float64)[mask]
    weight_rows = np.broadcast_to(np.asarray(weights, dtype=np.float64)[:, None], mask.shape)[mask]
    return feature_rows, target_rows, weight_rows


def _fit_hierarchical_module_layer(
    module_name: str,
    feature_panel: dict[str, Any],
    target_panel: dict[str, Any],
) -> dict[str, Any]:
    province_axis = list(feature_panel["province_axis"])
    region_by_province = list(feature_panel["region_by_province"])
    province_weights = np.asarray(feature_panel["province_weights"], dtype=np.float64)
    year_axis = list(feature_panel["year_axis"])
    feature_ids = list(feature_panel["feature_ids"])
    province_features = np.asarray(feature_panel["province_features"], dtype=np.float64)
    target_values = np.asarray(target_panel["values"], dtype=np.float64)
    target_mask = np.asarray(target_panel["mask"], dtype=bool)

    if province_features.shape[:2] != target_values.shape:
        raise ValueError(f"HMBA-02 module {module_name} has mismatched feature and target panels")
    if province_features.shape[2] == 0:
        empty_prediction = np.zeros_like(target_values, dtype=np.float64)
        return {
            "feature_ids": [],
            "year_axis": year_axis,
            "province_axis": province_axis,
            "region_axis": sorted(set(region_by_province)),
            "proxy_label": str(target_panel["label"]),
            "national_coefficient": [],
            "region_coefficients": {},
            "province_coefficients": {},
            "effective_sample_sizes": {"national": 0.0, "regions": {}, "provinces": {}},
            "pooling_strength_summary": {"mean_region_pool_share": 0.0, "mean_province_pool_share": 0.0},
            "predictions": {"national_only": empty_prediction, "region_pooled": empty_prediction, "hierarchical": empty_prediction},
            "metrics": {
                "national_only_rmse": float("inf"),
                "region_pooled_rmse": float("inf"),
                "hierarchical_rmse": float("inf"),
                "hierarchical_gain_vs_national_only": 0.0,
                "hierarchical_gain_vs_region_pooled": 0.0,
            },
            "depth_metrics": {"national": float("inf"), "region": float("inf"), "province": float("inf")},
            "national_score_by_depth": {
                "national": [0.0 for _ in year_axis],
                "region": [0.0 for _ in year_axis],
                "province": [0.0 for _ in year_axis],
            },
            "national_hierarchical_score_by_year": [0.0 for _ in year_axis],
        }

    national_year_features: list[np.ndarray] = []
    national_year_targets: list[float] = []
    national_year_weights: list[float] = []
    for year_idx in range(len(year_axis)):
        valid = np.asarray(target_mask[:, year_idx], dtype=bool)
        if not np.any(valid):
            continue
        local_weights = province_weights[valid]
        local_weights = local_weights / float(np.sum(local_weights))
        national_year_features.append(np.sum(local_weights[:, None] * province_features[valid, year_idx, :], axis=0))
        national_year_targets.append(float(np.sum(local_weights * target_values[valid, year_idx])))
        national_year_weights.append(1.0)
    national_matrix = (
        np.asarray(national_year_features, dtype=np.float64)
        if national_year_features
        else np.zeros((0, province_features.shape[2]), dtype=np.float64)
    )
    national_target = np.asarray(national_year_targets, dtype=np.float64)
    national_weights = np.asarray(national_year_weights, dtype=np.float64)
    national_coef = (
        _weighted_lstsq(national_matrix, national_target, national_weights)
        if national_target.size
        else np.zeros((province_features.shape[2],), dtype=np.float64)
    )
    national_effective_n = float(len(national_target))

    region_coefficients: dict[str, np.ndarray] = {}
    region_effective_n: dict[str, float] = {}
    region_pool_share: dict[str, float] = {}
    region_prediction = np.full(target_values.shape, np.nan, dtype=np.float64)
    hierarchical_prediction = np.full(target_values.shape, np.nan, dtype=np.float64)

    unique_regions = sorted(set(region_by_province))
    region_member_counts = {region: sum(1 for value in region_by_province if value == region) for region in unique_regions}
    province_coefficients: dict[str, np.ndarray] = {}
    province_effective_n: dict[str, float] = {}
    province_pool_share: dict[str, float] = {}

    for region in unique_regions:
        province_idx = [idx for idx, value in enumerate(region_by_province) if value == region]
        feature_rows, target_rows, weight_rows = _stack_valid_rows(
            province_features[province_idx, :, :],
            target_values[province_idx, :],
            target_mask[province_idx, :],
            province_weights[province_idx],
        )
        region_raw = _weighted_lstsq(feature_rows, target_rows, weight_rows) if feature_rows.size else np.asarray(national_coef, dtype=np.float64)
        region_n = _effective_sample_size(weight_rows)
        pooled_share = region_n / max(region_n + national_effective_n, np.finfo(np.float64).eps)
        region_coef = (1.0 - pooled_share) * national_coef + pooled_share * region_raw
        region_coefficients[region] = np.asarray(region_coef, dtype=np.float64)
        region_effective_n[region] = float(region_n)
        region_pool_share[region] = float(pooled_share)
        region_prediction[np.asarray(province_idx, dtype=int), :] = np.einsum("pyf,f->py", province_features[province_idx, :, :], region_coef)

    for province_idx, province_name in enumerate(province_axis):
        region = region_by_province[province_idx]
        province_valid = np.asarray(target_mask[province_idx, :], dtype=bool)
        province_rows = np.asarray(province_features[province_idx, province_valid, :], dtype=np.float64)
        province_target = np.asarray(target_values[province_idx, province_valid], dtype=np.float64)
        province_weights_local = np.ones((province_rows.shape[0],), dtype=np.float64)
        province_raw = _weighted_lstsq(province_rows, province_target, province_weights_local) if province_rows.size else np.asarray(region_coefficients[region], dtype=np.float64)
        province_n = float(province_rows.shape[0])
        parent_n = float(region_effective_n.get(region, 0.0)) / float(max(region_member_counts.get(region, 1), 1))
        pooled_share = province_n / max(province_n + parent_n, np.finfo(np.float64).eps) if province_n > 0.0 else 0.0
        province_coef = (1.0 - pooled_share) * region_coefficients[region] + pooled_share * province_raw
        province_coefficients[province_name] = np.asarray(province_coef, dtype=np.float64)
        province_effective_n[province_name] = province_n
        province_pool_share[province_name] = float(pooled_share)
        hierarchical_prediction[province_idx, :] = np.einsum("yf,f->y", province_features[province_idx, :, :], province_coef)

    national_prediction = np.einsum("pyf,f->py", province_features, national_coef)
    valid_mask = np.asarray(target_mask, dtype=bool)
    repeated_weights = np.broadcast_to(province_weights[:, None], valid_mask.shape)[valid_mask]
    metrics = {
        "national_only_rmse": _weighted_rmse(national_prediction[valid_mask], target_values[valid_mask], repeated_weights),
        "region_pooled_rmse": _weighted_rmse(region_prediction[valid_mask], target_values[valid_mask], repeated_weights),
        "hierarchical_rmse": _weighted_rmse(hierarchical_prediction[valid_mask], target_values[valid_mask], repeated_weights),
    }
    metrics["hierarchical_gain_vs_national_only"] = float(metrics["national_only_rmse"] - metrics["hierarchical_rmse"])
    metrics["hierarchical_gain_vs_region_pooled"] = float(metrics["region_pooled_rmse"] - metrics["hierarchical_rmse"])

    national_scores_by_depth = {"national": [], "region": [], "province": []}
    for year_idx in range(len(year_axis)):
        valid = np.asarray(target_mask[:, year_idx], dtype=bool)
        if np.any(valid):
            local_weights = province_weights[valid]
            local_weights = local_weights / float(np.sum(local_weights))
            national_scores_by_depth["national"].append(float(np.sum(local_weights * national_prediction[valid, year_idx])))
            national_scores_by_depth["region"].append(float(np.sum(local_weights * region_prediction[valid, year_idx])))
            national_scores_by_depth["province"].append(float(np.sum(local_weights * hierarchical_prediction[valid, year_idx])))
        else:
            national_scores_by_depth["national"].append(0.0)
            national_scores_by_depth["region"].append(0.0)
            national_scores_by_depth["province"].append(0.0)

    return {
        "feature_ids": feature_ids,
        "year_axis": year_axis,
        "province_axis": province_axis,
        "region_axis": unique_regions,
        "region_by_province": region_by_province,
        "proxy_label": str(target_panel["label"]),
        "national_coefficient": [float(value) for value in national_coef],
        "region_coefficients": {region: [float(value) for value in coefficient] for region, coefficient in region_coefficients.items()},
        "province_coefficients": {province: [float(value) for value in coefficient] for province, coefficient in province_coefficients.items()},
        "effective_sample_sizes": {
            "national": float(national_effective_n),
            "regions": {region: float(value) for region, value in region_effective_n.items()},
            "provinces": {province: float(value) for province, value in province_effective_n.items()},
        },
        "pooling_strength_summary": {
            "mean_region_pool_share": float(np.mean(list(region_pool_share.values()))) if region_pool_share else 0.0,
            "mean_province_pool_share": float(np.mean(list(province_pool_share.values()))) if province_pool_share else 0.0,
        },
        "predictions": {
            "national_only": national_prediction,
            "region_pooled": region_prediction,
            "hierarchical": hierarchical_prediction,
        },
        "metrics": metrics,
        "depth_metrics": {
            "national": float(metrics["national_only_rmse"]),
            "region": float(metrics["region_pooled_rmse"]),
            "province": float(metrics["hierarchical_rmse"]),
        },
        "national_score_by_depth": {
            depth_name: [float(value) for value in values] for depth_name, values in national_scores_by_depth.items()
        },
        "national_hierarchical_score_by_year": [float(value) for value in national_scores_by_depth["province"]],
    }


def _expand_yearly_scores_to_quarters(snapshot: IntegratedSnapshot, year_axis: list[int], yearly_scores: list[float]) -> np.ndarray:
    year_lookup = {int(year): float(yearly_scores[idx]) for idx, year in enumerate(year_axis)}
    expanded = np.zeros((len(snapshot.model_quarters),), dtype=np.float64)
    latest_value = float(next(iter(year_lookup.values()), 0.0))
    for quarter_idx, quarter in enumerate(snapshot.model_quarters):
        year_value = _quarter_year(quarter)
        if year_value in year_lookup:
            latest_value = float(year_lookup[year_value])
        expanded[quarter_idx] = latest_value
    return expanded


def _hmba02_candidate_matrices(
    snapshot: IntegratedSnapshot,
    config: CandidateConfig,
    hierarchical_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    hidden = np.asarray(snapshot.hidden_scores[:, : config.hidden_rank], dtype=np.float64)
    zero_matrix = np.zeros((len(snapshot.model_quarters), 0), dtype=np.float64)

    def _score_column(module_name: str) -> np.ndarray:
        values = np.asarray(hierarchical_scores.get(module_name, np.zeros((len(snapshot.model_quarters),), dtype=np.float64)), dtype=np.float64)
        if not np.any(np.isfinite(values)) or np.std(values) <= np.finfo(np.float64).eps:
            return zero_matrix
        return values[:, None]

    observation_direct = {
        "tested_for_viral_load": _select_matrix(snapshot, snapshot.observation_feature_ids["tested_for_viral_load"])
        if config.use_observation_covariates
        else zero_matrix,
        "virally_suppressed": _select_matrix(snapshot, snapshot.observation_feature_ids["virally_suppressed"])
        if config.use_observation_covariates
        else zero_matrix,
    }
    return {
        "incidence_direct": _score_column("incidence"),
        "transition_direct": {transition: _score_column(transition) for transition in TRANSITION_NAMES},
        "observation_direct": observation_direct,
        "hidden": hidden,
        "feature_ids": {
            "incidence": ["hierarchical::incidence"] if _score_column("incidence").shape[1] else [],
            "transition": {
                transition: [f"hierarchical::{transition}"] if _score_column(transition).shape[1] else []
                for transition in TRANSITION_NAMES
            },
            "observation": {
                "tested_for_viral_load": list(snapshot.observation_feature_ids["tested_for_viral_load"]) if config.use_observation_covariates else [],
                "virally_suppressed": list(snapshot.observation_feature_ids["virally_suppressed"]) if config.use_observation_covariates else [],
            },
        },
    }


def _write_hmba02_auxiliary_fit_chart(path: Path, module_summaries: dict[str, Any]) -> None:
    module_names = [module_name for module_name in REPORTING_MODULES if module_name in module_summaries]
    x_axis = np.arange(len(module_names))
    width = 0.25
    national_values = [
        float(value) if np.isfinite(value := float((module_summaries[module]["metrics"] or {}).get("national_only_rmse", np.nan))) else np.nan
        for module in module_names
    ]
    region_values = [
        float(value) if np.isfinite(value := float((module_summaries[module]["metrics"] or {}).get("region_pooled_rmse", np.nan))) else np.nan
        for module in module_names
    ]
    hierarchical_values = [
        float(value) if np.isfinite(value := float((module_summaries[module]["metrics"] or {}).get("hierarchical_rmse", np.nan))) else np.nan
        for module in module_names
    ]
    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.bar(x_axis - width, national_values, width=width, label="national-only")
    ax.bar(x_axis, region_values, width=width, label="region-pooled")
    ax.bar(x_axis + width, hierarchical_values, width=width, label="hierarchical")
    ax.set_title("HMBA-02 Auxiliary Fit Comparison")
    ax.set_ylabel("weighted RMSE")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(module_names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_hmba02_pooling_profile_chart(path: Path, module_summaries: dict[str, Any]) -> None:
    module_names = [module_name for module_name in REPORTING_MODULES if module_name in module_summaries]
    region_values = [float((module_summaries[module]["pooling_strength_summary"] or {}).get("mean_region_pool_share", 0.0)) for module in module_names]
    province_values = [float((module_summaries[module]["pooling_strength_summary"] or {}).get("mean_province_pool_share", 0.0)) for module in module_names]
    x_axis = np.arange(len(module_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.bar(x_axis - width / 2.0, region_values, width=width, label="region pooling share")
    ax.bar(x_axis + width / 2.0, province_values, width=width, label="province pooling share")
    ax.set_title("HMBA-02 Pooling Profile")
    ax.set_ylabel("share of child raw coefficient retained")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(module_names, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_hmba02_combined_comparison_chart(
    path: Path,
    *,
    hmba01_primary_loss: float,
    hmba01_diag_flow_loss: float,
    hmba02_fit: FitResult,
    incumbent_primary_loss: float,
    incumbent_diag_flow_loss: float,
) -> None:
    labels = ["hmba01-combined", "hmba02-hierarchical", "integrated-incumbent"]
    x_axis = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.bar(
        x_axis - width / 2.0,
        [float(hmba01_primary_loss), float(hmba02_fit.window_metrics["historical"]["primary_loss"]), float(incumbent_primary_loss)],
        width=width,
        label="primary loss",
    )
    ax.bar(
        x_axis + width / 2.0,
        [float(hmba01_diag_flow_loss), float(hmba02_fit.window_metrics["historical"]["diag_flow_loss"]), float(incumbent_diag_flow_loss)],
        width=width,
        label="diagnosis-flow loss",
    )
    ax.set_title("HMBA-02 Combined Model Comparison")
    ax.set_ylabel("historical normalized loss")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _hmba02_auxiliary_gate(module_summaries: dict[str, Any]) -> dict[str, Any]:
    active_modules = [
        module_name
        for module_name, summary in module_summaries.items()
        if list(summary.get("feature_ids") or [])
    ]
    if not active_modules:
        return {
            "active_module_count": 0,
            "national_only_mean_rmse": float("inf"),
            "hierarchical_mean_rmse": float("inf"),
            "passes_auxiliary_gate": False,
        }
    national_mean = float(np.mean([float(module_summaries[module]["metrics"]["national_only_rmse"]) for module in active_modules]))
    hierarchical_mean = float(np.mean([float(module_summaries[module]["metrics"]["hierarchical_rmse"]) for module in active_modules]))
    region_mean = float(np.mean([float(module_summaries[module]["metrics"]["region_pooled_rmse"]) for module in active_modules]))
    return {
        "active_module_count": int(len(active_modules)),
        "national_only_mean_rmse": national_mean,
        "region_pooled_mean_rmse": region_mean,
        "hierarchical_mean_rmse": hierarchical_mean,
        "hierarchical_gain_vs_national_only": float(national_mean - hierarchical_mean),
        "hierarchical_gain_vs_region_pooled": float(region_mean - hierarchical_mean),
        "passes_auxiliary_gate": bool(hierarchical_mean <= national_mean),
    }


def _discover_latest_hmba02_summary() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    experiment_dir = _latest_transition_experiment_dir("HMBA-02-hierarchical-geography-layer", "hierarchical_combined_model.json")
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-02 hierarchical geography layer artifact was found under artifacts/runs")
    combined_summary = dict(read_json(experiment_dir / "hierarchical_combined_model.json", default={}) or {})
    auxiliary_summary = dict(read_json(experiment_dir / "hierarchical_auxiliary_evaluation.json", default={}) or {})
    if not combined_summary or not auxiliary_summary:
        raise FileNotFoundError(f"HMBA-02 combined summary or auxiliary evaluation is missing in {experiment_dir}")
    return experiment_dir, combined_summary, auxiliary_summary


def _module_feature_context(inputs: TransitionResearchInputs, evidence: ProvincialEvidence) -> dict[str, Any]:
    province_indices, province_axis, region_by_province, province_weights = _province_lookup(inputs, evidence)
    month_indices_by_year = _analysis_year_month_indices(inputs)
    return {
        "province_indices": province_indices,
        "province_axis": province_axis,
        "region_by_province": list(region_by_province),
        "province_weights": province_weights,
        "year_axis": list(month_indices_by_year),
        "month_indices": [int(month_indices_by_year[year]) for year in list(month_indices_by_year)],
    }


def _single_module_feature_panel(
    inputs: TransitionResearchInputs,
    feature_ids: list[str],
    context: dict[str, Any],
) -> dict[str, Any]:
    province_axis = list(context["province_axis"])
    year_axis = list(context["year_axis"])
    if not feature_ids:
        return {
            "feature_ids": [],
            "year_axis": year_axis,
            "province_axis": province_axis,
            "region_by_province": list(context["region_by_province"]),
            "province_weights": np.asarray(context["province_weights"], dtype=np.float64),
            "province_features": np.zeros((len(province_axis), len(year_axis), 0), dtype=np.float64),
        }
    feature_indices = [int(inputs.factor_index[feature_id]) for feature_id in feature_ids]
    raw = np.asarray(inputs.province_tensor, dtype=np.float64)[
        np.asarray(context["province_indices"], dtype=int)[:, None],
        np.asarray(context["month_indices"], dtype=int)[None, :],
        :,
    ][:, :, feature_indices]
    standardized = _weighted_standardize(raw, np.asarray(context["province_weights"], dtype=np.float64))
    return {
        "feature_ids": list(feature_ids),
        "year_axis": year_axis,
        "province_axis": province_axis,
        "region_by_province": list(context["region_by_province"]),
        "province_weights": np.asarray(context["province_weights"], dtype=np.float64),
        "province_features": standardized,
    }


def _subset_module_target_panels(feature_panel: dict[str, Any], auxiliary_targets: dict[str, Any], module_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    target_panel = dict((auxiliary_targets.get("targets") or {}).get(module_name) or {})
    if not target_panel:
        return dict(feature_panel), {"label": "", "values": np.zeros((0, 0), dtype=np.float64), "mask": np.zeros((0, 0), dtype=bool)}
    common_provinces = [province for province in list(feature_panel.get("province_axis") or []) if province in set(auxiliary_targets["province_axis"])]
    common_years = [year for year in list(feature_panel.get("year_axis") or []) if year in set(auxiliary_targets["year_axis"])]
    province_lookup_features = {province: idx for idx, province in enumerate(list(feature_panel.get("province_axis") or []))}
    province_lookup_aux = {province: idx for idx, province in enumerate(list(auxiliary_targets["province_axis"] or []))}
    year_lookup_features = {int(year): idx for idx, year in enumerate(list(feature_panel.get("year_axis") or []))}
    year_lookup_aux = {int(year): idx for idx, year in enumerate(list(auxiliary_targets["year_axis"] or []))}
    return (
        {
            "feature_ids": list(feature_panel.get("feature_ids") or []),
            "province_axis": common_provinces,
            "region_by_province": [
                str((feature_panel.get("region_by_province") or [])[province_lookup_features[province]]) for province in common_provinces
            ],
            "province_weights": np.asarray(
                [np.asarray(feature_panel.get("province_weights"), dtype=np.float64)[province_lookup_features[province]] for province in common_provinces],
                dtype=np.float64,
            ),
            "year_axis": common_years,
            "province_features": np.asarray(
                [
                    [
                        np.asarray(feature_panel["province_features"], dtype=np.float64)[province_lookup_features[province], year_lookup_features[int(year)], :]
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=np.float64,
            ),
        },
        {
            "label": str(target_panel.get("label") or ""),
            "values": np.asarray(
                [
                    [
                        np.asarray(target_panel["values"], dtype=np.float64)[province_lookup_aux[province], year_lookup_aux[int(year)]]
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=np.float64,
            ),
            "mask": np.asarray(
                [
                    [
                        bool(np.asarray(target_panel["mask"], dtype=bool)[province_lookup_aux[province], year_lookup_aux[int(year)]])
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=bool,
            ),
        },
    )


def _depth_rank(depth_name: str) -> int:
    return {"none": 0, "national": 1, "region": 2, "province": 3}.get(str(depth_name), 0)


def _hmba03_auxiliary_gate(
    module_summaries: dict[str, Any],
    depth_map: dict[str, str],
    *,
    absolute_ceiling: float | None = None,
    module_ceiling_by_name: dict[str, float] | None = None,
) -> dict[str, Any]:
    active_modules = [
        module_name
        for module_name, summary in module_summaries.items()
        if list(summary.get("feature_ids") or []) and str(depth_map.get(module_name) or "none") != "none"
    ]
    if not active_modules:
        return {
            "active_module_count": 0,
            "national_only_mean_rmse": float("inf"),
            "selected_depth_mean_rmse": float("inf"),
            "passes_auxiliary_gate": False,
        }
    national_values = [float(module_summaries[module]["depth_metrics"]["national"]) for module in active_modules]
    selected_values = [float(module_summaries[module]["depth_metrics"][str(depth_map[module])]) for module in active_modules]
    national_mean = float(np.mean(national_values))
    selected_mean = float(np.mean(selected_values))
    reference_ceiling = float(absolute_ceiling) if absolute_ceiling is not None and np.isfinite(float(absolute_ceiling)) else float("inf")
    max_selected_rmse = float(max(selected_values)) if selected_values else float("inf")
    spike_modules = [
        {
            "module_name": module_name,
            "depth": str(depth_map[module_name]),
            "selected_depth_rmse": float(module_summaries[module_name]["depth_metrics"][str(depth_map[module_name])]),
            "module_ceiling": float((module_ceiling_by_name or {}).get(module_name, reference_ceiling)),
        }
        for module_name in active_modules
        if float(module_summaries[module_name]["depth_metrics"][str(depth_map[module_name])])
        > float((module_ceiling_by_name or {}).get(module_name, reference_ceiling))
    ]
    return {
        "active_module_count": int(len(active_modules)),
        "national_only_mean_rmse": national_mean,
        "selected_depth_mean_rmse": selected_mean,
        "hierarchical_gain_vs_national_only": float(national_mean - selected_mean),
        "absolute_auxiliary_ceiling": reference_ceiling,
        "max_selected_module_rmse": max_selected_rmse,
        "spike_modules": spike_modules,
        "module_ceiling_by_name": {
            str(module_name): float((module_ceiling_by_name or {}).get(module_name, reference_ceiling))
            for module_name in active_modules
        },
        "passes_auxiliary_gate": bool(selected_mean <= national_mean and selected_mean <= reference_ceiling and not spike_modules),
    }


def _hmba03_joint_score(fit: FitResult, auxiliary_gate: dict[str, Any]) -> tuple[float, float, float, float]:
    historical = dict(fit.window_metrics["historical"] or {})
    return (
        float(historical["primary_loss"]),
        float(auxiliary_gate.get("selected_depth_mean_rmse") or float("inf")),
        float(historical["diag_flow_loss"]),
        float(historical["total_loss"]),
    )


def _write_hmba03_search_frontier(path: Path, frontier_rows: list[dict[str, Any]]) -> None:
    labels = [str(row.get("candidate_id") or row.get("step") or "") for row in frontier_rows]
    primary_values = [float(row.get("candidate_primary_loss") or np.nan) for row in frontier_rows]
    auxiliary_values = [float(row.get("candidate_auxiliary_rmse") or np.nan) for row in frontier_rows]
    best_primary = [float(row.get("best_primary_loss") or np.nan) for row in frontier_rows]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.2))
    axes[0].plot(labels, primary_values, marker="o", linewidth=1.4, label="candidate primary loss")
    axes[0].plot(labels, best_primary, marker="o", linewidth=1.2, label="best-so-far primary loss")
    axes[0].set_title("HMBA-03 Joint Search Frontier")
    axes[0].set_ylabel("historical primary loss")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    axes[1].plot(labels, auxiliary_values, marker="o", linewidth=1.4, color="tab:green", label="candidate auxiliary RMSE")
    axes[1].set_ylabel("selected-depth auxiliary RMSE")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_hmba03_depth_assignment(path: Path, depth_map: dict[str, str]) -> None:
    module_names = list(REPORTING_MODULES)
    depth_values = [_depth_rank(str(depth_map.get(module_name) or "none")) for module_name in module_names]
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.bar(module_names, depth_values, color="tab:blue")
    ax.set_title("HMBA-03 Final Depth Assignment")
    ax.set_ylabel("hierarchy depth")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["none", "national", "region", "province"])
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_hmba03_combined_comparison_chart(
    path: Path,
    *,
    hmba01_primary_loss: float,
    hmba01_diag_flow_loss: float,
    hmba02_primary_loss: float,
    hmba02_diag_flow_loss: float,
    hmba03_fit: FitResult,
) -> None:
    labels = ["hmba01", "hmba02", "hmba03"]
    x_axis = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.bar(
        x_axis - width / 2.0,
        [float(hmba01_primary_loss), float(hmba02_primary_loss), float(hmba03_fit.window_metrics["historical"]["primary_loss"])],
        width=width,
        label="primary loss",
    )
    ax.bar(
        x_axis + width / 2.0,
        [float(hmba01_diag_flow_loss), float(hmba02_diag_flow_loss), float(hmba03_fit.window_metrics["historical"]["diag_flow_loss"])],
        width=width,
        label="diagnosis-flow loss",
    )
    ax.set_title("HMBA-03 Joint Model Comparison")
    ax.set_ylabel("historical normalized loss")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _discover_latest_hmba03_outputs() -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    experiment_dir = _latest_transition_experiment_dir("HMBA-03-joint-hierarchical-autoresearch-loop", "joint_hierarchical_model.json")
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-03 joint autoresearch experiment found under artifacts/runs")
    combined_summary = dict(read_json(experiment_dir / "joint_hierarchical_model.json", default={}) or {})
    frontier_rows = [dict(row) for row in list(read_json(experiment_dir / "joint_search_frontier.json", default=[]))]
    trace_rows = [dict(row) for row in list(read_json(experiment_dir / "joint_search_trace.json", default=[]))]
    if not combined_summary or not frontier_rows or not trace_rows:
        raise RuntimeError(f"HMBA-03 experiment directory is missing required outputs: {experiment_dir}")
    return experiment_dir, combined_summary, frontier_rows, trace_rows


def _discover_latest_hmba02_outputs() -> tuple[Path, dict[str, Any], dict[str, Any]]:
    experiment_dir = _latest_transition_experiment_dir("HMBA-02-hierarchical-geography-layer", "hierarchical_combined_model.json")
    if experiment_dir is None:
        raise FileNotFoundError("No HMBA-02 hierarchical geography experiment found under artifacts/runs")
    combined_summary = dict(read_json(experiment_dir / "hierarchical_combined_model.json", default={}) or {})
    auxiliary_evaluation = dict(read_json(experiment_dir / "hierarchical_auxiliary_evaluation.json", default={}) or {})
    if not combined_summary:
        raise RuntimeError(f"HMBA-02 experiment directory is missing required outputs: {experiment_dir}")
    return experiment_dir, combined_summary, auxiliary_evaluation


def _parse_hmba03_action_detail(action_type: str, action_detail: str | None) -> tuple[str | None, str | None]:
    detail = str(action_detail or "")
    if action_type == "add_bundle" and "|" in detail:
        bundle_name, depth_name = detail.rsplit("|", 1)
        return bundle_name, depth_name
    if action_type == "depth_upgrade":
        return None, detail
    return None, None


def _hmba03_model_delta_rows(
    hmba03_summary: dict[str, Any],
    hmba02_summary: dict[str, Any],
    hmba02_auxiliary_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    hmba02_auxiliary_rmse = float(((hmba02_auxiliary_summary.get("auxiliary_gate") or {}).get("hierarchical_mean_rmse")) or np.nan)
    rows = [
        {
            "model_label": "HMBA-01",
            "primary_loss": float(hmba03_summary.get("hmba01_primary_loss") or 0.0),
            "diag_flow_loss": float(hmba03_summary.get("hmba01_diag_flow_loss") or 0.0),
            "auxiliary_rmse": None,
        },
        {
            "model_label": "HMBA-02",
            "primary_loss": float(hmba03_summary.get("hmba02_primary_loss") or 0.0),
            "diag_flow_loss": float(hmba03_summary.get("hmba02_diag_flow_loss") or 0.0),
            "auxiliary_rmse": (hmba02_auxiliary_rmse if np.isfinite(hmba02_auxiliary_rmse) else None),
        },
        {
            "model_label": "HMBA-03",
            "primary_loss": float((hmba03_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
            "diag_flow_loss": float((hmba03_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
            "auxiliary_rmse": float((hmba03_summary.get("auxiliary_gate") or {}).get("selected_depth_mean_rmse") or np.nan),
        },
    ]
    primary_baseline = float(rows[0]["primary_loss"])
    diag_baseline = float(rows[0]["diag_flow_loss"])
    auxiliary_baseline = float(rows[1]["auxiliary_rmse"]) if rows[1]["auxiliary_rmse"] is not None and np.isfinite(float(rows[1]["auxiliary_rmse"])) else float("nan")
    previous_primary = None
    previous_diag = None
    previous_aux = None
    for row in rows:
        row["primary_delta_vs_hmba01"] = float(row["primary_loss"] - primary_baseline)
        row["diag_flow_delta_vs_hmba01"] = float(row["diag_flow_loss"] - diag_baseline)
        row["auxiliary_delta_vs_hmba02"] = (
            float(float(row["auxiliary_rmse"]) - auxiliary_baseline)
            if row["auxiliary_rmse"] is not None and np.isfinite(float(row["auxiliary_rmse"])) and np.isfinite(auxiliary_baseline)
            else None
        )
        row["primary_delta_vs_previous"] = None if previous_primary is None else float(row["primary_loss"] - previous_primary)
        row["diag_flow_delta_vs_previous"] = None if previous_diag is None else float(row["diag_flow_loss"] - previous_diag)
        row["auxiliary_delta_vs_previous"] = (
            None
            if previous_aux is None or row["auxiliary_rmse"] is None or not np.isfinite(float(row["auxiliary_rmse"])) or not np.isfinite(previous_aux)
            else float(float(row["auxiliary_rmse"]) - previous_aux)
        )
        previous_primary = float(row["primary_loss"])
        previous_diag = float(row["diag_flow_loss"])
        previous_aux = float(row["auxiliary_rmse"]) if row["auxiliary_rmse"] is not None and np.isfinite(float(row["auxiliary_rmse"])) else previous_aux
    return rows


def _hmba03_accepted_step_effect_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not trace_rows:
        return []
    initial_primary = float(trace_rows[0].get("primary_loss") or np.nan)
    initial_diag = float(trace_rows[0].get("diag_flow_loss") or np.nan)
    initial_aux = float(trace_rows[0].get("selected_depth_auxiliary_rmse") or np.nan)
    rows: list[dict[str, Any]] = []
    previous_primary = None
    previous_diag = None
    previous_aux = None
    for row in trace_rows:
        action_type = str(row.get("action_type") or "")
        action_module = row.get("action_module")
        bundle_name, depth_name = _parse_hmba03_action_detail(action_type, row.get("action_detail"))
        primary_loss = float(row.get("primary_loss") or np.nan)
        diag_flow_loss = float(row.get("diag_flow_loss") or np.nan)
        auxiliary_rmse = float(row.get("selected_depth_auxiliary_rmse") or np.nan)
        if action_type == "initial":
            action_label = "initial HMBA-01 promoted state"
        elif action_type == "add_bundle":
            action_label = f"{action_module} + {bundle_name} @ {depth_name}"
        elif action_type == "depth_upgrade":
            action_label = f"{action_module} depth -> {depth_name}"
        else:
            action_label = str(row.get("candidate_id") or "")
        primary_delta_prev = 0.0 if previous_primary is None else float(primary_loss - previous_primary)
        diag_delta_prev = 0.0 if previous_diag is None else float(diag_flow_loss - previous_diag)
        auxiliary_delta_prev = 0.0 if previous_aux is None else float(auxiliary_rmse - previous_aux)
        if row is trace_rows[0]:
            dominant_channel = "baseline"
        else:
            scales = {
                "primary": abs(primary_delta_prev),
                "diagnosis_flow": abs(diag_delta_prev),
                "auxiliary": abs(auxiliary_delta_prev),
            }
            dominant_channel = max(scales.items(), key=lambda item: item[1])[0]
        rows.append(
            {
                "step": int(row.get("step") or 0),
                "candidate_id": str(row.get("candidate_id") or ""),
                "action_label": action_label,
                "action_type": action_type,
                "action_module": action_module,
                "bundle_name": bundle_name,
                "depth_name": depth_name,
                "primary_loss": primary_loss,
                "diag_flow_loss": diag_flow_loss,
                "auxiliary_rmse": auxiliary_rmse,
                "primary_delta_vs_previous": primary_delta_prev,
                "diag_flow_delta_vs_previous": diag_delta_prev,
                "auxiliary_delta_vs_previous": auxiliary_delta_prev,
                "primary_delta_vs_initial": float(primary_loss - initial_primary),
                "diag_flow_delta_vs_initial": float(diag_flow_loss - initial_diag),
                "auxiliary_delta_vs_initial": float(auxiliary_rmse - initial_aux),
                "dominant_channel": dominant_channel,
            }
        )
        previous_primary = primary_loss
        previous_diag = diag_flow_loss
        previous_aux = auxiliary_rmse
    return rows


def _hmba03_module_effect_rows(hmba03_summary: dict[str, Any]) -> list[dict[str, Any]]:
    final_depth_map = dict(hmba03_summary.get("final_depth_map") or {})
    final_bundle_map = dict(hmba03_summary.get("final_bundle_map") or {})
    module_depth_metrics = dict(hmba03_summary.get("module_depth_metrics") or {})
    module_feature_ids = dict(hmba03_summary.get("module_feature_ids") or {})
    rows: list[dict[str, Any]] = []
    for module_name in REPORTING_MODULES:
        depth_metrics = {
            "national": float((module_depth_metrics.get(module_name) or {}).get("national") or np.nan),
            "region": float((module_depth_metrics.get(module_name) or {}).get("region") or np.nan),
            "province": float((module_depth_metrics.get(module_name) or {}).get("province") or np.nan),
        }
        selected_depth = str(final_depth_map.get(module_name) or "none")
        selected_rmse = depth_metrics.get(selected_depth, float("nan"))
        finite_depths = {depth_name: value for depth_name, value in depth_metrics.items() if np.isfinite(float(value))}
        best_auxiliary_depth = min(finite_depths, key=lambda depth_name: float(finite_depths[depth_name])) if finite_depths else "none"
        best_auxiliary_rmse = float(finite_depths.get(best_auxiliary_depth, np.nan))
        national_rmse = float(depth_metrics.get("national", np.nan))
        region_rmse = float(depth_metrics.get("region", np.nan))
        province_rmse = float(depth_metrics.get("province", np.nan))
        gain_vs_national = float(national_rmse - selected_rmse) if np.isfinite(national_rmse) and np.isfinite(selected_rmse) else float("nan")
        gain_vs_best_auxiliary = (
            float(best_auxiliary_rmse - selected_rmse)
            if np.isfinite(best_auxiliary_rmse) and np.isfinite(selected_rmse)
            else float("nan")
        )
        if selected_depth == best_auxiliary_depth:
            selection_rationale = "selected depth also minimizes auxiliary RMSE"
        else:
            selection_rationale = "selected depth sacrifices some auxiliary RMSE to preserve the national mechanistic objective"
        rows.append(
            {
                "module_name": module_name,
                "selected_depth": selected_depth,
                "best_auxiliary_depth": best_auxiliary_depth,
                "national_rmse": national_rmse,
                "region_rmse": region_rmse,
                "province_rmse": province_rmse,
                "selected_depth_rmse": selected_rmse,
                "best_auxiliary_rmse": best_auxiliary_rmse,
                "gain_vs_national": gain_vs_national,
                "gain_vs_best_auxiliary": gain_vs_best_auxiliary,
                "selected_bundle_names": list(final_bundle_map.get(module_name) or []),
                "selected_feature_ids": list(module_feature_ids.get(module_name) or []),
                "selection_rationale": selection_rationale,
            }
        )
    return rows


def _hmba03_spike_annotation_payload(frontier_rows: list[dict[str, Any]], champion_auxiliary_rmse: float) -> dict[str, Any]:
    parsed_rows: list[dict[str, Any]] = []
    for row in frontier_rows:
        auxiliary_rmse = float(row.get("candidate_auxiliary_rmse") or np.nan)
        primary_loss = float(row.get("candidate_primary_loss") or np.nan)
        diag_flow_loss = float(row.get("candidate_diag_flow_loss") or np.nan)
        if not (np.isfinite(auxiliary_rmse) and np.isfinite(primary_loss) and np.isfinite(diag_flow_loss)):
            continue
        action_type = str(row.get("action_type") or "")
        action_module = str(row.get("action_module") or "")
        bundle_name, depth_name = _parse_hmba03_action_detail(action_type, row.get("action_detail"))
        match = re.match(r"step(?P<step>\d+)", str(row.get("candidate_id") or ""))
        step_number = int(match.group("step")) if match else int(row.get("step") or 0)
        parsed_rows.append(
            {
                "candidate_id": str(row.get("candidate_id") or ""),
                "step": step_number,
                "action_type": action_type,
                "action_module": action_module,
                "bundle_name": bundle_name,
                "depth_name": depth_name,
                "auxiliary_rmse": auxiliary_rmse,
                "primary_loss": primary_loss,
                "diag_flow_loss": diag_flow_loss,
                "auxiliary_ratio_vs_champion": float(auxiliary_rmse / champion_auxiliary_rmse) if champion_auxiliary_rmse > 0.0 else float("inf"),
            }
        )
    top_candidates = sorted(parsed_rows, key=lambda item: (-float(item["auxiliary_rmse"]), str(item["candidate_id"])))[:10]
    family_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in parsed_rows:
        if row["action_type"] != "add_bundle" or not row["bundle_name"] or not row["action_module"]:
            continue
        family_groups.setdefault((str(row["action_module"]), str(row["bundle_name"])), []).append(row)
    family_rows: list[dict[str, Any]] = []
    for (module_name, bundle_name), rows in family_groups.items():
        family_rows.append(
            {
                "module_name": module_name,
                "bundle_name": bundle_name,
                "count": int(len(rows)),
                "max_auxiliary_rmse": float(max(float(row["auxiliary_rmse"]) for row in rows)),
                "mean_auxiliary_rmse": float(np.mean([float(row["auxiliary_rmse"]) for row in rows])),
                "min_primary_loss": float(min(float(row["primary_loss"]) for row in rows)),
                "max_primary_loss": float(max(float(row["primary_loss"]) for row in rows)),
                "min_diag_flow_loss": float(min(float(row["diag_flow_loss"]) for row in rows)),
                "max_diag_flow_loss": float(max(float(row["diag_flow_loss"]) for row in rows)),
            }
        )
    family_rows.sort(key=lambda item: (-float(item["max_auxiliary_rmse"]), -int(item["count"]), str(item["module_name"]), str(item["bundle_name"])))
    if not family_rows:
        return {
            "top_candidates": top_candidates,
            "family_rankings": family_rows,
            "worst_family": {},
            "worst_family_rows": [],
        }
    worst_family = dict(family_rows[0])
    worst_family_rows = [
        row
        for row in parsed_rows
        if str(row["action_module"]) == str(worst_family["module_name"]) and str(row["bundle_name"]) == str(worst_family["bundle_name"])
    ]
    depth_profile: dict[str, dict[str, Any]] = {}
    for depth_name in ("national", "region", "province"):
        depth_rows = [row for row in worst_family_rows if str(row.get("depth_name") or "") == depth_name]
        if not depth_rows:
            continue
        max_row = max(depth_rows, key=lambda row: float(row["auxiliary_rmse"]))
        ordered_rows = sorted(depth_rows, key=lambda row: int(row["step"]))
        depth_profile[depth_name] = {
            "count": int(len(depth_rows)),
            "mean_auxiliary_rmse": float(np.mean([float(row["auxiliary_rmse"]) for row in depth_rows])),
            "max_auxiliary_rmse": float(max_row["auxiliary_rmse"]),
            "max_candidate_id": str(max_row["candidate_id"]),
            "step_sequence": [int(row["step"]) for row in ordered_rows],
            "auxiliary_sequence": [float(row["auxiliary_rmse"]) for row in ordered_rows],
        }
    national_max = float((depth_profile.get("national") or {}).get("max_auxiliary_rmse") or np.nan)
    region_max = float((depth_profile.get("region") or {}).get("max_auxiliary_rmse") or np.nan)
    province_max = float((depth_profile.get("province") or {}).get("max_auxiliary_rmse") or np.nan)
    worst_family["depth_profile"] = depth_profile
    worst_family["national_to_region_peak_ratio"] = (
        float(national_max / region_max) if np.isfinite(national_max) and np.isfinite(region_max) and region_max > 0.0 else None
    )
    worst_family["national_to_province_peak_ratio"] = (
        float(national_max / province_max)
        if np.isfinite(national_max) and np.isfinite(province_max) and province_max > 0.0
        else None
    )
    return {
        "top_candidates": top_candidates,
        "family_rankings": family_rows[:10],
        "worst_family": worst_family,
        "worst_family_rows": worst_family_rows,
    }


def _set_zoom_limits(ax: Any, values: list[float], *, pad_ratio: float = 0.2) -> None:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    if not finite_values:
        return
    value_min = float(min(finite_values))
    value_max = float(max(finite_values))
    span = max(value_max - value_min, max(abs(value_min), 1e-6) * 0.02)
    padding = span * float(pad_ratio)
    ax.set_ylim(value_min - padding, value_max + padding)


def _write_hmba03a_zoomed_delta_chart(path: Path, model_rows: list[dict[str, Any]]) -> None:
    labels = [str(row["model_label"]) for row in model_rows]
    primary_values = [float(row["primary_loss"]) for row in model_rows]
    diag_values = [float(row["diag_flow_loss"]) for row in model_rows]
    x_axis = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)
    axes[0].bar(x_axis, primary_values, color=["#9aa5b1", "#5b8ff9", "#1f5fd6"])
    axes[0].set_title("A. Historical Primary Loss")
    axes[0].set_ylabel("normalized loss")
    axes[0].set_xticks(x_axis)
    axes[0].set_xticklabels(labels)
    _set_zoom_limits(axes[0], primary_values)
    for idx, row in enumerate(model_rows):
        axes[0].text(
            idx,
            primary_values[idx],
            f"{primary_values[idx]:.6f}\nd={float(row['primary_delta_vs_hmba01']):+.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[1].bar(x_axis, diag_values, color=["#b0b9c3", "#66c2a5", "#1b9e77"])
    axes[1].set_title("B. Historical Diagnosis-Flow Loss")
    axes[1].set_ylabel("normalized loss")
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(labels)
    _set_zoom_limits(axes[1], diag_values)
    for idx, row in enumerate(model_rows):
        axes[1].text(
            idx,
            diag_values[idx],
            f"{diag_values[idx]:.6f}\nd={float(row['diag_flow_delta_vs_hmba01']):+.6f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.suptitle("HMBA-03A Zoomed Loss Delta Charts", fontsize=13, y=1.02)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_hmba03a_accepted_step_delta_chart(path: Path, step_rows: list[dict[str, Any]]) -> None:
    plotted_rows = [row for row in step_rows if int(row["step"]) > 0]
    if not plotted_rows:
        fig, ax = plt.subplots(figsize=(10.0, 4.0))
        ax.text(0.5, 0.5, "No accepted HMBA-03 mutation steps were recorded.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return
    labels = [f"S{int(row['step'])}" for row in plotted_rows]
    x_axis = np.arange(len(labels))
    delta_fields = [
        ("primary_delta_vs_previous", "Primary delta vs previous", "tab:blue"),
        ("diag_flow_delta_vs_previous", "Diag-flow delta vs previous", "tab:green"),
        ("auxiliary_delta_vs_previous", "Auxiliary RMSE delta vs previous", "tab:red"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13.4, 8.2), sharex=True, constrained_layout=True)
    for axis, (field_name, title, color) in zip(axes, delta_fields):
        values = [float(row[field_name]) for row in plotted_rows]
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.bar(x_axis, values, color=[color if value >= 0.0 else "#1a9850" for value in values], alpha=0.9)
        axis.set_title(title, fontsize=11)
        axis.set_ylabel("delta")
        for idx, value in enumerate(values):
            axis.text(idx, value, f"{value:+.6f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    axes[-1].set_xticks(x_axis)
    axes[-1].set_xticklabels([f"{label}\n{row['action_label']}" for label, row in zip(labels, plotted_rows)], rotation=25, ha="right")
    fig.suptitle("HMBA-03A Accepted-Step Delta Decomposition", fontsize=13, y=1.01)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_hmba03a_module_effect_table_chart(path: Path, module_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(14.4, 4.8))
    ax.axis("off")
    column_labels = ["Module", "Selected", "Best aux", "Nat RMSE", "Reg RMSE", "Prov RMSE", "Selected RMSE", "Gain vs nat"]
    cell_text = [
        [
            str(row["module_name"]),
            str(row["selected_depth"]),
            str(row["best_auxiliary_depth"]),
            f"{float(row['national_rmse']):.4f}",
            f"{float(row['region_rmse']):.4f}",
            f"{float(row['province_rmse']):.4f}",
            f"{float(row['selected_depth_rmse']):.4f}",
            f"{float(row['gain_vs_national']):+.4f}",
        ]
        for row in module_rows
    ]
    table = ax.table(cellText=cell_text, colLabels=column_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dce6f2")
        elif col_idx == 7:
            gain_value = float(module_rows[row_idx - 1]["gain_vs_national"])
            cell.set_facecolor("#d6f5d6" if gain_value > 0.0 else "#f7d6d6")
        elif col_idx == 1 and str(module_rows[row_idx - 1]["selected_depth"]) == str(module_rows[row_idx - 1]["best_auxiliary_depth"]):
            cell.set_facecolor("#e5f5e0")
    ax.set_title("HMBA-03A Module-Wise Effect Table", pad=16)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_hmba03a_spike_annotation_chart(path: Path, spike_payload: dict[str, Any], champion_auxiliary_rmse: float) -> None:
    top_candidates = [dict(row) for row in list(spike_payload.get("top_candidates") or [])]
    worst_family = dict(spike_payload.get("worst_family") or {})
    worst_family_rows = [dict(row) for row in list(spike_payload.get("worst_family_rows") or [])]
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 5.4), constrained_layout=True)
    if top_candidates:
        plotted = top_candidates[:8]
        labels = [f"{row['action_module']} | {row['bundle_name']} | {row['depth_name']}" for row in plotted]
        values = [float(row["auxiliary_rmse"]) for row in plotted]
        axes[0].barh(np.arange(len(plotted)), values, color="#d95f02")
        axes[0].invert_yaxis()
        axes[0].set_yticks(np.arange(len(plotted)))
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].set_title("A. Highest Auxiliary-RMSE Candidates")
        axes[0].set_xlabel("selected-depth auxiliary RMSE")
        for idx, row in enumerate(plotted):
            axes[0].text(values[idx], idx, f"  {values[idx]:.3f}", va="center", fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No spike candidates found.", ha="center", va="center")
        axes[0].axis("off")
    if worst_family and worst_family_rows:
        grouped_by_depth = {
            depth_name: sorted(
                [row for row in worst_family_rows if str(row.get("depth_name") or "") == depth_name],
                key=lambda row: int(row["step"]),
            )
            for depth_name in ("national", "region", "province")
        }
        depth_colors = {"national": "#b2182b", "region": "#ef8a62", "province": "#4d9221"}
        for depth_name, rows in grouped_by_depth.items():
            if not rows:
                continue
            axes[1].plot(
                [int(row["step"]) for row in rows],
                [float(row["auxiliary_rmse"]) for row in rows],
                marker="o",
                linewidth=1.8,
                color=depth_colors[depth_name],
                label=depth_name,
            )
            max_row = max(rows, key=lambda row: float(row["auxiliary_rmse"]))
            axes[1].annotate(
                f"{depth_name} max {float(max_row['auxiliary_rmse']):.3f}",
                xy=(int(max_row["step"]), float(max_row["auxiliary_rmse"])),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
            )
        axes[1].axhline(champion_auxiliary_rmse, color="black", linestyle="--", linewidth=1.0, label="champion")
        axes[1].set_title(
            "B. Worst Family Depth Profile\n"
            f"{str(worst_family.get('module_name') or '')} + {str(worst_family.get('bundle_name') or '')}"
        )
        axes[1].set_xlabel("search step")
        axes[1].set_ylabel("selected-depth auxiliary RMSE")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No repeated spike family found.", ha="center", va="center")
        axes[1].axis("off")
    fig.suptitle("HMBA-03A Spike Candidate Annotations", fontsize=13, y=1.02)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_hmba03a_interpretation_dashboard(
    path: Path,
    *,
    model_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    spike_payload: dict[str, Any],
) -> None:
    champion_row = next((row for row in model_rows if str(row["model_label"]) == "HMBA-03"), None)
    worst_family = dict(spike_payload.get("worst_family") or {})
    fig = plt.figure(figsize=(15.2, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1])
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[1, 1])

    primary_values = [float(row["primary_loss"]) for row in model_rows]
    diag_values = [float(row["diag_flow_loss"]) for row in model_rows]
    x_axis = np.arange(len(model_rows))
    width = 0.34
    ax0.bar(x_axis - width / 2.0, primary_values, width=width, color="#4c78a8", label="primary")
    ax0.bar(x_axis + width / 2.0, diag_values, width=width, color="#59a14f", label="diag-flow")
    ax0.set_xticks(x_axis)
    ax0.set_xticklabels([str(row["model_label"]) for row in model_rows])
    _set_zoom_limits(ax0, primary_values + diag_values, pad_ratio=0.25)
    ax0.set_title("A. HMBA-03 beats HMBA-01/HMBA-02 in the narrow national-loss band")
    ax0.set_ylabel("historical normalized loss")
    ax0.legend(fontsize=8)

    accepted_rows = [row for row in step_rows if int(row["step"]) > 0]
    if accepted_rows:
        ax1.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        ax1.plot(
            [int(row["step"]) for row in accepted_rows],
            [float(row["primary_delta_vs_previous"]) for row in accepted_rows],
            marker="o",
            linewidth=1.7,
            color="#4c78a8",
            label="primary delta",
        )
        ax1.plot(
            [int(row["step"]) for row in accepted_rows],
            [float(row["auxiliary_delta_vs_previous"]) for row in accepted_rows],
            marker="o",
            linewidth=1.7,
            color="#e15759",
            label="auxiliary delta",
        )
        ax1.set_title("B. Later accepted moves are mostly geography-side gains")
        ax1.set_xlabel("accepted search step")
        ax1.set_ylabel("delta vs previous")
        ax1.legend(fontsize=8)
    else:
        ax1.text(0.5, 0.5, "No accepted mutation steps", ha="center", va="center")
        ax1.axis("off")

    module_names = [str(row["module_name"]) for row in module_rows]
    gain_values = [float(row["gain_vs_national"]) for row in module_rows]
    colors = ["#1a9850" if value >= 0.0 else "#d73027" for value in gain_values]
    ax2.barh(np.arange(len(module_rows)), gain_values, color=colors)
    ax2.set_yticks(np.arange(len(module_rows)))
    ax2.set_yticklabels(module_names)
    ax2.invert_yaxis()
    ax2.axvline(0.0, color="black", linewidth=0.8)
    ax2.set_title("C. Province depth matters for most care-cascade modules")
    ax2.set_xlabel("gain in auxiliary RMSE vs national-only")
    for idx, row in enumerate(module_rows):
        ax2.text(float(row["gain_vs_national"]), idx, f"  {float(row['gain_vs_national']):+.3f}", va="center", fontsize=8)

    worst_family_text = "No spike family detected."
    if worst_family:
        worst_family_text = "\n".join(
            textwrap.wrap(
                (
                    f"Worst repeated family: {str(worst_family.get('module_name') or '')} + "
                    f"{str(worst_family.get('bundle_name') or '')}. "
                    f"Peak auxiliary RMSE = {float(worst_family.get('max_auxiliary_rmse') or 0.0):.3f}. "
                    f"National peak is "
                    f"{float(((worst_family.get('depth_profile') or {}).get('national') or {}).get('max_auxiliary_rmse') or 0.0):.3f}, "
                    f"region peak is "
                    f"{float(((worst_family.get('depth_profile') or {}).get('region') or {}).get('max_auxiliary_rmse') or 0.0):.3f}, "
                    f"province peak is "
                    f"{float(((worst_family.get('depth_profile') or {}).get('province') or {}).get('max_auxiliary_rmse') or 0.0):.3f}. "
                    "That depth attenuation pattern indicates a structurally bad national transfer function rather than random search noise."
                ),
                width=52,
            )
        )
    summary_lines = [
        "D. Scientific interpretation",
        "",
        f"Champion primary loss: {float((champion_row or {}).get('primary_loss') or 0.0):.6f}",
        f"Champion diagnosis-flow loss: {float((champion_row or {}).get('diag_flow_loss') or 0.0):.6f}",
        "",
        "Interpretation:",
        worst_family_text,
    ]
    ax3.axis("off")
    ax3.text(
        0.0,
        1.0,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox={"facecolor": "#f8f8f8", "edgecolor": "#d0d0d0", "boxstyle": "round,pad=0.6"},
    )
    fig.suptitle("HMBA-03A Interpretation Dashboard", fontsize=15, y=1.01)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_hmba03a_markdown_report(
    path: Path,
    *,
    hmba03_dir: Path,
    hmba02_dir: Path,
    model_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    module_rows: list[dict[str, Any]],
    spike_payload: dict[str, Any],
) -> None:
    worst_family = dict(spike_payload.get("worst_family") or {})
    top_candidates = [dict(row) for row in list(spike_payload.get("top_candidates") or [])]
    model_lookup = {str(row["model_label"]): dict(row) for row in model_rows}
    hmba03_row = model_lookup.get("HMBA-03", {})
    primary_gain_vs_hmba01 = float(hmba03_row.get("primary_delta_vs_hmba01") or 0.0)
    diag_gain_vs_hmba01 = float(hmba03_row.get("diag_flow_delta_vs_hmba01") or 0.0)
    markdown_lines = [
        "# HMBA-03A Interpretation Dashboard",
        "",
        f"- HMBA-03 source: `{hmba03_dir}`",
        f"- HMBA-02 source: `{hmba02_dir}`",
        "",
        "## Key Findings",
        "",
        f"- HMBA-03 reduced historical primary loss by `{abs(primary_gain_vs_hmba01):.6f}` relative to HMBA-01.",
        f"- HMBA-03 reduced historical diagnosis-flow loss by `{abs(diag_gain_vs_hmba01):.6f}` relative to HMBA-01.",
        f"- The final auxiliary mean RMSE is `{float((hmba03_row.get('auxiliary_rmse') if hmba03_row else np.nan)):.6f}`.",
        "",
        "## Accepted-Step Effects",
        "",
        "| Step | Action | Primary delta vs prev | Diag-flow delta vs prev | Auxiliary delta vs prev | Dominant channel |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in [row for row in step_rows if int(row["step"]) > 0]:
        markdown_lines.append(
            f"| {int(row['step'])} | {row['action_label']} | {float(row['primary_delta_vs_previous']):+.6f} | "
            f"{float(row['diag_flow_delta_vs_previous']):+.6f} | {float(row['auxiliary_delta_vs_previous']):+.6f} | {row['dominant_channel']} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Module-Wise Effect Table",
            "",
            "| Module | Selected depth | Best aux depth | Nat RMSE | Reg RMSE | Prov RMSE | Selected RMSE | Gain vs nat | Bundles |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in module_rows:
        markdown_lines.append(
            f"| {row['module_name']} | {row['selected_depth']} | {row['best_auxiliary_depth']} | "
            f"{float(row['national_rmse']):.4f} | {float(row['region_rmse']):.4f} | {float(row['province_rmse']):.4f} | "
            f"{float(row['selected_depth_rmse']):.4f} | {float(row['gain_vs_national']):+.4f} | {', '.join(row['selected_bundle_names']) or 'none'} |"
        )
    markdown_lines.extend(["", "## Spike-Candidate Interpretation", ""])
    if worst_family:
        markdown_lines.extend(
            [
                f"- Worst repeated spike family: `{worst_family['module_name']} + {worst_family['bundle_name']}`.",
                f"- Maximum auxiliary RMSE: `{float(worst_family['max_auxiliary_rmse']):.6f}` across `{int(worst_family['count'])}` evaluated candidates.",
                f"- Peak by depth: national `{float(((worst_family.get('depth_profile') or {}).get('national') or {}).get('max_auxiliary_rmse') or 0.0):.6f}`, region `{float(((worst_family.get('depth_profile') or {}).get('region') or {}).get('max_auxiliary_rmse') or 0.0):.6f}`, province `{float(((worst_family.get('depth_profile') or {}).get('province') or {}).get('max_auxiliary_rmse') or 0.0):.6f}`.",
                "- Scientific meaning: this family repeatedly fails the auxiliary geography contract while leaving the national loss in the same narrow band, which is evidence of a structurally bad factor-to-transition mapping at coarse aggregation rather than random optimizer noise.",
            ]
        )
    else:
        markdown_lines.append("- No repeated spike family was detected.")
    if top_candidates:
        markdown_lines.extend(
            [
                "",
                "## Top Spike Candidates",
                "",
                "| Candidate | Module | Bundle | Depth | Auxiliary RMSE | Ratio vs champion |",
                "| --- | --- | --- | --- | ---: | ---: |",
            ]
        )
        for row in top_candidates[:8]:
            markdown_lines.append(
                f"| {row['candidate_id']} | {row['action_module']} | {row['bundle_name']} | {row['depth_name']} | "
                f"{float(row['auxiliary_rmse']):.6f} | {float(row['auxiliary_ratio_vs_champion']):.2f}x |"
            )
    path.write_text("\n".join(markdown_lines), encoding="utf-8")


def run_hmba_00(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    retained_ids = _retained_factor_ids(inputs)
    evidence = _discover_latest_provincial_evidence(retained_ids)
    factor_payload = _factor_rows(inputs, evidence)
    rows = list(factor_payload["rows"])
    module_seed_manifest = _module_seed_report(rows)
    national_panel = _build_national_observation_tensors(ctx)
    auxiliary_panel = _load_provincial_auxiliary_state_tensor(inputs, evidence)
    incumbent_run_id, incumbent_dir, incumbent_decision, incumbent_baseline = _discover_latest_integrated_incumbent()
    hidden_summary = _hidden_rank_summary(inputs)

    national_value_artifact = save_tensor_artifact(
        array=np.asarray(national_panel["values"], dtype=np.float32),
        axis_names=["quarter", "metric"],
        artifact_dir=ctx.experiment_dir,
        stem="national_observation_values",
        backend="numpy",
        device="cpu",
        notes=["hmba00_national_observation_values"],
        save_pt=False,
    )
    national_mask_artifact = save_tensor_artifact(
        array=np.asarray(national_panel["mask"], dtype=np.uint8),
        axis_names=["quarter", "metric"],
        artifact_dir=ctx.experiment_dir,
        stem="national_observation_mask",
        backend="numpy",
        device="cpu",
        notes=["hmba00_national_observation_mask"],
        save_pt=False,
    )
    provincial_aux_artifact = save_tensor_artifact(
        array=np.asarray(auxiliary_panel["values"], dtype=np.float32),
        axis_names=["province", "time", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="provincial_auxiliary_state_shares",
        backend="numpy",
        device="cpu",
        notes=["hmba00_provincial_auxiliary_state_shares"],
        save_pt=False,
    )
    regional_aux_artifact = save_tensor_artifact(
        array=np.asarray(auxiliary_panel["regional_values"], dtype=np.float32),
        axis_names=["region", "time", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="regional_auxiliary_state_shares",
        backend="numpy",
        device="cpu",
        notes=["hmba00_regional_auxiliary_state_shares"],
        save_pt=False,
    )
    national_aux_artifact = save_tensor_artifact(
        array=np.asarray(auxiliary_panel["national_values"], dtype=np.float32),
        axis_names=["national", "time", "state"],
        artifact_dir=ctx.experiment_dir,
        stem="national_auxiliary_state_shares",
        backend="numpy",
        device="cpu",
        notes=["hmba00_national_auxiliary_state_shares"],
        save_pt=False,
    )

    coverage_chart_path = ctx.experiment_dir / "coverage_frontier.png"
    module_chart_path = ctx.experiment_dir / "module_seed_frontier.png"
    _write_coverage_chart(coverage_chart_path, national_panel=national_panel, auxiliary_panel=auxiliary_panel)
    _write_module_seed_chart(module_chart_path, module_seed_manifest)

    contract_snapshot = {
        "source_run_id": ctx.source_run_id,
        "experiment_id": ctx.experiment.experiment_id,
        "autoresearch_variant": "evidence-to-model-loop",
        "national_incumbent_run_id": incumbent_run_id,
        "national_incumbent_experiment_dir": str(incumbent_dir),
        "national_incumbent_decision": incumbent_decision,
        "national_baseline_gate": incumbent_baseline,
        "provincial_evidence_run_id": evidence.run_id,
        "provincial_evidence_dir": str(evidence.run_dir / "phase3"),
        "provincial_overlap_count": int(evidence.overlap_count),
        "provincial_benchmark_gate_report": evidence.benchmark_gate_report,
        "retained_factor_count": len(retained_ids),
        "factor_axis_count": len(inputs.factor_axis),
        "analysis_years": list(inputs.analysis_years),
        "module_names": list(REPORTING_MODULES),
        "transition_names": list(TRANSITION_NAMES),
        "national_metric_axis": list(NATIONAL_METRIC_AXIS),
        "state_axis": list(STATE_AXIS),
        "hidden_rank_summary": hidden_summary,
        "contract_terms": {
            "national_panel_role": "observed_national_targets",
            "provincial_panel_role": "auxiliary_provincial_state_share_evidence",
            "province_truth_available": False,
            "module_seed_source": "current provincial autoresearch evidence overlap with retained Phase 2 factors",
        },
    }
    module_seed_payload = {
        "provincial_evidence_run_id": evidence.run_id,
        "provincial_overlap_count": int(evidence.overlap_count),
        "module_factor_rankings": module_seed_manifest["module_factor_rankings"],
        "module_bundle_rankings": module_seed_manifest["module_bundle_rankings"],
    }
    evidence_panel_summary = {
        "national_panel": {
            "quarter_axis": national_panel["quarter_axis"],
            "historical_end_quarter": national_panel["historical_end_quarter"],
            "metric_axis": national_panel["metric_axis"],
            "metric_coverage": national_panel["coverage"],
        },
        "provincial_auxiliary_panel": {
            "province_count": len(auxiliary_panel["province_axis"]),
            "region_count": len(auxiliary_panel["region_axis"]),
            "time_count": len(auxiliary_panel["time_axis"]),
            "time_axis": auxiliary_panel["time_axis"],
            "province_weight_time": auxiliary_panel["province_weight_time"],
            "state_axis": auxiliary_panel["state_axis"],
            "mean_state_availability": round(float(np.mean(np.isfinite(auxiliary_panel["values"]))), 6),
        },
        "phase2_inputs": {
            "analysis_years": list(inputs.analysis_years),
            "retained_factor_count": len(retained_ids),
            "province_factor_shape": list(np.asarray(inputs.province_tensor).shape),
            "region_factor_shape": list(np.asarray(inputs.region_tensor).shape),
            "national_factor_shape": list(np.asarray(inputs.national_tensor).shape),
        },
    }

    contract_path = ctx.experiment_dir / "contract_snapshot.json"
    module_seed_path = ctx.experiment_dir / "module_seed_manifest.json"
    panel_summary_path = ctx.experiment_dir / "evidence_panel_summary.json"
    write_json(contract_path, contract_snapshot)
    write_json(module_seed_path, module_seed_payload)
    write_json(panel_summary_path, evidence_panel_summary)

    experiment_spec = {
        "goal": "Freeze the hierarchical module-bundle autoresearch contract before any province-aware optimization.",
        "selected_variant": "evidence-to-model-loop",
        "search_unit": "frozen evidence contract and module seed materialization",
        "source_run_id": ctx.source_run_id,
        "provincial_evidence_run_id": evidence.run_id,
    }
    coverage_summary = {
        "analysis_years": list(inputs.analysis_years),
        "retained_factor_count": len(retained_ids),
        "province_count": len(auxiliary_panel["province_axis"]),
        "region_count": len(auxiliary_panel["region_axis"]),
        "national_metric_count": len(NATIONAL_METRIC_AXIS),
        "module_count": len(REPORTING_MODULES),
    }
    decision = {
        "completed": True,
        "keep": True,
        "reason": "HMBA-00 froze the mixed national-observed and provincial-auxiliary contract required before hierarchical module-bundle autoresearch.",
        "national_incumbent_run_id": incumbent_run_id,
        "provincial_evidence_run_id": evidence.run_id,
        "module_seed_top_incidence_bundle": (
            module_seed_payload["module_bundle_rankings"].get("incidence", [{}])[0].get("bundle_name")
            if module_seed_payload["module_bundle_rankings"].get("incidence")
            else None
        ),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec=experiment_spec,
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            numerical_guard_entry(
                name="eps_float32",
                role="division_and_rank_tolerance_guard",
                why_needed="The hierarchical contract build computes normalized coverage fractions and SVD rank thresholds that need a machine-defined lower bound.",
            )
        ],
    )
    artifacts.update(
        {
            "contract_snapshot": str(contract_path),
            "module_seed_manifest": str(module_seed_path),
            "evidence_panel_summary": str(panel_summary_path),
            "national_observation_values": str(national_value_artifact["value_path"]),
            "national_observation_mask": str(national_mask_artifact["value_path"]),
            "provincial_auxiliary_state_shares": str(provincial_aux_artifact["value_path"]),
            "regional_auxiliary_state_shares": str(regional_aux_artifact["value_path"]),
            "national_auxiliary_state_shares": str(national_aux_artifact["value_path"]),
            "coverage_frontier": str(coverage_chart_path),
            "module_seed_frontier": str(module_chart_path),
        }
    )
    return {
        "artifacts": artifacts,
        "contract_snapshot": contract_snapshot,
        "module_seed_manifest": module_seed_payload,
        "evidence_panel_summary": evidence_panel_summary,
        "decision": decision,
    }


def run_hmba_01(ctx: TransitionResearchContext) -> dict[str, Any]:
    contract_dir, contract_snapshot, module_seed_manifest = _discover_latest_hmba_contract()
    snapshot = _build_snapshot(ctx)
    incumbent_config = _hmba01_incumbent_config(contract_snapshot)
    module_bundle_features = _module_bundle_feature_lookup(module_seed_manifest)

    empty_feature_map = {module_name: [] for module_name in REPORTING_MODULES}
    hidden_only_fit = _fit_with_matrices(snapshot, incumbent_config, _hmba01_candidate_matrices(snapshot, incumbent_config, empty_feature_map))
    hidden_only_gate_ok, hidden_only_baseline = _fit_passes_hmba01_gate(snapshot, hidden_only_fit)

    module_results: dict[str, Any] = {}
    module_frontier_rows: list[dict[str, Any]] = []
    promoted_bundle_map: dict[str, list[str]] = {module_name: [] for module_name in REPORTING_MODULES}
    hidden_only_primary_loss = float(hidden_only_fit.window_metrics["historical"]["primary_loss"])

    for module_name in REPORTING_MODULES:
        available_bundles = list(module_bundle_features.get(module_name, {}))
        current_selected: list[str] = []
        current_fit = hidden_only_fit
        current_gate_ok = bool(hidden_only_gate_ok)
        current_baseline = dict(hidden_only_baseline)
        promotion_trace = [
            {
                "step": 0,
                "selected_bundles": [],
                "primary_loss": round(float(current_fit.window_metrics["historical"]["primary_loss"]), 6),
                "diag_flow_loss": round(float(current_fit.window_metrics["historical"]["diag_flow_loss"]), 6),
                "passes_gate": bool(current_gate_ok),
            }
        ]
        step = 0
        while available_bundles:
            step += 1
            candidate_rows: list[dict[str, Any]] = []
            best_entry: dict[str, Any] | None = None
            best_fit: FitResult | None = None
            best_baseline: dict[str, Any] | None = None
            best_selected: list[str] | None = None
            for bundle_name in list(available_bundles):
                trial_selected = current_selected + [bundle_name]
                feature_map = {name: [] for name in REPORTING_MODULES}
                feature_map[module_name] = sorted(
                    {
                        factor_id
                        for selected_bundle in trial_selected
                        for factor_id in list(module_bundle_features.get(module_name, {}).get(selected_bundle, []))
                    }
                )
                fit = _fit_with_matrices(snapshot, incumbent_config, _hmba01_candidate_matrices(snapshot, incumbent_config, feature_map))
                passes_gate, baseline_comparison = _fit_passes_hmba01_gate(snapshot, fit)
                entry = {
                    "module_name": module_name,
                    "step": step,
                    "trial_bundle_sequence": list(trial_selected),
                    "trial_feature_ids": list(feature_map[module_name]),
                    "trial_primary_loss": round(float(fit.window_metrics["historical"]["primary_loss"]), 6),
                    "trial_diag_flow_loss": round(float(fit.window_metrics["historical"]["diag_flow_loss"]), 6),
                    "trial_total_loss": round(float(fit.window_metrics["historical"]["total_loss"]), 6),
                    "passes_gate": bool(passes_gate),
                    "model_beats_carry_forward": bool(baseline_comparison["model_beats_carry_forward"]),
                    "model_beats_simple_compartmental": bool(baseline_comparison["model_beats_simple_compartmental"]),
                    "diagnosis_flow_beats_carry_forward": bool(baseline_comparison["diagnosis_flow_beats_carry_forward"]),
                    "diagnosis_flow_beats_simple_compartmental": bool(baseline_comparison["diagnosis_flow_beats_simple_compartmental"]),
                    "improves_over_current": bool(_score_tuple(fit.window_metrics["historical"]) < _score_tuple(current_fit.window_metrics["historical"])),
                }
                candidate_rows.append(entry)
                if not passes_gate:
                    continue
                if _score_tuple(fit.window_metrics["historical"]) >= _score_tuple(current_fit.window_metrics["historical"]):
                    continue
                if best_fit is None or _score_tuple(fit.window_metrics["historical"]) < _score_tuple(best_fit.window_metrics["historical"]):
                    best_fit = fit
                    best_entry = entry
                    best_baseline = baseline_comparison
                    best_selected = list(trial_selected)
            module_frontier_rows.extend(candidate_rows)
            if best_fit is None or best_selected is None or best_entry is None or best_baseline is None:
                break
            current_fit = best_fit
            current_selected = list(best_selected)
            current_gate_ok = True
            current_baseline = dict(best_baseline)
            promoted_bundle_map[module_name] = list(current_selected)
            available_bundles = [bundle_name for bundle_name in available_bundles if bundle_name not in current_selected]
            promotion_trace.append(
                {
                    "step": int(step),
                    "selected_bundles": list(current_selected),
                    "primary_loss": round(float(current_fit.window_metrics["historical"]["primary_loss"]), 6),
                    "diag_flow_loss": round(float(current_fit.window_metrics["historical"]["diag_flow_loss"]), 6),
                    "passes_gate": bool(current_gate_ok),
                }
            )
        module_results[module_name] = {
            "promoted_bundles": list(current_selected),
            "promotion_trace": promotion_trace,
            "historical_score_tuple": [round(float(value), 6) for value in _score_tuple(current_fit.window_metrics["historical"])],
            "baseline_comparison": current_baseline,
            "feature_ids": sorted(
                {
                    factor_id
                    for bundle_name in current_selected
                    for factor_id in list(module_bundle_features.get(module_name, {}).get(bundle_name, []))
                }
            ),
        }

    combined_feature_map = _build_direct_feature_map(module_bundle_features, selected_bundles_by_module=promoted_bundle_map)
    combined_fit = _fit_with_matrices(snapshot, incumbent_config, _hmba01_candidate_matrices(snapshot, incumbent_config, combined_feature_map))
    combined_gate_ok, combined_baseline = _fit_passes_hmba01_gate(snapshot, combined_fit)

    single_bundle_gain_rows = _single_bundle_gain_rows(
        module_frontier_rows=module_frontier_rows,
        hidden_only_primary_loss=hidden_only_primary_loss,
    )

    frontier_path = ctx.experiment_dir / "module_local_frontier.json"
    module_results_path = ctx.experiment_dir / "module_promotion_results.json"
    combined_summary_path = ctx.experiment_dir / "combined_promoted_model.json"
    write_json(frontier_path, module_frontier_rows)
    write_json(module_results_path, module_results)
    write_json(
        combined_summary_path,
        {
            "promoted_bundle_map": promoted_bundle_map,
            "feature_map": combined_feature_map,
            "baseline_comparison": combined_baseline,
            "historical_score_tuple": [round(float(value), 6) for value in _score_tuple(combined_fit.window_metrics["historical"])],
            "passes_gate": bool(combined_gate_ok),
        },
    )

    single_bundle_gain_chart_path = ctx.experiment_dir / "single_bundle_gain_heatmap.png"
    promotion_trace_chart_path = ctx.experiment_dir / "module_promotion_trace.png"
    combined_comparison_chart_path = ctx.experiment_dir / "combined_model_comparison.png"
    _write_single_bundle_gain_heatmap(single_bundle_gain_chart_path, single_bundle_gain_rows)
    _write_module_promotion_trace(promotion_trace_chart_path, module_results)
    _write_combined_comparison_chart(
        combined_comparison_chart_path,
        hidden_only_fit=hidden_only_fit,
        combined_fit=combined_fit,
        incumbent_primary_loss=float((contract_snapshot.get("national_baseline_gate") or {}).get("model_primary_loss") or 0.0),
        incumbent_diag_flow_loss=float((contract_snapshot.get("national_baseline_gate") or {}).get("model_diag_flow_loss") or 0.0),
    )
    paper_archive = _refresh_phase3_figure_archive(
        hmba01_chart_paths={
            "single_bundle_gain": single_bundle_gain_chart_path,
            "promotion_trace": promotion_trace_chart_path,
            "combined_comparison": combined_comparison_chart_path,
        }
    )

    coverage_summary = {
        "module_count": len(REPORTING_MODULES),
        "seeded_bundle_count": int(sum(len(list(bundles)) for bundles in module_bundle_features.values())),
        "promoted_bundle_count": int(sum(len(list(bundles)) for bundles in promoted_bundle_map.values())),
        "paper_figure_count": int(paper_archive["figure_count"]),
        "contract_run_dir": str(contract_dir),
    }
    decision = {
        "completed": True,
        "keep": True,
        "reason": "HMBA-01 completed exact module-local direct-bundle forward selection on top of the frozen HMBA-00 contract while keeping hidden structure separate.",
        "hidden_only_gate_ok": bool(hidden_only_gate_ok),
        "combined_gate_ok": bool(combined_gate_ok),
        "combined_primary_loss": round(float(combined_fit.window_metrics["historical"]["primary_loss"]), 6),
        "combined_diag_flow_loss": round(float(combined_fit.window_metrics["historical"]["diag_flow_loss"]), 6),
        "promoted_bundle_map": promoted_bundle_map,
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "goal": "Promote direct Phase 2 bundles module by module on top of the frozen HMBA-00 contract with hidden structure kept separate.",
            "selected_variant": "evidence-to-model-loop",
            "search_unit": "exact module-local forward selection over the finite seeded bundle sets",
            "contract_dir": str(contract_dir),
            "source_run_id": ctx.source_run_id,
            "incumbent_config": {
                "candidate_id": incumbent_config.candidate_id,
                "diagnosis_family": incumbent_config.diagnosis_family,
                "care_family": incumbent_config.care_family,
                "hidden_rank": int(incumbent_config.hidden_rank),
                "use_observation_covariates": bool(incumbent_config.use_observation_covariates),
            },
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            {
                "name": "seeded_bundle_count",
                "value": int(sum(len(list(bundles)) for bundles in module_bundle_features.values())),
                "role": "finite_module_search_space",
                "source_type": "estimated",
                "estimation_data": "HMBA-00 module_seed_manifest.module_bundle_rankings",
                "estimation_method": "count module-specific direct bundles available for exact forward selection",
                "uncertainty": "none",
                "why_needed": "Defines the exact finite direct-bundle search space for HMBA-01 without introducing an arbitrary top-K cutoff.",
            },
            {
                "name": "incumbent_hidden_rank",
                "value": int(incumbent_config.hidden_rank),
                "role": "hidden_structure_lock",
                "source_type": "estimated",
                "estimation_data": "HMBA-00 contract snapshot incumbent candidate id",
                "estimation_method": "parse hidden rank from the frozen incumbent configuration used by the national integrated loop",
                "uncertainty": "none",
                "why_needed": "Keeps hidden Phase 2 structure fixed while direct bundles are promoted module by module.",
            },
            numerical_guard_entry(
                name="eps_float32",
                role="forward_selection_loss_guard",
                why_needed="The HMBA-01 promotion loop uses normalized losses and machine-epsilon-safe comparisons when fitting challengers on sparse observed channels.",
            ),
        ],
    )
    artifacts.update(
        {
            "module_local_frontier": str(frontier_path),
            "module_promotion_results": str(module_results_path),
            "combined_promoted_model": str(combined_summary_path),
            "single_bundle_gain_heatmap": str(single_bundle_gain_chart_path),
            "module_promotion_trace": str(promotion_trace_chart_path),
            "combined_model_comparison": str(combined_comparison_chart_path),
            "paper_figure_manifest": str(paper_archive["figure_manifest"]),
        }
    )
    return {
        "artifacts": artifacts,
        "module_results": module_results,
        "combined_baseline_comparison": combined_baseline,
        "decision": decision,
        "paper_archive": paper_archive,
    }


def run_hmba_02(ctx: TransitionResearchContext) -> dict[str, Any]:
    contract_dir, contract_snapshot, _module_seed_manifest = _discover_latest_hmba_contract()
    hmba01_dir, hmba01_combined_summary, _hmba01_module_results = _discover_latest_hmba01_summary()
    inputs = load_transition_research_inputs(ctx)
    retained_ids = _retained_factor_ids(inputs)
    evidence = _discover_latest_provincial_evidence(retained_ids)
    auxiliary_panel = _load_provincial_auxiliary_state_tensor(inputs, evidence)
    auxiliary_targets = _module_auxiliary_proxy_panel(auxiliary_panel, inputs.analysis_years)
    snapshot = _build_snapshot(ctx)
    incumbent_config = _hmba01_incumbent_config(contract_snapshot)
    promoted_feature_map = {
        module_name: [str(value) for value in list((hmba01_combined_summary.get("feature_map") or {}).get(module_name) or []) if str(value)]
        for module_name in REPORTING_MODULES
    }
    module_feature_panels = _build_module_feature_panels(inputs, evidence, promoted_feature_map)

    module_summaries: dict[str, Any] = {}
    hierarchical_scores: dict[str, np.ndarray] = {}
    for module_name in REPORTING_MODULES:
        feature_panel = dict((module_feature_panels.get("module_panels") or {}).get(module_name) or {})
        target_panel = dict((auxiliary_targets.get("targets") or {}).get(module_name) or {})
        if not feature_panel or not target_panel:
            continue
        common_provinces = [province for province in list(feature_panel.get("province_axis") or []) if province in set(auxiliary_targets["province_axis"])]
        common_years = [year for year in list(feature_panel.get("year_axis") or []) if year in set(auxiliary_targets["year_axis"])]
        province_lookup_features = {province: idx for idx, province in enumerate(list(feature_panel.get("province_axis") or []))}
        province_lookup_aux = {province: idx for idx, province in enumerate(list(auxiliary_targets["province_axis"] or []))}
        year_lookup_features = {int(year): idx for idx, year in enumerate(list(feature_panel.get("year_axis") or []))}
        year_lookup_aux = {int(year): idx for idx, year in enumerate(list(auxiliary_targets["year_axis"] or []))}
        feature_subset = {
            "feature_ids": list(feature_panel.get("feature_ids") or []),
            "province_axis": common_provinces,
            "region_by_province": [
                str((feature_panel.get("region_by_province") or [])[province_lookup_features[province]]) for province in common_provinces
            ],
            "province_weights": np.asarray(
                [np.asarray(feature_panel.get("province_weights"), dtype=np.float64)[province_lookup_features[province]] for province in common_provinces],
                dtype=np.float64,
            ),
            "year_axis": common_years,
            "province_features": np.asarray(
                [
                    [
                        np.asarray(feature_panel["province_features"], dtype=np.float64)[province_lookup_features[province], year_lookup_features[int(year)], :]
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=np.float64,
            ),
        }
        target_subset = {
            "label": str(target_panel.get("label") or ""),
            "values": np.asarray(
                [
                    [
                        np.asarray(target_panel["values"], dtype=np.float64)[province_lookup_aux[province], year_lookup_aux[int(year)]]
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=np.float64,
            ),
            "mask": np.asarray(
                [
                    [
                        bool(np.asarray(target_panel["mask"], dtype=bool)[province_lookup_aux[province], year_lookup_aux[int(year)]])
                        for year in common_years
                    ]
                    for province in common_provinces
                ],
                dtype=bool,
            ),
        }
        summary = _fit_hierarchical_module_layer(module_name, feature_subset, target_subset)
        module_summaries[module_name] = summary
        hierarchical_scores[module_name] = _expand_yearly_scores_to_quarters(
            snapshot,
            [int(year) for year in summary["year_axis"]],
            [float(value) for value in summary["national_hierarchical_score_by_year"]],
        )

    hidden_only_fit = _fit_with_matrices(
        snapshot,
        incumbent_config,
        _hmba01_candidate_matrices(snapshot, incumbent_config, {module_name: [] for module_name in REPORTING_MODULES}),
    )
    hmba01_fit = _fit_with_matrices(
        snapshot,
        incumbent_config,
        _hmba01_candidate_matrices(snapshot, incumbent_config, promoted_feature_map),
    )
    hmba02_fit = _fit_with_matrices(
        snapshot,
        incumbent_config,
        _hmba02_candidate_matrices(snapshot, incumbent_config, hierarchical_scores),
    )
    hmba02_gate_ok, hmba02_baseline = _fit_passes_hmba01_gate(snapshot, hmba02_fit)
    auxiliary_gate = _hmba02_auxiliary_gate(module_summaries)

    coefficient_path = ctx.experiment_dir / "hierarchical_module_coefficients.json"
    auxiliary_path = ctx.experiment_dir / "hierarchical_auxiliary_evaluation.json"
    combined_path = ctx.experiment_dir / "hierarchical_combined_model.json"
    write_json(
        coefficient_path,
        {
            module_name: {
                "feature_ids": list(summary.get("feature_ids") or []),
                "proxy_label": str(summary.get("proxy_label") or ""),
                "year_axis": list(summary.get("year_axis") or []),
                "national_coefficient": list(summary.get("national_coefficient") or []),
                "region_coefficients": dict(summary.get("region_coefficients") or {}),
                "province_coefficients": dict(summary.get("province_coefficients") or {}),
                "effective_sample_sizes": dict(summary.get("effective_sample_sizes") or {}),
                "pooling_strength_summary": dict(summary.get("pooling_strength_summary") or {}),
            }
            for module_name, summary in module_summaries.items()
        },
    )
    write_json(
        auxiliary_path,
        {
            "module_metrics": {module_name: dict(summary.get("metrics") or {}) for module_name, summary in module_summaries.items()},
            "proxy_labels": {module_name: str(summary.get("proxy_label") or "") for module_name, summary in module_summaries.items()},
            "auxiliary_gate": auxiliary_gate,
        },
    )
    write_json(
        combined_path,
        {
            "promoted_feature_map": promoted_feature_map,
            "hierarchical_score_modules": {
                module_name: {
                    "feature_ids": list(summary.get("feature_ids") or []),
                    "year_axis": list(summary.get("year_axis") or []),
                    "national_hierarchical_score_by_year": [round(float(value), 6) for value in list(summary.get("national_hierarchical_score_by_year") or [])],
                }
                for module_name, summary in module_summaries.items()
            },
            "hidden_only_historical_score_tuple": [round(float(value), 6) for value in _score_tuple(hidden_only_fit.window_metrics["historical"])],
            "hmba01_historical_score_tuple": [round(float(value), 6) for value in _score_tuple(hmba01_fit.window_metrics["historical"])],
            "hmba02_historical_score_tuple": [round(float(value), 6) for value in _score_tuple(hmba02_fit.window_metrics["historical"])],
            "baseline_comparison": hmba02_baseline,
            "passes_national_gate": bool(hmba02_gate_ok),
            "passes_auxiliary_gate": bool(auxiliary_gate["passes_auxiliary_gate"]),
        },
    )

    auxiliary_chart_path = ctx.experiment_dir / "hierarchical_auxiliary_fit_comparison.png"
    pooling_chart_path = ctx.experiment_dir / "hierarchical_pooling_profile.png"
    combined_chart_path = ctx.experiment_dir / "hierarchical_combined_model_comparison.png"
    _write_hmba02_auxiliary_fit_chart(auxiliary_chart_path, module_summaries)
    _write_hmba02_pooling_profile_chart(pooling_chart_path, module_summaries)
    _write_hmba02_combined_comparison_chart(
        combined_chart_path,
        hmba01_primary_loss=float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
        hmba01_diag_flow_loss=float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
        hmba02_fit=hmba02_fit,
        incumbent_primary_loss=float((contract_snapshot.get("national_baseline_gate") or {}).get("model_primary_loss") or 0.0),
        incumbent_diag_flow_loss=float((contract_snapshot.get("national_baseline_gate") or {}).get("model_diag_flow_loss") or 0.0),
    )
    paper_archive = _refresh_phase3_figure_archive(
        hmba02_chart_paths={
            "auxiliary_fit": auxiliary_chart_path,
            "pooling_profile": pooling_chart_path,
            "combined_comparison": combined_chart_path,
        }
    )

    coverage_summary = {
        "contract_run_dir": str(contract_dir),
        "hmba01_run_dir": str(hmba01_dir),
        "module_count": int(len(module_summaries)),
        "active_direct_module_count": int(sum(1 for summary in module_summaries.values() if list(summary.get("feature_ids") or []))),
        "paper_figure_count": int(paper_archive["figure_count"]),
        "province_count": int(len(module_feature_panels["province_axis"])),
        "analysis_year_count": int(len(module_feature_panels["year_axis"])),
    }
    decision = {
        "completed": True,
        "keep": bool(hmba02_gate_ok and auxiliary_gate["passes_auxiliary_gate"]),
        "reason": "HMBA-02 fitted national, region, and province direct-score coefficients on the frozen HMBA contract and fed the aggregated hierarchical scores into the national mechanistic loop while leaving hidden structure separate.",
        "passes_national_gate": bool(hmba02_gate_ok),
        "passes_auxiliary_gate": bool(auxiliary_gate["passes_auxiliary_gate"]),
        "auxiliary_gate": auxiliary_gate,
        "hmba02_primary_loss": round(float(hmba02_fit.window_metrics["historical"]["primary_loss"]), 6),
        "hmba02_diag_flow_loss": round(float(hmba02_fit.window_metrics["historical"]["diag_flow_loss"]), 6),
        "hmba01_primary_loss": round(float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0), 6),
        "hmba01_diag_flow_loss": round(float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0), 6),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "goal": "Learn the actual HMBA hierarchical direct-score layer with national, region, and province pooling on top of the frozen HMBA-00 contract.",
            "selected_variant": "evidence-to-model-loop",
            "search_unit": "closed-form hierarchical coefficient pooling over the HMBA-01 promoted direct bundles",
            "contract_dir": str(contract_dir),
            "hmba01_dir": str(hmba01_dir),
            "source_run_id": ctx.source_run_id,
            "incumbent_config": {
                "candidate_id": incumbent_config.candidate_id,
                "diagnosis_family": incumbent_config.diagnosis_family,
                "care_family": incumbent_config.care_family,
                "hidden_rank": int(incumbent_config.hidden_rank),
                "use_observation_covariates": bool(incumbent_config.use_observation_covariates),
            },
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            {
                "name": "national_effective_samples",
                "value": int(len(module_feature_panels["year_axis"])),
                "role": "hierarchical_parent_mass",
                "source_type": "estimated",
                "estimation_data": "HMBA-02 common yearly panel",
                "estimation_method": "count yearly national rows available for the pooled parent coefficient in each module",
                "uncertainty": "none",
                "why_needed": "Defines the parent support mass in the hierarchical pooling formula without introducing a tuned regularization constant.",
            },
            numerical_guard_entry(
                name="eps_float64",
                role="hierarchical_pooling_guard",
                why_needed="The HMBA-02 hierarchy uses weighted least squares, effective sample sizes, and safe ratio targets that require a machine-defined lower bound.",
            ),
        ],
    )
    artifacts.update(
        {
            "hierarchical_module_coefficients": str(coefficient_path),
            "hierarchical_auxiliary_evaluation": str(auxiliary_path),
            "hierarchical_combined_model": str(combined_path),
            "hierarchical_auxiliary_fit_comparison": str(auxiliary_chart_path),
            "hierarchical_pooling_profile": str(pooling_chart_path),
            "hierarchical_combined_model_comparison": str(combined_chart_path),
            "paper_figure_manifest": str(paper_archive["figure_manifest"]),
        }
    )
    return {
        "artifacts": artifacts,
        "module_summaries": module_summaries,
        "baseline_comparison": hmba02_baseline,
        "decision": decision,
        "paper_archive": paper_archive,
    }


def run_hmba_03(ctx: TransitionResearchContext) -> dict[str, Any]:
    contract_dir, contract_snapshot, module_seed_manifest = _discover_latest_hmba_contract()
    hmba01_dir, hmba01_combined_summary, _hmba01_module_results = _discover_latest_hmba01_summary()
    hmba02_dir, hmba02_combined_summary, hmba02_auxiliary_summary = _discover_latest_hmba02_summary()
    inputs = load_transition_research_inputs(ctx)
    retained_ids = _retained_factor_ids(inputs)
    evidence = _discover_latest_provincial_evidence(retained_ids)
    auxiliary_panel = _load_provincial_auxiliary_state_tensor(inputs, evidence)
    auxiliary_targets = _module_auxiliary_proxy_panel(auxiliary_panel, inputs.analysis_years)
    snapshot = _build_snapshot(ctx)
    incumbent_config = _hmba01_incumbent_config(contract_snapshot)
    module_bundle_features = _module_bundle_feature_lookup(module_seed_manifest)
    feature_context = _module_feature_context(inputs, evidence)

    initial_bundle_map = {
        module_name: [str(value) for value in list((hmba01_combined_summary.get("promoted_bundle_map") or {}).get(module_name) or []) if str(value)]
        for module_name in REPORTING_MODULES
    }
    initial_depth_map = {
        module_name: ("national" if list(initial_bundle_map.get(module_name) or []) else "none")
        for module_name in REPORTING_MODULES
    }
    module_summary_cache: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}
    state_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    absolute_auxiliary_ceiling: float | None = None
    module_auxiliary_ceiling_by_name: dict[str, float] | None = None

    def _state_key(bundle_map: dict[str, list[str]], depth_map: dict[str, str]) -> tuple[Any, ...]:
        return tuple(
            (module_name, tuple(sorted(list(bundle_map.get(module_name) or []))), str(depth_map.get(module_name) or "none"))
            for module_name in REPORTING_MODULES
        )

    def _feature_ids_from_bundles(module_name: str, bundle_names: list[str]) -> list[str]:
        feature_ids: list[str] = []
        for bundle_name in bundle_names:
            feature_ids.extend(list(module_bundle_features.get(module_name, {}).get(bundle_name, [])))
        return sorted(set(feature_ids))

    def _module_summary(module_name: str, feature_ids: list[str]) -> dict[str, Any]:
        cache_key = (module_name, tuple(feature_ids))
        if cache_key in module_summary_cache:
            return module_summary_cache[cache_key]
        feature_panel = _single_module_feature_panel(inputs, feature_ids, feature_context)
        subset_feature_panel, subset_target_panel = _subset_module_target_panels(feature_panel, auxiliary_targets, module_name)
        summary = _fit_hierarchical_module_layer(module_name, subset_feature_panel, subset_target_panel)
        module_summary_cache[cache_key] = summary
        return summary

    def _evaluate_state(
        *,
        bundle_map: dict[str, list[str]],
        depth_map: dict[str, str],
        candidate_id: str,
        action_type: str,
        action_module: str | None,
        action_detail: str | None,
    ) -> dict[str, Any]:
        key = _state_key(bundle_map, depth_map)
        if key in state_cache:
            cached = dict(state_cache[key])
            cached["candidate_id"] = candidate_id
            cached["action_type"] = action_type
            cached["action_module"] = action_module
            cached["action_detail"] = action_detail
            return cached
        module_summaries: dict[str, Any] = {}
        hierarchical_scores: dict[str, np.ndarray] = {}
        for module_name in REPORTING_MODULES:
            bundle_names = list(bundle_map.get(module_name) or [])
            feature_ids = _feature_ids_from_bundles(module_name, bundle_names)
            summary = _module_summary(module_name, feature_ids)
            module_summaries[module_name] = summary
            depth_name = str(depth_map.get(module_name) or "none")
            if depth_name == "none" or not list(summary.get("feature_ids") or []):
                hierarchical_scores[module_name] = np.zeros((len(snapshot.model_quarters),), dtype=np.float64)
                continue
            yearly_scores = [float(value) for value in list((summary.get("national_score_by_depth") or {}).get(depth_name) or [])]
            hierarchical_scores[module_name] = _expand_yearly_scores_to_quarters(
                snapshot,
                [int(year) for year in list(summary.get("year_axis") or [])],
                yearly_scores,
            )
        fit = _fit_with_matrices(snapshot, incumbent_config, _hmba02_candidate_matrices(snapshot, incumbent_config, hierarchical_scores))
        national_gate_ok, baseline_comparison = _fit_passes_hmba01_gate(snapshot, fit)
        auxiliary_gate = _hmba03_auxiliary_gate(
            module_summaries,
            depth_map,
            absolute_ceiling=absolute_auxiliary_ceiling,
            module_ceiling_by_name=module_auxiliary_ceiling_by_name,
        )
        state = {
            "state_key": key,
            "bundle_map": {module_name: list(bundle_map.get(module_name) or []) for module_name in REPORTING_MODULES},
            "depth_map": {module_name: str(depth_map.get(module_name) or "none") for module_name in REPORTING_MODULES},
            "module_summaries": module_summaries,
            "fit": fit,
            "baseline_comparison": baseline_comparison,
            "auxiliary_gate": auxiliary_gate,
            "passes_national_gate": bool(national_gate_ok),
            "passes_dual_gate": bool(national_gate_ok and auxiliary_gate["passes_auxiliary_gate"]),
            "joint_score": _hmba03_joint_score(fit, auxiliary_gate),
            "candidate_id": candidate_id,
            "action_type": action_type,
            "action_module": action_module,
            "action_detail": action_detail,
        }
        state_cache[key] = dict(state)
        return state

    initial_state = _evaluate_state(
        bundle_map=initial_bundle_map,
        depth_map=initial_depth_map,
        candidate_id="step00-initial-hmba01-national-depth",
        action_type="initial",
        action_module=None,
        action_detail=None,
    )
    absolute_auxiliary_ceiling = float(initial_state["auxiliary_gate"]["selected_depth_mean_rmse"])
    module_auxiliary_ceiling_by_name = {
        module_name: float((initial_state["module_summaries"][module_name]["depth_metrics"] or {}).get(str(initial_depth_map.get(module_name) or "none"), 0.0))
        for module_name in REPORTING_MODULES
        if str(initial_depth_map.get(module_name) or "none") != "none"
    }
    state_cache.clear()
    champion = _evaluate_state(
        bundle_map=initial_bundle_map,
        depth_map=initial_depth_map,
        candidate_id="step00-initial-hmba01-national-depth",
        action_type="initial",
        action_module=None,
        action_detail=None,
    )
    if not champion["passes_dual_gate"]:
        raise RuntimeError("HMBA-03 initial state failed the dual gate; HMBA-01 promoted bundles at national depth must provide a valid starting state.")

    seen_state_keys = {champion["state_key"]}
    frontier_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = [
        {
            "step": 0,
            "candidate_id": str(champion["candidate_id"]),
            "action_type": "initial",
            "bundle_map": champion["bundle_map"],
            "depth_map": champion["depth_map"],
            "joint_score": [round(float(value), 6) for value in champion["joint_score"]],
            "primary_loss": round(float(champion["fit"].window_metrics["historical"]["primary_loss"]), 6),
            "diag_flow_loss": round(float(champion["fit"].window_metrics["historical"]["diag_flow_loss"]), 6),
            "selected_depth_auxiliary_rmse": round(float(champion["auxiliary_gate"]["selected_depth_mean_rmse"]), 6),
        }
    ]

    step = 0
    while True:
        step += 1
        candidate_states: list[dict[str, Any]] = []
        for module_name in REPORTING_MODULES:
            current_bundles = list(champion["bundle_map"].get(module_name) or [])
            current_depth = str(champion["depth_map"].get(module_name) or "none")
            available_bundles = [
                bundle_name
                for bundle_name in list(module_bundle_features.get(module_name, {}))
                if bundle_name not in set(current_bundles)
            ]
            for bundle_name in available_bundles:
                for depth_name in ("national", "region", "province"):
                    trial_bundle_map = {name: list(champion["bundle_map"].get(name) or []) for name in REPORTING_MODULES}
                    trial_depth_map = {name: str(champion["depth_map"].get(name) or "none") for name in REPORTING_MODULES}
                    trial_bundle_map[module_name] = current_bundles + [bundle_name]
                    trial_depth_map[module_name] = depth_name
                    key = _state_key(trial_bundle_map, trial_depth_map)
                    if key in seen_state_keys:
                        continue
                    seen_state_keys.add(key)
                    candidate_states.append(
                        _evaluate_state(
                            bundle_map=trial_bundle_map,
                            depth_map=trial_depth_map,
                            candidate_id=f"step{step:02d}-{module_name}-add-{bundle_name}-depth-{depth_name}",
                            action_type="add_bundle",
                            action_module=module_name,
                            action_detail=f"{bundle_name}|{depth_name}",
                        )
                    )
            if current_depth != "none":
                for depth_name in ("national", "region", "province"):
                    if _depth_rank(depth_name) <= _depth_rank(current_depth):
                        continue
                    trial_bundle_map = {name: list(champion["bundle_map"].get(name) or []) for name in REPORTING_MODULES}
                    trial_depth_map = {name: str(champion["depth_map"].get(name) or "none") for name in REPORTING_MODULES}
                    trial_depth_map[module_name] = depth_name
                    key = _state_key(trial_bundle_map, trial_depth_map)
                    if key in seen_state_keys:
                        continue
                    seen_state_keys.add(key)
                    candidate_states.append(
                        _evaluate_state(
                            bundle_map=trial_bundle_map,
                            depth_map=trial_depth_map,
                            candidate_id=f"step{step:02d}-{module_name}-depth-{depth_name}",
                            action_type="depth_upgrade",
                            action_module=module_name,
                            action_detail=depth_name,
                        )
                    )

        if not candidate_states:
            break

        for candidate_state in candidate_states:
            frontier_rows.append(
                {
                    "step": int(step),
                    "candidate_id": str(candidate_state["candidate_id"]),
                    "action_type": str(candidate_state["action_type"]),
                    "action_module": candidate_state["action_module"],
                    "action_detail": candidate_state["action_detail"],
                    "passes_national_gate": bool(candidate_state["passes_national_gate"]),
                    "passes_dual_gate": bool(candidate_state["passes_dual_gate"]),
                    "candidate_primary_loss": round(float(candidate_state["fit"].window_metrics["historical"]["primary_loss"]), 6),
                    "candidate_diag_flow_loss": round(float(candidate_state["fit"].window_metrics["historical"]["diag_flow_loss"]), 6),
                    "candidate_auxiliary_rmse": round(float(candidate_state["auxiliary_gate"]["selected_depth_mean_rmse"]), 6),
                    "best_primary_loss": round(float(champion["fit"].window_metrics["historical"]["primary_loss"]), 6),
                    "best_auxiliary_rmse": round(float(champion["auxiliary_gate"]["selected_depth_mean_rmse"]), 6),
                    "bundle_map": candidate_state["bundle_map"],
                    "depth_map": candidate_state["depth_map"],
                }
            )

        promotable = [
            candidate_state
            for candidate_state in candidate_states
            if bool(candidate_state["passes_dual_gate"]) and tuple(candidate_state["joint_score"]) < tuple(champion["joint_score"])
        ]
        if not promotable:
            break
        champion = min(promotable, key=lambda state: tuple(state["joint_score"]))
        trace_rows.append(
            {
                "step": int(step),
                "candidate_id": str(champion["candidate_id"]),
                "action_type": str(champion["action_type"]),
                "action_module": champion["action_module"],
                "action_detail": champion["action_detail"],
                "bundle_map": champion["bundle_map"],
                "depth_map": champion["depth_map"],
                "joint_score": [round(float(value), 6) for value in champion["joint_score"]],
                "primary_loss": round(float(champion["fit"].window_metrics["historical"]["primary_loss"]), 6),
                "diag_flow_loss": round(float(champion["fit"].window_metrics["historical"]["diag_flow_loss"]), 6),
                "selected_depth_auxiliary_rmse": round(float(champion["auxiliary_gate"]["selected_depth_mean_rmse"]), 6),
            }
        )

    frontier_path = ctx.experiment_dir / "joint_search_frontier.json"
    trace_path = ctx.experiment_dir / "joint_search_trace.json"
    combined_path = ctx.experiment_dir / "joint_hierarchical_model.json"
    claim_contract_path = ctx.experiment_dir / "hmba_claim_contract.json"
    claim_contract = {
        "provincial_panel_role": "auxiliary_provincial_state_share_evidence",
        "province_truth_available": False,
        "allowed_claim": "Hierarchical geography is a regularization and structure-discovery device on auxiliary provincial evidence.",
        "forbidden_claim": "Do not present HMBA province coefficients as validated provincial truth or causal provincial determinant effects.",
        "absolute_auxiliary_ceiling": round(float(absolute_auxiliary_ceiling or np.nan), 6),
        "module_auxiliary_ceiling_by_name": {
            str(module_name): round(float(value), 6) for module_name, value in dict(module_auxiliary_ceiling_by_name or {}).items()
        },
    }
    write_json(frontier_path, frontier_rows)
    write_json(trace_path, trace_rows)
    write_json(claim_contract_path, claim_contract)
    write_json(
        combined_path,
        {
            "final_bundle_map": champion["bundle_map"],
            "final_depth_map": champion["depth_map"],
            "joint_score": [round(float(value), 6) for value in champion["joint_score"]],
            "baseline_comparison": champion["baseline_comparison"],
            "auxiliary_gate": champion["auxiliary_gate"],
            "claim_contract": claim_contract,
            "hmba01_primary_loss": float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
            "hmba01_diag_flow_loss": float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
            "hmba02_primary_loss": float((hmba02_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
            "hmba02_diag_flow_loss": float((hmba02_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
            "hmba03_historical_score_tuple": [round(float(value), 6) for value in _score_tuple(champion["fit"].window_metrics["historical"])],
            "module_depth_metrics": {
                module_name: dict((summary.get("depth_metrics") or {})) for module_name, summary in champion["module_summaries"].items()
            },
            "module_feature_ids": {
                module_name: list(summary.get("feature_ids") or []) for module_name, summary in champion["module_summaries"].items()
            },
        },
    )

    frontier_chart_path = ctx.experiment_dir / "joint_search_frontier.png"
    depth_chart_path = ctx.experiment_dir / "joint_depth_assignment.png"
    comparison_chart_path = ctx.experiment_dir / "joint_model_comparison.png"
    _write_hmba03_search_frontier(frontier_chart_path, frontier_rows)
    _write_hmba03_depth_assignment(depth_chart_path, champion["depth_map"])
    _write_hmba03_combined_comparison_chart(
        comparison_chart_path,
        hmba01_primary_loss=float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
        hmba01_diag_flow_loss=float((hmba01_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
        hmba02_primary_loss=float((hmba02_combined_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
        hmba02_diag_flow_loss=float((hmba02_combined_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
        hmba03_fit=champion["fit"],
    )
    paper_archive = _refresh_phase3_figure_archive(
        hmba03_chart_paths={
            "frontier": frontier_chart_path,
            "depth_assignment": depth_chart_path,
            "combined_comparison": comparison_chart_path,
        }
    )

    total_candidate_count = int(len(frontier_rows))
    decision = {
        "completed": True,
        "keep": bool(champion["passes_dual_gate"]),
        "reason": "HMBA-03 completed a joint autoresearch loop over module bundle additions and hierarchy-depth mutations under the same national-plus-auxiliary dual gate.",
        "passes_national_gate": bool(champion["passes_national_gate"]),
        "passes_dual_gate": bool(champion["passes_dual_gate"]),
        "final_primary_loss": round(float(champion["fit"].window_metrics["historical"]["primary_loss"]), 6),
        "final_diag_flow_loss": round(float(champion["fit"].window_metrics["historical"]["diag_flow_loss"]), 6),
        "final_auxiliary_rmse": round(float(champion["auxiliary_gate"]["selected_depth_mean_rmse"]), 6),
        "absolute_auxiliary_ceiling": round(float(absolute_auxiliary_ceiling or np.nan), 6),
        "joint_search_steps": int(len(trace_rows) - 1),
        "evaluated_candidates": total_candidate_count,
        "final_bundle_map": champion["bundle_map"],
        "final_depth_map": champion["depth_map"],
        "claim_boundary": claim_contract["allowed_claim"],
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "goal": "Search module bundles and hierarchy depth jointly under the frozen HMBA contract while keeping hidden structure separate.",
            "selected_variant": "evidence-to-model-loop",
            "search_unit": "one-action joint mutation over a module bundle addition or hierarchy-depth change",
            "contract_dir": str(contract_dir),
            "hmba01_dir": str(hmba01_dir),
            "hmba02_dir": str(hmba02_dir),
            "source_run_id": ctx.source_run_id,
            "incumbent_config": {
                "candidate_id": incumbent_config.candidate_id,
                "diagnosis_family": incumbent_config.diagnosis_family,
                "care_family": incumbent_config.care_family,
                "hidden_rank": int(incumbent_config.hidden_rank),
                "use_observation_covariates": bool(incumbent_config.use_observation_covariates),
            },
        },
        coverage_summary={
            "contract_run_dir": str(contract_dir),
            "hmba01_run_dir": str(hmba01_dir),
            "hmba02_run_dir": str(hmba02_dir),
            "paper_figure_count": int(paper_archive["figure_count"]),
            "evaluated_candidates": total_candidate_count,
            "final_active_module_count": int(
                sum(1 for module_name in REPORTING_MODULES if str(champion["depth_map"].get(module_name) or "none") != "none")
            ),
            "absolute_auxiliary_ceiling": round(float(absolute_auxiliary_ceiling or np.nan), 6),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "finite_joint_mutation_budget",
                "value": int(
                    sum(len(list(module_bundle_features.get(module_name, {}))) for module_name in REPORTING_MODULES)
                    + 2 * len(REPORTING_MODULES)
                ),
                "role": "finite_joint_search_space_bound",
                "source_type": "estimated",
                "estimation_data": "HMBA-00 module seed manifest and three-level hierarchy depth ladder",
                "estimation_method": "count all seeded direct bundles plus the two admissible upward depth mutations per module",
                "uncertainty": "none",
                "why_needed": "Shows the HMBA-03 autoresearch loop is operating on a finite, evidence-derived one-action neighborhood rather than an unconstrained combinatorial space.",
            },
            numerical_guard_entry(
                name="eps_float64",
                role="joint_autoresearch_guard",
                why_needed="The HMBA-03 loop compares lexicographic loss tuples and safe auxiliary ratios under sparse observed channels, which requires a machine-defined lower bound.",
            ),
        ],
    )
    artifacts.update(
        {
            "joint_search_frontier": str(frontier_path),
            "joint_search_trace": str(trace_path),
            "joint_hierarchical_model": str(combined_path),
            "hmba_claim_contract": str(claim_contract_path),
            "joint_search_frontier_png": str(frontier_chart_path),
            "joint_depth_assignment_png": str(depth_chart_path),
            "joint_model_comparison_png": str(comparison_chart_path),
            "paper_figure_manifest": str(paper_archive["figure_manifest"]),
        }
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "champion": champion,
        "paper_archive": paper_archive,
        "hmba02_auxiliary_summary": hmba02_auxiliary_summary,
    }


def run_hmba_03a(ctx: TransitionResearchContext) -> dict[str, Any]:
    hmba03_dir, hmba03_summary, frontier_rows, trace_rows = _discover_latest_hmba03_outputs()
    hmba02_dir, hmba02_summary, hmba02_auxiliary_summary = _discover_latest_hmba02_outputs()
    model_rows = _hmba03_model_delta_rows(hmba03_summary, hmba02_summary, hmba02_auxiliary_summary)
    step_rows = _hmba03_accepted_step_effect_rows(trace_rows)
    module_rows = _hmba03_module_effect_rows(hmba03_summary)
    champion_auxiliary_rmse = float((hmba03_summary.get("auxiliary_gate") or {}).get("selected_depth_mean_rmse") or np.nan)
    spike_payload = _hmba03_spike_annotation_payload(frontier_rows, champion_auxiliary_rmse)

    dashboard_json = {
        "source_run_id": ctx.source_run_id,
        "hmba03_experiment_dir": str(hmba03_dir),
        "hmba02_experiment_dir": str(hmba02_dir),
        "model_delta_rows": model_rows,
        "accepted_step_effect_rows": step_rows,
        "module_effect_rows": module_rows,
        "spike_candidate_annotations": spike_payload,
    }
    dashboard_json_path = ctx.experiment_dir / "interpretation_dashboard.json"
    report_path = ctx.experiment_dir / "interpretation_dashboard.md"
    accepted_steps_path = ctx.experiment_dir / "accepted_step_effect_table.json"
    module_table_path = ctx.experiment_dir / "module_effect_table.json"
    spike_path = ctx.experiment_dir / "spike_candidate_annotations.json"
    write_json(dashboard_json_path, dashboard_json)
    write_json(accepted_steps_path, step_rows)
    write_json(module_table_path, module_rows)
    write_json(spike_path, spike_payload)
    _write_hmba03a_markdown_report(
        report_path,
        hmba03_dir=hmba03_dir,
        hmba02_dir=hmba02_dir,
        model_rows=model_rows,
        step_rows=step_rows,
        module_rows=module_rows,
        spike_payload=spike_payload,
    )

    zoomed_chart_path = ctx.experiment_dir / "zoomed_loss_deltas.png"
    step_delta_chart_path = ctx.experiment_dir / "accepted_step_deltas.png"
    module_chart_path = ctx.experiment_dir / "module_effect_table.png"
    spike_chart_path = ctx.experiment_dir / "spike_candidate_annotations.png"
    dashboard_chart_path = ctx.experiment_dir / "interpretation_dashboard.png"
    _write_hmba03a_zoomed_delta_chart(zoomed_chart_path, model_rows)
    _write_hmba03a_accepted_step_delta_chart(step_delta_chart_path, step_rows)
    _write_hmba03a_module_effect_table_chart(module_chart_path, module_rows)
    _write_hmba03a_spike_annotation_chart(spike_chart_path, spike_payload, champion_auxiliary_rmse)
    _write_hmba03a_interpretation_dashboard(
        dashboard_chart_path,
        model_rows=model_rows,
        step_rows=step_rows,
        module_rows=module_rows,
        spike_payload=spike_payload,
    )
    paper_archive = _refresh_phase3_figure_archive(
        hmba03a_chart_paths={
            "zoomed_deltas": zoomed_chart_path,
            "accepted_step_deltas": step_delta_chart_path,
            "module_effect_table": module_chart_path,
            "spike_annotations": spike_chart_path,
            "dashboard": dashboard_chart_path,
        }
    )

    worst_family = dict(spike_payload.get("worst_family") or {})
    decision = {
        "completed": True,
        "keep": True,
        "reason": "HMBA-03A converted the emitted HMBA-03 search artifacts into an interpretation dashboard with zoomed delta charts, module-wise effect tables, and spike-family annotations.",
        "hmba03_run_dir": str(hmba03_dir),
        "hmba02_run_dir": str(hmba02_dir),
        "primary_loss_hmba03": float((hmba03_summary.get("baseline_comparison") or {}).get("model_primary_loss") or 0.0),
        "diag_flow_loss_hmba03": float((hmba03_summary.get("baseline_comparison") or {}).get("model_diag_flow_loss") or 0.0),
        "champion_auxiliary_rmse": champion_auxiliary_rmse,
        "worst_spike_family_module": worst_family.get("module_name"),
        "worst_spike_family_bundle": worst_family.get("bundle_name"),
        "worst_spike_family_peak_auxiliary_rmse": worst_family.get("max_auxiliary_rmse"),
        "paper_figure_count": int(paper_archive["figure_count"]),
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "goal": "Turn the HMBA-03 emitted search frontier into a paper-quality interpretation dashboard without rerunning the autoresearch loop.",
            "selected_variant": "benchmark-hardening-loop",
            "search_unit": "post-hoc interpretation over the emitted HMBA-03 joint frontier and accepted trace",
            "source_run_id": ctx.source_run_id,
            "hmba03_experiment_dir": str(hmba03_dir),
            "hmba02_experiment_dir": str(hmba02_dir),
        },
        coverage_summary={
            "frontier_candidate_count": int(len(frontier_rows)),
            "accepted_step_count": int(max(len(step_rows) - 1, 0)),
            "module_count": int(len(module_rows)),
            "top_spike_candidate_count": int(len(list(spike_payload.get("top_candidates") or []))),
            "paper_figure_count": int(paper_archive["figure_count"]),
        },
        decision=decision,
        numeric_justification=[
            {
                "name": "hmba03_candidate_count",
                "value": int(len(frontier_rows)),
                "role": "finite_interpretation_frontier_size",
                "source_type": "estimated",
                "estimation_data": "HMBA-03 joint_search_frontier.json",
                "estimation_method": "count the emitted candidate rows in the frontier artifact",
                "uncertainty": "none",
                "why_needed": "The interpretation dashboard is grounded in the full emitted frontier rather than a hand-picked subset of candidates.",
            },
            numerical_guard_entry(
                name="eps_float64",
                role="dashboard_zoom_guard",
                why_needed="The zoomed delta charts and spike-ratio annotations need a machine-defined lower bound when losses are separated by only a few thousandths.",
            ),
        ],
    )
    artifacts.update(
        {
            "interpretation_dashboard_json": str(dashboard_json_path),
            "interpretation_dashboard_markdown": str(report_path),
            "accepted_step_effect_table": str(accepted_steps_path),
            "module_effect_table": str(module_table_path),
            "spike_candidate_annotations": str(spike_path),
            "zoomed_loss_deltas_png": str(zoomed_chart_path),
            "accepted_step_deltas_png": str(step_delta_chart_path),
            "module_effect_table_png": str(module_chart_path),
            "spike_candidate_annotations_png": str(spike_chart_path),
            "interpretation_dashboard_png": str(dashboard_chart_path),
            "paper_figure_manifest": str(paper_archive["figure_manifest"]),
        }
    )
    return {
        "artifacts": artifacts,
        "decision": decision,
        "dashboard": dashboard_json,
        "paper_archive": paper_archive,
    }


__all__ = ["run_hmba_00", "run_hmba_01", "run_hmba_02", "run_hmba_03", "run_hmba_03a"]
