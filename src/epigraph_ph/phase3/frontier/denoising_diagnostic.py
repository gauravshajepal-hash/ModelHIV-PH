from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize

from epigraph_ph.runtime import ROOT_DIR, read_json, write_json

from .aggregation_diagnostic import (
    REPORTING_MODULES,
    ProvincialEvidence,
    _discover_latest_provincial_evidence,
    _factor_rows,
    _module_seed_report,
)
from .artifacts import TransitionResearchContext, write_experiment_artifacts
from .hierarchical_autoresearch import _retained_factor_ids
from .numeric_policy import numerical_guard_entry
from .sources import TransitionResearchInputs, load_transition_research_inputs


def _mp_density(x: float, beta: float) -> float:
    beta_value = float(min(max(beta, np.finfo(np.float64).eps), 1.0))
    a_edge = (1.0 - math.sqrt(beta_value)) ** 2
    b_edge = (1.0 + math.sqrt(beta_value)) ** 2
    if x <= a_edge or x >= b_edge:
        return 0.0
    numerator = math.sqrt(max((b_edge - x) * (x - a_edge), 0.0))
    denominator = 2.0 * math.pi * beta_value * x
    return numerator / max(denominator, np.finfo(np.float64).eps)


def _mp_median(beta: float) -> float:
    beta_value = float(min(max(beta, np.finfo(np.float64).eps), 1.0))
    a_edge = (1.0 - math.sqrt(beta_value)) ** 2
    b_edge = (1.0 + math.sqrt(beta_value)) ** 2

    def cdf(x: float) -> float:
        integral, _error = integrate.quad(lambda value: _mp_density(value, beta_value), a_edge, x, limit=200)
        return float(integral) - 0.5

    return float(optimize.brentq(cdf, a_edge + 1e-12, b_edge - 1e-12, maxiter=200))


def _optimal_svht_unknown_sigma_threshold(singular_values: np.ndarray, rows: int, cols: int) -> float:
    singular_array = np.asarray(singular_values, dtype=np.float64)
    if singular_array.size == 0:
        return float("inf")
    beta = float(min(rows, cols)) / float(max(rows, cols))
    lambda_star = math.sqrt(
        2.0 * (beta + 1.0)
        + (8.0 * beta) / max(beta + 1.0 + math.sqrt(beta**2 + 14.0 * beta + 1.0), np.finfo(np.float64).eps)
    )
    mp_median = _mp_median(beta)
    omega = lambda_star / math.sqrt(max(mp_median, np.finfo(np.float64).eps))
    return float(omega * np.median(singular_array))


def _weighted_residual_denoise(matrix: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(matrix, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    eps = np.finfo(np.float64).eps
    if values.ndim != 2:
        raise ValueError("partial denoising expects a 2D province-by-time matrix")
    if values.shape[0] != weight_array.shape[0]:
        raise ValueError("province weights must align with the province axis")
    if float(np.sum(weight_array)) <= eps:
        weight_array = np.full((values.shape[0],), 1.0 / float(values.shape[0]), dtype=np.float64)
    else:
        weight_array = weight_array / float(np.sum(weight_array))

    national_series = np.sum(weight_array[:, None] * values, axis=0)
    residual = values - national_series[None, :]
    singular_values = np.linalg.svd(residual, compute_uv=False)
    threshold = _optimal_svht_unknown_sigma_threshold(singular_values, residual.shape[0], residual.shape[1])
    u_matrix, s_values, vh_matrix = np.linalg.svd(residual, full_matrices=False)
    keep_mask = s_values > threshold
    kept_rank = int(np.sum(keep_mask))
    if kept_rank <= 0:
        denoised_residual = np.zeros_like(residual, dtype=np.float64)
    else:
        denoised_residual = (u_matrix[:, keep_mask] * s_values[keep_mask]) @ vh_matrix[keep_mask, :]
    weighted_mean = np.sum(weight_array[:, None] * denoised_residual, axis=0)
    denoised_residual = denoised_residual - weighted_mean[None, :]
    denoised = national_series[None, :] + denoised_residual
    residual_energy = float(np.sum(np.square(residual)))
    denoised_energy = float(np.sum(np.square(denoised_residual)))
    leading_energy_share = float(np.sum(np.square(s_values[: max(kept_rank, 1)])) / max(np.sum(np.square(s_values)), eps)) if s_values.size else 0.0
    return denoised, {
        "svht_threshold": float(threshold),
        "raw_rank": int(np.sum(s_values > eps)),
        "kept_rank": kept_rank,
        "raw_residual_energy": residual_energy,
        "denoised_residual_energy": denoised_energy,
        "retained_residual_energy_share": float(denoised_energy / max(residual_energy, eps)) if residual_energy > eps else 0.0,
        "leading_energy_share_after_threshold": leading_energy_share,
    }


def _denoise_retained_province_tensor(inputs: TransitionResearchInputs, evidence: ProvincialEvidence) -> tuple[np.ndarray, list[dict[str, Any]]]:
    province_tensor = np.asarray(inputs.province_tensor, dtype=np.float64).copy()
    retained_ids = [factor_id for factor_id in sorted(_retained_factor_ids(inputs)) if factor_id in inputs.factor_index]
    province_weights = np.asarray(
        [
            float(evidence.province_weights.get(str(province), 0.0))
            if str(province) not in {"", "Philippines", "unknown"}
            else 0.0
            for province in inputs.province_axis
        ],
        dtype=np.float64,
    )
    diagnostics: list[dict[str, Any]] = []
    for factor_id in retained_ids:
        factor_idx = int(inputs.factor_index[factor_id])
        matrix = np.asarray(province_tensor[:, :, factor_idx], dtype=np.float64)
        denoised, diag = _weighted_residual_denoise(matrix, province_weights)
        province_tensor[:, :, factor_idx] = denoised
        diagnostics.append({"factor_id": factor_id, **diag})
    return np.asarray(province_tensor, dtype=np.float32), diagnostics


def _row_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("factor_id") or ""): dict(row) for row in rows}


def _factor_delta_rows(raw_rows: list[dict[str, Any]], denoised_rows: list[dict[str, Any]], denoise_diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_lookup = _row_lookup(raw_rows)
    denoised_lookup = _row_lookup(denoised_rows)
    diagnostic_lookup = {str(row.get("factor_id") or ""): dict(row) for row in denoise_diagnostics}
    rows: list[dict[str, Any]] = []
    for factor_id in sorted(raw_lookup):
        raw_row = raw_lookup[factor_id]
        denoised_row = denoised_lookup.get(factor_id, {})
        diag_row = diagnostic_lookup.get(factor_id, {})
        row = {
            "factor_id": factor_id,
            "factor_name": str(raw_row.get("factor_name") or factor_id),
            "block_name": str(raw_row.get("block_name") or ""),
            "raw_hierarchical_priority": float(raw_row.get("hierarchical_priority_score") or 0.0),
            "denoised_hierarchical_priority": float(denoised_row.get("hierarchical_priority_score") or 0.0),
            "hierarchical_priority_delta": float(denoised_row.get("hierarchical_priority_score") or 0.0)
            - float(raw_row.get("hierarchical_priority_score") or 0.0),
            "raw_aggregation_loss": float(raw_row.get("aggregation_loss") or 0.0),
            "denoised_aggregation_loss": float(denoised_row.get("aggregation_loss") or 0.0),
            "aggregation_loss_delta": float(denoised_row.get("aggregation_loss") or 0.0) - float(raw_row.get("aggregation_loss") or 0.0),
            "raw_region_gain": float(raw_row.get("region_gain") or 0.0),
            "denoised_region_gain": float(denoised_row.get("region_gain") or 0.0),
            "region_gain_delta": float(denoised_row.get("region_gain") or 0.0) - float(raw_row.get("region_gain") or 0.0),
            "raw_alignment_rmse": float(raw_row.get("provided_national_alignment_rmse") or 0.0),
            "denoised_alignment_rmse": float(denoised_row.get("provided_national_alignment_rmse") or 0.0),
            "alignment_rmse_delta": float(denoised_row.get("provided_national_alignment_rmse") or 0.0)
            - float(raw_row.get("provided_national_alignment_rmse") or 0.0),
            "raw_alignment_correlation": float(raw_row.get("provided_national_alignment_correlation") or 0.0),
            "denoised_alignment_correlation": float(denoised_row.get("provided_national_alignment_correlation") or 0.0),
            "alignment_correlation_delta": float(denoised_row.get("provided_national_alignment_correlation") or 0.0)
            - float(raw_row.get("provided_national_alignment_correlation") or 0.0),
            "svht_threshold": float(diag_row.get("svht_threshold") or 0.0),
            "raw_rank": int(diag_row.get("raw_rank") or 0),
            "kept_rank": int(diag_row.get("kept_rank") or 0),
            "retained_residual_energy_share": float(diag_row.get("retained_residual_energy_share") or 0.0),
        }
        rows.append(row)
    rows.sort(key=lambda row: (-abs(float(row["hierarchical_priority_delta"])), str(row["factor_id"])))
    return rows


def _bundle_score_lookup(module_seed_manifest: dict[str, Any]) -> dict[str, dict[str, float]]:
    rankings = dict(module_seed_manifest.get("module_bundle_rankings") or {})
    return {
        str(module_name): {
            str(bundle_row.get("bundle_name") or ""): float(bundle_row.get("bundle_score") or 0.0)
            for bundle_row in list(bundle_rows or [])
        }
        for module_name, bundle_rows in rankings.items()
    }


def _module_delta_rows(raw_manifest: dict[str, Any], denoised_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_lookup = _bundle_score_lookup(raw_manifest)
    denoised_lookup = _bundle_score_lookup(denoised_manifest)
    rows: list[dict[str, Any]] = []
    for module_name in REPORTING_MODULES:
        bundle_names = sorted(set(raw_lookup.get(module_name, {})) | set(denoised_lookup.get(module_name, {})))
        if not bundle_names:
            continue
        for bundle_name in bundle_names:
            raw_score = float(raw_lookup.get(module_name, {}).get(bundle_name, 0.0))
            denoised_score = float(denoised_lookup.get(module_name, {}).get(bundle_name, 0.0))
            rows.append(
                {
                    "module_name": module_name,
                    "bundle_name": bundle_name,
                    "raw_bundle_score": raw_score,
                    "denoised_bundle_score": denoised_score,
                    "bundle_score_delta": denoised_score - raw_score,
                }
            )
    rows.sort(key=lambda row: (str(row["module_name"]), -abs(float(row["bundle_score_delta"])), str(row["bundle_name"])))
    return rows


def _summary_payload(
    factor_delta_rows: list[dict[str, Any]],
    raw_module_seed_manifest: dict[str, Any],
    denoised_module_seed_manifest: dict[str, Any],
) -> dict[str, Any]:
    def _mean(key: str) -> float:
        values = [float(row[key]) for row in factor_delta_rows]
        return float(np.mean(values)) if values else 0.0

    module_delta_rows = _module_delta_rows(raw_module_seed_manifest, denoised_module_seed_manifest)
    back_half_modules = {"A_to_V", "A_to_L", "L_to_A"}
    return {
        "factor_count": len(factor_delta_rows),
        "mean_hierarchical_priority_delta": round(_mean("hierarchical_priority_delta"), 6),
        "mean_aggregation_loss_delta": round(_mean("aggregation_loss_delta"), 6),
        "mean_region_gain_delta": round(_mean("region_gain_delta"), 6),
        "mean_alignment_rmse_delta": round(_mean("alignment_rmse_delta"), 6),
        "mean_alignment_correlation_delta": round(_mean("alignment_correlation_delta"), 6),
        "mean_retained_residual_energy_share": round(_mean("retained_residual_energy_share"), 6),
        "mean_raw_rank": round(_mean("raw_rank"), 6),
        "mean_kept_rank": round(_mean("kept_rank"), 6),
        "mean_back_half_bundle_delta": round(
            float(
                np.mean(
                    [
                        float(row["bundle_score_delta"])
                        for row in module_delta_rows
                        if str(row["module_name"]) in back_half_modules
                    ]
                )
            )
            if any(str(row["module_name"]) in back_half_modules for row in module_delta_rows)
            else 0.0,
            6,
        ),
        "raw_top_incidence_bundle": (
            (raw_module_seed_manifest.get("module_bundle_rankings") or {}).get("incidence", [{}])[0].get("bundle_name")
            if (raw_module_seed_manifest.get("module_bundle_rankings") or {}).get("incidence")
            else None
        ),
        "denoised_top_incidence_bundle": (
            (denoised_module_seed_manifest.get("module_bundle_rankings") or {}).get("incidence", [{}])[0].get("bundle_name")
            if (denoised_module_seed_manifest.get("module_bundle_rankings") or {}).get("incidence")
            else None
        ),
    }


def _write_module_delta_chart(path: Path, module_delta_rows: list[dict[str, Any]]) -> None:
    module_names = [module_name for module_name in REPORTING_MODULES if any(str(row["module_name"]) == module_name for row in module_delta_rows)]
    bundle_names = sorted({str(row["bundle_name"]) for row in module_delta_rows})
    matrix = np.zeros((len(module_names), len(bundle_names)), dtype=np.float64)
    for module_idx, module_name in enumerate(module_names):
        for bundle_idx, bundle_name in enumerate(bundle_names):
            match = next(
                (
                    float(row["bundle_score_delta"])
                    for row in module_delta_rows
                    if str(row["module_name"]) == module_name and str(row["bundle_name"]) == bundle_name
                ),
                0.0,
            )
            matrix[module_idx, bundle_idx] = match
    fig, ax = plt.subplots(figsize=(14, 5.2))
    limit = float(np.max(np.abs(matrix))) if matrix.size else 1.0
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_title("AN-03B Partial Denoising Bundle Score Delta")
    ax.set_yticks(range(len(module_names)))
    ax.set_yticklabels(module_names)
    ax.set_xticks(range(len(bundle_names)))
    ax.set_xticklabels(bundle_names, rotation=45, ha="right")
    fig.colorbar(image, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_rank_chart(path: Path, factor_delta_rows: list[dict[str, Any]]) -> None:
    labels = [str(row["factor_id"]) for row in factor_delta_rows[:12]]
    raw_ranks = [float(row["raw_rank"]) for row in factor_delta_rows[:12]]
    kept_ranks = [float(row["kept_rank"]) for row in factor_delta_rows[:12]]
    x_axis = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.bar(x_axis - width / 2.0, raw_ranks, width=width, label="raw rank")
    ax.bar(x_axis + width / 2.0, kept_ranks, width=width, label="kept rank")
    ax.set_title("AN-03B Partial Denoising Rank Compression")
    ax.set_ylabel("Effective matrix rank")
    ax.set_xticks(x_axis)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_priority_chart(path: Path, factor_delta_rows: list[dict[str, Any]]) -> None:
    top_rows = factor_delta_rows[:15]
    labels = [str(row["factor_id"]) for row in reversed(top_rows)]
    raw_scores = [float(row["raw_hierarchical_priority"]) for row in reversed(top_rows)]
    denoised_scores = [float(row["denoised_hierarchical_priority"]) for row in reversed(top_rows)]
    fig, ax = plt.subplots(figsize=(12, 6))
    y_axis = np.arange(len(labels))
    ax.barh(y_axis - 0.2, raw_scores, height=0.38, label="raw")
    ax.barh(y_axis + 0.2, denoised_scores, height=0.38, label="denoised")
    ax.set_yticks(y_axis)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hierarchical priority score")
    ax.set_title("AN-03B Raw vs Partial-Denoised Factor Priority")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _latest_experiment_dir(experiment_id: str, sentinel_filename: str) -> Path | None:
    runs_root = ROOT_DIR / "artifacts" / "runs"
    candidates: list[tuple[float, Path]] = []
    for run_dir in runs_root.iterdir():
        if not run_dir.is_dir():
            continue
        experiment_dir = run_dir / "transition_research" / experiment_id
        sentinel_path = experiment_dir / sentinel_filename
        if sentinel_path.exists():
            candidates.append((sentinel_path.stat().st_mtime, experiment_dir))
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row[0]))[1]


def _export_paper_figure_archive(
    *,
    ctx: TransitionResearchContext,
    denoising_chart_paths: dict[str, Path],
) -> dict[str, Any]:
    archive_dir = ROOT_DIR / "artifacts" / "paper_figures" / "phase3_hmba_20260409"
    archive_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    figure_specs: list[tuple[str, str, str, str, Path | None]] = [
        (
            "fig01",
            "Aggregation Loss Priority",
            "Top retained Phase 2 factors ranked by aggregation-loss priority using provincial evidence overlap.",
            "phase2_aggregation_loss_priority.png",
            (_latest_experiment_dir("AN-03A-phase2-aggregation-loss-diagnostic", "aggregation_loss_top_factors.png") or Path()) / "aggregation_loss_top_factors.png",
        ),
        (
            "fig02",
            "Aggregation Loss Module Heatmap",
            "Module-specific aggregation-loss scores for the highest-priority retained factors.",
            "phase2_aggregation_loss_module_heatmap.png",
            (_latest_experiment_dir("AN-03A-phase2-aggregation-loss-diagnostic", "aggregation_loss_module_heatmap.png") or Path()) / "aggregation_loss_module_heatmap.png",
        ),
        (
            "fig03",
            "HMBA Coverage Frontier",
            "Observed national metric coverage versus provincial auxiliary state-share coverage used by the HMBA contract.",
            "hmba_contract_coverage_frontier.png",
            (_latest_experiment_dir("HMBA-00-hierarchical-contract-freeze", "coverage_frontier.png") or Path()) / "coverage_frontier.png",
        ),
        (
            "fig04",
            "HMBA Module Seed Frontier",
            "Evidence-backed module-specific Phase 2 bundle seed frontier for HMBA search initialization.",
            "hmba_module_seed_frontier.png",
            (_latest_experiment_dir("HMBA-00-hierarchical-contract-freeze", "module_seed_frontier.png") or Path()) / "module_seed_frontier.png",
        ),
        (
            "fig05",
            "Partial Denoising Module Delta",
            "Bundle-score change under data-driven partial denoising of the province factor field.",
            "partial_denoising_module_delta.png",
            denoising_chart_paths.get("module_delta"),
        ),
        (
            "fig06",
            "Partial Denoising Rank Compression",
            "Effective factor-matrix rank before and after partial denoising for the highest-priority retained factors.",
            "partial_denoising_rank_compression.png",
            denoising_chart_paths.get("rank"),
        ),
        (
            "fig07",
            "Partial Denoising Priority Comparison",
            "Raw versus partial-denoised hierarchical priority scores for the highest-priority retained factors.",
            "partial_denoising_priority_comparison.png",
            denoising_chart_paths.get("priority"),
        ),
    ]

    for figure_id, title, caption, filename, source_path in figure_specs:
        if source_path is None or not source_path.exists():
            continue
        destination = archive_dir / filename
        shutil.copy2(source_path, destination)
        manifest.append(
            {
                "figure_id": figure_id,
                "title": title,
                "caption": caption,
                "source_path": str(source_path),
                "archived_path": str(destination),
            }
        )

    manifest_path = archive_dir / "figure_manifest.json"
    write_json(manifest_path, manifest)
    return {"archive_dir": str(archive_dir), "figure_manifest": str(manifest_path), "figure_count": len(manifest)}


def run_an03b(ctx: TransitionResearchContext) -> dict[str, Any]:
    inputs = load_transition_research_inputs(ctx)
    retained_ids = _retained_factor_ids(inputs)
    evidence = _discover_latest_provincial_evidence(retained_ids)

    raw_factor_payload = _factor_rows(inputs, evidence)
    raw_rows = list(raw_factor_payload["rows"])
    raw_module_seed_manifest = _module_seed_report(raw_rows)

    denoised_tensor, denoise_diagnostics = _denoise_retained_province_tensor(inputs, evidence)
    denoised_inputs = replace(inputs, province_tensor=np.asarray(denoised_tensor, dtype=np.float32))
    denoised_factor_payload = _factor_rows(denoised_inputs, evidence)
    denoised_rows = list(denoised_factor_payload["rows"])
    denoised_module_seed_manifest = _module_seed_report(denoised_rows)

    factor_delta_rows = _factor_delta_rows(raw_rows, denoised_rows, denoise_diagnostics)
    module_delta_rows = _module_delta_rows(raw_module_seed_manifest, denoised_module_seed_manifest)
    summary_payload = _summary_payload(factor_delta_rows, raw_module_seed_manifest, denoised_module_seed_manifest)

    raw_rows_path = ctx.experiment_dir / "raw_factor_rows.json"
    denoised_rows_path = ctx.experiment_dir / "denoised_factor_rows.json"
    factor_delta_path = ctx.experiment_dir / "partial_denoising_factor_delta.json"
    module_delta_path = ctx.experiment_dir / "partial_denoising_module_delta.json"
    summary_path = ctx.experiment_dir / "partial_denoising_summary.json"
    raw_seed_path = ctx.experiment_dir / "raw_module_seed_manifest.json"
    denoised_seed_path = ctx.experiment_dir / "denoised_module_seed_manifest.json"
    diagnostics_path = ctx.experiment_dir / "partial_denoising_rank_diagnostics.json"
    write_json(raw_rows_path, raw_rows)
    write_json(denoised_rows_path, denoised_rows)
    write_json(factor_delta_path, factor_delta_rows)
    write_json(module_delta_path, module_delta_rows)
    write_json(summary_path, summary_payload)
    write_json(raw_seed_path, raw_module_seed_manifest)
    write_json(denoised_seed_path, denoised_module_seed_manifest)
    write_json(diagnostics_path, denoise_diagnostics)

    module_delta_chart_path = ctx.experiment_dir / "partial_denoising_module_delta.png"
    rank_chart_path = ctx.experiment_dir / "partial_denoising_rank_compression.png"
    priority_chart_path = ctx.experiment_dir / "partial_denoising_priority_comparison.png"
    _write_module_delta_chart(module_delta_chart_path, module_delta_rows)
    _write_rank_chart(rank_chart_path, factor_delta_rows)
    _write_priority_chart(priority_chart_path, factor_delta_rows)

    paper_archive = _export_paper_figure_archive(
        ctx=ctx,
        denoising_chart_paths={
            "module_delta": module_delta_chart_path,
            "rank": rank_chart_path,
            "priority": priority_chart_path,
        },
    )

    coverage_summary = {
        "analysis_years": list(inputs.analysis_years),
        "retained_factor_count": len(retained_ids),
        "provincial_overlap_count": int(evidence.overlap_count),
        "module_count": len(REPORTING_MODULES),
        "paper_figure_count": int(paper_archive["figure_count"]),
    }
    decision = {
        "completed": True,
        "keep": True,
        "reason": "AN-03B tested data-driven partial denoising on the retained province factor field without wiring it into HMBA-01.",
        "provincial_evidence_run_id": evidence.run_id,
        "mean_alignment_rmse_delta": float(summary_payload["mean_alignment_rmse_delta"]),
        "mean_region_gain_delta": float(summary_payload["mean_region_gain_delta"]),
        "mean_back_half_bundle_delta": float(summary_payload["mean_back_half_bundle_delta"]),
        "raw_top_incidence_bundle": summary_payload["raw_top_incidence_bundle"],
        "denoised_top_incidence_bundle": summary_payload["denoised_top_incidence_bundle"],
    }
    artifacts = write_experiment_artifacts(
        ctx=ctx,
        experiment_spec={
            "experiment_id": ctx.experiment.experiment_id,
            "description": ctx.experiment.description,
            "source_run_id": ctx.source_run_id,
            "denoising_method": "weighted_residual_svht_unknown_sigma",
            "paper_figure_archive": paper_archive,
            "provincial_evidence_run_id": evidence.run_id,
        },
        coverage_summary=coverage_summary,
        decision=decision,
        numeric_justification=[
            {
                "name": "retained_factor_count",
                "value": int(len(retained_ids)),
                "role": "denoising_problem_count",
                "source_type": "estimated",
                "estimation_data": "current retained Phase 2 factor ids",
                "estimation_method": "count retained factor ids present in the province factor tensor",
                "uncertainty": "none",
                "why_needed": "Defines the number of province-by-time factor matrices tested by the denoising diagnostic.",
            },
            {
                "name": "provincial_overlap_count",
                "value": int(evidence.overlap_count),
                "role": "evidence_overlap_gate",
                "source_type": "estimated",
                "estimation_data": "intersection between retained Phase 2 factors and provincial selected determinant modifiers",
                "estimation_method": "count shared factor ids across the current source run and the kept provincial evidence run",
                "uncertainty": "none",
                "why_needed": "Restricts the denoising test to the factor universe that is actually supported by the current provincial evidence line.",
            },
            numerical_guard_entry(
                name="float64_machine_epsilon",
                role="svht_and_weight_normalization_guard",
                why_needed="Prevents undefined thresholds or weight normalizations when a factor residual matrix or province weight vector is nearly degenerate.",
            ),
        ],
    )
    artifacts.update(
        {
            "raw_factor_rows": str(raw_rows_path),
            "denoised_factor_rows": str(denoised_rows_path),
            "partial_denoising_factor_delta": str(factor_delta_path),
            "partial_denoising_module_delta": str(module_delta_path),
            "partial_denoising_summary": str(summary_path),
            "partial_denoising_rank_diagnostics": str(diagnostics_path),
            "raw_module_seed_manifest": str(raw_seed_path),
            "denoised_module_seed_manifest": str(denoised_seed_path),
            "partial_denoising_module_delta_chart": str(module_delta_chart_path),
            "partial_denoising_rank_chart": str(rank_chart_path),
            "partial_denoising_priority_chart": str(priority_chart_path),
            "paper_figure_manifest": str(paper_archive["figure_manifest"]),
        }
    )
    return {
        "summary": summary_payload,
        "factor_delta_rows": factor_delta_rows,
        "module_delta_rows": module_delta_rows,
        "paper_archive": paper_archive,
        "artifacts": artifacts,
        "decision": decision,
    }


__all__ = ["run_an03b"]
