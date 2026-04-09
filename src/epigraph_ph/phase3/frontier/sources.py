from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import load_tensor_artifact, read_json

from .artifacts import TransitionResearchContext


@dataclass(slots=True)
class TransitionResearchInputs:
    ctx: TransitionResearchContext
    analysis_years: list[int]
    month_axis: list[str]
    region_axis: list[str]
    factor_axis: list[str]
    retained_factor_rows: list[dict[str, Any]]
    retained_factor_lookup: dict[str, dict[str, Any]]
    factor_catalog_rows: list[dict[str, Any]]
    factor_catalog_lookup: dict[str, dict[str, Any]]
    factor_index: dict[str, int]
    national_tensor: np.ndarray
    region_tensor: np.ndarray
    factor_support_counts: dict[str, int]
    phase3_target_blankets: dict[str, Any]
    archive_support: dict[str, Any]
    subgroup_weight_summary: dict[str, Any]
    subgroup_prior_learning_summary: dict[str, Any]
    subgroup_anchor_pack: dict[str, Any]


def _merge_retained_factor_rows(phase2_dir: Path) -> list[dict[str, Any]]:
    predictive_rows = list(read_json(phase2_dir / "retained_predictive_factor_set.json", default=[]))
    context_rows = list(read_json(phase2_dir / "retained_context_factor_set.json", default=[]))
    merged: dict[str, dict[str, Any]] = {}
    for row in predictive_rows + context_rows:
        factor_id = str(row.get("factor_id") or "")
        if factor_id:
            merged[factor_id] = dict(row)
    if merged:
        return [merged[key] for key in sorted(merged)]
    retained_catalog = read_json(phase2_dir / "retained_mesoscopic_factor_catalog.json", default={})
    rows = list(retained_catalog.get("rows", [])) if isinstance(retained_catalog, dict) else []
    return [dict(row) for row in rows]


def _analysis_years(month_axis: list[str], archive_support: dict[str, Any]) -> list[int]:
    month_years = [int(str(month).split("-", maxsplit=1)[0]) for month in month_axis]
    reference_points = list(((archive_support.get("harp_program_support") or {}).get("reference_points") or []))
    supported_years = [
        int(str(row.get("effective_month") or row.get("month") or "").split("-", maxsplit=1)[0])
        for row in reference_points
        if str(row.get("effective_month") or row.get("month") or "").strip()
    ]
    max_supported_year = max(supported_years) if supported_years else max(month_years)
    return sorted({year for year in month_years if year <= max_supported_year})


def load_transition_research_inputs(ctx: TransitionResearchContext) -> TransitionResearchInputs:
    axes = read_json(ctx.phase15_dir / "multiscale_factor_axes.json", default={})
    factor_catalog_rows = list(read_json(ctx.phase15_dir / "multiscale_factor_catalog.json", default=[]))
    factor_catalog_lookup = {str(row.get("factor_id") or ""): dict(row) for row in factor_catalog_rows}
    factor_axis = [str(value) for value in list(axes.get("factor", []))]
    month_axis = [str(value) for value in list(axes.get("month", []))]
    region_axis = [str(value) for value in list(axes.get("region", []))]
    factor_index = {factor_id: idx for idx, factor_id in enumerate(factor_axis)}
    retained_factor_rows = _merge_retained_factor_rows(ctx.phase2_dir)
    retained_factor_lookup = {str(row.get("factor_id") or ""): dict(row) for row in retained_factor_rows}
    national_tensor = np.asarray(load_tensor_artifact(ctx.phase15_dir / "multiscale_national_factor_tensor.npz"), dtype=np.float32)
    region_tensor = np.asarray(load_tensor_artifact(ctx.phase15_dir / "multiscale_region_factor_tensor.npz"), dtype=np.float32)
    phase3_target_blankets = read_json(ctx.phase2_dir / "multiscale_phase3_target_blankets.json", default={})
    factor_support_counts = {
        str(row.get("factor_id") or ""): int(row.get("support_count") or 0)
        for row in list(phase3_target_blankets.get("factor_support_rows", []) or [])
    }
    archive_support = read_json(ctx.phase15_dir / "archive_observation_target_support.json", default={})
    subgroup_weight_summary = read_json((ctx.phase3_dir or ctx.source_run_dir) / "subgroup_weight_summary.json", default={}) if ctx.phase3_dir else {}
    subgroup_prior_learning_summary = read_json((ctx.phase3_dir or ctx.source_run_dir) / "subgroup_prior_learning_summary.json", default={}) if ctx.phase3_dir else {}
    subgroup_anchor_pack = read_json(ctx.source_run_dir / "harp_archive" / "subgroup_anchor_pack.json", default={})
    analysis_years = _analysis_years(month_axis, archive_support)
    return TransitionResearchInputs(
        ctx=ctx,
        analysis_years=analysis_years,
        month_axis=month_axis,
        region_axis=region_axis,
        factor_axis=factor_axis,
        retained_factor_rows=retained_factor_rows,
        retained_factor_lookup=retained_factor_lookup,
        factor_catalog_rows=factor_catalog_rows,
        factor_catalog_lookup=factor_catalog_lookup,
        factor_index=factor_index,
        national_tensor=national_tensor,
        region_tensor=region_tensor,
        factor_support_counts=factor_support_counts,
        phase3_target_blankets=phase3_target_blankets,
        archive_support=archive_support,
        subgroup_weight_summary=subgroup_weight_summary,
        subgroup_prior_learning_summary=subgroup_prior_learning_summary,
        subgroup_anchor_pack=subgroup_anchor_pack,
    )
