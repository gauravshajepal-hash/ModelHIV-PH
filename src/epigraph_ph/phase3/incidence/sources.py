from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, read_json

from .artifacts import IncidenceResearchContext


@dataclass(slots=True)
class IncidenceAuditInputs:
    ctx: IncidenceResearchContext
    historical_metric_rows: list[dict[str, Any]]
    ground_truth_summary: dict[str, Any]
    population_candidate_bank_path: Path | None
    population_candidate_rows: list[dict[str, Any]]
    normalized_subparameters_path: Path | None
    normalized_subparameter_rows: list[dict[str, Any]]


def _candidate_bank_paths() -> list[Path]:
    explicit_default = ROOT_DIR / "artifacts" / "runs" / "audit-phase0-reuse-s00-20260331" / "phase0" / "extracted" / "candidate_banks" / "candidate_bank_populationmeasure.json"
    discovered = sorted((ROOT_DIR / "artifacts" / "runs").glob("*/phase0/extracted/candidate_banks/candidate_bank_populationmeasure.json"))
    ordered: list[Path] = []
    if explicit_default.exists():
        ordered.append(explicit_default)
    for path in discovered:
        if path not in ordered:
            ordered.append(path)
    return ordered


def _resolve_population_candidate_bank(ctx: IncidenceResearchContext) -> Path | None:
    if ctx.phase0_dir is not None:
        direct = ctx.phase0_dir / "extracted" / "candidate_banks" / "candidate_bank_populationmeasure.json"
        if direct.exists():
            return direct
    for path in _candidate_bank_paths():
        if path.exists():
            return path
    return None


def _normalized_subparameter_paths() -> list[Path]:
    explicit_default = ROOT_DIR / "artifacts" / "runs" / "audit-phase0-reuse-s00-20260331" / "phase1" / "normalized_subparameters.json"
    discovered = sorted((ROOT_DIR / "artifacts" / "runs").glob("*/phase1/normalized_subparameters.json"))
    ordered: list[Path] = []
    if explicit_default.exists():
        ordered.append(explicit_default)
    for path in discovered:
        if path not in ordered:
            ordered.append(path)
    return ordered


def _resolve_normalized_subparameters(ctx: IncidenceResearchContext) -> Path | None:
    if ctx.phase1_dir is not None:
        direct = ctx.phase1_dir / "normalized_subparameters.json"
        if direct.exists():
            return direct
    for path in _normalized_subparameter_paths():
        if path.exists():
            return path
    return None


def _load_population_candidate_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = read_json(path, default={})
    if isinstance(payload, dict):
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return []


def _load_normalized_subparameter_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = read_json(path, default=[])
    if isinstance(payload, list):
        return [dict(row) for row in payload if isinstance(row, dict)]
    return []


def load_incidence_audit_inputs(ctx: IncidenceResearchContext) -> IncidenceAuditInputs:
    historical_metric_rows = read_json(ctx.harp_archive_dir / "historical_metric_rows.json", default=[])
    ground_truth_summary = read_json(ctx.harp_archive_dir / "ground_truth_summary.json", default={})
    population_candidate_bank_path = _resolve_population_candidate_bank(ctx)
    population_candidate_rows = _load_population_candidate_rows(population_candidate_bank_path)
    normalized_subparameters_path = _resolve_normalized_subparameters(ctx)
    normalized_subparameter_rows = _load_normalized_subparameter_rows(normalized_subparameters_path)
    return IncidenceAuditInputs(
        ctx=ctx,
        historical_metric_rows=[dict(row) for row in historical_metric_rows if isinstance(row, dict)],
        ground_truth_summary=dict(ground_truth_summary) if isinstance(ground_truth_summary, dict) else {},
        population_candidate_bank_path=population_candidate_bank_path,
        population_candidate_rows=population_candidate_rows,
        normalized_subparameters_path=normalized_subparameters_path,
        normalized_subparameter_rows=normalized_subparameter_rows,
    )
