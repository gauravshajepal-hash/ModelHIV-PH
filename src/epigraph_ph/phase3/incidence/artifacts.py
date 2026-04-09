from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, read_json, write_json

from epigraph_ph.phase3.frontier.numeric_policy import write_numeric_justification
from .registry import ExperimentDefinition, get_experiment_definition


@dataclass(slots=True)
class IncidenceResearchContext:
    experiment: ExperimentDefinition
    run_context: RunContext
    experiment_dir: Path
    source_run_id: str
    source_run_dir: Path
    harp_archive_dir: Path
    phase0_dir: Path | None
    phase1_dir: Path | None


def build_incidence_research_context(
    *,
    run_id: str,
    plugin_id: str,
    experiment_id: str,
    source_run_id: str,
) -> IncidenceResearchContext:
    experiment = get_experiment_definition(experiment_id)
    run_context = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    experiment_dir = ensure_dir(run_context.run_dir / "incidence_research" / experiment_id)
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / source_run_id
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run directory does not exist: {source_run_dir}")
    harp_archive_dir = source_run_dir / "harp_archive"
    if not harp_archive_dir.exists():
        raise FileNotFoundError(f"harp_archive directory does not exist: {harp_archive_dir}")
    phase0_dir = source_run_dir / "phase0"
    if not phase0_dir.exists():
        phase0_dir = None
    phase1_dir = source_run_dir / "phase1"
    if not phase1_dir.exists():
        phase1_dir = None
    return IncidenceResearchContext(
        experiment=experiment,
        run_context=run_context,
        experiment_dir=experiment_dir,
        source_run_id=source_run_id,
        source_run_dir=source_run_dir,
        harp_archive_dir=harp_archive_dir,
        phase0_dir=phase0_dir,
        phase1_dir=phase1_dir,
    )


def write_experiment_artifacts(
    *,
    ctx: IncidenceResearchContext,
    experiment_spec: dict[str, Any],
    coverage_summary: dict[str, Any],
    decision: dict[str, Any],
    numeric_justification: list[dict[str, Any]],
) -> dict[str, str]:
    experiment_spec_path = ctx.experiment_dir / "experiment_spec.json"
    coverage_summary_path = ctx.experiment_dir / "coverage_summary.json"
    decision_path = ctx.experiment_dir / "decision.json"
    numeric_justification_path = ctx.experiment_dir / "numeric_justification.json"
    write_json(experiment_spec_path, experiment_spec)
    write_json(coverage_summary_path, coverage_summary)
    write_json(decision_path, decision)
    write_numeric_justification(numeric_justification_path, numeric_justification)
    manifest_payload = read_json(ctx.run_context.manifest_path(), default={})
    if not isinstance(manifest_payload, dict):
        manifest_payload = {}
    incidence_runs = dict(manifest_payload.get("incidence_research", {}))
    incidence_runs[ctx.experiment.experiment_id] = {
        "experiment_dir": str(ctx.experiment_dir),
        "source_run_id": ctx.source_run_id,
        "source_run_dir": str(ctx.source_run_dir),
        "harp_archive_dir": str(ctx.harp_archive_dir),
        "phase0_dir": str(ctx.phase0_dir) if ctx.phase0_dir is not None else None,
        "phase1_dir": str(ctx.phase1_dir) if ctx.phase1_dir is not None else None,
        "artifacts": {
            "experiment_spec": str(experiment_spec_path),
            "coverage_summary": str(coverage_summary_path),
            "decision": str(decision_path),
            "numeric_justification": str(numeric_justification_path),
        },
    }
    manifest_payload["incidence_research"] = incidence_runs
    write_json(ctx.run_context.manifest_path(), manifest_payload)
    return {
        "experiment_spec": str(experiment_spec_path),
        "coverage_summary": str(coverage_summary_path),
        "decision": str(decision_path),
        "numeric_justification": str(numeric_justification_path),
    }
