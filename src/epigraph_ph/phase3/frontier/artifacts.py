from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ROOT_DIR, RunContext, ensure_dir, read_json, write_json

from ..shared.evaluation_regimes import resolve_broad_phase3_result_dir_name
from .numeric_policy import write_numeric_justification
from .registry import ExperimentDefinition, get_experiment_definition


@dataclass(slots=True)
class TransitionResearchContext:
    experiment: ExperimentDefinition
    run_context: RunContext
    experiment_dir: Path
    source_run_id: str
    source_run_dir: Path
    phase15_dir: Path
    phase2_dir: Path
    phase3_dir: Path | None


def _resolve_phase3_result_dir_name(source_run_dir: Path, explicit_name: str | None = None) -> str | None:
    return resolve_broad_phase3_result_dir_name(source_run_dir, explicit_name=explicit_name)


def build_transition_research_context(
    *,
    run_id: str,
    plugin_id: str,
    experiment_id: str,
    source_run_id: str,
    phase3_result_dir_name: str | None = None,
) -> TransitionResearchContext:
    experiment = get_experiment_definition(experiment_id)
    run_context = RunContext.create(run_id=run_id, plugin_id=plugin_id)
    experiment_dir = ensure_dir(run_context.run_dir / "transition_research" / experiment_id)
    source_run_dir = ROOT_DIR / "artifacts" / "runs" / source_run_id
    if not source_run_dir.exists():
        raise FileNotFoundError(f"source run directory does not exist: {source_run_dir}")
    phase15_dir = source_run_dir / "phase15"
    phase2_dir = source_run_dir / "phase2"
    if not phase15_dir.exists():
        raise FileNotFoundError(f"phase15 directory does not exist: {phase15_dir}")
    if not phase2_dir.exists():
        raise FileNotFoundError(f"phase2 directory does not exist: {phase2_dir}")
    resolved_phase3_name = _resolve_phase3_result_dir_name(source_run_dir, explicit_name=phase3_result_dir_name)
    phase3_dir = (source_run_dir / resolved_phase3_name) if resolved_phase3_name else None
    if phase3_dir is not None and not phase3_dir.exists():
        phase3_dir = None
    return TransitionResearchContext(
        experiment=experiment,
        run_context=run_context,
        experiment_dir=experiment_dir,
        source_run_id=source_run_id,
        source_run_dir=source_run_dir,
        phase15_dir=phase15_dir,
        phase2_dir=phase2_dir,
        phase3_dir=phase3_dir,
    )


def write_experiment_artifacts(
    *,
    ctx: TransitionResearchContext,
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
    transition_runs = dict(manifest_payload.get("transition_research", {}))
    transition_runs[ctx.experiment.experiment_id] = {
        "experiment_dir": str(ctx.experiment_dir),
        "source_run_id": ctx.source_run_id,
        "phase15_dir": str(ctx.phase15_dir),
        "phase2_dir": str(ctx.phase2_dir),
        "phase3_dir": str(ctx.phase3_dir) if ctx.phase3_dir is not None else None,
        "artifacts": {
            "experiment_spec": str(experiment_spec_path),
            "coverage_summary": str(coverage_summary_path),
            "decision": str(decision_path),
            "numeric_justification": str(numeric_justification_path),
        },
    }
    manifest_payload["transition_research"] = transition_runs
    write_json(ctx.run_context.manifest_path(), manifest_payload)
    return {
        "experiment_spec": str(experiment_spec_path),
        "coverage_summary": str(coverage_summary_path),
        "decision": str(decision_path),
        "numeric_justification": str(numeric_justification_path),
    }
