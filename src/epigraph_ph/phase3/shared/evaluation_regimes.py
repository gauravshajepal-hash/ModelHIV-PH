from __future__ import annotations

from pathlib import Path
from typing import Any

from epigraph_ph.runtime import read_json


BROAD_FROZEN_HISTORY_REGIME = "broad_frozen_history_rescue_core"
TRANSITION_RESEARCH_REGIME = "quarter_level_transition_research"
INCIDENCE_LOCKED_REGIME = "diagnosis_locked_incidence_branch"


def broad_frozen_tournament_artifact_path(run_dir: Path) -> Path:
    return run_dir / "phase3_frozen_backtest_tournament" / "representation_tournament.json"


def load_broad_frozen_tournament_payload(run_dir: Path) -> dict[str, Any]:
    return read_json(broad_frozen_tournament_artifact_path(run_dir), default={})


def load_broad_representation_trial_rows(run_dir: Path, representation_mode: str) -> list[dict[str, Any]]:
    payload = load_broad_frozen_tournament_payload(run_dir)
    return [
        dict(row)
        for row in list(payload.get("trial_rows") or [])
        if str(row.get("representation") or "") == str(representation_mode or "")
    ]


def resolve_broad_phase3_result_dir_name(source_run_dir: Path, explicit_name: str | None = None) -> str | None:
    if explicit_name:
        return explicit_name
    payload = load_broad_frozen_tournament_payload(source_run_dir)
    summary = dict(payload.get("summary") or {})
    winner_representation = str(summary.get("winner_representation") or "").strip()
    if winner_representation:
        candidate = f"phase3_frozen_backtest_{winner_representation}"
        if (source_run_dir / candidate).exists():
            return candidate
    fallback = "phase3_frozen_backtest"
    if (source_run_dir / fallback).exists():
        return fallback
    return None
