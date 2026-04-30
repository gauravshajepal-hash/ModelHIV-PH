from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .runtime import ensure_dir, read_json, write_json


AUTORESEARCH_EXPERIMENT_RE = re.compile(
    r"^### (?P<experiment>[A-Za-z0-9][A-Za-z0-9_-]*)\s*$",
    flags=re.MULTILINE,
)


def _utc_from_timestamp(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _path_times(path: Path) -> dict[str, str | None]:
    if not path.exists():
        return {"modified_at": None, "metadata_changed_at": None, "birth_at": None}
    stat = path.stat()
    birth = getattr(stat, "st_birthtime", None)
    return {
        "modified_at": _utc_from_timestamp(float(stat.st_mtime)),
        "metadata_changed_at": _utc_from_timestamp(float(stat.st_ctime)),
        "birth_at": _utc_from_timestamp(float(birth)) if birth is not None else None,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _first_dict_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _load_json_reports(run_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    reports: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((run_dir / "analysis").glob("*.json")):
        payload = read_json(path, default=None)
        if isinstance(payload, dict):
            reports.append((path, payload))
    return reports


def _choose_primary_report(reports: list[tuple[Path, dict[str, Any]]]) -> tuple[Path, dict[str, Any]] | None:
    if not reports:
        return None
    for path, payload in reports:
        if path.name.endswith("_report.json") and (
            "best_candidate" in payload or "family_name" in payload or "decision" in payload
        ):
            return path, payload
    return reports[0]


def _extract_score(payload: dict[str, Any]) -> dict[str, Any]:
    best_candidate = payload.get("best_candidate")
    if not isinstance(best_candidate, dict):
        return {}
    score = best_candidate.get("score")
    if not isinstance(score, dict):
        return {}
    return {
        "candidate_mean_mae": _first_dict_value(score, ("candidate_mean_mae", "dynamic_mean_mae")),
        "candidate_worst_mae": _first_dict_value(score, ("candidate_worst_mae", "dynamic_worst_mae")),
        "carry_forward_mean_mae": score.get("carry_forward_mean_mae"),
        "carry_forward_worst_mae": score.get("carry_forward_worst_mae"),
        "candidate_mean_smape": _first_dict_value(score, ("candidate_mean_smape", "dynamic_mean_smape")),
        "carry_forward_mean_smape": score.get("carry_forward_mean_smape"),
    }


def _extract_gate(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("hidden_channel_gate", "tr_v3_04b_gate", "tr_v3_04c_gate", "tr_v3_04_gate"):
        gate = payload.get(key)
        if isinstance(gate, dict):
            return {"gate_name": key, **gate}
    return {}


def _claim_promotion_review(payload: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    model_contract = payload.get("model_contract")
    hazard_semantics = payload.get("hazard_semantics")
    benchmark_contract = payload.get("benchmark_contract")
    data_provenance = payload.get("data_provenance")
    observation_score_ledger_summary = payload.get("observation_score_ledger_summary")
    phase2_context = payload.get("phase2_context")
    phase2_prior_contract = payload.get("phase2_prior_contract")
    claim_card = payload.get("claim_card")
    blockers: list[str] = []
    warnings: list[str] = []
    decision = str(payload.get("decision") or "")
    non_model_stage = decision in {"contract_only", "diagnostic_only", "not_identifiable", "auxiliary_only"}
    if non_model_stage:
        blockers.append("non_model_or_diagnostic_stage")
    if not non_model_stage:
        if not isinstance(model_contract, dict):
            blockers.append("missing_model_contract")
        if not isinstance(hazard_semantics, dict):
            blockers.append("missing_hazard_semantics")
        if not isinstance(benchmark_contract, dict):
            blockers.append("missing_benchmark_contract")
        if not isinstance(data_provenance, dict):
            blockers.append("missing_data_provenance")
        if not isinstance(observation_score_ledger_summary, dict):
            blockers.append("missing_observation_score_ledger_summary")
        elif int(observation_score_ledger_summary.get("entry_count") or 0) <= 0:
            blockers.append("empty_observation_score_ledger")
    candidate_mean_mae = score.get("candidate_mean_mae")
    candidate_worst_mae = score.get("candidate_worst_mae")
    carry_forward_mean_mae = score.get("carry_forward_mean_mae")
    carry_forward_worst_mae = score.get("carry_forward_worst_mae")
    if not non_model_stage and (candidate_mean_mae is None or candidate_worst_mae is None):
        blockers.append("missing_primary_score")
    if not non_model_stage and (carry_forward_mean_mae is None or carry_forward_worst_mae is None):
        blockers.append("missing_carry_forward_score")
    if (
        not non_model_stage
        and
        candidate_mean_mae is not None
        and carry_forward_mean_mae is not None
        and math.isfinite(float(candidate_mean_mae))
        and math.isfinite(float(carry_forward_mean_mae))
        and float(candidate_mean_mae) > float(carry_forward_mean_mae)
    ):
        blockers.append("fails_carry_forward_mean_gate")
    if (
        not non_model_stage
        and
        candidate_worst_mae is not None
        and carry_forward_worst_mae is not None
        and math.isfinite(float(candidate_worst_mae))
        and math.isfinite(float(carry_forward_worst_mae))
        and float(candidate_worst_mae) > float(carry_forward_worst_mae)
    ):
        blockers.append("fails_carry_forward_worst_gate")
    if isinstance(phase2_context, dict):
        direct_count = int(phase2_context.get("direct_prior_feature_count") or 0)
        hidden_count = int(phase2_context.get("hidden_driver_feature_count") or 0)
        if (direct_count > 0 or hidden_count > 0) and not isinstance(phase2_prior_contract, dict):
            blockers.append("missing_phase2_prior_contract")
    if isinstance(hazard_semantics, dict):
        diagnostic_status = str(((hazard_semantics.get("diagnostic_derived_hazard") or {}).get("status")) or "")
        if diagnostic_status == "emitted":
            warnings.append("diagnostic_derived_hazard_not_mechanistic")
    if "lockbox_contract" not in payload:
        warnings.append("no_final_lockbox_contract")
    if "raw_endpoint_error" not in payload and "raw_endpoint_errors" not in payload:
        warnings.append("missing_raw_endpoint_error_report")
    if "archive_drift_gate" not in payload:
        warnings.append("missing_archive_drift_gate")
    if isinstance(claim_card, dict):
        if claim_card.get("promotion_eligible") is False:
            blockers.extend(
                [
                    str(value)
                    for value in list(claim_card.get("blockers") or [])
                    if str(value) not in blockers
                ]
            )
    promotion_eligible = not blockers and payload.get("decision") == "keep"
    return {
        "schema_version": "phase3_dynamic_claim_promotion_review.v1",
        "promotion_eligible": bool(promotion_eligible),
        "publication_eligible": False,
        "promotion_status": "eligible_development_champion" if promotion_eligible else "quarantined_or_diagnostic",
        "blockers": blockers,
        "warnings": warnings,
        "required_before_publication": [
            "final_lockbox_contract",
            "raw_endpoint_error_report",
            "residual_p90_report",
            "archive_drift_gate",
            "support_tier_stability_gate",
        ],
    }


def _infer_family_name(run_id: str, payload: dict[str, Any]) -> str | None:
    family_name = payload.get("family_name")
    if isinstance(family_name, str) and family_name:
        return family_name
    match = re.search(r"(tr-v3-[0-9]{2}[a-z]?)", run_id, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def _classify_run(run_id: str, payload: dict[str, Any]) -> str:
    if payload.get("benchmark_kind") == "early_history_partial_observation" or "early-history" in run_id:
        return "early_history_partial"
    if "family-comparison" in run_id:
        return "family_comparison"
    if "provenance" in run_id:
        return "provenance_diagnostic"
    if "sandbox" in run_id:
        return "blocked_time_autoresearch"
    return "artifact_run"


def _status_from_payload(run_id: str, payload: dict[str, Any]) -> str:
    decision = payload.get("decision")
    if isinstance(decision, str) and decision:
        return decision
    if payload.get("benchmark_kind") == "early_history_partial_observation" or "early-history" in run_id:
        return "diagnostic"
    if "family-comparison" in run_id:
        return "comparison"
    return "unknown"


def _collect_run_record(run_dir: Path, sandbox_root: Path) -> dict[str, Any]:
    run_id = run_dir.name
    json_reports = _load_json_reports(run_dir)
    primary = _choose_primary_report(json_reports)
    primary_path: Path | None = primary[0] if primary else None
    primary_payload: dict[str, Any] = primary[1] if primary else {}
    markdown_reports = sorted((run_dir / "analysis").glob("*.md"))
    report_files = [path for path, _ in json_reports] + markdown_reports
    report_times = [_path_times(path) for path in report_files]
    modified_values = [entry["modified_at"] for entry in report_times if entry["modified_at"]]
    score = _extract_score(primary_payload)
    gate = _extract_gate(primary_payload)
    claim_review = _claim_promotion_review(primary_payload, score)
    best_candidate = primary_payload.get("best_candidate")
    best_config = best_candidate.get("config") if isinstance(best_candidate, dict) else None
    return {
        "run_id": run_id,
        "run_kind": _classify_run(run_id, primary_payload),
        "status": _status_from_payload(run_id, primary_payload),
        "decision": primary_payload.get("decision"),
        "decision_reason": primary_payload.get("decision_reason"),
        "family_name": _infer_family_name(run_id, primary_payload),
        "generated_at": primary_payload.get("generated_at"),
        "source_run_id": primary_payload.get("source_run_id"),
        "loop_variant": primary_payload.get("loop_variant"),
        "benchmark_contract": primary_payload.get("benchmark_contract"),
        "benchmark_kind": primary_payload.get("benchmark_kind"),
        "split_count": primary_payload.get("split_count"),
        "candidate_count": primary_payload.get("candidate_count"),
        "phase2_context": primary_payload.get("phase2_context"),
        "model_contract": primary_payload.get("model_contract"),
        "hazard_semantics": primary_payload.get("hazard_semantics"),
        "phase2_prior_contract": primary_payload.get("phase2_prior_contract"),
        "observation_score_ledger_summary": primary_payload.get("observation_score_ledger_summary"),
        "score": score,
        "gate": gate,
        "claim_promotion_review": claim_review,
        "best_config": best_config,
        "filesystem": {
            "run_dir": _path_times(run_dir),
            "first_report_modified_at": min(modified_values) if modified_values else None,
            "last_report_modified_at": max(modified_values) if modified_values else None,
        },
        "paths": {
            "run_dir": _relative(run_dir, sandbox_root),
            "primary_json_report": _relative(primary_path, sandbox_root) if primary_path else None,
            "markdown_reports": [_relative(path, sandbox_root) for path in markdown_reports],
            "json_reports": [_relative(path, sandbox_root) for path, _ in json_reports],
        },
    }


def collect_artifact_runs(sandbox_root: Path) -> list[dict[str, Any]]:
    runs_dir = sandbox_root / "artifacts" / "runs"
    if not runs_dir.exists():
        return []
    records = [
        _collect_run_record(path, sandbox_root)
        for path in sorted(runs_dir.iterdir())
        if path.is_dir() and path.name != ".gitkeep"
    ]
    def _sort_time(record: dict[str, Any]) -> str:
        return str(
            record.get("generated_at")
            or record.get("filesystem", {}).get("first_report_modified_at")
            or ""
        )

    return sorted(records, key=lambda record: (_sort_time(record), str(record.get("run_id") or "")))


def _extract_section(markdown: str, start: int, end: int) -> str:
    return markdown[start:end].strip()


def _section_field(section: str, heading: str) -> str | None:
    marker = f"{heading}:\n"
    start = section.find(marker)
    if start < 0:
        return None
    rest = section[start + len(marker) :]
    next_heading = re.search(r"\n[A-Z][A-Za-z -]+:\n", rest)
    value = rest[: next_heading.start()] if next_heading else rest
    lines = [line.strip() for line in value.strip().splitlines()]
    return "\n".join(line for line in lines if line)


def _first_bullet_text(value: str | None) -> str | None:
    if not value:
        return None
    for line in value.splitlines():
        if line.startswith("- "):
            return line[2:].strip()
    return value.splitlines()[0].strip() if value.splitlines() else None


def parse_declared_experiments(autoresearch_path: Path) -> list[dict[str, Any]]:
    if not autoresearch_path.exists():
        return []
    markdown = autoresearch_path.read_text(encoding="utf-8")
    matches = list(AUTORESEARCH_EXPERIMENT_RE.finditer(markdown))
    records: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        section = _extract_section(markdown, start, end)
        objective = _first_bullet_text(_section_field(section, "Objective"))
        status = _first_bullet_text(_section_field(section, "Current implementation status"))
        model = _section_field(section, "Model")
        contract = _section_field(section, "Contract")
        records.append(
            {
                "experiment_id": match.group("experiment"),
                "objective": objective,
                "implementation_status": status,
                "model_or_contract": model or contract,
            }
        )
    return records


def _float_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown_link(path: str | None) -> str:
    if not path:
        return ""
    return f"[report]({path})"


def render_experiment_registry_markdown(registry: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Phase3(dynamic) Experiment Registry",
        "",
        f"- Generated at: `{registry['generated_at']}`",
        f"- Sandbox root: `{registry['sandbox_root']}`",
        f"- Artifact runs tracked: `{len(registry['artifact_runs'])}`",
        f"- Declared experiments tracked: `{len(registry['declared_experiments'])}`",
        "",
        "## Evidence Contract",
        "",
        "- JSON reports are the primary source for decisions, scores, and model metadata.",
        "- Markdown reports are retained as human-readable evidence links.",
        "- Filesystem timestamps are audit hints only; copied files can preserve older modification times.",
        "- `metadata_changed_at` is not a scientific timestamp; use `generated_at` when present.",
        "- `champion_by_kept_mean_mae` is retained as a legacy diagnostic only.",
        "- `champion_by_claim_aware_promotion` is the only registry field allowed to support new champion language.",
        "",
        "## Artifact Run Timeline",
        "",
        "| Run | Family | Kind | Status | Claim Promotion | Generated | First Report Modified | Mean MAE | Worst MAE | Report |",
        "|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    for record in registry["artifact_runs"]:
        score = record.get("score") or {}
        fs = record.get("filesystem") or {}
        paths = record.get("paths") or {}
        claim_review = record.get("claim_promotion_review") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record.get('run_id') or ''}`",
                    f"`{record.get('family_name') or ''}`",
                    f"`{record.get('run_kind') or ''}`",
                    f"`{record.get('status') or ''}`",
                    f"`{claim_review.get('promotion_status') or ''}`",
                    f"`{record.get('generated_at') or ''}`",
                    f"`{fs.get('first_report_modified_at') or ''}`",
                    _float_text(score.get("candidate_mean_mae")),
                    _float_text(score.get("candidate_worst_mae")),
                    _markdown_link(paths.get("primary_json_report")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Declared Experiment Ladder",
            "",
            "| Experiment | Implementation Status | Objective |",
            "|---|---|---|",
        ]
    )
    for record in registry["declared_experiments"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{record.get('experiment_id') or ''}`",
                    str(record.get("implementation_status") or ""),
                    str(record.get("objective") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def build_experiment_registry(sandbox_root: Path) -> dict[str, Any]:
    sandbox_root = sandbox_root.resolve()
    artifact_runs = collect_artifact_runs(sandbox_root)
    declared_experiments = parse_declared_experiments(sandbox_root / "AUTORESEARCH.md")
    kept = [
        record
        for record in artifact_runs
        if record.get("status") == "keep" and record.get("score", {}).get("candidate_mean_mae") is not None
    ]
    champion = min(kept, key=lambda record: float(record["score"]["candidate_mean_mae"])) if kept else None
    claim_eligible = [
        record
        for record in kept
        if (record.get("claim_promotion_review") or {}).get("promotion_eligible") is True
    ]
    claim_aware_champion = min(
        claim_eligible,
        key=lambda record: (
            float(record["score"]["candidate_mean_mae"]),
            float(record["score"]["candidate_worst_mae"]),
        ),
    ) if claim_eligible else None
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": "phase3_dynamic_experiment_registry.v2",
        "sandbox_root": sandbox_root.as_posix(),
        "champion_by_kept_mean_mae": champion,
        "champion_by_claim_aware_promotion": claim_aware_champion,
        "champion_selection_contract": {
            "legacy_mean_mae_field": "diagnostic_only",
            "claim_aware_field": "authoritative_for_new_champion_claims",
            "publication_eligibility": "requires lockbox, raw endpoint, residual p90, drift, and support-tier gates",
        },
        "artifact_runs": artifact_runs,
        "declared_experiments": declared_experiments,
    }


def write_experiment_registry(*, sandbox_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    registry = build_experiment_registry(sandbox_root)
    target_dir = output_dir or sandbox_root / "artifacts" / "experiment_tracking"
    ensure_dir(target_dir)
    write_json(target_dir / "experiment_registry.json", registry)
    (target_dir / "experiment_registry.md").write_text(
        render_experiment_registry_markdown(registry),
        encoding="utf-8",
    )
    return registry
