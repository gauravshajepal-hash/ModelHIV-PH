from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from epigraph_ph.phase0.official_determinant_bridge import build_official_determinant_bridge_rows
from epigraph_ph.phase0.phase3_target_contract import PHASE3_MODULE_CONTRACT
from epigraph_ph.phase0.pipeline import _phase0_alignment_bundle
from epigraph_ph.phase1.pipeline import run_phase1_build
from epigraph_ph.phase2.pipeline import run_phase2_build
from epigraph_ph.runtime import RunContext, ensure_dir, read_json, utc_now_iso, write_json


OFFICIAL_AUGMENTED_BASELINE_SCHEMA_VERSION = "phase2_official_augmented_baseline.v1"

INCIDENCE_ADJACENT_MODULES: tuple[str, ...] = (
    "incidence_pressure",
    "diagnosis_delay",
)

BLOCKED_CANONICAL_NAMES: frozenset[str] = frozenset(
    {
        "advanced_hiv_disease_share",
        "case_report_timeliness",
        "late_hiv_diagnosis_percent",
        "median_cd4_at_diagnosis",
        "median_cd4_at_enrollment",
        "registry_backlog",
        "reporting_delay",
        "surveillance_completeness",
    }
)

BLOCKED_METRIC_TOKENS: tuple[str, ...] = (
    "new_hiv_infections",
    "hiv_incidence",
    "incidence_prevalence",
    "incidence_mortality",
)


def _safe_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _module_canonical_names(module_ids: tuple[str, ...]) -> set[str]:
    names: set[str] = set()
    for module_id in module_ids:
        names.update(str(value) for value in list((PHASE3_MODULE_CONTRACT.get(module_id) or {}).get("canonical_names") or []))
    return names


def _source_family(row: Mapping[str, Any]) -> str:
    platform = _safe_text(row.get("platform")).lower()
    source_bank = _safe_text(row.get("source_bank")).lower()
    source_id = _safe_text(row.get("source_id")).lower()
    if platform:
        return platform
    if "unaids" in source_id:
        return "unaids"
    if "world_bank" in source_id or "wdi" in source_id:
        return "world_bank_wdi"
    return source_bank or "unknown"


def _is_incidence_safe_official_row(row: Mapping[str, Any], allowed_names: set[str]) -> bool:
    canonical = _safe_text(row.get("canonical_name"))
    if canonical not in allowed_names or canonical in BLOCKED_CANONICAL_NAMES:
        return False
    text = " ".join(
        _safe_text(row.get(key)).lower()
        for key in ("canonical_name", "source_metric_name", "source_id", "parameter_text", "evidence_span")
    )
    if any(token in text for token in BLOCKED_METRIC_TOKENS):
        return False
    value = row.get("value")
    if value in ("", None):
        return False
    return True


def _source_manifest_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = _safe_text(row.get("source_id"))
        if not source_id or source_id in by_id:
            continue
        by_id[source_id] = {
            "source_id": source_id,
            "label": _safe_text(row.get("source_title") or row.get("source_label") or source_id),
            "source_kind": "official_determinant_bridge_candidate",
            "platform": _safe_text(row.get("platform")) or _source_family(row),
            "source_tier": _safe_text(row.get("source_tier")),
            "source_url": _safe_text(row.get("source_url")),
            "source_family": _source_family(row),
        }
    return [by_id[key] for key in sorted(by_id)]


def build_official_augmented_baseline(
    *,
    source_run_id: str,
    output_run_id: str,
    plugin_id: str = "hiv",
    profile: str = "hiv_rescue_v2",
    module_ids: tuple[str, ...] = INCIDENCE_ADJACENT_MODULES,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_ctx = RunContext.create(run_id=source_run_id, plugin_id=plugin_id)
    output_ctx = RunContext.create(run_id=output_run_id, plugin_id=plugin_id)
    source_run_dir = source_ctx.run_dir
    output_run_dir = output_ctx.run_dir
    if output_run_dir.exists() and any(output_run_dir.iterdir()):
        if not overwrite:
            manifest_path = output_run_dir / "phase2" / "official_augmented_baseline_manifest.json"
            if manifest_path.exists():
                return dict(read_json(manifest_path, default={}) or {})
            raise FileExistsError(f"Output run already exists and is not an official augmented run: {output_run_dir}")
        shutil.rmtree(output_run_dir)
    ensure_dir(output_run_dir)

    if (source_run_dir / "harp_archive").exists():
        shutil.copytree(source_run_dir / "harp_archive", output_run_dir / "harp_archive")

    source_phase0 = source_run_dir / "phase0"
    target_phase0 = ensure_dir(output_run_dir / "phase0")
    target_raw = ensure_dir(target_phase0 / "raw")
    target_extracted = ensure_dir(target_phase0 / "extracted")
    source_manifest = list(read_json(source_phase0 / "raw" / "source_manifest.json", default=[]) or [])
    base_candidates = list(read_json(source_phase0 / "extracted" / "canonical_parameter_candidates.json", default=[]) or [])
    if not base_candidates:
        raise FileNotFoundError(f"Missing baseline canonical candidates under {source_phase0 / 'extracted'}")

    official_dir = source_run_dir / "phase0" / "evidence_ledger"
    official_rows_path = official_dir / "official_determinant_candidate_rows.json"
    if not official_rows_path.exists():
        build_official_determinant_bridge_rows(run_dir=source_run_dir, out_dir=official_dir)
    official_rows = list(read_json(official_rows_path, default=[]) or [])
    allowed_names = _module_canonical_names(module_ids)
    selected_official_rows = [
        dict(row)
        for row in official_rows
        if _is_incidence_safe_official_row(dict(row), allowed_names)
    ]
    by_candidate_id: dict[str, dict[str, Any]] = {}
    for row in base_candidates:
        candidate_id = _safe_text((row or {}).get("candidate_id") or (row or {}).get("subparameter_id"))
        if candidate_id:
            by_candidate_id[candidate_id] = dict(row)
    added = []
    for row in selected_official_rows:
        candidate = dict(row)
        candidate["source_run_id"] = source_run_id
        candidate["augmentation_role"] = "incidence_safe_official_determinant_candidate"
        candidate["allowed_phase3_use"] = "incidence_covariate_candidate_not_training_target"
        candidate_id = _safe_text(candidate.get("candidate_id") or candidate.get("subparameter_id"))
        if not candidate_id or candidate_id in by_candidate_id:
            continue
        by_candidate_id[candidate_id] = candidate
        added.append(candidate)
    augmented_candidates = list(by_candidate_id.values())
    source_rows_by_id = {
        _safe_text(row.get("source_id")): dict(row)
        for row in source_manifest
        if _safe_text(row.get("source_id"))
    }
    for row in _source_manifest_rows(added):
        source_rows_by_id.setdefault(_safe_text(row.get("source_id")), row)
    augmented_source_manifest = [source_rows_by_id[key] for key in sorted(source_rows_by_id)]

    write_json(target_raw / "source_manifest.json", augmented_source_manifest)
    write_json(target_extracted / "canonical_parameter_candidates.json", augmented_candidates)
    alignment_paths = _phase0_alignment_bundle(
        candidate_rows=augmented_candidates,
        source_rows=source_rows_by_id,
        plugin_id=plugin_id,
        artifact_dir=target_extracted,
    )
    phase0_manifest = {
        "schema_version": OFFICIAL_AUGMENTED_BASELINE_SCHEMA_VERSION,
        "stage_status": {"official_augmented_phase0": "completed"},
        "source_run_id": source_run_id,
        "output_run_id": output_run_id,
        "module_ids": list(module_ids),
        "base_candidate_row_count": len(base_candidates),
        "official_candidate_row_count": len(official_rows),
        "selected_official_candidate_row_count": len(selected_official_rows),
        "added_official_candidate_row_count": len(added),
        "augmented_candidate_row_count": len(augmented_candidates),
        "artifact_paths": {
            "source_manifest": str(target_raw / "source_manifest.json"),
            "canonical_parameter_candidates": str(target_extracted / "canonical_parameter_candidates.json"),
            **{str(key): str(value) for key, value in alignment_paths.items()},
        },
    }
    write_json(target_phase0 / "phase0_manifest.json", phase0_manifest)
    phase1 = run_phase1_build(run_id=output_run_id, plugin_id=plugin_id, profile=profile)
    if (source_run_dir / "phase15").exists():
        if (output_run_dir / "phase15").exists():
            shutil.rmtree(output_run_dir / "phase15")
        shutil.copytree(source_run_dir / "phase15", output_run_dir / "phase15")
        phase15_status = {"phase15": "reused_from_source_run"}
    else:
        phase15_status = {"phase15": "missing_from_source_run"}
    phase2 = run_phase2_build(run_id=output_run_id, plugin_id=plugin_id, profile=profile)
    by_canonical = Counter(_safe_text(row.get("canonical_name")) for row in added)
    by_family = Counter(_source_family(row) for row in added)
    manifest = {
        "schema_version": OFFICIAL_AUGMENTED_BASELINE_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "output_run_id": output_run_id,
        "module_ids": list(module_ids),
        "base_candidate_row_count": len(base_candidates),
        "official_candidate_row_count": len(official_rows),
        "selected_official_candidate_row_count": len(selected_official_rows),
        "added_official_candidate_row_count": len(added),
        "augmented_candidate_row_count": len(augmented_candidates),
        "added_by_canonical_name": dict(sorted(by_canonical.items())),
        "added_by_source_family": dict(sorted(by_family.items())),
        "blocked_canonical_names": sorted(BLOCKED_CANONICAL_NAMES),
        "blocked_metric_tokens": list(BLOCKED_METRIC_TOKENS),
        "phase1_stage_status": dict(phase1.get("stage_status") or {}),
        "phase15_stage_status": phase15_status,
        "phase2_stage_status": dict(phase2.get("stage_status") or {}),
        "artifact_paths": {
            "source_manifest": str(target_raw / "source_manifest.json"),
            "canonical_parameter_candidates": str(target_extracted / "canonical_parameter_candidates.json"),
            **{str(key): str(value) for key, value in alignment_paths.items()},
            "phase2_structural_payload": str(output_run_dir / "phase2" / "phase2_structural_payload.json"),
        },
        "contract": {
            "purpose": "Make official incidence-adjacent determinant rows visible to Phase1/Phase2 without promoting incidence observations as covariates.",
            "target_modules": list(module_ids),
            "incidence_safety_rule": "exclude direct incidence metrics, late-diagnosis outcome constraints, reporting outcomes, and CD4/AHD outcome rows.",
        },
    }
    manifest_path = ensure_dir(output_run_dir / "phase2") / "official_augmented_baseline_manifest.json"
    write_json(manifest_path, manifest)
    return manifest


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build an official-determinant augmented Phase2 baseline run.")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--plugin", default="hiv")
    parser.add_argument("--profile", default="hiv_rescue_v2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = build_official_augmented_baseline(
        source_run_id=args.source_run_id,
        output_run_id=args.output_run_id,
        plugin_id=args.plugin,
        profile=args.profile,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(manifest.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
