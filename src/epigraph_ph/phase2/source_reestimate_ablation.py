from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from epigraph_ph.phase0.pipeline import _phase0_alignment_bundle
from epigraph_ph.phase1.pipeline import run_phase1_build
from epigraph_ph.phase2.pipeline import run_phase2_build
from epigraph_ph.runtime import RunContext, ensure_dir, read_json, utc_now_iso, write_json


SOURCE_REESTIMATION_SCHEMA_VERSION = "phase2_source_family_reestimated_ablation.v1"
DEFAULT_SOURCE_FAMILIES: tuple[str, ...] = (
    "psa",
    "world_bank_wdi",
    "google_mobility",
    "philhealth",
    "philhealth_open_portal",
    "fies",
    "unaids",
)


def infer_source_family(row: Mapping[str, Any]) -> str:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "source_family",
            "platform",
            "source_id",
            "document_id",
            "source_bank",
            "source_tier",
            "extraction_method",
        )
    ).lower()
    if "philhealth_open_portal" in text:
        return "philhealth_open_portal"
    if "philhealth" in text:
        return "philhealth"
    if "google_mobility" in text or "google_community_mobility" in text:
        return "google_mobility"
    if "world_bank_wdi" in text or "worldbank" in text or "wdi_" in text:
        return "world_bank_wdi"
    if "unaids" in text or "aidsinfo" in text:
        return "unaids"
    if "psa-" in text or "philippine_statistics_authority" in text or "psa_" in text:
        return "psa"
    if "fies" in text or "family income and expenditure" in text:
        return "fies"
    if "yafs" in text:
        return "yafs"
    if "doh_harp_hasp" in text or "harp" in text or "hasp" in text:
        return "doh_harp_hasp"
    if "openalex" in text:
        return "openalex"
    if "crossref" in text:
        return "crossref"
    if "pubmed" in text:
        return "pubmed"
    if "biorxiv" in text:
        return "biorxiv"
    if "arxiv" in text:
        return "arxiv"
    platform = str(row.get("platform") or "").strip().lower()
    if platform:
        return platform
    source_bank = str(row.get("source_bank") or "").strip().lower()
    return source_bank or "unknown"


def _edge_key(row: Mapping[str, Any], *, kind: str) -> str:
    return f"{kind}:{row.get('source')}->{row.get('target')}:lag{int(row.get('lag') or 0)}"


def _edge_sign(value: Any) -> int:
    try:
        weight = float(value)
    except Exception:
        return 0
    if weight > 0.0:
        return 1
    if weight < 0.0:
        return -1
    return 0


def _edge_rows(phase2_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for kind, key in (("direct", "direct_temporal_edge_rows"), ("hidden", "hidden_driver_rows")):
        for row in list(phase2_payload.get(key) or []):
            edge = dict(row)
            edge["edge_kind"] = kind
            rows[_edge_key(edge, kind=kind)] = edge
    return rows


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        return
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)


def _prepare_phase0_candidate_ablation(
    *,
    source_run_dir: Path,
    ablation_run_dir: Path,
    excluded_family: str,
    plugin_id: str,
) -> dict[str, Any]:
    source_phase0 = source_run_dir / "phase0"
    target_phase0 = ensure_dir(ablation_run_dir / "phase0")
    target_raw = ensure_dir(target_phase0 / "raw")
    target_extracted = ensure_dir(target_phase0 / "extracted")

    source_manifest = list(read_json(source_phase0 / "raw" / "source_manifest.json", default=[]) or [])
    source_rows = {
        str(row.get("source_id") or ""): dict(row)
        for row in source_manifest
        if str(row.get("source_id") or "")
    }
    candidates = list(read_json(source_phase0 / "extracted" / "canonical_parameter_candidates.json", default=[]) or [])
    if not candidates:
        raise FileNotFoundError(f"No canonical_parameter_candidates.json in {source_phase0 / 'extracted'}")

    family_counts = Counter(infer_source_family(row) for row in candidates)
    filtered_candidates = [dict(row) for row in candidates if infer_source_family(row) != excluded_family]
    retained_source_ids = {str(row.get("source_id") or "") for row in filtered_candidates if str(row.get("source_id") or "")}
    filtered_source_manifest = [
        dict(row)
        for row in source_manifest
        if not str(row.get("source_id") or "") or str(row.get("source_id") or "") in retained_source_ids
    ]
    filtered_source_rows = {
        str(row.get("source_id") or ""): dict(row)
        for row in filtered_source_manifest
        if str(row.get("source_id") or "")
    }

    write_json(target_raw / "source_manifest.json", filtered_source_manifest)
    write_json(target_extracted / "canonical_parameter_candidates.json", filtered_candidates)
    alignment_paths = _phase0_alignment_bundle(
        candidate_rows=filtered_candidates,
        source_rows=filtered_source_rows or source_rows,
        plugin_id=plugin_id,
        artifact_dir=target_extracted,
    )
    manifest = {
        "schema_version": SOURCE_REESTIMATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_dir.name,
        "excluded_source_family": excluded_family,
        "phase0_candidate_row_count_before": len(candidates),
        "phase0_candidate_row_count_after": len(filtered_candidates),
        "removed_candidate_row_count": len(candidates) - len(filtered_candidates),
        "source_family_counts_before": dict(sorted(family_counts.items())),
        "artifact_paths": {
            "source_manifest": str(target_raw / "source_manifest.json"),
            "canonical_parameter_candidates": str(target_extracted / "canonical_parameter_candidates.json"),
            **{str(key): str(value) for key, value in alignment_paths.items()},
        },
    }
    write_json(target_phase0 / "source_family_ablation_manifest.json", manifest)
    write_json(target_phase0 / "phase0_manifest.json", {"stage_status": {"phase0_candidate_ablation": "completed"}, **manifest})
    return manifest


def _run_single_reestimate(
    *,
    source_run_dir: Path,
    ablation_run_id: str,
    excluded_family: str,
    plugin_id: str,
    profile: str,
    overwrite: bool,
) -> dict[str, Any]:
    ctx = RunContext.create(run_id=ablation_run_id, plugin_id=plugin_id)
    ablation_run_dir = ctx.run_dir
    if ablation_run_dir.exists() and any(ablation_run_dir.iterdir()) and not overwrite:
        phase2_payload = ablation_run_dir / "phase2" / "phase2_structural_payload.json"
        phase0_manifest_path = ablation_run_dir / "phase0" / "source_family_ablation_manifest.json"
        if phase2_payload.exists() and phase0_manifest_path.exists():
            return {
                "ablation_run_id": ablation_run_id,
                "excluded_source_family": excluded_family,
                "status": "completed",
                "reason": "reused_existing_run",
                "phase0_manifest": read_json(phase0_manifest_path, default={}) or {},
                "phase1_stage_status": dict(
                    (read_json(ablation_run_dir / "phase1" / "phase1_manifest.json", default={}) or {}).get("stage_status")
                    or {}
                ),
                "phase15_stage_status": dict(
                    (read_json(ablation_run_dir / "phase15" / "phase15_manifest.json", default={}) or {}).get("stage_status")
                    or {}
                ),
                "phase2_stage_status": dict(
                    (read_json(ablation_run_dir / "phase2" / "phase2_manifest.json", default={}) or {}).get("stage_status")
                    or {}
                ),
            }
        return {
            "ablation_run_id": ablation_run_id,
            "excluded_source_family": excluded_family,
            "status": "skipped_existing_run",
            "reason": "pass overwrite=True to rebuild this ablation run",
        }
    if overwrite and ablation_run_dir.exists():
        shutil.rmtree(ablation_run_dir)
        ablation_run_dir.mkdir(parents=True, exist_ok=True)
    ensure_dir(ablation_run_dir)
    _copy_if_exists(source_run_dir / "harp_archive", ablation_run_dir / "harp_archive")
    phase0_manifest = _prepare_phase0_candidate_ablation(
        source_run_dir=source_run_dir,
        ablation_run_dir=ablation_run_dir,
        excluded_family=excluded_family,
        plugin_id=plugin_id,
    )
    if int(phase0_manifest["removed_candidate_row_count"]) <= 0:
        return {
            "ablation_run_id": ablation_run_id,
            "excluded_source_family": excluded_family,
            "status": "skipped_no_rows_removed",
            "phase0_manifest": phase0_manifest,
        }
    phase1 = run_phase1_build(run_id=ablation_run_id, plugin_id=plugin_id, profile=profile)
    if (source_run_dir / "phase15").exists():
        target_phase15 = ablation_run_dir / "phase15"
        if target_phase15.exists():
            shutil.rmtree(target_phase15)
        shutil.copytree(source_run_dir / "phase15", target_phase15)
        phase15_stage_status = {"phase15": "reused_from_source_run"}
    else:
        phase15_stage_status = {"phase15": "missing_from_source_run"}
    phase2 = run_phase2_build(run_id=ablation_run_id, plugin_id=plugin_id, profile=profile)
    return {
        "ablation_run_id": ablation_run_id,
        "excluded_source_family": excluded_family,
        "status": "completed",
        "phase0_manifest": phase0_manifest,
        "phase1_stage_status": dict(phase1.get("stage_status") or {}),
        "phase15_stage_status": phase15_stage_status,
        "phase2_stage_status": dict(phase2.get("stage_status") or {}),
    }


def build_source_reestimated_ablation_report(
    *,
    source_run_id: str,
    output_run_id: str,
    source_families: list[str] | None = None,
    plugin_id: str = "hiv",
    profile: str = "hiv_rescue_v2",
    overwrite: bool = False,
) -> dict[str, Any]:
    source_ctx = RunContext.create(run_id=source_run_id, plugin_id=plugin_id)
    output_ctx = RunContext.create(run_id=output_run_id, plugin_id=plugin_id)
    output_dir = ensure_dir(output_ctx.run_dir / "phase2" / "source_reestimated_ablation")
    source_families = list(source_families or DEFAULT_SOURCE_FAMILIES)
    baseline_phase2 = read_json(source_ctx.run_dir / "phase2" / "phase2_structural_payload.json", default={}) or {}
    baseline_edges = _edge_rows(baseline_phase2)
    family_rows: list[dict[str, Any]] = []
    edge_family_rows: list[dict[str, Any]] = []
    for family in source_families:
        ablation_run_id = f"{output_run_id}-without-{family.replace('_', '-')}"
        try:
            run_row = _run_single_reestimate(
                source_run_dir=source_ctx.run_dir,
                ablation_run_id=ablation_run_id,
                excluded_family=family,
                plugin_id=plugin_id,
                profile=profile,
                overwrite=overwrite,
            )
        except Exception as exc:
            run_row = {
                "ablation_run_id": ablation_run_id,
                "excluded_source_family": family,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        ablated_phase2 = read_json(
            RunContext.create(run_id=ablation_run_id, plugin_id=plugin_id).run_dir
            / "phase2"
            / "phase2_structural_payload.json",
            default={},
        ) or {}
        ablated_edges = _edge_rows(ablated_phase2)
        family_rows.append(
            {
                **run_row,
                "baseline_edge_count": len(baseline_edges),
                "ablated_edge_count": len(ablated_edges),
            }
        )
        for edge_key, baseline_edge in sorted(baseline_edges.items()):
            ablated_edge = ablated_edges.get(edge_key)
            if ablated_edge is None:
                status = "absent"
                sign_preserved = False
            else:
                sign_preserved = _edge_sign(ablated_edge.get("weight")) == _edge_sign(baseline_edge.get("weight"))
                status = "survived" if sign_preserved else "sign_conflict"
            edge_family_rows.append(
                {
                    "edge_key": edge_key,
                    "edge_kind": str(baseline_edge.get("edge_kind") or ""),
                    "excluded_source_family": family,
                    "ablation_run_id": ablation_run_id,
                    "ablation_status": str(run_row.get("status") or ""),
                    "reestimated_status": status if str(run_row.get("status")) == "completed" else "not_evaluated",
                    "sign_preserved": bool(sign_preserved),
                    "baseline_weight": baseline_edge.get("weight"),
                    "reestimated_weight": None if ablated_edge is None else ablated_edge.get("weight"),
                }
            )
    edge_summary: dict[str, Any] = {}
    for edge_key in sorted(baseline_edges):
        rows = [row for row in edge_family_rows if str(row.get("edge_key")) == edge_key]
        completed = [row for row in rows if str(row.get("ablation_status")) == "completed"]
        edge_summary[edge_key] = {
            "evaluated_family_count": len(completed),
            "survived_family_count": sum(1 for row in completed if str(row.get("reestimated_status")) == "survived"),
            "absent_family_count": sum(1 for row in completed if str(row.get("reestimated_status")) == "absent"),
            "sign_conflict_family_count": sum(1 for row in completed if str(row.get("reestimated_status")) == "sign_conflict"),
            "passed": bool(completed) and all(str(row.get("reestimated_status")) == "survived" for row in completed),
        }
    report = {
        "schema_version": SOURCE_REESTIMATION_SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "source_run_id": source_run_id,
        "output_run_id": output_run_id,
        "contract": {
            "ablation_unit": "source_family",
            "reestimation_scope": "Phase0 canonical candidates filtered, then Phase1, Phase15, and Phase2 rebuilt.",
            "promotion_use": "A Phase2 edge cannot be a Phase3 prior unless it survives every completed source-family re-estimation ablation with sign preserved.",
        },
        "source_families": source_families,
        "baseline_edge_count": len(baseline_edges),
        "family_rows": family_rows,
        "edge_family_rows": edge_family_rows,
        "edge_summary": edge_summary,
    }
    report_path = output_dir / "phase2_source_reestimated_ablation_report.json"
    write_json(report_path, report)
    md_path = output_dir / "phase2_source_reestimated_ablation_report.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    report["artifact_paths"] = {
        "report_json": str(report_path),
        "report_md": str(md_path),
    }
    write_json(report_path, report)
    return report


def _markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 2 Source-Family Re-Estimated Ablation",
        "",
        f"- Source run: `{report.get('source_run_id')}`",
        f"- Output run: `{report.get('output_run_id')}`",
        f"- Baseline edges: `{report.get('baseline_edge_count')}`",
        "",
        "## Family Runs",
        "",
        "| Family | Status | Removed Phase0 Rows | Ablated Edge Count |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in list(report.get("family_rows") or []):
        manifest = dict(row.get("phase0_manifest") or {})
        lines.append(
            f"| `{row.get('excluded_source_family')}` | `{row.get('status')}` | "
            f"{int(manifest.get('removed_candidate_row_count') or 0)} | {int(row.get('ablated_edge_count') or 0)} |"
        )
    lines.extend(["", "## Edge Summary", "", "| Edge | Evaluated Families | Survived | Absent | Sign Conflict | Passed |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for edge_key, row in sorted(dict(report.get("edge_summary") or {}).items()):
        lines.append(
            f"| `{edge_key}` | {int(row.get('evaluated_family_count') or 0)} | "
            f"{int(row.get('survived_family_count') or 0)} | {int(row.get('absent_family_count') or 0)} | "
            f"{int(row.get('sign_conflict_family_count') or 0)} | `{bool(row.get('passed'))}` |"
        )
    return "\n".join(lines) + "\n"


def _main() -> int:
    parser = argparse.ArgumentParser(description="Re-estimate Phase2 after excluding source families upstream.")
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--source-family", action="append", dest="source_families")
    parser.add_argument("--plugin", default="hiv")
    parser.add_argument("--profile", default="hiv_rescue_v2")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = build_source_reestimated_ablation_report(
        source_run_id=args.source_run_id,
        output_run_id=args.output_run_id,
        source_families=args.source_families,
        plugin_id=args.plugin,
        profile=args.profile,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(report.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
