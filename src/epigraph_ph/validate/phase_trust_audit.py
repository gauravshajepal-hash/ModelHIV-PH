from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.semantic_benchmark import run_phase0_semantic_benchmark
from epigraph_ph.runtime import ensure_dir, read_json, write_json


def _phase_manifest_path(run_dir: Path, phase_name: str) -> Path:
    return run_dir / phase_name / f"{phase_name}_manifest.json"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path, default=[])
    return payload if isinstance(payload, list) else []


def _read_dict(path: Path) -> dict[str, Any]:
    payload = read_json(path, default={})
    return payload if isinstance(payload, dict) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _phase15_engine_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase15_cfg = dict((plugin.constraint_settings or {}).get("phase15", {}) or {})
    latent_v2_cfg = dict(phase15_cfg.get("latent_blocks_v2") or {})
    engine_cfg = dict(latent_v2_cfg.get("engine") or {})
    return {
        "regional_precision_ceiling": float(engine_cfg.get("regional_precision_ceiling") or 25.0),
        "loading_magnitude_ceiling": float(engine_cfg.get("loading_magnitude_ceiling") or 4.0),
        "minimum_loading_magnitude": float(engine_cfg.get("minimum_loading_magnitude") or 0.05),
    }


def _ensure_phase0_semantic_benchmark(run_dir: Path, plugin_id: str) -> dict[str, Any]:
    benchmark_path = run_dir / "phase0" / "analysis" / "semantic_quality_benchmark.json"
    payload = _read_dict(benchmark_path)
    if payload:
        return payload
    candidate_count = len(_read_rows(run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json"))
    if candidate_count > 2500:
        return {
            "available": False,
            "reason": "candidate_count_too_large_for_inline_refresh",
            "candidate_count": candidate_count,
        }
    try:
        return run_phase0_semantic_benchmark(run_id=run_dir.name, plugin_id=plugin_id)
    except Exception as exc:  # pragma: no cover - best effort only
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _update_phase_manifest_status(
    *,
    run_dir: Path,
    phase_name: str,
    artifact_status: str,
    trust_status: str,
    summary: dict[str, Any],
) -> None:
    manifest_path = _phase_manifest_path(run_dir, phase_name)
    manifest = _read_dict(manifest_path)
    if not manifest:
        return
    manifest["artifact_status"] = artifact_status
    manifest["trust_status"] = trust_status
    manifest["trust_summary"] = summary
    write_json(manifest_path, manifest)


def _phase0_status(run_dir: Path, plugin_id: str) -> dict[str, Any]:
    phase_dir = run_dir / "phase0"
    manifest = _read_dict(_phase_manifest_path(run_dir, "phase0"))
    candidates = _read_rows(phase_dir / "extracted" / "canonical_parameter_candidates.json")
    numeric_rows = _read_rows(phase_dir / "extracted" / "numeric_observations.json")
    extraction_audit = _read_dict(run_dir / "analysis" / "extraction_quality_audit.json")
    semantic_benchmark = _ensure_phase0_semantic_benchmark(run_dir, plugin_id)
    human_labeled_benchmark_present = (phase_dir / "analysis" / "human_labeled_semantic_benchmark.json").exists()
    checks = {
        "manifest_present": bool(manifest),
        "candidate_count_matches_artifact": int(manifest.get("canonical_candidate_count") or 0) == len(candidates),
        "numeric_count_matches_artifact": int(manifest.get("numeric_observation_count") or 0) == len(numeric_rows),
        "extraction_quality_audit_passed": bool(extraction_audit.get("overall_passed")),
    }
    artifact_status = "complete" if all(checks.values()) else "inconsistent"
    if artifact_status != "complete":
        trust_status = "blocked_artifact_inconsistency"
    elif not human_labeled_benchmark_present:
        trust_status = "provisional_no_human_labeled_semantic_benchmark"
    else:
        trust_status = "provisionally_trusted"
    summary = {
        "candidate_count": len(candidates),
        "numeric_observation_count": len(numeric_rows),
        "extraction_quality_passed": bool(extraction_audit.get("overall_passed")),
        "semantic_winner": str(dict(semantic_benchmark.get("interpretation") or {}).get("semantic_winner") or ""),
        "semantic_benchmark_ndcg": _safe_float(
            dict(dict(semantic_benchmark.get("systems") or {}).get("local_embedder_faiss") or {}).get("mean_ndcg_at_k"),
            default=float("nan"),
        ),
        "human_labeled_benchmark_present": human_labeled_benchmark_present,
        "checks": checks,
    }
    return {"artifact_status": artifact_status, "trust_status": trust_status, "summary": summary}


def _phase1_status(run_dir: Path) -> dict[str, Any]:
    phase_dir = run_dir / "phase1"
    manifest = _read_dict(_phase_manifest_path(run_dir, "phase1"))
    normalized_rows = _read_rows(phase_dir / "normalized_subparameters.json")
    tensor_rows = _read_rows(phase_dir / "tensor_rows.json")
    audit = _read_dict(phase_dir / "latent_observability_audit.json")
    report = _read_dict(phase_dir / "normalization_report.json")
    checks = {
        "manifest_present": bool(manifest),
        "normalized_count_matches_artifact": int(manifest.get("canonical_candidate_count") or 0) == len(normalized_rows),
        "tensor_count_matches_artifact": int(manifest.get("numeric_observation_count") or 0) == len(tensor_rows),
        "latent_observability_present": bool(audit.get("rows")),
        "normalization_report_present": bool(report),
    }
    artifact_status = "complete" if all(checks.values()) else "inconsistent"
    missing_fraction = _safe_float(report.get("missing_mask_fraction"), default=float("nan"))
    eligible_subnational = int(dict(audit.get("summary") or {}).get("eligible_for_subnational_inference_count") or 0)
    if artifact_status != "complete":
        trust_status = "blocked_artifact_inconsistency"
    elif not np.isfinite(missing_fraction) or missing_fraction >= 0.99:
        trust_status = "provisional_extreme_missingness"
    else:
        trust_status = "provisional_heuristic_noise_model"
    summary = {
        "normalized_row_count": len(normalized_rows),
        "tensor_row_count": len(tensor_rows),
        "eligible_for_subnational_inference_count": eligible_subnational,
        "missing_mask_fraction": missing_fraction,
        "checks": checks,
    }
    return {"artifact_status": artifact_status, "trust_status": trust_status, "summary": summary}


def _phase15_status(run_dir: Path, plugin_id: str) -> dict[str, Any]:
    phase_dir = run_dir / "phase15"
    manifest = _read_dict(_phase_manifest_path(run_dir, "phase15"))
    fit_summary = _read_dict(phase_dir / "phase15_v2_fit_summary.json")
    indicator_params = _read_rows(phase_dir / "phase15_v2_indicator_parameters.json")
    aggregation = _read_dict(phase_dir / "phase15_v2_aggregation_weights.json")
    cfg = _phase15_engine_cfg(plugin_id)
    rows = list(fit_summary.get("rows") or [])
    checks = {
        "manifest_present": bool(manifest),
        "fit_summary_present": bool(rows),
        "indicator_parameters_present": bool(indicator_params),
        "aggregation_weights_present": bool(aggregation),
    }
    artifact_status = "complete" if all(checks.values()) else "inconsistent"
    regional_ceiling = float(cfg["regional_precision_ceiling"])
    regional_saturation_count = sum(
        1
        for row in rows
        if _safe_float(row.get("regional_precision"), default=0.0) >= regional_ceiling - 1e-6
    )
    loading_floor = float(cfg["minimum_loading_magnitude"])
    loading_ceiling = float(cfg["loading_magnitude_ceiling"])
    loadings = np.asarray([abs(_safe_float(row.get("lambda"), default=0.0)) for row in indicator_params], dtype=np.float64)
    loading_floor_fraction = float(np.mean(loadings <= loading_floor + 1e-6)) if loadings.size else float("nan")
    loading_ceiling_fraction = float(np.mean(loadings >= loading_ceiling - 1e-6)) if loadings.size else float("nan")
    weight_dispersion = _safe_float(dict(aggregation.get("diagnostics") or {}).get("weight_dispersion"), default=float("nan"))
    if artifact_status != "complete":
        trust_status = "blocked_artifact_inconsistency"
    elif regional_saturation_count > 0 or (np.isfinite(loading_ceiling_fraction) and loading_ceiling_fraction > 0.20):
        trust_status = "partial_identification_pooling_warning"
    elif np.isfinite(loading_floor_fraction) and loading_floor_fraction > 0.50:
        trust_status = "partial_identification_loading_floor_warning"
    else:
        trust_status = "provisional_latent_summary"
    summary = {
        "block_count": len(rows),
        "indicator_parameter_count": len(indicator_params),
        "regional_precision_saturation_count": regional_saturation_count,
        "loading_floor_fraction": loading_floor_fraction,
        "loading_ceiling_fraction": loading_ceiling_fraction,
        "weight_dispersion": weight_dispersion,
        "checks": checks,
    }
    return {"artifact_status": artifact_status, "trust_status": trust_status, "summary": summary}


def _phase2_status(run_dir: Path) -> dict[str, Any]:
    phase_dir = run_dir / "phase2"
    manifest = _read_dict(_phase_manifest_path(run_dir, "phase2"))
    latent_bundle = _read_dict(phase_dir / "latent_temporal_graph_bundle.json")
    latent_blanket = _read_dict(phase_dir / "latent_temporal_phase3_target_blankets.json")
    latent_validation = _read_dict(phase_dir / "latent_temporal_graph_validation.json")
    multiscale_bundle = _read_dict(phase_dir / "multiscale_dag_bundle.json")
    multiscale_blanket = _read_dict(phase_dir / "multiscale_phase3_target_blankets.json")
    compatibility_payload = _read_dict(phase_dir / "phase3_compatibility_payload.json")
    latent_scales = dict(latent_bundle.get("scales") or {})
    multiscale_scales = dict(multiscale_bundle.get("scales") or {})
    latent_completed_scales = sum(1 for row in latent_scales.values() if row.get("status") == "completed")
    multiscale_completed_scales = sum(1 for row in multiscale_scales.values() if row.get("status") == "completed")
    checks = {
        "manifest_present": bool(manifest),
        "latent_bundle_present": bool(latent_bundle),
        "latent_blankets_present": bool(latent_blanket),
        "latent_validation_present": bool(latent_validation),
        "multiscale_bundle_present": bool(multiscale_bundle),
        "multiscale_blankets_present": bool(multiscale_blanket),
        "compatibility_payload_present": bool(compatibility_payload),
        "latent_scales_emitted": set(latent_scales.keys()) == {"province", "region", "national"},
        "multiscale_scales_emitted": set(multiscale_scales.keys()) == {"province", "region", "national"},
    }
    artifact_status = "complete" if all(checks.values()) else "inconsistent"
    if artifact_status != "complete":
        trust_status = "blocked_artifact_inconsistency"
    else:
        trust_status = "temporal_hypothesis_only"
    summary = {
        "latent_completed_scale_count": latent_completed_scales,
        "multiscale_completed_scale_count": multiscale_completed_scales,
        "latent_edge_counts": {scale: int(dict(payload).get("edge_count", 0) or 0) for scale, payload in latent_scales.items()},
        "latent_hidden_driver_counts": {scale: int(dict(payload).get("hidden_driver_count", 0) or 0) for scale, payload in latent_scales.items()},
        "multiscale_edge_counts": {scale: int(dict(payload).get("edge_count", 0) or 0) for scale, payload in multiscale_scales.items()},
        "multiscale_hidden_driver_counts": {scale: int(dict(payload).get("hidden_driver_count", 0) or 0) for scale, payload in multiscale_scales.items()},
        "checks": checks,
    }
    return {"artifact_status": artifact_status, "trust_status": trust_status, "summary": summary}


def build_phase_trust_audit(
    *,
    run_dir: str | Path,
    plugin_id: str,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    analysis_dir = ensure_dir(Path(output_dir) if output_dir is not None else run_path / "analysis")

    phase_rows = {
        "phase0": _phase0_status(run_path, plugin_id),
        "phase1": _phase1_status(run_path),
        "phase15": _phase15_status(run_path, plugin_id),
        "phase2": _phase2_status(run_path),
    }
    for phase_name, payload in phase_rows.items():
        _update_phase_manifest_status(
            run_dir=run_path,
            phase_name=phase_name,
            artifact_status=str(payload["artifact_status"]),
            trust_status=str(payload["trust_status"]),
            summary=dict(payload["summary"]),
        )

    artifact_complete = all(str(payload["artifact_status"]) == "complete" for payload in phase_rows.values())
    trust_statuses = [str(payload["trust_status"]) for payload in phase_rows.values()]
    overall_trust_status = "scientifically_provisional"
    if any(status.startswith("blocked_") for status in trust_statuses):
        overall_trust_status = "blocked_by_artifact_or_audit_failure"
    report = {
        "run_dir": str(run_path),
        "plugin_id": plugin_id,
        "overall_artifact_status": "complete" if artifact_complete else "inconsistent",
        "overall_trust_status": overall_trust_status,
        "phase_rows": {
            phase_name: {
                "artifact_status": payload["artifact_status"],
                "trust_status": payload["trust_status"],
                "summary": payload["summary"],
            }
            for phase_name, payload in phase_rows.items()
        },
        "recommended_next_steps": [
            "Add a hand-labeled semantic benchmark for unstructured Phase 0 extraction.",
            "Calibrate Phase 15 uncertainty and pooling sensitivity under synthetic recovery.",
            "Propagate Phase 15 state uncertainty into Phase 2 edge confidence.",
            "Keep Phase 2 edges labeled as temporal hypotheses rather than mechanisms.",
        ],
    }
    json_path = analysis_dir / "phase_trust_audit.json"
    md_path = analysis_dir / "phase_trust_audit.md"
    write_json(json_path, report)
    md_lines = [
        "# Phase Trust Audit",
        "",
        f"- overall artifact status: `{report['overall_artifact_status']}`",
        f"- overall trust status: `{report['overall_trust_status']}`",
        "",
        "## Phase Status",
        "",
    ]
    for phase_name, payload in phase_rows.items():
        md_lines.append(f"### {phase_name}")
        md_lines.append("")
        md_lines.append(f"- artifact status: `{payload['artifact_status']}`")
        md_lines.append(f"- trust status: `{payload['trust_status']}`")
        for key, value in dict(payload["summary"]).items():
            if key == "checks":
                continue
            md_lines.append(f"- `{key}`: `{value}`")
        checks = dict(payload["summary"]).get("checks") or {}
        for check_name, check_value in checks.items():
            md_lines.append(f"- check `{check_name}`: `{check_value}`")
        md_lines.append("")
    md_lines.append("## Recommended Next Steps")
    md_lines.append("")
    for row in report["recommended_next_steps"]:
        md_lines.append(f"- {row}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    report["artifacts"] = {
        "json": str(json_path),
        "md": str(md_path),
    }
    return report
