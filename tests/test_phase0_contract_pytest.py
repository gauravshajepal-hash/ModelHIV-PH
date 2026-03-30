from __future__ import annotations

import numpy as np

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import load_tensor_artifact, read_json


def test_phase0_constraint_settings_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase0_cfg = (plugin.constraint_settings or {}).get("phase0", {})

    assert phase0_cfg.get("source_budget_weights", {}).get("openalex") is not None
    assert phase0_cfg.get("mode_max_queries", {}).get("default") is not None
    assert phase0_cfg.get("wide_sweep_scoring", {}).get("accepted_hiv_threshold") is not None
    assert phase0_cfg.get("alignment", {}).get("semantic_threshold") is not None
    assert phase0_cfg.get("alignment", {}).get("temporal_decay") is not None
    assert phase0_cfg.get("ocr", {}).get("render_dpi") is not None
    assert phase0_cfg.get("confidence", {}).get("structured_anchor") is not None
    assert phase0_cfg.get("request_sleep_seconds") is not None
    assert phase0_cfg.get("retry_pause_seconds") is not None


def test_phase0_manifest_backend_and_overload_contract(phase0_registry_run_dir) -> None:
    manifest = read_json(phase0_registry_run_dir / "phase0" / "phase0_manifest.json", default={})
    truth_manifest = read_json(phase0_registry_run_dir / "phase0" / "ground_truth_manifest.json", default={})
    truth_checks = read_json(phase0_registry_run_dir / "phase0" / "ground_truth_checks.json", default=[])
    truth_summary = read_json(phase0_registry_run_dir / "phase0" / "ground_truth_summary.json", default={})
    notes = manifest.get("notes", [])
    backend_status = manifest.get("backend_status", {})
    assert manifest.get("stage_status", {}).get("harvest") == "completed"
    assert any(str(note).startswith("corpus_mode:") for note in notes)
    assert any(str(note).startswith("query_count:") for note in notes)
    # Smoke runs must stay bounded so the suite cannot overload the machine.
    assert 0 < int(manifest.get("document_count", 0)) <= 100
    assert "duckdb" in backend_status
    assert truth_manifest.get("phase_name") == "phase0"
    assert truth_checks
    assert truth_summary.get("phase_name") == "phase0"
    assert truth_summary.get("literature_silo_count", 0) >= 10


def test_phase0_parsed_documents_do_not_hide_parser_choice(phase0_registry_run_dir) -> None:
    document_manifest = read_json(phase0_registry_run_dir / "phase0" / "parsed" / "document_manifest.json", default=[])
    assert document_manifest
    for row in document_manifest:
        if row.get("parse_status") == "parsed":
            assert row.get("parser_used")
        assert row.get("snapshot_type")


def test_phase0_candidate_contract_and_alignment_sanity(phase0_registry_run_dir) -> None:
    candidates = read_json(phase0_registry_run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    alignment_summary = read_json(phase0_registry_run_dir / "phase0" / "extracted" / "alignment_summary.json", default={})
    aligned_tensor = load_tensor_artifact(phase0_registry_run_dir / "phase0" / "extracted" / "aligned_tensor.npz")
    quality_tensor = load_tensor_artifact(phase0_registry_run_dir / "phase0" / "extracted" / "quality_weights.npz")

    assert candidates
    for row in candidates:
        assert row.get("candidate_id")
        assert row.get("source_id")
        assert row.get("canonical_name")
        assert row.get("geo") is not None
        assert row.get("time") is not None
    assert len(alignment_summary.get("province_axis", [])) == aligned_tensor.shape[0]
    assert len(alignment_summary.get("month_axis", [])) == aligned_tensor.shape[1]
    assert len(alignment_summary.get("canonical_name_axis", [])) == aligned_tensor.shape[2]
    assert np.isfinite(aligned_tensor).all()
    assert np.isfinite(quality_tensor).all()
    assert np.all(quality_tensor >= 0.0)


def test_phase0_index_contract_has_embeddings_and_manifest(phase0_registry_run_dir) -> None:
    index_manifest = read_json(phase0_registry_run_dir / "phase0" / "index" / "index_manifest.json", default={})
    embeddings = np.load(phase0_registry_run_dir / "phase0" / "index" / "embeddings.npy")
    assert index_manifest.get("vector_count", embeddings.shape[0]) == embeddings.shape[0]
    assert embeddings.ndim == 2
    assert embeddings.shape[1] > 0
    assert np.isfinite(embeddings).all()
    assert index_manifest.get("embedding_backend") in {"sentence_transformers", "transformers", "hashed_local"}
    assert "chroma_available" in index_manifest
    if index_manifest.get("embedding_backend") == "sentence_transformers":
        assert index_manifest.get("fallback_used") is False


def test_phase0_literature_review_contract_has_quality_and_promotion_fields(phase0_registry_run_dir) -> None:
    review = read_json(phase0_registry_run_dir / "phase0" / "literature_review" / "literature_review_by_silo.json", default={})
    silos = review.get("silos", [])
    assert silos
    for row in silos:
        assert row.get("silo_id")
        assert row.get("display_name")
        assert "top_documents" in row
        assert "extracted_candidate_subparameters" in row
        assert "source_quality_ladder" in row
        assert "promotion_eligibility" in row
