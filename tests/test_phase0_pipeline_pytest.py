from __future__ import annotations

import json
from pathlib import Path

from epigraph_ph.cli.main import build_parser
from epigraph_ph.phase0.pipeline import (
    _extract_who_core_team_anchors,
    _docling_should_parse_pdf,
    _lighton_ocr_vllm_ready,
    _local_official_anchor_specs,
    _openalex_abstract_from_inverted_index,
    _phase0_ocr_backend,
    _select_diverse_working_set,
    _sniff_local_document_type,
    _source_harvest_budgets,
    _extend_query_bank,
)
from epigraph_ph.phase0.semantic_benchmark import run_phase0_semantic_benchmark
from epigraph_ph.runtime import read_json


def test_query_bank_and_budgets_include_new_sources() -> None:
    queries = _extend_query_bank("massive")
    assert len(queries) > 10
    silos = {row.get("query_silo") for row in queries}
    assert "cash_instability" in silos
    assert "labor_migration" in silos
    assert "housing_precarity" in silos
    assert "social_capital" in silos
    assert "policy_implementation_weakness" in silos
    budgets = _source_harvest_budgets(queries, target_records=500, max_results=20)
    assert "openalex" in budgets
    assert "semanticscholar" in budgets
    assert budgets["openalex"] >= budgets["arxiv"]


def test_openalex_abstract_reconstruction() -> None:
    abstract = _openalex_abstract_from_inverted_index({"HIV": [0], "care": [1], "cascade": [2]})
    assert abstract == "HIV care cascade"


def test_docling_guard_is_conservative(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    assert _docling_should_parse_pdf(pdf_path) is False


def test_sniff_local_document_type_handles_html_renamed_pdf(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "landing.pdf"
    fake_pdf.write_text("<html><body>not a real pdf</body></html>", encoding="utf-8")
    assert _sniff_local_document_type(fake_pdf) == "html"


def test_heavy_working_set_cap(tmp_path: Path, monkeypatch) -> None:
    docs = []
    for idx in range(700):
        path = tmp_path / f"doc_{idx}.pdf"
        path.write_bytes(b"%PDF-1.4\n")
        docs.append({"document_id": f"doc-{idx}", "source_id": f"src-{idx}", "local_path": str(path), "platform": "crossref", "query_geo_focus": ""})
    monkeypatch.setattr("epigraph_ph.phase0.pipeline._is_heavy_parse_document", lambda doc: True)
    selected = _select_diverse_working_set(docs, limit=1000)
    assert len(selected) <= 250


def test_offline_build_and_registry_smoke(phase0_registry_run_dir: Path) -> None:
    run_dir = phase0_registry_run_dir
    phase0_manifest = read_json(run_dir / "phase0" / "phase0_manifest.json", default={})
    assert phase0_manifest.get("stage_status", {}).get("harvest") == "completed"
    candidates = read_json(run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    assert len(candidates) >= 0
    alignment_summary = read_json(run_dir / "phase0" / "extracted" / "alignment_summary.json", default={})
    assert len(alignment_summary.get("province_axis", [])) > 1
    sources = read_json(run_dir / "registry" / "source_registry.json", default={})
    subparameters = read_json(run_dir / "registry" / "subparameter_registry.json", default={})
    assert sources["source_count"] > 0
    assert subparameters["subparameter_count"] >= subparameters["by_source_bank"]["phase0_wide_sweep_literature"]
    assert read_json(run_dir / "phase0" / "raw" / "harvested_sweep_records.json", default=[]) != []
    literature_review = read_json(run_dir / "phase0" / "literature_review" / "literature_review_by_silo.json", default={})
    assert literature_review.get("silo_count", 0) >= 10
    assert (run_dir / "phase0" / "literature_review" / "silos" / "testing_uptake.json").exists()
    adapter_manifest = read_json(run_dir / "phase0" / "raw" / "structured_source_adapter_manifest.json", default=[])
    adapter_ids = {row.get("adapter_id") for row in adapter_manifest}
    assert {"ndhs", "yafs", "fies", "philgis_boundary_proxy", "transport_network_proxies"}.issubset(adapter_ids)
    assert (run_dir / "phase0" / "raw" / "source_manifest.parquet").exists()
    assert (run_dir / "phase0" / "parsed" / "parsed_document_blocks.parquet").exists()
    assert (run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.parquet").exists()


def test_local_official_anchor_specs_pick_up_local_files(tmp_path: Path, monkeypatch) -> None:
    core_pdf = tmp_path / "2025 PH HIV Estimates_Core team_for WHO.pdf"
    surveillance_pdf = tmp_path / "The Philippine HIV_STI Surveillance.pdf"
    core_pdf.write_bytes(b"%PDF-1.4\n%anchor\n")
    surveillance_pdf.write_bytes(b"%PDF-1.4\n%anchor\n")
    monkeypatch.setattr("epigraph_ph.phase0.pipeline._desktop_candidates", lambda: [tmp_path])
    specs = _local_official_anchor_specs()
    assert len(specs) == 2
    assert all(row.get("is_anchor_eligible") for row in specs)
    assert all(row.get("local_document_path") for row in specs)


def test_targeted_core_team_anchor_extraction_produces_structured_rows() -> None:
    source = {
        "source_id": "who-core",
        "source_tier": "tier1_official_anchor",
        "title": "2025 PH HIV Estimates Core Team for WHO",
        "year": 2025,
        "url": "file:///who.pdf",
    }
    block = {
        "block_id": "block-doc-who-page-13",
        "document_id": "doc-who",
        "source_id": "who-core",
        "page_number": 13,
        "text": (
            "Philippine HIV Care Cascade as of December 2024 "
            "216,900 135,026 90,854 43,534 41,164 Estimated PLHIV Diagnosed PLHIV "
            "Alive on ART Tested for Viral Load Virally Suppressed 62% 67% 48% 95%"
        ),
    }
    observations, candidates = _extract_who_core_team_anchors(block, source)
    assert len(observations) >= 9
    assert any(row["parameter_text"] == "Diagnosed PLHIV" and row["time"] == "2024-12" for row in observations)
    assert any(row["parameter_text"] == "Alive on ART among diagnosed PLHIV" and row["unit"] == "percent" for row in observations)
    assert len(candidates) == len(observations)


def test_semantic_benchmark_runs_on_phase0_smoke_corpus(phase0_registry_run_dir: Path) -> None:
    summary = run_phase0_semantic_benchmark(run_id=phase0_registry_run_dir.name, plugin_id="hiv", top_k=5)
    systems = summary.get("systems", {})
    assert "hashed_local_faiss" in systems
    assert "local_embedder_faiss" in systems
    assert systems["local_embedder_faiss"]["mean_ndcg_at_k"] >= systems["hashed_local_faiss"]["mean_ndcg_at_k"]


def test_phase0_ocr_backend_auto_stays_disabled_without_local_or_reachable_vllm(monkeypatch) -> None:
    _lighton_ocr_vllm_ready.cache_clear()
    monkeypatch.delenv("EPIGRAPH_LIGHTON_OCR_ENDPOINT", raising=False)
    monkeypatch.setattr("epigraph_ph.phase0.pipeline.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("unreachable")))
    assert _phase0_ocr_backend("auto") == "disabled"


def test_phase0_ocr_backend_auto_uses_explicit_vllm_endpoint(monkeypatch) -> None:
    _lighton_ocr_vllm_ready.cache_clear()
    monkeypatch.setenv("EPIGRAPH_LIGHTON_OCR_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions")
    monkeypatch.setattr("epigraph_ph.phase0.pipeline.requests.get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    assert _phase0_ocr_backend("auto") == "lighton_vllm"


def test_phase0_cli_accepts_ocr_sidecar_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase0",
            "build",
            "--run-id",
            "demo-run",
            "--plugin",
            "hiv",
            "--enable-ocr-sidecar",
            "--ocr-backend",
            "lighton_vllm",
        ]
    )
    assert args.enable_ocr_sidecar is True
    assert args.ocr_backend == "lighton_vllm"
