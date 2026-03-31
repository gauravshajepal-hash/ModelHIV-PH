from __future__ import annotations

import json
from pathlib import Path

from epigraph_ph.cli.main import build_parser
from epigraph_ph.phase0.literature_candidates import wide_sweep_candidate_rows, wide_sweep_record_canonical_names
from epigraph_ph.phase0.pipeline import (
    _phase0_alignment_bundle,
    _extract_who_core_team_anchors,
    _extract_chunk_soft_candidates,
    _chunk_text,
    _drop_low_signal_numeric_candidate,
    _document_metadata_text,
    _docling_should_parse_pdf,
    _filter_min_literature_year,
    _lighton_ocr_vllm_extract,
    _lighton_ocr_vllm_ready,
    _local_official_anchor_specs,
    _normalize_lighton_ocr_endpoint,
    _openalex_abstract_from_inverted_index,
    _parse_html_document,
    _parse_non_pdf_document,
    _phase0_ocr_backend,
    _phase0_observation_time,
    _second_pass_canonical_name,
    _select_diverse_working_set,
    _should_run_ocr_sidecar,
    _skip_regex_numeric_extraction,
    _sniff_local_document_type,
    _source_harvest_budgets,
    _extend_query_bank,
)
from epigraph_ph.phase0.semantic_benchmark import run_phase0_semantic_benchmark
from epigraph_ph.runtime import read_json


def test_query_bank_and_budgets_include_new_sources() -> None:
    queries = _extend_query_bank("massive")
    assert len(queries) > 10
    assert len(_extend_query_bank("ultra")) >= len(queries)
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


def test_wide_sweep_candidate_rows_expand_into_real_canonical_names() -> None:
    record = {
        "record_id": "rec-1",
        "title": "Transport friction, housing precarity, and delayed HIV testing uptake in migrant workers",
        "query_geo_focus": "philippines",
        "query_silo": "transport_friction",
        "determinant_silos": ["transport_friction", "housing_precarity"],
        "soft_subparameter_hints": ["labor_migration", "cash_instability"],
        "linkage_targets": ["testing_uptake", "linkage_to_care"],
        "soft_ontology_tags": ["logistics", "policy"],
    }
    canonical_names = wide_sweep_record_canonical_names(record)
    assert "transport_friction" in canonical_names
    assert "housing_precarity" in canonical_names
    assert "labor_migration" in canonical_names
    assert "testing_uptake" in canonical_names
    rows = wide_sweep_candidate_rows(record, bank_name="phase0_wide_sweep_literature")
    assert len(rows) >= 4
    assert {row["canonical_name"] for row in rows} == set(canonical_names)


def test_openalex_abstract_reconstruction() -> None:
    abstract = _openalex_abstract_from_inverted_index({"HIV": [0], "care": [1], "cascade": [2]})
    assert abstract == "HIV care cascade"


def test_phase0_filters_literature_before_2010() -> None:
    rows = [
        {"source_id": "pubmed-old", "platform": "pubmed", "year": 2009},
        {"source_id": "pubmed-new", "platform": "pubmed", "year": 2014},
        {"source_id": "arxiv-new", "platform": "arxiv", "year": 2010},
        {"source_id": "kaggle-seed", "platform": "kaggle", "year": None},
    ]
    filtered = _filter_min_literature_year(rows)
    assert {row["source_id"] for row in filtered} == {"pubmed-new", "arxiv-new", "kaggle-seed"}


def test_phase0_chunk_text_builds_overlapping_chunks() -> None:
    text = " ".join(f"token{idx}" for idx in range(800))
    chunks = _chunk_text(text)
    assert len(chunks) >= 2
    assert all(len(chunk) >= 100 for chunk in chunks)


def test_docling_guard_is_conservative(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    assert _docling_should_parse_pdf(pdf_path) is False


def test_sniff_local_document_type_handles_html_renamed_pdf(tmp_path: Path) -> None:
    fake_pdf = tmp_path / "landing.pdf"
    fake_pdf.write_text("<html><body>not a real pdf</body></html>", encoding="utf-8")
    assert _sniff_local_document_type(fake_pdf) == "html"


def test_parse_html_document_strips_markup(tmp_path: Path) -> None:
    landing = tmp_path / "landing.html"
    landing.write_text(
        "<!DOCTYPE html><html><head><title>Ignored</title><script>var x = 1;</script></head>"
        "<body><h1>Retention in care</h1><p>Clinic continuity and linkage support.</p></body></html>",
        encoding="utf-8",
    )
    parsed = _parse_html_document(landing)
    assert "Retention in care" in parsed
    assert "Clinic continuity" in parsed
    assert "<html" not in parsed.lower()
    assert "var x = 1" not in parsed


def test_html_documents_prefer_metadata_over_landing_page_markup(tmp_path: Path) -> None:
    landing = tmp_path / "landing.html"
    landing.write_text(
        "<!DOCTYPE html><html><body><script>window.foo=1</script><p>publisher access page</p></body></html>",
        encoding="utf-8",
    )
    text, parser_used = _parse_non_pdf_document(
        landing,
        doc_type="html",
        document_row={"title": "Filipino HIV testing uptake", "abstract": "Fear and stigma reduce testing uptake."},
    )
    assert parser_used == "html_metadata"
    assert text == "Filipino HIV testing uptake\nFear and stigma reduce testing uptake."


def test_document_metadata_text_joins_title_and_abstract() -> None:
    assert _document_metadata_text({"title": "Title", "abstract": "Abstract"}) == "Title\nAbstract"


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
    collector_manifest = read_json(run_dir / "phase0" / "raw" / "collector_manifest.json", default=[])
    adapter_ids = {row.get("adapter_id") for row in adapter_manifest}
    assert {
        "ndhs",
        "yafs",
        "fies",
        "philgis_boundary_proxy",
        "transport_network_proxies",
        "google_mobility",
        "world_bank_wdi",
        "philhealth_reports",
        "doh_facility_stats",
    }.issubset(adapter_ids)
    assert collector_manifest
    collector_ids = {row.get("collector_id") for row in collector_manifest}
    assert {"who", "unaids", "doh_philippines", "un", "ndhs", "yafs", "fies", "google_mobility", "world_bank_wdi", "philhealth_reports", "doh_facility_stats"}.issubset(collector_ids)
    assert (run_dir / "phase0" / "raw" / "source_manifest.parquet").exists()
    assert (run_dir / "phase0" / "parsed" / "parsed_document_blocks.parquet").exists()
    assert (run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.parquet").exists()
    assert (run_dir / "phase0" / "analysis" / "prompt_library.json").exists()
    assert (run_dir / "phase0" / "analysis" / "tool_stack_manifest.json").exists()
    assert (run_dir / "phase0" / "analysis" / "top_candidate_variables.json").exists()
    assert (run_dir / "phase0" / "analysis" / "curated_bibliography.json").exists()
    assert (run_dir / "phase0" / "analysis" / "schema_validation_summary.json").exists()
    assert (run_dir / "phase0" / "analysis" / "source_rights_manifest.json").exists()
    assert (run_dir / "phase0" / "analysis" / "human_review_queue.json").exists()
    assert (run_dir / "phase0" / "analysis" / "resource_usage_manifest.json").exists()
    assert (run_dir / "phase0" / "analysis" / "implementation_reality_summary.json").exists()
    assert (run_dir / "phase0" / "extracted" / "canonicalization_summary.json").exists()
    assert (run_dir / "phase0" / "boundary_shape_manifest.json").exists()
    assert (run_dir / "phase0" / "boundary_shape_checks.json").exists()
    assert (run_dir / "phase0" / "boundary_shape_summary.json").exists()


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
    assert "mean_ndcg_at_k" in systems["local_embedder_faiss"]
    assert systems["local_embedder_faiss"]["mean_ndcg_at_k"] > 0.0


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


def test_phase0_normalizes_lighton_base_endpoint() -> None:
    assert _normalize_lighton_ocr_endpoint("http://127.0.0.1:8000") == "http://127.0.0.1:8000/v1/chat/completions"
    assert _normalize_lighton_ocr_endpoint("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1/chat/completions"
    assert _normalize_lighton_ocr_endpoint("http://127.0.0.1:8000/v1/responses") == "http://127.0.0.1:8000/v1/responses"


def test_phase0_lighton_extract_falls_back_to_responses(monkeypatch) -> None:
    class _FakeResponse:
        def __init__(self, *, status_code: int, payload: dict[str, object]) -> None:
            self.status_code = status_code
            self._payload = payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http_{self.status_code}")

        def json(self) -> dict[str, object]:
            return self._payload

    calls: list[str] = []

    def _fake_post(url: str, json=None, headers=None, timeout=None):  # type: ignore[override]
        calls.append(url)
        if url.endswith("/chat/completions"):
            return _FakeResponse(status_code=404, payload={})
        return _FakeResponse(
            status_code=200,
            payload={"output": [{"content": [{"text": "| Col A | Col B |\\n|---|---|\\n| 1 | 2 |"}]}]},
        )

    monkeypatch.setenv("EPIGRAPH_LIGHTON_OCR_ENDPOINT", "http://127.0.0.1:8000")
    monkeypatch.setattr("epigraph_ph.phase0.pipeline.requests.post", _fake_post)
    text = _lighton_ocr_vllm_extract("ZmFrZQ==")
    assert "| Col A | Col B |" in text
    assert any(url.endswith("/chat/completions") for url in calls)
    assert any(url.endswith("/responses") for url in calls)


def test_phase0_observation_time_uses_source_year_when_old_literature_year_is_detected() -> None:
    source = {"platform": "crossref", "year": 2018}
    assert _phase0_observation_time(source, "1993-01") == "2018-01"


def test_phase0_skips_regex_numeric_extraction_for_metadata_only_nonanchors() -> None:
    block = {"document_type": "json_metadata", "parser_used": "json_metadata"}
    source = {"is_anchor_eligible": False}
    assert _skip_regex_numeric_extraction(block, source) is True


def test_phase0_extracts_chunk_soft_candidates_from_parsed_chunks() -> None:
    parsed_chunks = [
        {
            "chunk_id": "chunk-1",
            "block_id": "block-1",
            "document_id": "doc-1",
            "source_id": "src-1",
            "text": (
                "Testing uptake remains low because stigma, delayed diagnosis, and travel time to the clinic "
                "reduce linkage to care and viral load monitoring in remote island provinces."
            ),
            "query_silo": "testing_uptake",
            "source_tier": "tier2_scientific_literature",
        }
    ]
    source_rows = {
        "src-1": {
            "source_id": "src-1",
            "title": "Testing uptake and travel burden",
            "platform": "crossref",
            "source_tier": "tier2_scientific_literature",
            "year": 2022,
            "query_geo_focus": "philippines",
            "query_silo": "testing_uptake",
            "url": "https://doi.org/example",
        }
    }
    rows = _extract_chunk_soft_candidates(parsed_chunks, source_rows)
    canonical_names = {row["canonical_name"] for row in rows}
    assert "testing_uptake" in canonical_names
    assert "linkage_to_care" in canonical_names
    assert all(row["source_bank"] == "phase0_chunk_soft_candidates" for row in rows)
    assert all(row["value"] is None for row in rows)


def test_phase0_second_pass_canonicalizer_maps_google_mobility_numeric_generic() -> None:
    name, reason = _second_pass_canonical_name(
        canonical_name="numeric_observation",
        parameter_text="12",
        local_context="Retail and recreation mobility fell by 12 percent in Metro Manila.",
        source_text="Google Community Mobility Reports mobility and commuting change for the Philippines",
        source={
            "platform": "google_mobility",
            "query_silo": "mobility_network_mixing",
            "title": "Google Community Mobility Reports",
            "query_geo_focus": "philippines",
        },
        unit="percent",
        semantics={"measurement_type": "rate", "denominator_type": "population"},
    )
    assert name == "mobility_network_mixing"
    assert reason in {"platform_mapping", "contextual_keywords", "query_silo_mapping", "mobility_mapping"}


def test_phase0_drops_low_signal_small_unitless_generic_numeric() -> None:
    drop, reason = _drop_low_signal_numeric_candidate(
        canonical_name="numeric_observation",
        value=8.0,
        unit="",
        parameter_text="8",
        local_context="8",
        source={
            "source_tier": "tier2_scientific_literature",
            "title": "Thinking clearly about social aspects of infectious disease transmission",
            "abstract": "Generic methodological paper",
            "query_silo": "",
        },
        geo_label="",
    )
    assert drop is True
    assert reason in {"low_signal_small_unitless", "unresolved_numeric_stub"}


def test_phase0_should_run_ocr_sidecar_respects_force_flag() -> None:
    assert _should_run_ocr_sidecar(enable_ocr_sidecar=True, force_ocr_sidecar=True, pdf_blocks=[{"text": "full text"}]) is True
    assert _should_run_ocr_sidecar(enable_ocr_sidecar=False, force_ocr_sidecar=True, pdf_blocks=[]) is False


def test_phase0_skips_regex_extraction_for_json_metadata_even_if_anchor_eligible() -> None:
    block = {"document_type": "json_metadata", "parser_used": "json_metadata"}
    source = {"is_anchor_eligible": True}
    assert _skip_regex_numeric_extraction(block, source) is True


def test_phase0_drops_year_fragment_from_metadata_titles() -> None:
    drop, reason = _drop_low_signal_numeric_candidate(
        canonical_name="numeric_observation",
        value=202.0,
        unit="",
        parameter_text="202",
        local_context="",
        source={
            "source_tier": "",
            "title": "HIV/AIDS and ART Registry of the Philippines 2022",
            "abstract": "",
            "query_silo": "",
        },
        geo_label="Philippines",
    )
    assert drop is True
    assert reason == "metadata_year_fragment"


def test_phase0_alignment_bundle_excludes_pre2010_literature_months(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "phase0_alignment"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = [
        {
            "candidate_id": "cand-1",
            "source_id": "src-1",
            "canonical_name": "testing_uptake",
            "geo": "Philippines",
            "time": "1993-01",
            "value": None,
        }
    ]
    source_rows = {
        "src-1": {
            "source_id": "src-1",
            "platform": "crossref",
            "year": 2018,
            "query_geo_focus": "philippines",
        }
    }
    bundle = _phase0_alignment_bundle(
        candidate_rows=candidate_rows,
        source_rows=source_rows,
        plugin_id="hiv",
        artifact_dir=artifact_dir,
    )
    summary = read_json(Path(bundle["alignment_summary"]), default={})
    assert "1993-01" not in set(summary.get("month_axis", []))
    assert "2018-01" in set(summary.get("month_axis", []))


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
            "--force-ocr-sidecar",
        ]
    )
    assert args.enable_ocr_sidecar is True
    assert args.ocr_backend == "lighton_vllm"
    assert args.force_ocr_sidecar is True


def test_phase0_cli_accepts_large_corpus_shard_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase0",
            "harvest",
            "--run-id",
            "demo-run",
            "--plugin",
            "hiv",
            "--corpus-mode",
            "megacrawl",
            "--metadata-only",
            "--query-shard-count",
            "12",
            "--query-shard-index",
            "3",
        ]
    )
    assert args.metadata_only is True
    assert args.query_shard_count == 12
    assert args.query_shard_index == 3


def test_phase0_cli_accepts_merge_shards_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase0",
            "merge-shards",
            "--run-id",
            "merged-run",
            "--plugin",
            "hiv",
            "--source-run-ids",
            "shard-00",
            "shard-01",
            "shard-02",
        ]
    )
    assert args.source_run_ids == ["shard-00", "shard-01", "shard-02"]


def test_phase0_cli_accepts_pilot_report_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase0",
            "pilot-report",
            "--run-id",
            "pilot-run",
            "--plugin",
            "hiv",
            "--top-k",
            "12",
        ]
    )
    assert args.phase0_command == "pilot-report"
    assert args.top_k == 12
