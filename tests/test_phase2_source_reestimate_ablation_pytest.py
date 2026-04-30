from __future__ import annotations

from epigraph_ph.phase2.source_reestimate_ablation import infer_source_family


def test_infer_source_family_recovers_official_platforms() -> None:
    assert infer_source_family({"source_id": "psa-psa-city-and-municipal-poverty-statistics"}) == "psa"
    assert infer_source_family({"platform": "google_mobility"}) == "google_mobility"
    assert infer_source_family({"document_id": "doc-world_bank_wdi_wdi_poverty_headcount"}) == "world_bank_wdi"
    assert infer_source_family({"source_id": "philhealth_open_portal_statistics-charts-2024"}) == "philhealth_open_portal"
    assert infer_source_family({"source_id": "openalex_example"}) == "openalex"
