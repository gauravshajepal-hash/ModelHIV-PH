from __future__ import annotations

from epigraph_ph.registry.models import has_verifiable_locator
from epigraph_ph.runtime import read_json


def test_source_registry_contract(phase0_registry_run_dir) -> None:
    payload = read_json(phase0_registry_run_dir / "registry" / "source_registry.json", default={})
    sources = payload.get("sources", [])
    adapters = payload.get("structured_source_adapters", [])
    plugin = payload.get("plugin", {})
    assert payload.get("source_count") == len(sources)
    assert sources
    assert plugin.get("plugin_id") == "hiv"
    assert adapters
    assert {"ndhs", "yafs", "fies", "philgis_boundary_proxy", "transport_network_proxies"}.issubset(
        {row.get("adapter_id") for row in adapters}
    )
    for row in sources:
        assert row.get("source_id")
        assert row.get("parse_status") in {"parsed", "empty", "pending"}
        assert row.get("extraction_status") in {"extracted", "pending", ""}
        notes = row.get("provenance_notes") or []
        assert any(str(note).startswith("parse:") for note in notes)


def test_subparameter_registry_contract(phase0_registry_run_dir) -> None:
    payload = read_json(phase0_registry_run_dir / "registry" / "subparameter_registry.json", default={})
    rows = payload.get("subparameters", [])
    assert payload.get("subparameter_count") == len(rows)
    assert rows
    assert payload.get("determinant_silos", {})
    assert payload.get("literature_review_path")
    allowed_banks = {
        "phase0_extracted",
        "phase0_wide_sweep_literature",
        "phase0_wide_sweep_hiv_direct",
        "phase0_wide_sweep_upstream_determinants",
    }
    for row in rows:
        assert row.get("subparameter_id")
        assert row.get("source_bank") in allowed_banks
        details = row.get("literature_ref_details") or []
        if row.get("source_bank") != "phase0_extracted" and details:
            assert all(has_verifiable_locator(detail) for detail in details)


def test_locator_truth_is_explicit() -> None:
    assert has_verifiable_locator({"url": "https://example.com"}) is True
    assert has_verifiable_locator({"doi": "10.1000/test"}) is True
    assert has_verifiable_locator({"pmid": "12345"}) is True
    assert has_verifiable_locator({"openalex_id": "W123"}) is True
    assert has_verifiable_locator({"title": "missing locator"}) is False
