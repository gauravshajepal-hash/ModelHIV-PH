from __future__ import annotations

from epigraph_ph.adapters.structured_sources import get_structured_source_adapters
from epigraph_ph.core.disease_plugin import get_disease_plugin, list_disease_plugins


def test_hiv_plugin_registry_contract() -> None:
    plugin = get_disease_plugin("hiv")
    assert plugin.plugin_id == "hiv"
    assert plugin.state_catalog == ["U", "D", "A", "V", "L"]
    assert "testing_uptake" in plugin.determinant_silos
    assert plugin.structured_source_adapters
    assert plugin.node_graph_defaults


def test_builtin_plugins_list_is_not_empty() -> None:
    plugin_ids = {plugin.plugin_id for plugin in list_disease_plugins()}
    assert "hiv" in plugin_ids


def test_structured_source_adapters_cover_requested_non_hiv_sources() -> None:
    adapter_ids = {row.adapter_id for row in get_structured_source_adapters("hiv")}
    assert {"ndhs", "yafs", "fies", "philgis_boundary_proxy", "transport_network_proxies"}.issubset(adapter_ids)
