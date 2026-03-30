from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LiteratureSiloDefinition:
    silo_id: str
    display_name: str
    description: str
    query_examples: list[str]
    promotion_track: str = "supporting_context"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StructuredSourceAdapterSpec:
    adapter_id: str
    source_name: str
    organization: str
    source_tier: str
    access_mode: str
    spatial_resolution: str
    temporal_resolution: str
    landing_url: str
    determinant_silos: list[str]
    promotion_track: str
    platform: str
    seed_queries: list[str] = field(default_factory=list)
    preferred_file_patterns: list[str] = field(default_factory=list)
    fallback_urls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DiseasePlugin:
    plugin_id: str
    display_name: str
    state_catalog: list[str]
    observation_families: list[str]
    intervention_channels: list[str]
    query_banks: dict[str, list[str]]
    determinant_silos: dict[str, LiteratureSiloDefinition]
    structured_source_adapters: list[StructuredSourceAdapterSpec]
    node_graph_defaults: list[dict[str, Any]]
    prior_hyperparameters: dict[str, Any] = field(default_factory=dict)
    numerical_stabilizers: dict[str, Any] = field(default_factory=dict)
    constraint_settings: dict[str, Any] = field(default_factory=dict)
    policy_settings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugin_id": self.plugin_id,
            "display_name": self.display_name,
            "state_catalog": list(self.state_catalog),
            "observation_families": list(self.observation_families),
            "intervention_channels": list(self.intervention_channels),
            "query_banks": {key: list(value) for key, value in self.query_banks.items()},
            "determinant_silos": {key: value.to_dict() for key, value in self.determinant_silos.items()},
            "structured_source_adapters": [row.to_dict() for row in self.structured_source_adapters],
            "node_graph_defaults": [dict(row) for row in self.node_graph_defaults],
            "prior_hyperparameters": dict(self.prior_hyperparameters),
            "numerical_stabilizers": dict(self.numerical_stabilizers),
            "constraint_settings": dict(self.constraint_settings),
            "policy_settings": dict(self.policy_settings),
            "notes": list(self.notes),
        }


_PLUGIN_REGISTRY: dict[str, DiseasePlugin] = {}


def register_disease_plugin(plugin: DiseasePlugin) -> DiseasePlugin:
    _PLUGIN_REGISTRY[plugin.plugin_id] = plugin
    return plugin


def _ensure_plugins_loaded() -> None:
    if _PLUGIN_REGISTRY:
        return
    from epigraph_ph.plugins import ensure_builtin_plugins_registered

    ensure_builtin_plugins_registered()


def get_disease_plugin(plugin_id: str) -> DiseasePlugin:
    _ensure_plugins_loaded()
    if plugin_id not in _PLUGIN_REGISTRY:
        raise KeyError(f"unknown disease plugin: {plugin_id}")
    return _PLUGIN_REGISTRY[plugin_id]


def list_disease_plugins() -> list[DiseasePlugin]:
    _ensure_plugins_loaded()
    return [plugin for _, plugin in sorted(_PLUGIN_REGISTRY.items())]
