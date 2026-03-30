from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Phase0BackendStatus:
    name: str
    available: bool
    selected: bool
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Phase0ManifestArtifact:
    plugin_id: str
    run_id: str
    generated_at: str
    raw_dir: str
    parsed_dir: str
    extracted_dir: str
    index_dir: str
    stage_status: dict[str, str]
    artifact_paths: dict[str, str]
    backend_status: dict[str, Phase0BackendStatus]
    source_count: int = 0
    document_count: int = 0
    parsed_block_count: int = 0
    table_count: int = 0
    numeric_observation_count: int = 0
    canonical_candidate_count: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend_status"] = {
            key: value.to_dict() if isinstance(value, Phase0BackendStatus) else value
            for key, value in self.backend_status.items()
        }
        return payload
