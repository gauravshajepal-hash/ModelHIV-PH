from __future__ import annotations

from pathlib import Path
from typing import Any

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.runtime import RunContext, read_json, write_json


def build_source_registry(*, plugin_id: str, output_path: str | Path, phase0_run_dir: str | Path) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase0_dir = Path(phase0_run_dir)
    raw_sources = read_json(phase0_dir / "phase0" / "raw" / "source_manifest.json", default=[])
    parsed_documents = {
        row.get("source_id"): row
        for row in read_json(phase0_dir / "phase0" / "parsed" / "document_manifest.json", default=[])
    }
    registry_rows: list[dict[str, Any]] = []
    for row in raw_sources:
        parsed = parsed_documents.get(row.get("source_id"), {})
        merged = dict(row)
        merged["parse_status"] = parsed.get("parse_status", "pending")
        merged["extraction_status"] = parsed.get("extraction_status", "pending")
        merged["parser_used"] = parsed.get("parser_used", "")
        notes = list(merged.get("provenance_notes") or [])
        notes.append(f"parse:{merged['parse_status']}")
        merged["provenance_notes"] = notes
        registry_rows.append(merged)
    payload = {
        "plugin_id": plugin_id,
        "plugin": plugin.to_dict(),
        "source_count": len(registry_rows),
        "structured_source_adapters": [row.to_dict() for row in plugin.structured_source_adapters],
        "sources": registry_rows,
    }
    write_json(Path(output_path), payload)
    return payload
