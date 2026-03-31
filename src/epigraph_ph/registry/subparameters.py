from __future__ import annotations

from pathlib import Path
from typing import Any

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase0.literature_candidates import wide_sweep_candidate_rows
from epigraph_ph.registry.models import LiteratureRefDetail, has_verifiable_locator
from epigraph_ph.runtime import read_json, write_json


def _wide_sweep_bank_rows(records: list[dict[str, Any]], *, bank_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        expanded = []
        for row in wide_sweep_candidate_rows(record, bank_name=bank_name):
            detail = LiteratureRefDetail(
                source_id=str(record.get("record_id") or ""),
                title=record.get("title"),
                year=record.get("year"),
                source_tier=record.get("source_tier"),
                url=record.get("url"),
                doi=record.get("doi"),
                pmid=record.get("pmid") if record.get("platform") == "pubmed" else None,
                openalex_id=record.get("openalex_id") if record.get("platform") == "openalex" else None,
            ).to_dict()
            patched = dict(row)
            patched["literature_ref_details"] = [detail] if has_verifiable_locator(detail) else []
            expanded.append(patched)
        rows.extend(expanded)
    return rows


def build_subparameter_registry(*, plugin_id: str, output_path: str | Path, phase0_run_dir: str | Path) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase0_dir = Path(phase0_run_dir)
    extracted_candidates = read_json(phase0_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    wide_dir = phase0_dir / "wide_sweep"
    union_records = read_json(wide_dir / "registry_eligible_records.json", default=[])
    hiv_records = read_json(wide_dir / "registry_eligible_hiv_direct_records.json", default=[])
    upstream_records = read_json(wide_dir / "registry_eligible_upstream_determinant_records.json", default=[])
    registry_rows: list[dict[str, Any]] = []
    for row in extracted_candidates:
        patched = dict(row)
        patched["source_bank"] = row.get("source_bank") or "phase0_extracted"
        patched["subparameter_id"] = row.get("candidate_id")
        registry_rows.append(patched)
    registry_rows.extend(_wide_sweep_bank_rows(union_records, bank_name="phase0_wide_sweep_literature"))
    registry_rows.extend(_wide_sweep_bank_rows(hiv_records, bank_name="phase0_wide_sweep_hiv_direct"))
    registry_rows.extend(_wide_sweep_bank_rows(upstream_records, bank_name="phase0_wide_sweep_upstream_determinants"))
    by_source_bank: dict[str, int] = {}
    for row in registry_rows:
        bank = str(row.get("source_bank") or "unknown")
        by_source_bank[bank] = by_source_bank.get(bank, 0) + 1
    payload = {
        "plugin_id": plugin_id,
        "subparameter_count": len(registry_rows),
        "by_source_bank": by_source_bank,
        "determinant_silos": {key: value.to_dict() for key, value in plugin.determinant_silos.items()},
        "literature_review_path": str(phase0_dir / "phase0" / "literature_review" / "literature_review_by_silo.json"),
        "subparameters": registry_rows,
    }
    write_json(Path(output_path), payload)
    return payload
