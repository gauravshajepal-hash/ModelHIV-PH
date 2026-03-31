from __future__ import annotations

import re
from typing import Any


_GENERIC_QUERY_SILOS = {
    "",
    "mixed",
    "modeling",
    "biology",
    "hiv_direct",
    "philippines_context",
    "philippines_subnational",
    "upstream_structural",
}

_ONTOLOGY_TO_CANONICAL = {
    "economics": "economic_access_constraint",
    "logistics": "transport_friction",
    "behavior": "collective_risk_behavior",
    "population": "mobility_network_mixing",
    "biology": "biological_progression_modifier",
    "policy": "policy_implementation_weakness",
}


def _canonical_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return token


def wide_sweep_record_canonical_names(record: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    query_silo = _canonical_token(str(record.get("query_silo") or ""))
    if query_silo and query_silo not in _GENERIC_QUERY_SILOS:
        names.add(query_silo)
    for key in ("determinant_silos", "linkage_targets", "soft_subparameter_hints"):
        for item in record.get(key) or []:
            token = _canonical_token(str(item))
            if token:
                names.add(token)
    for tag in record.get("soft_ontology_tags") or []:
        mapped = _ONTOLOGY_TO_CANONICAL.get(_canonical_token(str(tag)))
        if mapped:
            names.add(mapped)
    return sorted(names) or ["literature_seed"]


def wide_sweep_candidate_rows(record: dict[str, Any], *, bank_name: str) -> list[dict[str, Any]]:
    detail = {
        "source_id": str(record.get("record_id") or ""),
        "title": record.get("title"),
        "year": record.get("year"),
        "source_tier": record.get("source_tier"),
        "url": record.get("url"),
        "doi": record.get("doi"),
        "pmid": record.get("pmid") if record.get("platform") == "pubmed" else None,
        "openalex_id": record.get("openalex_id") if record.get("platform") == "openalex" else None,
    }
    rows: list[dict[str, Any]] = []
    for canonical_name in wide_sweep_record_canonical_names(record):
        rows.append(
            {
                "subparameter_id": f"{bank_name}:{record.get('record_id')}:{canonical_name}",
                "source_bank": bank_name,
                "candidate_text": (record.get("title") or ""),
                "canonical_name": canonical_name,
                "value": None,
                "unit": "",
                "geo": record.get("query_geo_focus") or "",
                "time": record.get("year") or "",
                "soft_ontology_tags": record.get("soft_ontology_tags") or [],
                "soft_subparameter_hints": record.get("soft_subparameter_hints") or [],
                "linkage_targets": record.get("linkage_targets") or [],
                "literature_ref_details": [detail] if any(detail.get(key) for key in ("url", "doi", "pmid", "openalex_id")) else [],
                "is_anchor_eligible": False,
                "is_direct_measurement": False,
                "is_prior_only": True,
            }
        )
    return rows
