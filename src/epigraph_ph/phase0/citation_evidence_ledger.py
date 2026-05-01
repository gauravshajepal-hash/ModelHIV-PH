from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from epigraph_ph.phase0.official_determinant_bridge import build_official_determinant_bridge_rows
from epigraph_ph.phase0.phase3_target_contract import PHASE3_MODULE_CONTRACT, phase3_target_canonical_names
from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json


LEDGER_SCHEMA_VERSION = "phase3_determinant_citation_evidence_ledger.v1"


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_id(row: Mapping[str, Any]) -> str:
    value = _clean_text(row.get("source_id"))
    if value:
        return value
    refs = list(row.get("literature_ref_details") or [])
    if refs and isinstance(refs[0], Mapping):
        return _clean_text(refs[0].get("source_id")) or _stable_hash(refs[0])[:24]
    return _stable_hash(row)[:24]


def _first_ref(row: Mapping[str, Any]) -> dict[str, Any]:
    refs = list(row.get("literature_ref_details") or [])
    if refs and isinstance(refs[0], Mapping):
        return dict(refs[0])
    return {}


def _source_family(*, row: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    platform = _clean_text(source.get("platform") or row.get("platform")).lower()
    organization = _clean_text(source.get("organization")).lower()
    source_id = _clean_text(source.get("source_id") or row.get("source_id")).lower()
    source_bank = _clean_text(row.get("source_bank")).lower()
    title = _clean_text(source.get("source_name") or source.get("title") or row.get("source_title")).lower()
    title_identity = title

    if platform == "unaids" or organization == "unaids" or source_id.startswith("unaids"):
        return "unaids"
    if platform in {"world_bank_wdi", "worldbank", "world_bank"} or "world bank" in organization or source_id.startswith(("world_bank", "wdi_")):
        return "world_bank_wdi"
    if platform == "fies" or source_id.startswith("fies") or "family income and expenditure" in title_identity:
        return "fies"
    if platform == "who" or organization in {"who", "world health organization"} or source_id.startswith("who_"):
        return "who"
    if platform in {"doh", "harp", "hasp", "harp_archive", "doh_harp_hasp"} or source_id.startswith(("doh_", "harp_archive", "historical_harp", "hasp_")):
        return "doh_harp_hasp"
    if platform == "psa" or source_id.startswith("psa") or "philippine statistics authority" in organization:
        return "psa"
    if platform == "google_mobility" or source_id.startswith("google_mobility"):
        return "google_mobility"
    if platform == "pubmed" or source_id.startswith("pubmed"):
        return "pubmed"
    if platform == "crossref" or source_id.startswith("crossref"):
        return "crossref"
    if platform == "openalex" or source_id.startswith("openalex"):
        return "openalex"
    fallback = _clean_text(source.get("platform") or row.get("platform") or row.get("source_bank"))
    return fallback or "unknown"


def _locator_payload(*, row: Mapping[str, Any], source: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, Any]:
    doi = _clean_text(source.get("doi") or ref.get("doi") or row.get("doi"))
    pmid = _clean_text(source.get("pmid") or ref.get("pmid") or row.get("pmid"))
    openalex_id = _clean_text(source.get("openalex_id") or ref.get("openalex_id") or row.get("openalex_id"))
    url = _clean_text(source.get("url") or ref.get("url") or row.get("url"))
    return {
        "doi": doi or None,
        "pmid": pmid or None,
        "openalex_id": openalex_id or None,
        "url": url or None,
        "has_verifiable_locator": bool(doi or pmid or openalex_id or url),
    }


def _allowed_use(*, row: Mapping[str, Any], directness: str, source_family: str, canonical_name: str) -> str:
    hinted = _clean_text(row.get("allowed_use_hint") or row.get("official_allowed_use_hint"))
    if hinted:
        return hinted
    if source_family == "unaids":
        return "validation_or_auxiliary_only"
    if source_family == "who":
        return "anchor_context_only"
    if directness == "direct_numeric_measurement":
        return "direct_determinant_covariate_candidate"
    if canonical_name in {"reporting_delay", "surveillance_completeness", "registry_backlog", "case_report_timeliness", "data_quality_audit"}:
        return "reporting_process_context_only"
    return "prior_context_only"


def _directness(row: Mapping[str, Any]) -> str:
    role = _clean_text(row.get("measurement_role"))
    has_value = row.get("model_numeric_value") is not None or row.get("value") is not None or row.get("raw_numeric_value") is not None
    if bool(row.get("is_direct_measurement")) and has_value:
        return "direct_numeric_measurement"
    if bool(row.get("is_anchor_eligible")):
        return "official_anchor_context"
    if role == "context_only":
        return "contextual_literature_support"
    return role or "unspecified_support"


def _measurement_semantics(*, canonical_name: str, directness: str) -> str:
    if canonical_name in {"reporting_delay", "surveillance_completeness", "registry_backlog", "case_report_timeliness", "vl_documentation_completeness", "data_quality_audit"}:
        return "reporting_process_covariate"
    if directness == "direct_numeric_measurement":
        return "determinant_covariate"
    return "prior_context"


def _target_index() -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for module_id, module in PHASE3_MODULE_CONTRACT.items():
        for canonical_name in list(module.get("canonical_names") or []):
            payload = index.setdefault(
                str(canonical_name),
                {"module_ids": set(), "transition_names": set(), "latent_blocks": set()},
            )
            payload["module_ids"].add(module_id)
            payload["transition_names"].update(str(value) for value in list(module.get("transitions") or []))
            payload["latent_blocks"].update(str(value) for value in list(module.get("latent_blocks") or []))
    return {
        key: {
            "module_ids": sorted(value["module_ids"]),
            "transition_names": sorted(value["transition_names"]),
            "latent_blocks": sorted(value["latent_blocks"]),
        }
        for key, value in index.items()
    }


def _known_latent_blocks() -> set[str]:
    blocks: set[str] = set()
    for module in PHASE3_MODULE_CONTRACT.values():
        blocks.update(str(value) for value in list(module.get("latent_blocks") or []) if str(value))
    return blocks


def _candidate_latent_block(row: Mapping[str, Any]) -> str:
    known_blocks = _known_latent_blocks()
    candidate = _clean_text(row.get("candidate_block") or row.get("block_id"))
    if candidate in known_blocks:
        return candidate
    canonical_name = _clean_text(row.get("canonical_name"))
    fallback_blocks = list((_target_index().get(canonical_name) or {}).get("latent_blocks") or [])
    fallback = _clean_text(fallback_blocks[0] if fallback_blocks else "")
    return fallback if fallback in known_blocks else "unassigned"


def _source_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_clean_text(row.get("source_id")): dict(row) for row in rows if _clean_text(row.get("source_id"))}


def _source_summary(source_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    for row in source_rows:
        family = _source_family(row={}, source=row)
        family_counts[family] += 1
        tier_counts[_clean_text(row.get("source_tier")) or "unknown"] += 1
    required = ["who", "fies", "world_bank_wdi", "unaids"]
    return {
        "source_family_counts": dict(sorted(family_counts.items())),
        "source_tier_counts": dict(sorted(tier_counts.items())),
        "required_official_family_counts": {family: int(family_counts.get(family, 0)) for family in required},
        "missing_required_official_families": [family for family in required if int(family_counts.get(family, 0)) == 0],
    }


def _example_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    evidence_span = _clean_text(row.get("evidence_span") or row.get("candidate_text") or row.get("parameter_text"))
    value = row.get("model_numeric_value")
    if value is None:
        value = row.get("value")
    if value is None:
        value = row.get("raw_numeric_value")
    return {
        "time": _clean_text(row.get("time") or row.get("year")),
        "geo": _clean_text(row.get("geo") or row.get("region") or row.get("province")),
        "value": value,
        "unit": _clean_text(row.get("normalized_unit") or row.get("unit") or row.get("original_unit")),
        "evidence_span": evidence_span[:500],
    }


def build_phase3_determinant_evidence_ledger(
    *,
    run_dir: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    target_names = set(phase3_target_canonical_names())
    target_meta = _target_index()
    source_rows = list(read_json(run_dir / "phase0" / "raw" / "source_manifest.json", default=[]) or [])
    sources = _source_map(source_rows)
    candidate_rows = list(read_json(run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[]) or [])
    base_candidate_row_count = len(candidate_rows)
    destination = ensure_dir(out_dir or (run_dir / "phase0" / "evidence_ledger"))
    official_bridge_summary = build_official_determinant_bridge_rows(run_dir=run_dir, out_dir=destination)
    official_bridge_rows = list(read_json(destination / "official_determinant_candidate_rows.json", default=[]) or [])
    candidate_rows.extend(dict(row) for row in official_bridge_rows if isinstance(row, Mapping))
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for row in candidate_rows:
        canonical_name = _clean_text(row.get("canonical_name"))
        if canonical_name not in target_names:
            continue
        source_id = _source_id(row)
        source = dict(sources.get(source_id) or {})
        ref = _first_ref(row)
        if not source:
            source = {
                "source_id": source_id,
                "title": ref.get("title") or row.get("source_title"),
                "source_name": ref.get("title") or row.get("source_title"),
                "source_tier": ref.get("source_tier") or row.get("source_tier"),
                "url": ref.get("url"),
                "doi": ref.get("doi"),
                "pmid": ref.get("pmid"),
                "openalex_id": ref.get("openalex_id"),
                "year": ref.get("year") or row.get("year"),
            }
        locators = _locator_payload(row=row, source=source, ref=ref)
        source_family = _source_family(row=row, source=source)
        directness = _directness(row)
        key = (canonical_name, source_id)
        group = groups.setdefault(
            key,
            {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "determinant_name": canonical_name,
                "module_ids": list(target_meta.get(canonical_name, {}).get("module_ids", [])),
                "transition_names": list(target_meta.get(canonical_name, {}).get("transition_names", [])),
                "contract_latent_blocks": list(target_meta.get(canonical_name, {}).get("latent_blocks", [])),
                "candidate_block": _candidate_latent_block(row),
                "expected_sign": _clean_text(row.get("expected_sign")),
                "source_id": source_id,
                "source_title": _clean_text(source.get("title") or source.get("source_name") or row.get("source_title")),
                "source_family": source_family,
                "source_tier": _clean_text(source.get("source_tier") or row.get("source_tier")) or "unknown",
                "organization": _clean_text(source.get("organization")),
                "platform": _clean_text(source.get("platform") or row.get("platform") or row.get("source_bank")),
                **locators,
                "evidence_directness_counts": Counter(),
                "measurement_role_counts": Counter(),
                "source_bank_counts": Counter(),
                "years": set(),
                "geographies": set(),
                "row_count": 0,
                "direct_numeric_row_count": 0,
                "example_observations": [],
                "mean_evidence_weight_sum": 0.0,
                "mean_observation_weight_sum": 0.0,
                "claim": "",
                "limitations": set(),
                "allowed_use_hints": set(),
            },
        )
        group["row_count"] += 1
        hinted_allowed_use = _clean_text(row.get("allowed_use_hint") or row.get("official_allowed_use_hint"))
        if hinted_allowed_use:
            group["allowed_use_hints"].add(hinted_allowed_use)
        group["evidence_directness_counts"][directness] += 1
        group["measurement_role_counts"][_clean_text(row.get("measurement_role")) or "unknown"] += 1
        group["source_bank_counts"][_clean_text(row.get("source_bank")) or "unknown"] += 1
        year = _clean_text(row.get("year") or row.get("time"))
        if year:
            group["years"].add(year[:4])
        geo = _clean_text(row.get("geo") or row.get("geo_scope") or row.get("region") or row.get("province"))
        if geo:
            group["geographies"].add(geo)
        if directness == "direct_numeric_measurement":
            group["direct_numeric_row_count"] += 1
        if len(group["example_observations"]) < 5:
            group["example_observations"].append(_example_observation(row))
        group["mean_evidence_weight_sum"] += float(row.get("evidence_weight") or row.get("confidence") or 0.0)
        group["mean_observation_weight_sum"] += float(row.get("observation_weight") or 0.0)
        if not bool(group["has_verifiable_locator"]):
            group["limitations"].add("no DOI, PMID, OpenAlex id, or URL locator")
        if directness != "direct_numeric_measurement":
            group["limitations"].add("contextual support; not a direct numeric determinant observation")
        if source_family == "unaids":
            group["limitations"].add("UNAIDS modeled or aggregated series must not be treated as direct determinant truth without an explicit allowed-use override")
        if source_family == "who":
            group["limitations"].add("WHO guidance/profile source is an anchor or context source, not a row-level determinant measurement")

    ledger_rows: list[dict[str, Any]] = []
    for (canonical_name, source_id), group in sorted(groups.items()):
        row_count = max(int(group["row_count"]), 1)
        evidence_counts = dict(sorted(group.pop("evidence_directness_counts").items()))
        role_counts = dict(sorted(group.pop("measurement_role_counts").items()))
        source_bank_counts = dict(sorted(group.pop("source_bank_counts").items()))
        years = sorted(value for value in group.pop("years") if value)
        geographies = sorted(value for value in group.pop("geographies") if value)
        limitations = sorted(group.pop("limitations"))
        allowed_use_hints = sorted(value for value in group.pop("allowed_use_hints") if value)
        directness = "direct_numeric_measurement" if evidence_counts.get("direct_numeric_measurement", 0) else next(iter(evidence_counts or {"unspecified_support": 1}))
        representative_row = {"allowed_use_hint": allowed_use_hints[0]} if allowed_use_hints else {}
        allowed_use = _allowed_use(
            row=representative_row,
            directness=directness,
            source_family=str(group["source_family"]),
            canonical_name=canonical_name,
        )
        measurement_semantics = _measurement_semantics(canonical_name=canonical_name, directness=directness)
        group["evidence_directness_counts"] = evidence_counts
        group["measurement_role_counts"] = role_counts
        group["source_bank_counts"] = source_bank_counts
        group["years"] = years
        group["geographies"] = geographies
        group["allowed_use"] = allowed_use
        group["allowed_use_hints"] = allowed_use_hints
        group["measurement_semantics"] = measurement_semantics
        group["mean_evidence_weight"] = round(float(group.pop("mean_evidence_weight_sum")) / row_count, 6)
        group["mean_observation_weight"] = round(float(group.pop("mean_observation_weight_sum")) / row_count, 6)
        group["citation_guard_status"] = "locator_present" if bool(group["has_verifiable_locator"]) else "unresolved_locator"
        quote_present = any(_clean_text(obs.get("evidence_span")) for obs in list(group.get("example_observations") or []))
        group["quote_span_status"] = "extracted_or_metadata_span_present" if quote_present else "missing_quote_span"
        group["claim"] = (
            f"{group['source_title'] or source_id} provides {directness.replace('_', ' ')} "
            f"for determinant {canonical_name} in modules {', '.join(group['module_ids'])}; "
            "this ledger row is evidence/provenance, not causal identification by itself."
        )
        group["limitations"] = limitations
        group["row_hash"] = _stable_hash(
            {
                "determinant_name": canonical_name,
                "source_id": source_id,
                "row_count": group["row_count"],
                "years": years,
                "allowed_use": allowed_use,
            }
        )
        group["ledger_id"] = f"det-ledger-{group['row_hash'][:16]}"
        ledger_rows.append(group)

    determinant_rows: list[dict[str, Any]] = []
    by_det: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger_rows:
        by_det[str(row["determinant_name"])].append(row)
    for determinant_name in sorted(target_names):
        rows = by_det.get(determinant_name, [])
        families = Counter(str(row.get("source_family") or "unknown") for row in rows)
        allowed = Counter(str(row.get("allowed_use") or "unknown") for row in rows)
        determinant_rows.append(
            {
                "determinant_name": determinant_name,
                "ledger_source_count": len(rows),
                "raw_row_count": int(sum(int(row.get("row_count") or 0) for row in rows)),
                "verifiable_source_count": int(sum(1 for row in rows if bool(row.get("has_verifiable_locator")))),
                "direct_numeric_source_count": int(sum(1 for row in rows if int(row.get("direct_numeric_row_count") or 0) > 0)),
                "source_family_counts": dict(sorted(families.items())),
                "allowed_use_counts": dict(sorted(allowed.items())),
            }
        )

    official_summary = _source_summary(source_rows)
    ledger_family_counts = Counter(str(row.get("source_family") or "unknown") for row in ledger_rows)
    row_compression = _row_compression_summary(ledger_rows)
    summary = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "generated_at": utc_now_iso(),
        "target_determinant_count": len(target_names),
        "covered_determinant_count": sum(1 for row in determinant_rows if int(row["ledger_source_count"]) > 0),
        "missing_determinants": [row["determinant_name"] for row in determinant_rows if int(row["ledger_source_count"]) == 0],
        "ledger_row_count": len(ledger_rows),
        "raw_candidate_row_count": len(candidate_rows),
        "base_candidate_row_count": base_candidate_row_count,
        "official_bridge_candidate_row_count": int(official_bridge_summary.get("candidate_row_count") or 0),
        "official_bridge_summary_path": str((official_bridge_summary.get("artifact_paths") or {}).get("summary_json") or ""),
        "target_candidate_row_count": int(sum(int(row.get("row_count") or 0) for row in ledger_rows)),
        "verifiable_ledger_row_count": int(sum(1 for row in ledger_rows if bool(row.get("has_verifiable_locator")))),
        "direct_numeric_ledger_row_count": int(sum(1 for row in ledger_rows if int(row.get("direct_numeric_row_count") or 0) > 0)),
        "ledger_source_family_counts": dict(sorted(ledger_family_counts.items())),
        "row_compression": row_compression,
        "official_source_manifest_summary": official_summary,
        "determinants": determinant_rows,
        "contract_note": "Rows are evidence/provenance. Causal or mechanistic Phase 3 prior use still requires Phase 2 edge falsification and model validation.",
    }

    ledger_path = destination / "phase3_determinant_evidence_ledger.jsonl"
    with ledger_path.open("w", encoding="utf-8") as handle:
        for row in ledger_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    summary_path = destination / "phase3_determinant_evidence_ledger_summary.json"
    write_json(summary_path, summary)
    markdown_path = destination / "phase3_determinant_evidence_ledger_summary.md"
    markdown_path.write_text(_summary_markdown(summary), encoding="utf-8")
    compression_path = destination / "phase3_determinant_row_compression_report.json"
    write_json(compression_path, row_compression)
    compression_md_path = destination / "phase3_determinant_row_compression_report.md"
    compression_md_path.write_text(_compression_markdown(row_compression), encoding="utf-8")
    compression_dashboard_path = destination / "phase3_determinant_row_compression_dashboard.png"
    compression_dashboard_written = _write_compression_dashboard(row_compression, compression_dashboard_path)
    dashboard_path = destination / "phase3_determinant_evidence_ledger_dashboard.png"
    dashboard_written = _write_dashboard(summary, dashboard_path)
    summary["artifact_paths"] = {
        "ledger_jsonl": str(ledger_path),
        "summary_json": str(summary_path),
        "summary_md": str(markdown_path),
        "row_compression_json": str(compression_path),
        "row_compression_md": str(compression_md_path),
        "row_compression_dashboard_png": str(compression_dashboard_path) if compression_dashboard_written else "",
        "dashboard_png": str(dashboard_path) if dashboard_written else "",
    }
    write_json(summary_path, summary)
    return summary


def _row_compression_summary(ledger_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    family_rows: dict[str, dict[str, Any]] = {}
    determinant_rows: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        family = _clean_text(row.get("source_family")) or "unknown"
        determinant = _clean_text(row.get("determinant_name")) or "unknown"
        raw_count = int(row.get("row_count") or 0)
        direct_count = int(row.get("direct_numeric_row_count") or 0)
        allowed_use = _clean_text(row.get("allowed_use")) or "unknown"
        for key, bucket_map in ((family, family_rows), (determinant, determinant_rows)):
            bucket = bucket_map.setdefault(
                key,
                {
                    "key": key,
                    "ledger_source_count": 0,
                    "raw_row_count": 0,
                    "direct_numeric_raw_row_count": 0,
                    "allowed_use_counts": Counter(),
                },
            )
            bucket["ledger_source_count"] += 1
            bucket["raw_row_count"] += raw_count
            bucket["direct_numeric_raw_row_count"] += direct_count
            bucket["allowed_use_counts"][allowed_use] += 1

    def finalize(bucket: Mapping[str, Any]) -> dict[str, Any]:
        ledger_source_count = max(int(bucket.get("ledger_source_count") or 0), 1)
        raw_row_count = int(bucket.get("raw_row_count") or 0)
        return {
            "key": str(bucket.get("key") or ""),
            "ledger_source_count": int(bucket.get("ledger_source_count") or 0),
            "raw_row_count": raw_row_count,
            "direct_numeric_raw_row_count": int(bucket.get("direct_numeric_raw_row_count") or 0),
            "rows_per_ledger_source": round(raw_row_count / ledger_source_count, 6),
            "allowed_use_counts": dict(sorted(dict(bucket.get("allowed_use_counts") or {}).items())),
        }

    family_final = [finalize(bucket) for bucket in family_rows.values()]
    determinant_final = [finalize(bucket) for bucket in determinant_rows.values()]
    family_final.sort(key=lambda row: (-int(row["raw_row_count"]), str(row["key"])))
    determinant_final.sort(key=lambda row: (-int(row["raw_row_count"]), str(row["key"])))
    return {
        "schema_version": "phase3_determinant_row_compression.v1",
        "source_family_rows": family_final,
        "determinant_rows": determinant_final,
        "interpretation": (
            "Ledger source counts are provenance groups. raw_row_count is the number of candidate observations compressed into those groups; "
            "rows_per_ledger_source explains why some official families look small in source-level plots despite many observations."
        ),
    }


def _compression_markdown(compression: Mapping[str, Any]) -> str:
    lines = [
        "# Phase 3 Determinant Row Compression Report",
        "",
        str(compression.get("interpretation") or ""),
        "",
        "## Source Families",
        "",
        "| Source Family | Ledger Sources | Raw Rows | Direct Numeric Raw Rows | Rows Per Source |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in list(compression.get("source_family_rows") or []):
        lines.append(
            f"| `{row.get('key')}` | {int(row.get('ledger_source_count') or 0)} | "
            f"{int(row.get('raw_row_count') or 0)} | {int(row.get('direct_numeric_raw_row_count') or 0)} | "
            f"{float(row.get('rows_per_ledger_source') or 0.0):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Top Determinants By Compressed Raw Rows",
            "",
            "| Determinant | Ledger Sources | Raw Rows | Direct Numeric Raw Rows | Rows Per Source |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in list(compression.get("determinant_rows") or [])[:25]:
        lines.append(
            f"| `{row.get('key')}` | {int(row.get('ledger_source_count') or 0)} | "
            f"{int(row.get('raw_row_count') or 0)} | {int(row.get('direct_numeric_raw_row_count') or 0)} | "
            f"{float(row.get('rows_per_ledger_source') or 0.0):.2f} |"
        )
    return "\n".join(lines) + "\n"


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    official = dict(summary.get("official_source_manifest_summary") or {}).get("required_official_family_counts") or {}
    lines = [
        "# Phase 3 Determinant Citation Evidence Ledger",
        "",
        f"- Run: `{summary.get('run_id')}`",
        f"- Target determinant coverage: `{summary.get('covered_determinant_count')}/{summary.get('target_determinant_count')}`",
        f"- Ledger rows: `{summary.get('ledger_row_count')}`",
        f"- Verifiable locator rows: `{summary.get('verifiable_ledger_row_count')}`",
        f"- Direct numeric determinant rows: `{summary.get('direct_numeric_ledger_row_count')}`",
        f"- Official bridge candidate rows: `{summary.get('official_bridge_candidate_row_count')}`",
        "",
        "## Required Official Sources",
        "",
        "| Family | Source Manifest Rows |",
        "| --- | ---: |",
    ]
    for family in ["who", "fies", "world_bank_wdi", "unaids"]:
        lines.append(f"| `{family}` | {int(official.get(family, 0))} |")
    lines.extend(["", "## Determinant Coverage", "", "| Determinant | Ledger Sources | Direct Numeric Sources | Verifiable Sources |", "| --- | ---: | ---: | ---: |"])
    for row in list(summary.get("determinants") or []):
        lines.append(
            f"| `{row['determinant_name']}` | {int(row['ledger_source_count'])} | "
            f"{int(row['direct_numeric_source_count'])} | {int(row['verifiable_source_count'])} |"
        )
    lines.extend(["", "## Use Constraint", "", str(summary.get("contract_note") or "")])
    return "\n".join(lines) + "\n"


def _write_dashboard(summary: Mapping[str, Any], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ensure_dir(path.parent)
    official_counts = dict((summary.get("official_source_manifest_summary") or {}).get("required_official_family_counts") or {})
    ledger_family_counts = dict(summary.get("ledger_source_family_counts") or {})
    determinants = list(summary.get("determinants") or [])
    direct_count = sum(1 for row in determinants if int(row.get("direct_numeric_source_count") or 0) > 0)
    context_only_count = sum(1 for row in determinants if int(row.get("ledger_source_count") or 0) > 0 and int(row.get("direct_numeric_source_count") or 0) == 0)
    missing_count = sum(1 for row in determinants if int(row.get("ledger_source_count") or 0) == 0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Phase 3 Determinant Evidence Ledger Coverage", fontsize=16, fontweight="bold")

    families = ["who", "fies", "world_bank_wdi", "unaids"]
    labels = ["WHO", "FIES", "World Bank WDI", "UNAIDS"]
    ax = axes[0, 0]
    ax.bar(labels, [int(official_counts.get(family, 0)) for family in families], color="#315f72")
    ax.set_title("Official Sources Present In Source Universe")
    ax.set_ylabel("source rows")
    ax.tick_params(axis="x", rotation=20)

    ax = axes[0, 1]
    mapped_families = ["fies", "world_bank_wdi", "unaids", "psa", "philhealth", "philhealth_open_portal", "google_mobility"]
    mapped_labels = ["FIES", "World Bank WDI", "UNAIDS", "PSA", "PhilHealth", "PhilHealth Portal", "Google Mobility"]
    ax.bar(mapped_labels, [int(ledger_family_counts.get(family, 0)) for family in mapped_families], color="#7a9a45")
    ax.set_title("Official Rows Mapped Into 64-Determinant Ledger")
    ax.set_ylabel("ledger source rows")
    ax.tick_params(axis="x", rotation=30)

    ax = axes[1, 0]
    ax.bar(["direct numeric", "context only", "missing"], [direct_count, context_only_count, missing_count], color=["#277f8e", "#d19b32", "#b7b7b7"])
    ax.set_title("Determinant Evidence Strength")
    ax.set_ylabel("determinants")

    ax = axes[1, 1]
    top_families = sorted(ledger_family_counts.items(), key=lambda item: int(item[1]), reverse=True)[:8]
    ax.barh([family for family, _count in reversed(top_families)], [int(count) for _family, count in reversed(top_families)], color="#7b4d8a")
    ax.set_title("Top Ledger Source Families")
    ax.set_xlabel("ledger source rows")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_compression_dashboard(compression: Mapping[str, Any], path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    ensure_dir(path.parent)
    family_rows = list(compression.get("source_family_rows") or [])[:12]
    determinant_rows = list(compression.get("determinant_rows") or [])[:12]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("Phase 3 Determinant Row Compression", fontsize=16, fontweight="bold")

    ax = axes[0]
    ax.barh(
        [str(row.get("key")) for row in reversed(family_rows)],
        [int(row.get("raw_row_count") or 0) for row in reversed(family_rows)],
        color="#2f6f73",
    )
    ax.set_title("Raw Candidate Rows By Source Family")
    ax.set_xlabel("raw candidate rows")

    ax = axes[1]
    ax.barh(
        [str(row.get("key")) for row in reversed(determinant_rows)],
        [float(row.get("rows_per_ledger_source") or 0.0) for row in reversed(determinant_rows)],
        color="#b5793a",
    )
    ax.set_title("Rows Compressed Per Ledger Source")
    ax.set_xlabel("raw rows / source")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def _main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 3 determinant citation evidence ledger.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = build_phase3_determinant_evidence_ledger(run_dir=args.run_dir, out_dir=args.out_dir)
    print(json.dumps(summary.get("artifact_paths", {}), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
