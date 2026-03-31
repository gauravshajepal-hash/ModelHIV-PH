from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from epigraph_ph.core.disease_plugin import DiseasePlugin
from epigraph_ph.phase0.literature_candidates import wide_sweep_candidate_rows
from epigraph_ph.runtime import ensure_dir, read_json, utc_now_iso, write_json


def _review_cfg(plugin: DiseasePlugin) -> dict[str, Any]:
    cfg = dict((plugin.constraint_settings or {}).get("literature_review", {}) or {})
    if not cfg:
        raise KeyError("Missing HIV literature_review constraint settings")
    return cfg


def _review_silos(source_row: dict[str, Any]) -> list[str]:
    silos = set()
    query_silo = str(source_row.get("query_silo") or "").strip()
    if query_silo:
        silos.add(query_silo)
    for item in source_row.get("determinant_silos") or []:
        token = str(item).strip()
        if token:
            silos.add(token)
    return sorted(silos)


def _quality_score(
    cfg: dict[str, Any],
    source_row: dict[str, Any],
    record: dict[str, Any],
    candidate_count: int,
    parsed: bool,
) -> float:
    tier_weights = dict(cfg["tier_quality_weight"])
    weight_cfg = dict(cfg["quality_score_weights"])
    tier = float(tier_weights.get(str(source_row.get("source_tier") or ""), tier_weights["default"]))
    domain = float(record.get("domain_quality_score") or 0.0)
    anchor = float(weight_cfg["anchor_bonus"]) if source_row.get("is_anchor_eligible") else 0.0
    extraction = min(float(weight_cfg["extraction_cap"]), float(weight_cfg["extraction_per_candidate"]) * candidate_count)
    parse_bonus = float(weight_cfg["parse_bonus"]) if parsed else 0.0
    return round(min(1.0, float(weight_cfg["tier"]) * tier + float(weight_cfg["domain"]) * domain + anchor + extraction + parse_bonus), 4)


def _promotion_status(cfg: dict[str, Any], candidate_rows: list[dict[str, Any]], top_documents: list[dict[str, Any]]) -> dict[str, Any]:
    thresholds = dict(cfg["promotion_thresholds"])
    direct_anchor = sum(1 for row in candidate_rows if row.get("is_anchor_eligible"))
    strong_numeric = sum(
        1
        for row in candidate_rows
        if row.get("value") is not None and float(row.get("confidence") or 0.0) >= float(thresholds["strong_numeric_confidence"]) and not row.get("is_prior_only")
    )
    supporting = sum(1 for row in candidate_rows if float(row.get("confidence") or 0.0) >= float(thresholds["supporting_confidence"]))
    if direct_anchor > 0 or strong_numeric >= int(thresholds["main_predictive_numeric_count"]):
        status = "main_predictive_possible"
    elif supporting > 0 or len(top_documents) >= int(thresholds["supporting_document_count"]):
        status = "supporting_context_only"
    else:
        status = "literature_seed_only"
    return {
        "status": status,
        "direct_anchor_candidate_count": direct_anchor,
        "strong_numeric_candidate_count": strong_numeric,
        "supporting_context_candidate_count": supporting,
        "document_count": len(top_documents),
    }


def build_phase0_literature_review(*, run_dir: str | Path, plugin: DiseasePlugin, output_dir: str | Path) -> dict[str, Any]:
    review_cfg = _review_cfg(plugin)
    run_root = Path(run_dir)
    out_dir = ensure_dir(Path(output_dir))
    silos_dir = ensure_dir(out_dir / "silos")
    source_manifest = read_json(run_root / "phase0" / "raw" / "source_manifest.json", default=[])
    document_manifest = {
        row.get("source_id"): row for row in read_json(run_root / "phase0" / "parsed" / "document_manifest.json", default=[])
    }
    scored_records = {
        row.get("record_id"): row for row in read_json(run_root / "wide_sweep" / "scored_records.json", default=[])
    }
    candidates = read_json(run_root / "phase0" / "extracted" / "canonical_parameter_candidates.json", default=[])
    by_source_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_source_candidates[str(row.get("source_id") or "")].append(row)
    for record in scored_records.values():
        source_id = str(record.get("record_id") or "")
        source_rows = wide_sweep_candidate_rows(record, bank_name="phase0_wide_sweep_literature")
        for row in source_rows:
            by_source_candidates[source_id].append(
                {
                    "canonical_name": row.get("canonical_name"),
                    "candidate_text": row.get("candidate_text"),
                    "confidence": float(record.get("domain_quality_score") or 0.0),
                    "is_direct_measurement": False,
                    "is_anchor_eligible": False,
                }
            )

    source_rows_by_silo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_manifest:
        for silo_id in _review_silos(row):
            if silo_id in plugin.determinant_silos:
                source_rows_by_silo[silo_id].append(row)

    adapter_rows_by_silo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for adapter in plugin.structured_source_adapters:
        for silo_id in adapter.determinant_silos:
            adapter_rows_by_silo[silo_id].append(adapter.to_dict())

    review_rows: list[dict[str, Any]] = []
    for silo_id, definition in plugin.determinant_silos.items():
        silo_source_rows = source_rows_by_silo.get(silo_id, [])
        ranked_documents: list[dict[str, Any]] = []
        silo_candidates: list[dict[str, Any]] = []
        for source_row in silo_source_rows:
            source_id = str(source_row.get("source_id") or "")
            source_candidates = by_source_candidates.get(source_id, [])
            silo_candidates.extend(source_candidates)
            parsed_row = document_manifest.get(source_id, {})
            scored_row = scored_records.get(source_id, {})
            ranked_documents.append(
                {
                    "source_id": source_id,
                    "title": source_row.get("title"),
                    "platform": source_row.get("platform"),
                    "source_tier": source_row.get("source_tier"),
                    "year": source_row.get("year"),
                    "url": source_row.get("url"),
                    "query_silo": source_row.get("query_silo"),
                    "determinant_silos": source_row.get("determinant_silos") or [],
                    "is_anchor_eligible": bool(source_row.get("is_anchor_eligible")),
                    "document_type": source_row.get("document_type", ""),
                    "candidate_count": len(source_candidates),
                    "domain_quality_score": float(scored_row.get("domain_quality_score") or 0.0),
                    "quality_score": _quality_score(
                        review_cfg,
                        source_row,
                        scored_row,
                        len(source_candidates),
                        parsed=str(parsed_row.get("parse_status")) == "parsed",
                    ),
                }
            )
        ranked_documents.sort(key=lambda row: (row["quality_score"], row["candidate_count"], row.get("year") or 0), reverse=True)
        top_documents = ranked_documents[:8]
        canonical_counter = Counter(str(row.get("canonical_name") or "") for row in silo_candidates if row.get("canonical_name"))
        candidate_summaries = []
        for canonical_name, count in canonical_counter.most_common(10):
            matching = [row for row in silo_candidates if row.get("canonical_name") == canonical_name]
            candidate_summaries.append(
                {
                    "canonical_name": canonical_name,
                    "count": count,
                    "mean_confidence": round(
                        sum(float(row.get("confidence") or 0.0) for row in matching) / max(1, len(matching)),
                        4,
                    ),
                    "example_candidate_text": next((row.get("candidate_text") for row in matching if row.get("candidate_text")), ""),
                    "direct_measurement_count": sum(1 for row in matching if row.get("is_direct_measurement")),
                    "anchor_eligible_count": sum(1 for row in matching if row.get("is_anchor_eligible")),
                }
            )
        tier_counts = Counter(str(row.get("source_tier") or "unknown") for row in silo_source_rows)
        source_quality_ladder = [
            {
                "source_tier": tier_name,
                "tier_weight": dict(review_cfg["tier_quality_weight"]).get(tier_name, dict(review_cfg["tier_quality_weight"])["default"]),
                "document_count": tier_counts[tier_name],
                "example_titles": [row["title"] for row in top_documents if row.get("source_tier") == tier_name][:3],
            }
            for tier_name, _ in sorted(
                tier_counts.items(),
                key=lambda item: (dict(review_cfg["tier_quality_weight"]).get(item[0], dict(review_cfg["tier_quality_weight"])["default"]), item[1]),
                reverse=True,
            )
        ]
        promotion = _promotion_status(review_cfg, silo_candidates, top_documents)
        review_row = {
            "silo_id": silo_id,
            "display_name": definition.display_name,
            "description": definition.description,
            "promotion_track": definition.promotion_track,
            "query_examples": list(definition.query_examples),
            "top_documents": top_documents,
            "extracted_candidate_subparameters": candidate_summaries,
            "source_quality_ladder": source_quality_ladder,
            "promotion_eligibility": promotion,
            "structured_source_adapters": adapter_rows_by_silo.get(silo_id, []),
            "document_count": len(silo_source_rows),
            "candidate_count": len(silo_candidates),
        }
        review_rows.append(review_row)
        write_json(silos_dir / f"{silo_id}.json", review_row)

    payload = {
        "plugin_id": plugin.plugin_id,
        "generated_at": utc_now_iso(),
        "silo_count": len(review_rows),
        "silos": review_rows,
    }
    write_json(out_dir / "literature_review_by_silo.json", payload)
    lines = [
        f"# Phase 0 Literature Review by Silo ({plugin.display_name})",
        "",
        f"Generated at: `{payload['generated_at']}`",
        "",
    ]
    for row in review_rows:
        promo = row["promotion_eligibility"]
        lines.extend(
            [
                f"## {row['display_name']}",
                "",
                row["description"],
                "",
                f"- Promotion track: `{row['promotion_track']}`",
                f"- Document count: `{row['document_count']}`",
                f"- Candidate count: `{row['candidate_count']}`",
                f"- Eligibility: `{promo['status']}`",
                f"- Direct anchor candidates: `{promo['direct_anchor_candidate_count']}`",
                f"- Strong numeric candidates: `{promo['strong_numeric_candidate_count']}`",
                "",
                "Top documents:",
            ]
        )
        for item in row["top_documents"][:5]:
            lines.append(
                f"- {item.get('title') or 'Untitled'} [{item.get('source_tier')}] quality={item.get('quality_score')} candidates={item.get('candidate_count')}"
            )
        lines.append("")
        lines.append("Top candidate subparameters:")
        for item in row["extracted_candidate_subparameters"][:5]:
            lines.append(
                f"- {item['canonical_name']} x{item['count']} mean_conf={item['mean_confidence']} anchors={item['anchor_eligible_count']}"
            )
        lines.append("")
    (out_dir / "literature_review_by_silo.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return payload
