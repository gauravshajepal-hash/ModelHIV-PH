from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

from epigraph_ph.latent_blocks import annotate_latent_indicator_fields


def _geo_resolution_from_row(row: dict[str, Any]) -> str:
    geo_value = str(row.get("geo") or "").strip().lower()
    province_value = str(row.get("province") or "").strip().lower()
    if province_value and geo_value and geo_value != province_value:
        return "city"
    if str(row.get("province") or "").strip():
        return "province"
    region = str(row.get("region") or "").strip().lower()
    if region and region not in {"national", "philippines"}:
        return "region"
    if geo_value in {"philippines", "national"}:
        return "national"
    return "unknown"


def _time_resolution_from_row(row: dict[str, Any]) -> str:
    time_value = str(row.get("time") or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}", time_value):
        return "monthly"
    if re.fullmatch(r"\d{4}", time_value):
        return "annual"
    return str(row.get("time_resolution") or "unknown")


def build_candidate_evidence_rows(*, validated_candidates: list[dict[str, Any]], plugin_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in validated_candidates:
        latent_fields = annotate_latent_indicator_fields(row, plugin_id)
        rows.append(
            {
                "evidence_indicator_id": str(row.get("candidate_id") or row.get("subparameter_id") or ""),
                "source_stage": "phase0_extract",
                "source_bank": str(row.get("source_bank") or "phase0_extracted"),
                "source_id": str(row.get("source_id") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "canonical_name": str(row.get("canonical_name") or ""),
                "candidate_block": str(latent_fields["candidate_block"]),
                "candidate_block_display_name": str(latent_fields["candidate_block_display_name"]),
                "block_source": str(latent_fields["block_source"]),
                "expected_sign": str(latent_fields["expected_sign"]),
                "sign_source": str(latent_fields["sign_source"]),
                "measurement_role": str(latent_fields["measurement_role"]),
                "role_reason": str(latent_fields["role_reason"]),
                "observation_operator": str(latent_fields["observation_operator"]),
                "geo_resolution": _geo_resolution_from_row(row),
                "time_resolution": _time_resolution_from_row(row),
                "is_numeric": row.get("value") is not None,
                "is_direct_measurement": bool(row.get("is_direct_measurement")),
                "is_anchor_eligible": bool(row.get("is_anchor_eligible")),
                "confidence": float(row.get("confidence") or 0.0),
                "evidence_weight_hint": round(float(row.get("confidence") or 0.0), 4),
                "literature_ref_count": len(list(row.get("literature_ref_details") or [])),
                "literature_basis": list(latent_fields.get("literature_basis") or []),
                "top_document_titles": [
                    str(detail.get("title") or "")
                    for detail in list(row.get("literature_ref_details") or [])
                    if str(detail.get("title") or "").strip()
                ][:3],
            }
        )
    return rows


def build_literature_context_rows(*, literature_review_payload: dict[str, Any], plugin_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for silo in list(literature_review_payload.get("silos") or []):
        silo_id = str(silo.get("silo_id") or "")
        promotion = dict(silo.get("promotion_eligibility") or {})
        top_document_titles = [
            str(item.get("title") or "")
            for item in list(silo.get("top_documents") or [])
            if str(item.get("title") or "").strip()
        ][:3]
        adapter_ids = [
            str(item.get("adapter_id") or "")
            for item in list(silo.get("structured_source_adapters") or [])
            if str(item.get("adapter_id") or "").strip()
        ]
        for candidate in list(silo.get("extracted_candidate_subparameters") or []):
            seed_row = {
                "canonical_name": candidate.get("canonical_name"),
                "value": None,
                "time": "",
                "source_bank": "phase0_literature_review",
                "source_title": silo.get("display_name"),
                "is_direct_measurement": False,
                "is_anchor_eligible": False,
                "is_prior_only": True,
            }
            latent_fields = annotate_latent_indicator_fields(seed_row, plugin_id)
            canonical_name = str(candidate.get("canonical_name") or "")
            rows.append(
                {
                    "evidence_indicator_id": f"literature_review:{silo_id}:{canonical_name}",
                    "source_stage": "phase0_literature_review",
                    "source_bank": "phase0_literature_review",
                    "source_id": silo_id,
                    "candidate_id": "",
                    "canonical_name": canonical_name,
                    "candidate_block": str(latent_fields["candidate_block"]),
                    "candidate_block_display_name": str(latent_fields["candidate_block_display_name"]),
                    "block_source": str(latent_fields["block_source"]),
                    "expected_sign": str(latent_fields["expected_sign"]),
                    "sign_source": str(latent_fields["sign_source"]),
                    "measurement_role": "context_only",
                    "role_reason": f"literature_review:{promotion.get('status') or 'supporting_context_only'}",
                    "observation_operator": "prior_only",
                    "geo_resolution": "mixed",
                    "time_resolution": "mixed",
                    "is_numeric": False,
                    "is_direct_measurement": False,
                    "is_anchor_eligible": False,
                    "confidence": float(candidate.get("mean_confidence") or 0.0),
                    "evidence_weight_hint": round(float(candidate.get("mean_confidence") or 0.0), 4),
                    "literature_ref_count": len(list(silo.get("top_documents") or [])),
                    "literature_basis": list(latent_fields.get("literature_basis") or []),
                    "top_document_titles": top_document_titles,
                    "supporting_silo_id": silo_id,
                    "promotion_track": str(silo.get("promotion_track") or ""),
                    "structured_adapter_ids": adapter_ids,
                    "query_examples": list(silo.get("query_examples") or []),
                }
            )
    return rows


def build_block_sign_priors(*, evidence_rows: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]:
    rollup: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "evidence_row_count": 0,
            "measurement_roles": Counter(),
            "expected_signs": Counter(),
            "literature_ref_count_total": 0,
            "literature_basis": set(),
            "supporting_silos": set(),
            "top_document_titles": set(),
        }
    )
    for row in evidence_rows:
        block_id = str(row.get("candidate_block") or "unassigned")
        canonical_name = str(row.get("canonical_name") or "")
        if not canonical_name:
            continue
        item = rollup[(block_id, canonical_name)]
        item["evidence_row_count"] += 1
        item["measurement_roles"][str(row.get("measurement_role") or "context_only")] += 1
        item["expected_signs"][str(row.get("expected_sign") or "neutral")] += 1
        item["literature_ref_count_total"] += int(row.get("literature_ref_count") or 0)
        item["literature_basis"].update(str(value) for value in list(row.get("literature_basis") or []) if str(value or "").strip())
        if str(row.get("supporting_silo_id") or "").strip():
            item["supporting_silos"].add(str(row.get("supporting_silo_id")))
        item["top_document_titles"].update(str(value) for value in list(row.get("top_document_titles") or []) if str(value or "").strip())

    rows: list[dict[str, Any]] = []
    by_block: dict[str, int] = Counter()
    for (block_id, canonical_name), item in sorted(rollup.items()):
        expected_sign = item["expected_signs"].most_common(1)[0][0] if item["expected_signs"] else "neutral"
        row = {
            "block_id": block_id,
            "canonical_name": canonical_name,
            "expected_sign": expected_sign,
            "evidence_row_count": int(item["evidence_row_count"]),
            "direct_indicator_count": int(item["measurement_roles"]["direct_indicator"]),
            "proxy_indicator_count": int(item["measurement_roles"]["proxy_indicator"]),
            "context_only_count": int(item["measurement_roles"]["context_only"]),
            "literature_ref_count_total": int(item["literature_ref_count_total"]),
            "literature_basis": sorted(item["literature_basis"]),
            "supporting_silos": sorted(item["supporting_silos"]),
            "top_document_titles": sorted(item["top_document_titles"])[:5],
        }
        rows.append(row)
        by_block[block_id] += 1
    return {
        "plugin_id": plugin_id,
        "row_count": len(rows),
        "block_count": len(by_block),
        "by_block": dict(by_block),
        "rows": rows,
    }


def build_measurement_manifest(*, evidence_rows: list[dict[str, Any]], plugin_id: str) -> dict[str, Any]:
    role_counts = Counter(str(row.get("measurement_role") or "context_only") for row in evidence_rows)
    block_counts = Counter(str(row.get("candidate_block") or "unassigned") for row in evidence_rows)
    sign_counts = Counter(str(row.get("expected_sign") or "neutral") for row in evidence_rows)
    operator_counts = Counter(str(row.get("observation_operator") or "unknown") for row in evidence_rows)
    stage_counts = Counter(str(row.get("source_stage") or "unknown") for row in evidence_rows)
    block_rows = []
    for block_id, count in sorted(block_counts.items()):
        block_rows.append(
            {
                "block_id": block_id,
                "row_count": int(count),
                "direct_indicator_count": int(
                    sum(
                        1
                        for row in evidence_rows
                        if str(row.get("candidate_block") or "unassigned") == block_id and str(row.get("measurement_role") or "") == "direct_indicator"
                    )
                ),
                "proxy_indicator_count": int(
                    sum(
                        1
                        for row in evidence_rows
                        if str(row.get("candidate_block") or "unassigned") == block_id and str(row.get("measurement_role") or "") == "proxy_indicator"
                    )
                ),
                "context_only_count": int(
                    sum(
                        1
                        for row in evidence_rows
                        if str(row.get("candidate_block") or "unassigned") == block_id and str(row.get("measurement_role") or "") == "context_only"
                    )
                ),
            }
        )
    return {
        "plugin_id": plugin_id,
        "row_count": len(evidence_rows),
        "measurement_role_counts": dict(role_counts),
        "candidate_block_counts": dict(block_counts),
        "expected_sign_counts": dict(sign_counts),
        "observation_operator_counts": dict(operator_counts),
        "source_stage_counts": dict(stage_counts),
        "blocks": block_rows,
    }
