from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from epigraph_ph.core.disease_plugin import get_disease_plugin


def _dominant_label(counter: Counter[str], default: str = "unassigned") -> str:
    if not counter:
        return default
    return counter.most_common(1)[0][0]


def _observability_gate_config(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase1_cfg = dict((plugin.constraint_settings or {}).get("phase1", {}) or {})
    gates = dict(phase1_cfg.get("observability_gates") or {})
    national_cfg = dict(gates.get("national_likelihood") or {})
    subnational_cfg = dict(gates.get("subnational_inference") or {})
    return {
        "national_likelihood": {
            "direct_indicator_min": int(national_cfg.get("direct_indicator_min") or 1),
            "numeric_row_min": int(national_cfg.get("numeric_row_min") or 1),
            "aggregate_support_score_min": float(national_cfg.get("aggregate_support_score_min") or 1.0),
        },
        "subnational_inference": {
            "indicator_support_min": int(subnational_cfg.get("indicator_support_min") or 1),
            "numeric_row_min": int(subnational_cfg.get("numeric_row_min") or 2),
            "province_support_min": int(subnational_cfg.get("province_support_min") or 2),
            "regional_support_min": int(subnational_cfg.get("regional_support_min") or 6),
            "regional_support_weight": float(subnational_cfg.get("regional_support_weight") or 0.5),
            "subnational_support_score_min": float(subnational_cfg.get("subnational_support_score_min") or 4.0),
        },
    }


def build_latent_observability_audit(
    *,
    normalized_rows: list[dict[str, Any]],
    parameter_catalog: list[dict[str, Any]],
    plugin_id: str,
) -> dict[str, Any]:
    gate_cfg = _observability_gate_config(plugin_id)
    rollup: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "row_count": 0,
            "numeric_row_count": 0,
            "direct_indicator_count": 0,
            "proxy_indicator_count": 0,
            "context_only_count": 0,
            "anchor_count": 0,
            "national_support_count": 0,
            "regional_support_count": 0,
            "province_support_count": 0,
            "monthly_support_count": 0,
            "annual_support_count": 0,
            "source_banks": Counter(),
            "candidate_blocks": Counter(),
            "expected_signs": Counter(),
            "literature_basis": set(),
        }
    )
    catalog_index = {str(row.get("canonical_name") or ""): dict(row) for row in parameter_catalog}
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        if not canonical_name:
            continue
        item = rollup[canonical_name]
        item["row_count"] += 1
        if row.get("model_numeric_value") is not None or row.get("raw_numeric_value") is not None:
            item["numeric_row_count"] += 1
        role = str(row.get("measurement_role") or "context_only")
        if role == "direct_indicator":
            item["direct_indicator_count"] += 1
        elif role == "proxy_indicator":
            item["proxy_indicator_count"] += 1
        else:
            item["context_only_count"] += 1
        if row.get("is_anchor_eligible"):
            item["anchor_count"] += 1
        geo_resolution = str(row.get("geo_resolution") or "unknown")
        if geo_resolution == "national":
            item["national_support_count"] += 1
        elif geo_resolution == "region":
            item["regional_support_count"] += 1
        elif geo_resolution in {"province", "city"}:
            item["province_support_count"] += 1
        time_resolution = str(row.get("time_resolution") or "unknown")
        if time_resolution == "monthly":
            item["monthly_support_count"] += 1
        elif time_resolution == "annual":
            item["annual_support_count"] += 1
        item["source_banks"][str(row.get("source_bank") or "")] += 1
        item["candidate_blocks"][str(row.get("candidate_block") or "unassigned")] += 1
        item["expected_signs"][str(row.get("expected_sign") or "neutral")] += 1
        item["literature_basis"].update(str(value) for value in list(row.get("literature_basis") or []) if str(value or "").strip())

    rows: list[dict[str, Any]] = []
    eligible_national_count = 0
    eligible_province_count = 0
    eligible_subnational_count = 0
    for canonical_name, item in sorted(rollup.items()):
        candidate_block = _dominant_label(item["candidate_blocks"])
        expected_sign = _dominant_label(item["expected_signs"], default="neutral")
        aggregate_support_score = float(
            int(item["national_support_count"])
            + 0.5 * int(item["annual_support_count"])
            + 0.5 * int(item["monthly_support_count"] > 0)
        )
        subnational_support_score = float(
            int(item["province_support_count"])
            + float(gate_cfg["subnational_inference"]["regional_support_weight"]) * int(item["regional_support_count"])
        )
        eligible_for_national_likelihood = bool(
            item["direct_indicator_count"] >= int(gate_cfg["national_likelihood"]["direct_indicator_min"])
            and item["numeric_row_count"] >= int(gate_cfg["national_likelihood"]["numeric_row_min"])
            and aggregate_support_score >= float(gate_cfg["national_likelihood"]["aggregate_support_score_min"])
            and (item["national_support_count"] > 0 or item["annual_support_count"] > 0 or item["monthly_support_count"] > 0)
        )
        eligible_for_province_graph = bool(
            (item["direct_indicator_count"] > 0 or item["proxy_indicator_count"] > 0)
            and (item["province_support_count"] > 0 or item["regional_support_count"] > 0 or item["national_support_count"] > 0)
        )
        eligible_for_subnational_inference = bool(
            (item["direct_indicator_count"] + item["proxy_indicator_count"]) >= int(gate_cfg["subnational_inference"]["indicator_support_min"])
            and item["numeric_row_count"] >= int(gate_cfg["subnational_inference"]["numeric_row_min"])
            and (
                int(item["province_support_count"]) >= int(gate_cfg["subnational_inference"]["province_support_min"])
                or int(item["regional_support_count"]) >= int(gate_cfg["subnational_inference"]["regional_support_min"])
                or subnational_support_score >= float(gate_cfg["subnational_inference"]["subnational_support_score_min"])
            )
        )
        eligible_national_count += 1 if eligible_for_national_likelihood else 0
        eligible_province_count += 1 if eligible_for_province_graph else 0
        eligible_subnational_count += 1 if eligible_for_subnational_inference else 0
        catalog_row = dict(catalog_index.get(canonical_name) or {})
        rows.append(
            {
                "canonical_name": canonical_name,
                "candidate_block": candidate_block,
                "expected_sign": expected_sign,
                "row_count": int(item["row_count"]),
                "numeric_row_count": int(item["numeric_row_count"]),
                "direct_indicator_count": int(item["direct_indicator_count"]),
                "proxy_indicator_count": int(item["proxy_indicator_count"]),
                "context_only_count": int(item["context_only_count"]),
                "anchor_count": int(item["anchor_count"]),
                "national_support_count": int(item["national_support_count"]),
                "regional_support_count": int(item["regional_support_count"]),
                "province_support_count": int(item["province_support_count"]),
                "monthly_support_count": int(item["monthly_support_count"]),
                "annual_support_count": int(item["annual_support_count"]),
                "aggregate_support_score": round(aggregate_support_score, 6),
                "subnational_support_score": round(subnational_support_score, 6),
                "source_bank_count": len([key for key, value in item["source_banks"].items() if key and value > 0]),
                "literature_basis": sorted(item["literature_basis"]),
                "eligible_for_national_likelihood": eligible_for_national_likelihood,
                "eligible_for_province_graph": eligible_for_province_graph,
                "eligible_for_subnational_inference": eligible_for_subnational_inference,
                "parameter_catalog_row_count": int(catalog_row.get("row_count") or 0),
            }
        )

    return {
        "plugin_id": plugin_id,
        "row_count": len(rows),
        "summary": {
            "eligible_for_national_likelihood_count": eligible_national_count,
            "eligible_for_province_graph_count": eligible_province_count,
            "eligible_for_subnational_inference_count": eligible_subnational_count,
            "direct_indicator_row_total": int(sum(row["direct_indicator_count"] for row in rows)),
            "proxy_indicator_row_total": int(sum(row["proxy_indicator_count"] for row in rows)),
            "context_only_row_total": int(sum(row["context_only_count"] for row in rows)),
        },
        "rows": rows,
    }


def build_direct_contextual_split(
    *,
    normalized_rows: list[dict[str, Any]],
    plugin_id: str,
) -> dict[str, Any]:
    row_counts = Counter(str(row.get("measurement_role") or "context_only") for row in normalized_rows)
    by_canonical: dict[str, Counter[str]] = defaultdict(Counter)
    by_block: dict[str, Counter[str]] = defaultdict(Counter)
    for row in normalized_rows:
        canonical_name = str(row.get("canonical_name") or "")
        role = str(row.get("measurement_role") or "context_only")
        block = str(row.get("candidate_block") or "unassigned")
        if canonical_name:
            by_canonical[canonical_name][role] += 1
        by_block[block][role] += 1

    rows = []
    for canonical_name, counter in sorted(by_canonical.items()):
        dominant_role = "context_only"
        if counter["direct_indicator"] > 0:
            dominant_role = "direct_indicator"
        elif counter["proxy_indicator"] > 0:
            dominant_role = "proxy_indicator"
        rows.append(
            {
                "canonical_name": canonical_name,
                "dominant_measurement_role": dominant_role,
                "direct_indicator_count": int(counter["direct_indicator"]),
                "proxy_indicator_count": int(counter["proxy_indicator"]),
                "context_only_count": int(counter["context_only"]),
            }
        )

    block_rows = []
    for block_id, counter in sorted(by_block.items()):
        block_rows.append(
            {
                "block_id": block_id,
                "direct_indicator_count": int(counter["direct_indicator"]),
                "proxy_indicator_count": int(counter["proxy_indicator"]),
                "context_only_count": int(counter["context_only"]),
            }
        )

    return {
        "plugin_id": plugin_id,
        "summary": {
            "row_counts": dict(row_counts),
            "canonical_count": len(rows),
            "direct_indicator_canonical_count": int(sum(1 for row in rows if row["direct_indicator_count"] > 0)),
            "proxy_indicator_canonical_count": int(sum(1 for row in rows if row["proxy_indicator_count"] > 0)),
            "context_only_canonical_count": int(sum(1 for row in rows if row["context_only_count"] > 0)),
        },
        "rows": rows,
        "blocks": block_rows,
    }
