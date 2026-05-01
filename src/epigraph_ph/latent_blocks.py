from __future__ import annotations

import re
from typing import Any, Mapping

from epigraph_ph.core.disease_plugin import get_disease_plugin


def _phase15_latent_cfg(plugin_id: str) -> dict[str, Any]:
    plugin = get_disease_plugin(plugin_id)
    phase15_cfg = dict((plugin.constraint_settings or {}).get("phase15", {}) or {})
    return dict(phase15_cfg.get("latent_blocks", {}) or {})


def _coerce_sign(value: Any) -> str:
    sign = str(value or "neutral").strip().lower()
    if sign not in {"positive", "negative", "neutral"}:
        return "neutral"
    return sign


def _coerce_indicator_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, str):
        return {
            "expected_sign": _coerce_sign(payload),
            "role_hint": "auto",
            "literature_basis": [],
        }
    raw = dict(payload or {})
    return {
        "expected_sign": _coerce_sign(raw.get("expected_sign")),
        "role_hint": str(raw.get("role_hint") or "auto"),
        "literature_basis": [str(item) for item in list(raw.get("literature_basis") or []) if str(item or "").strip()],
    }


def latent_block_specs(plugin_id: str) -> list[dict[str, Any]]:
    cfg = _phase15_latent_cfg(plugin_id)
    rows: list[dict[str, Any]] = []
    for raw_block in list(cfg.get("blocks") or []):
        block = dict(raw_block or {})
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        indicators = {
            str(canonical_name): _coerce_indicator_payload(payload)
            for canonical_name, payload in dict(block.get("indicators") or {}).items()
            if str(canonical_name or "").strip()
        }
        rows.append(
            {
                "block_id": block_id,
                "display_name": str(block.get("display_name") or block_id.replace("_", " ").title()),
                "description": str(block.get("description") or ""),
                "minimum_direct_indicators": int(block.get("minimum_direct_indicators") or 1),
                "minimum_indicator_count": int(block.get("minimum_indicator_count") or 2),
                "literature_basis": [str(item) for item in list(block.get("literature_basis") or []) if str(item or "").strip()],
                "indicators": indicators,
            }
        )
    return rows


def latent_indicator_lookup(plugin_id: str) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for block in latent_block_specs(plugin_id):
        for canonical_name, indicator in dict(block.get("indicators") or {}).items():
            lookup[str(canonical_name)] = {
                "block_id": str(block["block_id"]),
                "display_name": str(block["display_name"]),
                "description": str(block["description"]),
                "expected_sign": str(indicator["expected_sign"]),
                "role_hint": str(indicator["role_hint"]),
                "literature_basis": sorted(
                    {
                        *[str(item) for item in list(block.get("literature_basis") or []) if str(item or "").strip()],
                        *[str(item) for item in list(indicator.get("literature_basis") or []) if str(item or "").strip()],
                    }
                ),
            }
    return lookup


def _is_numeric_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value).strip()
    if not text:
        return False
    try:
        float(text)
    except Exception:
        return False
    return True


def classify_measurement_role(row: Mapping[str, Any]) -> tuple[str, str]:
    numeric_present = _is_numeric_value(row.get("model_numeric_value")) or _is_numeric_value(row.get("value"))
    is_direct = bool(row.get("is_direct_measurement"))
    is_anchor = bool(row.get("is_anchor_eligible"))
    is_prior_only = bool(row.get("is_prior_only"))
    if numeric_present and (is_direct or is_anchor):
        return "direct_indicator", "numeric_direct_or_anchor"
    if numeric_present and not is_prior_only:
        return "proxy_indicator", "numeric_non_anchor"
    if numeric_present and is_prior_only:
        return "proxy_indicator", "numeric_prior_proxy"
    return "context_only", "non_numeric_or_text_context"


def infer_candidate_block(canonical_name: str, plugin_id: str) -> tuple[str, str]:
    token = str(canonical_name or "").strip()
    lookup = latent_indicator_lookup(plugin_id)
    if token in lookup:
        return str(lookup[token]["block_id"]), "latent_block_spec"
    lowered = token.lower()
    if any(part in lowered for part in ("testing", "diagnos", "knowledge", "prevention", "stigma", "education", "cd4", "self_testing", "community_testing")):
        return "testing_prevention_reach", "heuristic_token"
    if any(part in lowered for part in ("linkage", "retention", "treatment", "art", "philhealth", "ltfu", "reengagement", "appointment")):
        return "care_access_continuity", "heuristic_token"
    if any(part in lowered for part in ("suppression", "viral", "clinic", "service", "policy", "lab", "reagent", "documentation")):
        return "suppression_capacity", "heuristic_token"
    if any(part in lowered for part in ("mobility", "migration", "sexual_risk", "risk_behavior", "mixing", "key_population", "msm", "tgw", "fsw", "pwid", "geosocial", "partner_seeking", "population_density", "urbanization")):
        return "mobility_exposure_pressure", "heuristic_token"
    if any(part in lowered for part in ("poverty", "friction", "travel", "remoteness", "precarity", "cash", "social_capital", "constraint", "stockout", "backlog", "reporting_delay")):
        return "structural_barrier_pressure", "heuristic_token"
    return "unassigned", "unassigned"


def infer_expected_sign(canonical_name: str, plugin_id: str) -> tuple[str, str]:
    token = str(canonical_name or "").strip()
    lookup = latent_indicator_lookup(plugin_id)
    if token in lookup:
        return str(lookup[token]["expected_sign"]), "latent_block_spec"
    lowered = token.lower()
    if any(part in lowered for part in ("uptake", "coverage", "knowledge", "suppression", "retention", "linkage", "clinic", "access", "capacity", "completeness", "reengagement", "refill", "education")):
        return "positive", "heuristic_token"
    if any(part in lowered for part in ("barrier", "friction", "cost", "poverty", "precarity", "weakness", "delay", "travel", "remoteness", "lapse", "stockout", "interruption", "loss_to_follow_up", "backlog", "fear")):
        return "negative", "heuristic_token"
    return "neutral", "unassigned"


def infer_observation_operator(row: Mapping[str, Any], measurement_role: str) -> str:
    if measurement_role == "context_only":
        return "prior_only"
    time_resolution = str(row.get("time_resolution") or "").strip().lower()
    time_value = str(row.get("time") or "").strip()
    source_bank = str(row.get("source_bank") or "").strip().lower()
    source_title = str(row.get("source_title") or "").strip().lower()
    if "survey" in source_bank or "survey" in source_title:
        return "survey_wave_mean"
    if time_resolution == "monthly" or re.fullmatch(r"\d{4}-\d{2}", time_value):
        return "monthly_snapshot"
    if time_resolution == "annual" or re.fullmatch(r"\d{4}", time_value):
        return "annual_snapshot"
    if measurement_role == "proxy_indicator":
        return "proxy_snapshot"
    return "snapshot_unknown"


def annotate_latent_indicator_fields(row: Mapping[str, Any], plugin_id: str) -> dict[str, Any]:
    canonical_name = str(row.get("canonical_name") or "")
    candidate_block, block_source = infer_candidate_block(canonical_name, plugin_id)
    expected_sign, sign_source = infer_expected_sign(canonical_name, plugin_id)
    measurement_role, role_reason = classify_measurement_role(row)
    lookup = latent_indicator_lookup(plugin_id)
    lookup_row = dict(lookup.get(canonical_name) or {})
    return {
        "candidate_block": candidate_block,
        "candidate_block_display_name": str(lookup_row.get("display_name") or candidate_block.replace("_", " ").title()),
        "block_source": block_source,
        "expected_sign": expected_sign,
        "sign_source": sign_source,
        "measurement_role": measurement_role,
        "role_reason": role_reason,
        "observation_operator": infer_observation_operator(row, measurement_role),
        "literature_basis": list(lookup_row.get("literature_basis") or []),
    }
