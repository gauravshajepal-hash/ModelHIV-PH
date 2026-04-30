from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


PHASE3_TARGET_CONTRACT_SCHEMA_VERSION = "phase3_targeted_determinant_contract.v1"


PHASE3_MODULE_CONTRACT: dict[str, dict[str, Any]] = {
    "incidence_pressure": {
        "transitions": ["S_eff_to_infections", "incidence_inflow"],
        "latent_blocks": ["mobility_exposure_pressure", "structural_barrier_pressure"],
        "canonical_names": [
            "key_population_burden",
            "msm_population_share",
            "tgw_population_share",
            "fsw_population_share",
            "pwid_population_share",
            "app_mediated_partner_seeking",
            "geosocial_networking",
            "sexual_risk",
            "collective_risk_behavior",
            "condom_use_barrier",
            "prep_active_refill",
            "prep_lapse",
            "population_density",
            "urbanization_pressure",
            "mobility_network_mixing",
            "labor_migration",
        ],
    },
    "diagnosis_delay": {
        "transitions": ["U_to_D"],
        "latent_blocks": ["testing_prevention_reach", "structural_barrier_pressure"],
        "canonical_names": [
            "testing_uptake",
            "testing_rate",
            "hiv_knowledge_index",
            "sex_education_reach",
            "stigma_barrier",
            "disclosure_fear",
            "late_hiv_diagnosis_percent",
            "advanced_hiv_disease_share",
            "median_cd4_at_diagnosis",
            "median_cd4_at_enrollment",
            "self_testing_access",
            "community_testing_reach",
            "reporting_delay",
            "surveillance_completeness",
        ],
    },
    "linkage_and_art": {
        "transitions": ["D_to_A"],
        "latent_blocks": ["care_access_continuity", "structural_barrier_pressure"],
        "canonical_names": [
            "linkage_to_care",
            "diagnosis_to_art_delay",
            "treatment_initiation_delay",
            "care_navigation",
            "treatment_hub_density",
            "clinics_per_capita",
            "philhealth_coverage",
            "transport_friction",
            "travel_time",
            "remoteness",
            "economic_access_constraint",
        ],
    },
    "retention_ltfu": {
        "transitions": ["A_to_L", "L_to_A"],
        "latent_blocks": ["care_access_continuity", "structural_barrier_pressure", "mobility_exposure_pressure"],
        "canonical_names": [
            "retention_adherence",
            "loss_to_follow_up",
            "treatment_interruption",
            "reengagement_in_care",
            "missed_appointment",
            "stockout_disruption",
            "cash_instability",
            "housing_precarity",
            "labor_migration",
            "transport_friction",
        ],
    },
    "vl_and_suppression": {
        "transitions": ["A_to_V"],
        "latent_blocks": ["suppression_capacity", "care_access_continuity"],
        "canonical_names": [
            "suppression_outcomes",
            "viral_suppression_rate",
            "viral_load_testing_coverage",
            "vl_testing_turnaround",
            "lab_capacity",
            "reagent_stockout",
            "documented_suppression",
            "service_delivery_reach",
            "health_system_reach",
            "health_expenditure",
            "policy_implementation_weakness",
        ],
    },
    "observation_reporting": {
        "transitions": ["observation_process"],
        "latent_blocks": ["testing_prevention_reach", "suppression_capacity"],
        "canonical_names": [
            "reporting_delay",
            "surveillance_completeness",
            "registry_backlog",
            "case_report_timeliness",
            "vl_documentation_completeness",
            "data_quality_audit",
        ],
    },
}


def phase3_target_canonical_names() -> list[str]:
    names: set[str] = set()
    for module in PHASE3_MODULE_CONTRACT.values():
        names.update(str(name) for name in list(module.get("canonical_names") or []) if str(name))
    return sorted(names)


def phase3_target_query_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for module_id, module in PHASE3_MODULE_CONTRACT.items():
        for canonical_name in list(module.get("canonical_names") or []):
            readable = str(canonical_name).replace("_", " ")
            rows.append(
                {
                    "query": f"HIV Philippines {readable}",
                    "query_domain": "phase3_targeted_determinants",
                    "query_lane": "upstream_determinant",
                    "query_geo_focus": "philippines",
                    "query_silo": str(canonical_name),
                    "phase3_module": module_id,
                }
            )
            rows.append(
                {
                    "query": f"HIV {readable} Southeast Asia cohort program implementation",
                    "query_domain": "phase3_targeted_determinants",
                    "query_lane": "upstream_determinant",
                    "query_geo_focus": "global",
                    "query_silo": str(canonical_name),
                    "phase3_module": module_id,
                }
            )
    return rows


def _canonical_name(row: Mapping[str, Any]) -> str:
    return str(row.get("canonical_name") or row.get("factor_name") or row.get("query_silo") or "").strip()


def _measurement_role(row: Mapping[str, Any]) -> str:
    return str(row.get("measurement_role") or row.get("observation_role") or "").strip() or "unknown"


def _source_bank(row: Mapping[str, Any]) -> str:
    return str(row.get("source_bank") or row.get("source_stage") or row.get("platform") or "").strip() or "unknown"


def _row_matches_module(row: Mapping[str, Any], module: Mapping[str, Any]) -> bool:
    canonical = _canonical_name(row)
    if canonical in set(str(name) for name in list(module.get("canonical_names") or [])):
        return True
    block = str(row.get("candidate_block") or row.get("block_id") or "").strip()
    return block in set(str(name) for name in list(module.get("latent_blocks") or []))


def build_phase3_targeted_extraction_audit(
    rows: Iterable[Mapping[str, Any]],
    *,
    artifact_name: str,
) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    all_target_names = phase3_target_canonical_names()
    rows_by_canonical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        canonical = _canonical_name(row)
        if canonical:
            rows_by_canonical[canonical].append(row)

    module_rows: list[dict[str, Any]] = []
    for module_id, module in PHASE3_MODULE_CONTRACT.items():
        target_names = [str(name) for name in list(module.get("canonical_names") or [])]
        matched_rows = [row for row in materialized if _row_matches_module(row, module)]
        covered_names = sorted({name for name in target_names if rows_by_canonical.get(name)})
        role_counts = Counter(_measurement_role(row) for row in matched_rows)
        source_bank_counts = Counter(_source_bank(row) for row in matched_rows)
        module_rows.append(
            {
                "module_id": module_id,
                "transitions": list(module.get("transitions") or []),
                "latent_blocks": list(module.get("latent_blocks") or []),
                "target_canonical_count": len(target_names),
                "covered_canonical_count": len(covered_names),
                "covered_canonical_names": covered_names,
                "missing_canonical_names": [name for name in target_names if name not in set(covered_names)],
                "matched_row_count": len(matched_rows),
                "measurement_role_counts": dict(role_counts),
                "source_bank_counts": dict(source_bank_counts),
            }
        )

    target_rows = [row for row in materialized if _canonical_name(row) in set(all_target_names)]
    return {
        "schema_version": PHASE3_TARGET_CONTRACT_SCHEMA_VERSION,
        "artifact_name": artifact_name,
        "row_count": len(materialized),
        "target_row_count": len(target_rows),
        "target_canonical_count": len(all_target_names),
        "covered_target_canonical_count": len({name for name in all_target_names if rows_by_canonical.get(name)}),
        "module_count": len(PHASE3_MODULE_CONTRACT),
        "modules": module_rows,
        "target_canonical_names": all_target_names,
        "covered_target_canonical_names": sorted({name for name in all_target_names if rows_by_canonical.get(name)}),
        "missing_target_canonical_names": [name for name in all_target_names if not rows_by_canonical.get(name)],
    }


def build_phase2_phase3_bridge_audit(
    *,
    structural_payload: Mapping[str, Any],
    transition_prior_map: Mapping[str, Any],
) -> dict[str, Any]:
    direct_edges = [dict(row) for row in list(structural_payload.get("direct_temporal_edge_rows") or [])]
    hidden_edges = [dict(row) for row in list(structural_payload.get("hidden_driver_rows") or [])]
    rows: list[dict[str, Any]] = []
    for transition, payload in sorted(dict(transition_prior_map).items()):
        target_blocks = dict((payload or {}).get("target_blocks") or {})
        admissible_keys: set[tuple[str, str, int]] = set()
        for target_block, target_payload in target_blocks.items():
            for source_block, source_payload in dict((target_payload or {}).get("source_blocks") or {}).items():
                for lag in list((source_payload or {}).get("lags") or []):
                    admissible_keys.add((str(source_block), str(target_block), int(lag)))
        direct_hits = [
            row
            for row in direct_edges
            if (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) in admissible_keys
        ]
        hidden_hits = [
            row
            for row in hidden_edges
            if (str(row.get("source") or ""), str(row.get("target") or ""), int(row.get("lag") or 0)) in admissible_keys
        ]
        rows.append(
            {
                "transition": str(transition),
                "admissible_edge_count": len(admissible_keys),
                "direct_hit_count": len(direct_hits),
                "hidden_hit_count": len(hidden_hits),
                "direct_hit_edges": direct_hits,
                "hidden_hit_edges": hidden_hits,
            }
        )
    return {
        "schema_version": "phase2_phase3_bridge_audit.v1",
        "transition_count": len(rows),
        "direct_edge_count": len(direct_edges),
        "hidden_edge_count": len(hidden_edges),
        "transition_rows": rows,
    }
