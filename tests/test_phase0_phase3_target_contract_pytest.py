from __future__ import annotations

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.latent_blocks import annotate_latent_indicator_fields
from epigraph_ph.phase0.phase3_target_contract import (
    build_phase2_phase3_bridge_audit,
    build_phase3_targeted_extraction_audit,
    phase3_target_canonical_names,
    phase3_target_query_rows,
)


def test_phase3_target_contract_covers_current_publication_modules() -> None:
    names = set(phase3_target_canonical_names())
    assert "key_population_burden" in names
    assert "app_mediated_partner_seeking" in names
    assert "advanced_hiv_disease_share" in names
    assert "viral_load_testing_coverage" in names
    assert "reporting_delay" in names

    queries = phase3_target_query_rows()
    assert any(row["phase3_module"] == "incidence_pressure" for row in queries)
    assert any(row["query_silo"] == "app_mediated_partner_seeking" for row in queries)


def test_phase3_targeted_extraction_audit_reports_missing_and_covered_names() -> None:
    audit = build_phase3_targeted_extraction_audit(
        [
            {
                "canonical_name": "key_population_burden",
                "candidate_block": "mobility_exposure_pressure",
                "measurement_role": "context_only",
                "source_bank": "phase0_literature_review",
            },
            {
                "canonical_name": "viral_load_testing_coverage",
                "candidate_block": "suppression_capacity",
                "measurement_role": "proxy_indicator",
                "source_bank": "phase0_extracted",
            },
        ],
        artifact_name="unit-test",
    )
    assert audit["covered_target_canonical_count"] == 2
    incidence = next(row for row in audit["modules"] if row["module_id"] == "incidence_pressure")
    assert "key_population_burden" in incidence["covered_canonical_names"]
    assert "app_mediated_partner_seeking" in incidence["missing_canonical_names"]
    suppression = next(row for row in audit["modules"] if row["module_id"] == "vl_and_suppression")
    assert suppression["measurement_role_counts"]["proxy_indicator"] == 1


def test_new_phase3_target_indicators_map_to_latent_blocks() -> None:
    plugin = get_disease_plugin("hiv")
    assert "key_population_burden" in plugin.determinant_silos
    assert "diagnosis_delay_backlog" in plugin.determinant_silos
    assert "vl_lab_capacity" in plugin.determinant_silos

    kp = annotate_latent_indicator_fields({"canonical_name": "key_population_burden"}, "hiv")
    assert kp["candidate_block"] == "mobility_exposure_pressure"
    cd4 = annotate_latent_indicator_fields({"canonical_name": "median_cd4_at_diagnosis"}, "hiv")
    assert cd4["candidate_block"] == "testing_prevention_reach"
    vl = annotate_latent_indicator_fields({"canonical_name": "viral_load_testing_coverage"}, "hiv")
    assert vl["candidate_block"] == "suppression_capacity"


def test_phase2_bridge_audit_counts_transition_admissible_hits() -> None:
    payload = {
        "direct_temporal_edge_rows": [
            {"source": "mobility_exposure_pressure", "target": "structural_barrier_pressure", "lag": 1},
            {"source": "care_access_continuity", "target": "suppression_capacity", "lag": 1},
        ],
        "hidden_driver_rows": [
            {"source": "structural_barrier_pressure", "target": "mobility_exposure_pressure", "lag": 2},
        ],
    }
    prior_map = {
        "U_to_D": {
            "target_blocks": {
                "structural_barrier_pressure": {
                    "source_blocks": {"mobility_exposure_pressure": {"lags": [1]}}
                }
            }
        },
        "A_to_V": {
            "target_blocks": {
                "suppression_capacity": {
                    "source_blocks": {"care_access_continuity": {"lags": [1]}}
                }
            }
        },
    }
    audit = build_phase2_phase3_bridge_audit(structural_payload=payload, transition_prior_map=prior_map)
    rows = {row["transition"]: row for row in audit["transition_rows"]}
    assert rows["U_to_D"]["direct_hit_count"] == 1
    assert rows["A_to_V"]["direct_hit_count"] == 1
