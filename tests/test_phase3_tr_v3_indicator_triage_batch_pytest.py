from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_indicator_triage_batch as batch


def test_provenance_class_marks_placeholder() -> None:
    row = {
        "indicator_name": "testing_uptake",
        "current_status": "unused_nonnumeric_source",
        "source_families": "source_phase1_all_geo|source_phase1_national",
        "national_numeric_obs": "0",
    }

    assert batch._provenance_class(row, set()) == "placeholder_nonnumeric"


def test_recommended_action_promotes_dense_unused_numeric() -> None:
    row = {
        "indicator_name": "youth_cases_15_24_period",
        "current_status": "unused_numeric_source",
        "cadence_class": "monthly_dense",
        "national_numeric_obs": "294",
    }

    action, rationale = batch._recommended_action(row, "raw_source_numeric", set())

    assert action == "promote_measurement_anchor"
    assert "dense national numeric support" in rationale


def test_raw_program_field_action_prioritizes_diagnosed_share() -> None:
    row = {"indicator_name": "diagnosed_share"}

    action, rationale = batch._raw_program_field_action(row)

    assert action == "candidate_first_class_canonical"
    assert "monthly dense ratio series" in rationale
