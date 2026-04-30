from __future__ import annotations

from epigraph_ph.phase2.official_augmented_baseline import (
    _is_incidence_safe_official_row,
    _module_canonical_names,
)


def test_incidence_official_augmentation_blocks_incidence_truth_rows() -> None:
    allowed = _module_canonical_names(("incidence_pressure", "diagnosis_delay"))
    assert not _is_incidence_safe_official_row(
        {
            "canonical_name": "key_population_burden",
            "source_metric_name": "new_hiv_infections_new_hiv_infections_all_ages",
            "source_id": "unaids_auto_new_hiv_infections",
            "value": 1000,
        },
        allowed,
    )


def test_incidence_official_augmentation_allows_kp_determinants() -> None:
    allowed = _module_canonical_names(("incidence_pressure", "diagnosis_delay"))
    assert _is_incidence_safe_official_row(
        {
            "canonical_name": "key_population_burden",
            "source_metric_name": "men_who_have_sex_with_men_size_estimate",
            "source_id": "unaids_auto_msm_size",
            "value": 10000,
        },
        allowed,
    )


def test_incidence_official_augmentation_blocks_late_diagnosis_outcome_rows() -> None:
    allowed = _module_canonical_names(("incidence_pressure", "diagnosis_delay"))
    assert not _is_incidence_safe_official_row(
        {
            "canonical_name": "late_hiv_diagnosis_percent",
            "source_metric_name": "treatment_cascade_late_hiv_diagnosis",
            "source_id": "unaids_late_hiv_diagnosis",
            "value": 30,
        },
        allowed,
    )
