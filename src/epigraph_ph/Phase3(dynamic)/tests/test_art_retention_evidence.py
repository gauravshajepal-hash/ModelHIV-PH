from __future__ import annotations

from phase3_dynamic.art_retention_evidence import build_art_retention_quarter_evidence


def test_art_retention_evidence_prefers_explicit_process_rows() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 120.0,
            "newly_enrolled_sum": 10.0,
            "newly_enrolled_count": 1,
            "deaths_sum": 0.0,
            "art_ltfu_period_sum": 7.0,
            "art_ltfu_period_count": 1,
            "art_reengagement_period_sum": 4.0,
            "art_reengagement_period_count": 1,
        },
        previous_summary={"art_last": 100.0},
        diagnosed_gap=50.0,
    )

    assert evidence["support_class"] == "direct_process_observed"
    assert evidence["interruption_count"] == 7.0
    assert evidence["reengagement_count"] == 4.0
    assert evidence["direct_reengagement_count"] == 4.0
    assert evidence["latent_reengagement_proxy_count"] == 0.0
    assert evidence["reengagement_evidence"]["publishable_process_claim"] is True
    assert evidence["direct_metric_counts"]["interruption"] == 1


def test_art_retention_evidence_uses_cohort_balance_proxy_when_direct_rows_absent() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 105.0,
            "newly_enrolled_sum": 20.0,
            "newly_enrolled_count": 1,
            "deaths_sum": 2.0,
        },
        previous_summary={"art_last": 100.0},
        diagnosed_gap=40.0,
    )

    assert evidence["support_class"] == "cohort_balance_proxy"
    assert evidence["interruption_count"] == 13.0
    assert evidence["reengagement_count"] == 0.0
    assert evidence["support_weight"] == 1.0


def test_art_retention_evidence_does_not_treat_stock_growth_as_reengagement_without_starts() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 140.0,
            "deaths_sum": 0.0,
        },
        previous_summary={"art_last": 100.0},
        diagnosed_gap=40.0,
    )

    assert evidence["support_class"] == "art_stock_balance_proxy"
    assert evidence["interruption_count"] == 0.0
    assert evidence["reengagement_count"] == 0.0


def test_art_retention_evidence_uses_cumulative_direct_process_deltas() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 100.0,
            "art_ltfu_cumulative_last": 125.0,
            "art_ltfu_cumulative_count": 1,
            "art_transfer_out_overseas_cumulative_last": 12.0,
            "art_transfer_out_overseas_cumulative_count": 1,
            "art_stopped_refused_cumulative_last": 5.0,
            "art_stopped_refused_cumulative_count": 1,
            "art_deaths_cumulative_last": 30.0,
            "art_deaths_cumulative_count": 1,
        },
        previous_summary={
            "art_last": 90.0,
            "art_ltfu_cumulative_last": 100.0,
            "art_ltfu_cumulative_count": 1,
            "art_transfer_out_overseas_cumulative_last": 10.0,
            "art_transfer_out_overseas_cumulative_count": 1,
            "art_stopped_refused_cumulative_last": 4.0,
            "art_stopped_refused_cumulative_count": 1,
            "art_deaths_cumulative_last": 20.0,
            "art_deaths_cumulative_count": 1,
        },
        diagnosed_gap=40.0,
    )

    assert evidence["support_class"] == "direct_process_observed"
    assert evidence["interruption_count"] == 25.0
    assert evidence["transfer_out_count"] == 2.0
    assert evidence["stopped_refused_count"] == 1.0
    assert evidence["art_death_count"] == 10.0
    assert evidence["process_removal_count"] == 13.0
    assert evidence["direct_reengagement_count"] == 0.0
    assert evidence["reengagement_evidence"]["publishable_process_claim"] is False


def test_art_retention_evidence_uses_public_stock_flow_reengagement_proxy_without_direct_restart_rows() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 115.0,
            "newly_enrolled_sum": 20.0,
            "newly_enrolled_count": 1,
            "art_ltfu_cumulative_last": 60.0,
            "art_ltfu_cumulative_count": 1,
        },
        previous_summary={
            "art_last": 100.0,
            "art_ltfu_cumulative_last": 50.0,
            "art_ltfu_cumulative_count": 1,
        },
        diagnosed_gap=45.0,
    )

    assert evidence["support_class"] == "direct_process_observed"
    assert evidence["interruption_count"] == 10.0
    assert evidence["direct_reengagement_count"] == 0.0
    assert evidence["latent_reengagement_proxy_count"] == 5.0
    assert evidence["reengagement_count"] == 5.0
    assert evidence["reengagement_evidence"]["claim_status"] == "proxy_only_no_public_treatment_cohort"
    assert evidence["reengagement_evidence"]["publishable_process_claim"] is False


def test_art_retention_evidence_zero_sensitivity_removes_proxy_reengagement() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 115.0,
            "newly_enrolled_sum": 20.0,
            "newly_enrolled_count": 1,
            "art_ltfu_cumulative_last": 60.0,
            "art_ltfu_cumulative_count": 1,
        },
        previous_summary={
            "art_last": 100.0,
            "art_ltfu_cumulative_last": 50.0,
            "art_ltfu_cumulative_count": 1,
        },
        diagnosed_gap=45.0,
        reengagement_sensitivity_mode="zero",
    )

    assert evidence["reengagement_count"] == 0.0
    assert evidence["latent_reengagement_proxy_count"] == 0.0
    assert evidence["latent_reengagement_public_point_count"] == 5.0
    assert evidence["reengagement_evidence"]["claim_status"] == "zero_sensitivity_no_public_treatment_cohort"
    assert evidence["reengagement_evidence"]["publishable_process_claim"] is False


def test_art_retention_evidence_upper_bound_sensitivity_uses_interrupted_pool_bound() -> None:
    evidence = build_art_retention_quarter_evidence(
        summary={
            "art_last": 115.0,
            "newly_enrolled_sum": 20.0,
            "newly_enrolled_count": 1,
            "art_ltfu_cumulative_last": 60.0,
            "art_ltfu_cumulative_count": 1,
        },
        previous_summary={
            "art_last": 100.0,
            "art_ltfu_cumulative_last": 50.0,
            "art_ltfu_cumulative_count": 1,
        },
        diagnosed_gap=45.0,
        reengagement_sensitivity_mode="upper_bound_proxy",
    )

    assert evidence["reengagement_count"] == 60.0
    assert evidence["latent_reengagement_proxy_count"] == 60.0
    assert evidence["latent_reengagement_public_point_count"] == 5.0
    assert evidence["latent_reengagement_upper_bound_count"] == 60.0
    assert evidence["reengagement_evidence"]["claim_status"] == "upper_bound_proxy_no_public_treatment_cohort"
