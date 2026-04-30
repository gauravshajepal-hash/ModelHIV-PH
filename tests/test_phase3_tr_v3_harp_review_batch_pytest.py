from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_harp_review_batch as batch


def test_monthly_core_table_merges_observed_rows_and_flow() -> None:
    observed_rows = [
        {
            "metric_name": "diagnosed_plhiv",
            "time": "2010-01",
            "value": 100.0,
            "source_id": "diag_a",
            "source_label": "Diagnosed A",
        },
        {
            "metric_name": "alive_on_art",
            "time": "2010-01",
            "value": 60.0,
            "source_id": "art_a",
            "source_label": "ART A",
        },
        {
            "metric_name": "tested_for_viral_load",
            "time": "2010-01",
            "value": 40.0,
            "source_id": "vl_a",
            "source_label": "VL A",
        },
        {
            "metric_name": "virally_suppressed",
            "time": "2010-01",
            "value": 30.0,
            "source_id": "sup_a",
            "source_label": "SUP A",
        },
    ]
    diagnosis_flow_points = [
        {
            "effective_month": "2010-01",
            "diagnosed_count": 12.0,
            "source_id": "flow_a",
            "source_label": "Flow A",
        }
    ]

    rows = batch._monthly_core_table(observed_rows, diagnosis_flow_points)

    assert len(rows) == 1
    row = rows[0]
    assert row["month"] == "2010-01"
    assert row["diagnosed_plhiv"] == 100.0
    assert row["new_diagnosed_cases_period"] == 12.0
    assert row["art_uptake_rate"] == 0.6
    assert row["viral_suppression_rate_tested"] == 0.75
    assert row["viral_suppression_rate_on_art"] == 0.5
    assert row["new_diagnosed_cases_period_source_id"] == "flow_a"


def test_source_transition_rows_detect_metric_handoffs() -> None:
    core_table = [
        {"month": "2010-01", "diagnosed_plhiv": 100.0, "diagnosed_plhiv_source_id": "a", "diagnosed_plhiv_source_label": "A"},
        {"month": "2010-02", "diagnosed_plhiv": 110.0, "diagnosed_plhiv_source_id": "a", "diagnosed_plhiv_source_label": "A"},
        {"month": "2010-03", "diagnosed_plhiv": 120.0, "diagnosed_plhiv_source_id": "b", "diagnosed_plhiv_source_label": "B"},
    ]

    transitions = batch._source_transition_rows(core_table)

    assert transitions == [
        {
            "metric_name": "diagnosed_plhiv",
            "month": "2010-03",
            "from_source_id": "a",
            "to_source_id": "b",
            "from_source_label": "A",
            "to_source_label": "B",
            "from_source_regime": "a",
            "to_source_regime": "b",
        }
    ]


def test_metric_inventory_rows_summarize_counts_and_ranges() -> None:
    metric_rows = [
        {"metric_name": "alive_on_art", "time": "2010-01", "source_id": "a", "unit": "count_people", "temporal_precision": "monthly_snapshot", "measurement_class": "program_observed_harp"},
        {"metric_name": "alive_on_art", "time": "2010-02", "source_id": "a", "unit": "count_people", "temporal_precision": "monthly_snapshot", "measurement_class": "program_observed_harp"},
        {"metric_name": "diagnosed_plhiv", "time": "2010-12", "source_id": "b", "unit": "count_people", "temporal_precision": "annual_snapshot", "measurement_class": "program_observed_harp"},
    ]

    rows = batch._metric_inventory_rows(metric_rows)

    assert rows[0]["metric_name"] == "alive_on_art"
    assert rows[0]["row_count"] == 2
    assert rows[0]["first_time"] == "2010-01"
    assert rows[0]["last_time"] == "2010-02"
    diagnosed = next(row for row in rows if row["metric_name"] == "diagnosed_plhiv")
    assert diagnosed["source_count"] == 1
    assert diagnosed["temporal_precisions"] == "annual_snapshot"


def test_suspicious_event_rows_flag_gap_handoff_and_jump() -> None:
    core_table = [
        {"month": "2010-01", "diagnosed_plhiv": 100.0, "diagnosed_plhiv_source_id": "doh_hiv_sti_2010_2010_january", "diagnosed_plhiv_source_label": "Jan"},
        {"month": "2010-02", "diagnosed_plhiv": 110.0, "diagnosed_plhiv_source_id": "doh_hiv_sti_2010_2010_february", "diagnosed_plhiv_source_label": "Feb"},
        {"month": "2010-03", "diagnosed_plhiv": "", "diagnosed_plhiv_source_id": "", "diagnosed_plhiv_source_label": ""},
        {"month": "2010-04", "diagnosed_plhiv": 300.0, "diagnosed_plhiv_source_id": "doh_hiv_sti_2010_2010_april", "diagnosed_plhiv_source_label": "Apr"},
        {"month": "2010-05", "diagnosed_plhiv": 330.0, "diagnosed_plhiv_source_id": "doh_official_cascade_ground_truth_2018_2025", "diagnosed_plhiv_source_label": "GT"},
    ]

    rows = batch._suspicious_event_rows(core_table)

    gap_open = next(row for row in rows if row["month"] == "2010-03")
    assert "gap_open" in gap_open["flags"]
    gap_close = next(row for row in rows if row["month"] == "2010-04")
    assert "gap_close" in gap_close["flags"]
    handoff = next(row for row in rows if row["month"] == "2010-05")
    assert "source_regime_handoff" in handoff["flags"]
