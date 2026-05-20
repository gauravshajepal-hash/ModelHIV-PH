from __future__ import annotations

from pathlib import Path

from phase3_dynamic.r94_2026_q1_hasp_intake_gate import (
    _comparison_gate,
    _comparison_rows,
    _direct_q1_target_row,
    extract_hasp_2026_q1_rows,
)


def test_r94_extracts_core_hasp_rows_from_text(tmp_path: Path) -> None:
    source_path = tmp_path / "hasp.pdf"
    text = """
    there would have been 288,000 estimated People Living with HIV (PLHIV)
    As of March 2026, 157,350 (55% of the estimated) PLHIV have been diagnosed.
    Further, 108,367 (69% of the diagnosed) PLHIV are currently on life-saving Antiretroviral Therapy (ART),
    of which 61,413 (57%) PLHIV have been tested for viral load (VL) in the past 12 months.
    Among those tested for VL, 59,540 (97%) were virally suppressed.
    In January to March 2026, there were 4,633 confirmed HIV-positive individuals reported.
    Of the recorded cases for this quarter, 1,104 (24%) had an advanced HIV disease.
    Cumulatively, 168,079 confirmed HIV cases have been reported.
    quarter average cases per day is 51.
    there were 4,716 people with HIV who were enrolled to treatment.
    median CD4 of these patients upon enrollment was at 182 cells/mm3.
    From January to March 2026, 477 deaths from any cause were reported.
    Since January 1984, a total of 10,727 deaths have been reported.
    2026    1690      1407      1536      1544
    Care Cascade by Region
    1 9,500 4,619 49% 3,217 70% 1,795 56% 1,735 97% 54%
    """

    rows, flags = extract_hasp_2026_q1_rows(text, source_path)
    target = _direct_q1_target_row(rows)

    assert target["diagnosed_plhiv"] == 157350.0
    assert target["alive_on_art"] == 108367.0
    assert target["tested_for_viral_load"] == 61413.0
    assert target["virally_suppressed"] == 59540.0
    assert target["new_diagnosed_cases_period"] == 4633.0
    assert any(row["time_granularity"] == "monthly" and row["value"] == 1690.0 for row in rows)
    assert any(row["geography"] == "1" and row["metric_id"] == "estimated_plhiv" for row in rows)
    assert any(row["metric_id"] == "estimated_plhiv" and row["observation_role"] == "validation_only" for row in rows)
    assert flags


def test_r94_comparison_gate_promotes_when_r41_beats_carry_forward() -> None:
    train_rows = [
        {
            "quarter": "2025-Q4",
            "diagnosed_plhiv": 150000.0,
            "alive_on_art": 100000.0,
            "tested_for_viral_load": 50000.0,
            "virally_suppressed": 48000.0,
            "new_diagnosed_cases_period": 4000.0,
        }
    ]
    target_row = {
        "quarter": "2026-Q1",
        "diagnosed_plhiv": 157350.0,
        "alive_on_art": 108367.0,
        "tested_for_viral_load": 61413.0,
        "virally_suppressed": 59540.0,
        "new_diagnosed_cases_period": 4633.0,
    }
    candidate_row = {
        "quarter": "2026-Q1",
        "diagnosed_plhiv": 157841.0,
        "alive_on_art": 109548.0,
        "tested_for_viral_load": 60384.0,
        "virally_suppressed": 58586.0,
        "new_diagnosed_cases_period": 4706.0,
    }
    carry_row = dict(train_rows[0], quarter="2026-Q1")

    rows = _comparison_rows(
        train_rows=train_rows,
        target_row=target_row,
        candidate_row=candidate_row,
        carry_row=carry_row,
    )
    gate = _comparison_gate(rows)

    assert len(rows) == 5
    assert gate["status"] == "r94_q1_hasp_anchor_promoted_for_future_initialization"
    assert gate["candidate_mean_norm_error"] < gate["carry_forward_mean_norm_error"]
