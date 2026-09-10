from __future__ import annotations

from pathlib import Path

from phase3_dynamic.r95_2026_q2_hasp_intake_gate import (
    _direct_q2_target_row,
    _gate,
    extract_hasp_2026_q2_rows,
)
from phase3_dynamic.r53_publication_claim_registry import _latest_hasp_q2_intake_claim


def test_r95_extracts_core_q2_rows(tmp_path: Path) -> None:
    source_path = tmp_path / "hasp_q2.pdf"
    text = """
    there would have been 288,000 estimated People Living with HIV (PLHIV)
    As of June 2026, 159,997 (56% of the estimated) PLHIV have been diagnosed.
    Further, 111,154 (69% of the diagnosed) PLHIV are currently on life-saving Antiretroviral Therapy (ART),
    of which 65,630 (59%) PLHIV have been tested for viral load (VL) in the past 12 months.
    Among those tested for VL, 63,850 (97%) were virally suppressed.
    From April to June 2026, there were 2,994 confirmed HIV-positive individuals reported.
    Of the recorded cases for this quarter, 828 (28%) had an advanced HIV disease.
    Cumulatively, 171,066 confirmed HIV cases have been reported.
    quarter average cases per day are 33.
    there were 4,291 people with HIV who were enrolled in treatment.
    median CD4 count among these patients at enrollment was 176 cells/mm3.
    From April to June 2026, 352 deaths from any cause were reported.
    Since January 1984, a total of 11,069 deaths have been reported.
    2026    1690      1407      1533          1217     851     926                                                          1271
    Care Cascade by Region
    1 9,500 4,765 50% 3,317 70% 1,910 58% 1,860 97% 56%
    """

    rows, flags = extract_hasp_2026_q2_rows(text, source_path)
    target = _direct_q2_target_row(rows)

    assert target["diagnosed_plhiv"] == 159997.0
    assert target["alive_on_art"] == 111154.0
    assert target["tested_for_viral_load"] == 65630.0
    assert target["virally_suppressed"] == 63850.0
    assert target["new_diagnosed_cases_period"] == 2994.0
    assert any(row["time_granularity"] == "monthly" and row["value"] == 926.0 for row in rows)
    assert any(row["geography"] == "1" and row["metric_id"] == "estimated_plhiv" for row in rows)
    assert any(row["metric_id"] == "estimated_plhiv" and row["observation_role"] == "validation_only" for row in rows)
    assert flags


def test_r95_gate_flags_flow_regression_without_blocking_stock_anchor() -> None:
    rows = [
        {"metric_name": "diagnosed_plhiv", "candidate_norm_error": 0.01, "carry_forward_norm_error": 0.02},
        {"metric_name": "alive_on_art", "candidate_norm_error": 0.02, "carry_forward_norm_error": 0.03},
        {"metric_name": "tested_for_viral_load", "candidate_norm_error": 0.03, "carry_forward_norm_error": 0.05},
        {"metric_name": "virally_suppressed", "candidate_norm_error": 0.03, "carry_forward_norm_error": 0.06},
        {"metric_name": "new_diagnosed_cases_period", "candidate_norm_error": 0.34, "carry_forward_norm_error": 0.29},
    ]

    gate = _gate(rows)

    assert gate["status"] == "r95_q2_stock_anchor_promoted_diagnosis_flow_shock_flagged"
    assert gate["diagnosis_flow_regresses_against_carry_forward"] is True
    assert not gate["blockers"]


def test_r95_registry_claim_keeps_diagnosis_flow_shock_boundary(tmp_path: Path) -> None:
    report_path = tmp_path / "r95.json"
    report_path.write_text("{}", encoding="utf-8")
    claim = _latest_hasp_q2_intake_claim(
        {
            "status": "r95_q2_stock_anchor_promoted_diagnosis_flow_shock_flagged",
            "candidate_family": "r41_monotone_growth_component_process",
            "extracted_row_count": 529,
            "observation_role_counts": {"direct_target": 83, "validation_only": 23},
            "q2_holdout_gate": {
                "status": "r95_q2_stock_anchor_promoted_diagnosis_flow_shock_flagged",
                "candidate_mean_norm_error": 0.09162295057208922,
                "carry_forward_mean_norm_error": 0.09338803940721494,
                "stock_candidate_mean_norm_error": 0.01964242684898145,
                "stock_carry_forward_mean_norm_error": 0.0433433729192372,
                "diagnosis_flow_regresses_against_carry_forward": True,
                "scored_metric_count": 5,
            },
        },
        report_path,
    )

    assert claim["claim_status"] == "stock_anchor_promoted_diagnosis_flow_shock_flagged"
    assert "Q2 stock row may initialize future stock forecasts" in claim["allowed_claim"]
    assert claim["key_metrics"]["diagnosis_flow_regresses_against_carry_forward"] is True
    assert "does not allow diagnosis-flow or incidence process claims" in claim["claim_limit"]
