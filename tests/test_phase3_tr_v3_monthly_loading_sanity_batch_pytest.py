from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_monthly_loading_sanity_batch as batch


def test_category_for_canonical_flags_champion_overlap() -> None:
    assert batch._category_for_canonical("diagnosed_plhiv") == "champion_overlap"
    assert batch._category_for_canonical("tested_for_viral_load") == "cascade_sidecar"
    assert batch._category_for_canonical("mobility_network_mixing") == "context_or_proxy"


def test_time_mix_label_detects_annual_only() -> None:
    assert batch._time_mix_label(monthly_support_count=0, annual_support_count=4) == "annual_only"
    assert batch._time_mix_label(monthly_support_count=2, annual_support_count=0) == "monthly_only"
    assert batch._time_mix_label(monthly_support_count=2, annual_support_count=3) == "mixed"


def test_build_block_summary_rows_computes_shares() -> None:
    rows = [
        {"block_id": "testing_engagement", "category": "champion_overlap", "abs_loading": 0.6, "singleton_support": False, "annual_only": False, "ppc_corr": 0.9, "risk_score": 4},
        {"block_id": "testing_engagement", "category": "context_or_proxy", "abs_loading": 0.4, "singleton_support": True, "annual_only": True, "ppc_corr": 0.7, "risk_score": 2},
    ]

    summary = batch._build_block_summary_rows(rows)

    assert len(summary) == 1
    row = summary[0]
    assert row["block_id"] == "testing_engagement"
    assert round(float(row["champion_loading_share"]), 6) == 0.6
    assert round(float(row["singleton_loading_share"]), 6) == 0.4
    assert round(float(row["annual_only_loading_share"]), 6) == 0.4
