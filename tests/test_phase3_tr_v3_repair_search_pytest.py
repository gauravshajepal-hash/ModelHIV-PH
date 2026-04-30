from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_repair_search as repair_search


def test_build_repair_search_specs_emits_bounded_search_space() -> None:
    specs = repair_search.build_repair_search_specs()
    ids = {row.search_id for row in specs}

    assert len(specs) == 92
    assert "SEARCH-R1-d2a-ridge1" in ids
    assert "SEARCH-R1-d2a-ridge8" in ids
    assert "SEARCH-R6-c4-f35" in ids
    assert "SEARCH-R7-c5-f45" in ids
    assert "SEARCH-R9-c3-f25-w50-repair" in ids
    assert "SEARCH-R9-c4-f35-w100-blend" in ids
    assert "SEARCH-R10-s25-f50" in ids
    assert "SEARCH-R10-s100-f100" in ids
    assert "SEARCH-R10E-a100-f25-delta-sup75" in ids
    assert "SEARCH-R10D-artdelta-ab50-f100" in ids

    spec_map = {row.search_id: row for row in specs}
    assert spec_map["SEARCH-R1-d2a-ridge4"].spec.transition_ridge_multipliers["D_to_A"] == 4.0
    assert spec_map["SEARCH-R6-c3-f25"].spec.repair_params == {
        "d_to_a_min_count": 3,
        "d_to_a_min_fraction": 0.25,
    }
    assert spec_map["SEARCH-R7-c5-f45"].base_experiment_id == "EXP-R7"
    assert spec_map["SEARCH-R9-c4-f35-w100-blend"].base_experiment_id == "EXP-R9"
    assert spec_map["SEARCH-R9-c4-f35-w100-blend"].spec.repair_params == {
        "d_to_a_min_count": 4,
        "d_to_a_min_fraction": 0.35,
        "d_to_a_use_blend": True,
        "art_delta_model_weight": 1.0,
    }
    assert spec_map["SEARCH-R10-s75-f50"].base_experiment_id == "EXP-R10"
    assert spec_map["SEARCH-R10-s75-f50"].spec.repair_params == {
        "diagnosed_weight": 0.75,
        "art_weight": 0.75,
        "flow_weight": 0.5,
    }
    assert spec_map["SEARCH-R10E-a100-f25-delta-sup75"].base_experiment_id == "EXP-R10-EXACT-CHAMPION"
    assert spec_map["SEARCH-R10E-a100-f25-delta-sup75"].spec.repair_params["flow_series_model"] == "delta"
    assert spec_map["SEARCH-R10D-artdelta-ab50-f100"].base_experiment_id == "EXP-R10-DENSE-CHAMPION"
    assert spec_map["SEARCH-R10D-artdelta-ab50-f100"].spec.repair_params["art_series_model"] == "delta"


def test_dominance_helpers_behave_for_frontier_search() -> None:
    assert repair_search._dominates((0.3, 0.2, 0.4), (0.4, 0.2, 0.5)) is True
    assert repair_search._dominates((0.4, 0.2, 0.5), (0.3, 0.2, 0.4)) is False
    assert repair_search._equivalent((0.3, 0.2), (0.3, 0.2)) is True
    assert repair_search._equivalent((0.3, 0.2), (0.3, 0.21)) is False
