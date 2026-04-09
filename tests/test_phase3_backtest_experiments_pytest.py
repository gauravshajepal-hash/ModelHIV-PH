from __future__ import annotations

from pathlib import Path

from epigraph_ph.phase3.shared.broad_backtest_support import (
    bias_metrics_from_evaluation,
    rank_representation_scores,
    rolling_origin_splits,
)
from epigraph_ph.phase3._lineage.rescue_core import (
    _merge_multiscale_blanket_factor_rows,
    _resolve_modifier_representation,
)
from epigraph_ph.runtime import write_json


def test_phase3_rolling_origin_splits_reserve_progressively_later_holdouts() -> None:
    splits = rolling_origin_splits([2017, 2018, 2019, 2020, 2021], min_train_years=3)

    assert splits == [
        {
            "split_label": "train_2017_2019__holdout_2020",
            "train_years": [2017, 2018, 2019],
            "holdout_years": [2020],
            "holdout_year": 2020,
        },
        {
            "split_label": "train_2017_2020__holdout_2021",
            "train_years": [2017, 2018, 2019, 2020],
            "holdout_years": [2021],
            "holdout_year": 2021,
        },
    ]


def test_phase3_representation_ranking_prefers_lower_mean_mae_then_smape() -> None:
    ranked = rank_representation_scores(
        [
            {"representation": "a", "model_mean_absolute_error": 0.09, "model_smape": 0.20},
            {"representation": "a", "model_mean_absolute_error": 0.11, "model_smape": 0.10},
            {"representation": "b", "model_mean_absolute_error": 0.10, "model_smape": 0.05},
            {"representation": "b", "model_mean_absolute_error": 0.10, "model_smape": 0.04},
        ]
    )

    assert ranked[0]["representation"] == "b"
    assert ranked[0]["mean_model_mae"] == 0.1
    assert ranked[0]["mean_model_smape"] == 0.045


def test_phase3_bias_metrics_extract_key_holdout_errors() -> None:
    metrics = bias_metrics_from_evaluation(
        {
            "holdout_reference_check": {
                "comparisons": [
                    {
                        "errors": {
                            "diagnosed_stock_abs_error": 0.12,
                            "art_stock_abs_error": 0.08,
                            "documented_suppression_abs_error": 0.05,
                            "viral_load_tested_among_art_abs_error": 0.14,
                            "suppressed_among_art_abs_error": 0.09,
                        }
                    }
                ]
            }
        }
    )

    assert metrics["diagnosed_stock_abs_error"] == 0.12
    assert metrics["documented_suppression_abs_error"] == 0.05
    assert metrics["viral_load_tested_among_art_abs_error"] == 0.14


def test_phase3_multiscale_blanket_merge_adds_supported_factor_rows(tmp_path: Path) -> None:
    phase2_dir = tmp_path / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        phase2_dir / "multiscale_phase3_target_blankets.json",
        {"merged_target_factor_ids": ["factor_0003", "factor_0011"]},
    )
    factor_catalog = [
        {"factor_id": "factor_0003", "factor_name": "cascade", "block_name": "epi", "transition_hooks": ["diagnosis_transitions"]},
        {"factor_id": "factor_0011", "factor_name": "policy", "block_name": "policy", "transition_hooks": ["linkage_transitions"]},
    ]

    promoted, supporting = _merge_multiscale_blanket_factor_rows(
        tmp_path,
        factor_catalog,
        [{"factor_id": "factor_0003", "factor_name": "cascade"}],
        [],
    )

    assert [row["factor_id"] for row in promoted] == ["factor_0003"]
    assert [row["factor_id"] for row in supporting] == ["factor_0011"]
    assert supporting[0]["promotion_class"] == "multiscale_supported"


def test_phase3_multiscale_modifier_representations_are_resolved() -> None:
    assert _resolve_modifier_representation("hiv_rescue_v2", "clumped_multiscale") == "clumped_multiscale"
    assert _resolve_modifier_representation("hiv_rescue_v2", "hybrid_multiscale") == "hybrid_multiscale"
