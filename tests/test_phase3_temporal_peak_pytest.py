from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from epigraph_ph.cli.main import build_parser
from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3.shared.mixed_frequency import build_mixed_frequency_observation_bundle, point_effective_month
from epigraph_ph.phase3._lineage.pipeline import _archive_points_from_program_points, _filter_diagnosis_flow_points_by_years
from epigraph_ph.phase3 import peak_search as peak_search_module
from epigraph_ph.phase3._lineage.peak_search import _group_transition_templates, _promotion_gate, _simulate_future_states, _target_series
from epigraph_ph.phase3._lineage.rescue_core import (
    _archive_report_availability_month,
    _apply_scaffold_dominance_gate,
    _diagnosis_flow_segment_sums_jax,
    _default_factor_transition_hooks,
    _frozen_vintage_incumbent_points,
    _harp_program_penalty_torch,
    _national_anchor_penalty_torch,
    _regrid_time_tensor_to_axis,
)
from epigraph_ph.phase3.shared.temporal_scaffold import build_shock_regime_basis, build_temporal_basis, derive_compartmental_scaffold
from epigraph_ph.runtime import read_json

try:
    import jax.numpy as jnp
except Exception:  # pragma: no cover
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def test_phase3_temporal_peak_sections_are_declared_in_plugin_contract() -> None:
    plugin = get_disease_plugin("hiv")
    phase3_priors = (plugin.prior_hyperparameters or {}).get("phase3", {})

    assert phase3_priors.get("temporal_decomposition", {}).get("slow_knot_months") == 12
    assert phase3_priors.get("temporal_decomposition", {}).get("medium_block_months") == 6
    assert phase3_priors.get("temporal_decomposition", {}).get("determinant_bound_scale") is not None
    assert phase3_priors.get("temporal_decomposition", {}).get("transition_floor") is not None
    assert phase3_priors.get("jax_svi_loss_scales", {}).get("national_anchor_penalty_scale") is not None
    assert phase3_priors.get("jax_svi_loss_scales", {}).get("diagnosis_flow_penalty_scale") is not None
    assert phase3_priors.get("torch_map_loss_scales", {}).get("diagnosis_flow_penalty_scale") is not None
    assert phase3_priors.get("mixed_frequency_observation", {}).get("annual_anchor_mode") == "year_end_snapshot"
    assert phase3_priors.get("shock_regimes", {}).get("regimes")
    assert phase3_priors.get("scaffold_dominance", {}).get("enabled") is True
    assert phase3_priors.get("scaffold_dominance", {}).get("fallback_mode") == "scaffold_only"
    assert phase3_priors.get("scaffold_dominance", {}).get("max_retained_determinants") == 4
    assert phase3_priors.get("peak_search", {}).get("search_method") == "elite_stochastic_peak_search"
    assert phase3_priors.get("peak_search", {}).get("initial_search_std", {}).get("slow") is not None
    assert phase3_priors.get("peak_search", {}).get("transition_group_templates", {}).get("general_service") is not None
    tournament_modes = set(phase3_priors.get("frozen_backtest", {}).get("representation_tournament", {}).get("modes") or [])
    assert "hybrid_temporal_multiscale" in tournament_modes


def test_harp_archive_build_parser_accepts_force_refresh() -> None:
    parser = build_parser()
    args = parser.parse_args(["harp-archive", "build", "--run-id", "pytest-run", "--force-refresh"])

    assert args.command == "harp-archive"
    assert args.harp_archive_command == "build"
    assert args.force_refresh is True


def test_archive_report_availability_month_parses_archive_quarter_end() -> None:
    assert _archive_report_availability_month("doh_hiv_sti_2025_2025_january_march", "2025 January - March") == "2025-03"
    assert _archive_report_availability_month("doh_hiv_sti_2024_2024_october_december", "2024 October - December") == "2024-12"
    assert _archive_report_availability_month("core_team_2025", "2025 PH HIV Estimates Core Team for WHO") is None


def test_frozen_vintage_incumbent_points_filters_post_cutoff_and_non_archive_sources(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    archive_dir = run_dir / "harp_archive"
    archive_dir.mkdir(parents=True)
    rows = [
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 215400.0,
            "source_id": "doh_hiv_sti_2024_2024_october_december",
            "source_label": "2024 October - December",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "diagnosed_plhiv",
            "value": 134892.0,
            "source_id": "doh_hiv_sti_2024_2024_october_december",
            "source_label": "2024 October - December",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "alive_on_art",
            "value": 90568.0,
            "source_id": "doh_hiv_sti_2024_2024_october_december",
            "source_label": "2024 October - December",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "virally_suppressed",
            "value": 36633.0,
            "source_id": "doh_hiv_sti_2024_2024_october_december",
            "source_label": "2024 October - December",
        },
        {
            "year": 2025,
            "time": "2025-03",
            "metric_name": "estimated_plhiv",
            "value": 230000.0,
            "source_id": "doh_hiv_sti_2025_2025_january_march",
            "source_label": "2025 January - March",
        },
        {
            "year": 2025,
            "time": "2025-03",
            "metric_name": "diagnosed_plhiv",
            "value": 140000.0,
            "source_id": "doh_hiv_sti_2025_2025_january_march",
            "source_label": "2025 January - March",
        },
        {
            "year": 2025,
            "time": "2025-03",
            "metric_name": "alive_on_art",
            "value": 93000.0,
            "source_id": "doh_hiv_sti_2025_2025_january_march",
            "source_label": "2025 January - March",
        },
        {
            "year": 2025,
            "time": "2025-03",
            "metric_name": "virally_suppressed",
            "value": 37000.0,
            "source_id": "doh_hiv_sti_2025_2025_january_march",
            "source_label": "2025 January - March",
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 999999.0,
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "source_label": "DOH 95-95-95 Ground Truth 2018-2025 From User-Supplied Official Slide",
        },
    ]
    (archive_dir / "historical_metric_rows.json").write_text(json.dumps(rows), encoding="utf-8")

    result = _frozen_vintage_incumbent_points(run_dir, "2024-12")

    assert set(result) == {2024}
    assert result[2024]["source_id"] == "doh_hiv_sti_2024_2024_october_december"
    assert result[2024]["availability_month"] == "2024-12"


def test_temporal_scaffold_and_basis_helpers_are_shape_safe() -> None:
    month_axis = [f"2024-{month:02d}" for month in range(1, 13)] + [f"2025-{month:02d}" for month in range(1, 13)]
    scaffold = derive_compartmental_scaffold(
        train_harp_points=[
            {"month": "2024-12", "estimated_plhiv": 215400, "diagnosed": 134892, "on_art": 90568, "viral_load_tested": 41746, "suppressed": 36633},
            {"month": "2025-12", "estimated_plhiv": 252800, "diagnosed": 153491, "on_art": 97943, "viral_load_tested": 53987, "suppressed": 52380},
        ],
        month_axis=month_axis,
        transition_prior=np.asarray([0.06, 0.14, 0.12, 0.05, 0.07], dtype=np.float32),
        floor=1e-5,
        transition_floor=0.001,
        transition_ceiling=0.95,
        probability_eps=1e-5,
    )
    basis = build_temporal_basis(month_axis, slow_knot_months=12, medium_block_months=6)

    assert scaffold["monthly_probs"].shape == (24, 5)
    assert scaffold["monthly_logits"].shape == (24, 5)
    assert np.all(scaffold["monthly_probs"] > 0.0)
    assert np.all(scaffold["monthly_probs"] < 1.0)
    assert basis["slow_basis"].shape[0] == 24
    assert basis["medium_basis"].shape[0] == 24
    assert basis["year_index"].shape == (24,)


def test_temporal_basis_single_month_retains_columns_for_jax_like_ops() -> None:
    month_axis = ["2025-12"]
    basis = build_temporal_basis(month_axis, slow_knot_months=12, medium_block_months=6)
    shock = build_shock_regime_basis(month_axis, [{"name": "single_window", "start_month": "2025-12", "end_month": "2025-12"}])

    assert basis["slow_basis"].shape == (1, 1)
    assert basis["medium_basis"].shape == (1, 1)
    assert shock["basis"].shape == (1, 1)
    assert int(basis["year_index"][0]) == 0


def test_mixed_frequency_bundle_maps_annual_points_to_year_end() -> None:
    month_axis = [f"2025-{month:02d}" for month in range(1, 13)]
    support_bundle = {
        "targets": {
            "diagnosed_stock": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "art_stock": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "documented_suppression": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "testing_coverage": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "deaths": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
        }
    }
    bundle = build_mixed_frequency_observation_bundle(
        month_axis=month_axis,
        province_axis=["Philippines"],
        support_bundle=support_bundle,
        official_points=[
            {
                "year": 2025,
                "temporal_precision": "annual_snapshot",
                "reference": {
                    "first95": 0.61,
                    "second95": 0.64,
                    "overall_suppressed": 0.2072,
                    "documented_suppression_among_art": 0.53,
                },
            }
        ],
        harp_points=[],
        region_axis=["NCR"],
    )

    weight = np.asarray(bundle["official_anchor_arrays"]["weight"], dtype=np.float32)
    assert float(weight[-1]) == 1.0
    assert float(np.sum(weight[:-1])) == 0.0
    assert bundle["summary_rows"][0]["anchor_aggregation_mode_used"] == "year_end_snapshot"


def test_point_effective_month_honors_effective_month_and_annual_precision() -> None:
    month_axis = [f"2024-{month:02d}" for month in range(1, 13)] + [f"2025-{month:02d}" for month in range(1, 13)]

    annual_month, annual_mode = point_effective_month(
        {
            "month": "2025-01",
            "effective_month": "2024-12",
            "temporal_precision": "annual_snapshot",
        },
        month_axis,
    )
    monthly_month, monthly_mode = point_effective_month(
        {
            "month": "2025-01",
            "effective_month": "2025-03",
            "temporal_precision": "monthly_snapshot",
        },
        month_axis,
    )

    assert annual_month == "2024-12"
    assert annual_mode == "year_end_snapshot"
    assert monthly_month == "2025-03"
    assert monthly_mode == "source_month"


def test_archive_points_from_program_points_preserve_quarterly_harp_support() -> None:
    month_axis = [f"2024-{month:02d}" for month in range(1, 13)]
    points = _archive_points_from_program_points(
        [
            {
                "label": "2024 October - December",
                "month": "2024-12",
                "effective_month": "2024-12",
                "source_month": "2024-12",
                "temporal_precision": "quarterly_snapshot",
                "estimated_plhiv": 215400,
                "diagnosed": 135026,
                "on_art": 90854,
                "viral_load_tested": 41860,
                "suppressed": 36723,
            },
            {
                "label": "Official 2024 annual",
                "month": "2024-01",
                "effective_month": "2024-01",
                "source_month": "2024-01",
                "temporal_precision": "annual_snapshot",
                "estimated_plhiv": 215400,
                "diagnosed": 134892,
                "on_art": 90568,
                "viral_load_tested": 41746,
                "suppressed": 36633,
            },
        ],
        [2024],
        month_axis,
    )

    assert len(points) == 2
    quarterly = next(row for row in points if row["label"] == "2024 October - December")
    annual = next(row for row in points if row["label"] == "Official 2024 annual")
    assert quarterly["year"] == 2024
    assert quarterly["month"] == "2024-12"
    assert quarterly["source_month"] == "2024-12"
    assert quarterly["temporal_precision"] == "quarterly_snapshot"
    assert annual["year"] == 2024
    assert annual["month"] == "2024-01"
    assert annual["temporal_precision"] == "annual_snapshot"


def test_filter_diagnosis_flow_points_by_years_uses_period_end() -> None:
    points = _filter_diagnosis_flow_points_by_years(
        [
            {
                "label": "2024 Q4 diagnosis flow",
                "period_start": "2024-10",
                "period_end": "2024-12",
                "effective_month": "2024-12",
                "diagnosed_share": 0.012,
            },
            {
                "label": "2025 Q1 diagnosis flow",
                "period_start": "2025-01",
                "period_end": "2025-03",
                "effective_month": "2025-03",
                "diagnosed_share": 0.013,
            },
        ],
        [2024],
    )

    assert len(points) == 1
    assert points[0]["label"] == "2024 Q4 diagnosis flow"


def test_diagnosis_flow_segment_sums_jax_is_vmap_safe() -> None:
    if jnp is None:
        return

    series = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    start_index = jnp.asarray([0, 1], dtype=jnp.int32)
    end_index = jnp.asarray([1, 3], dtype=jnp.int32)

    result = np.asarray(_diagnosis_flow_segment_sums_jax(series, start_index, end_index), dtype=np.float32)

    assert result.tolist() == [3.0, 9.0]


def test_mixed_frequency_bundle_marks_quarterly_harp_points_as_source_month() -> None:
    month_axis = [f"2024-{month:02d}" for month in range(1, 13)]
    support_bundle = {
        "targets": {
            "diagnosed_stock": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "art_stock": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "documented_suppression": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "testing_coverage": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
            "deaths": {"observed_mask": np.zeros((1, 12), dtype=np.float32), "latent_weight": np.ones((1, 12), dtype=np.float32)},
        }
    }
    bundle = build_mixed_frequency_observation_bundle(
        month_axis=month_axis,
        province_axis=["Philippines"],
        support_bundle=support_bundle,
        official_points=[],
        harp_points=[
            {
                "month": "2024-12",
                "effective_month": "2024-12",
                "source_month": "2024-12",
                "temporal_precision": "quarterly_snapshot",
                "estimated_plhiv": 215400,
                "diagnosed": 135026,
                "on_art": 90854,
                "viral_load_tested": 41860,
                "suppressed": 36723,
            }
        ],
        region_axis=["NCR"],
    )

    weight = np.asarray(bundle["harp_anchor_arrays"]["weight"], dtype=np.float32)
    assert float(weight[-1]) == 1.0
    assert float(np.sum(weight[:-1])) == 0.0
    assert bundle["aggregation_modes"]["harp"] == ["source_month"]
    assert bundle["summary_rows"][0]["anchor_aggregation_mode_used"] == "source_month"


def test_regrid_time_tensor_to_axis_expands_sparse_factor_support_without_dropping_columns() -> None:
    source_axis = ["2024-01", "2024-12"]
    target_axis = [f"2024-{month:02d}" for month in range(1, 13)]
    tensor = np.asarray([[[0.0], [1.0]]], dtype=np.float32)

    regridded = _regrid_time_tensor_to_axis(tensor, source_axis, target_axis)

    assert regridded.shape == (1, 12, 1)
    assert float(regridded[0, 0, 0]) == 0.0
    assert float(regridded[0, -1, 0]) == 1.0
    assert 0.0 < float(regridded[0, 5, 0]) < 1.0


def test_default_factor_transition_hooks_are_semantically_scoped() -> None:
    policy_hooks = _default_factor_transition_hooks({"block_name": "policy_implementation"})
    propagation_hooks = _default_factor_transition_hooks({"network_feature_family": "information_propagation"})

    assert "linkage_transitions" in policy_hooks
    assert "retention_attrition_transitions" in policy_hooks
    assert "diagnosis_transitions" in propagation_hooks
    assert "linkage_transitions" in propagation_hooks


def test_scaffold_dominance_gate_drops_determinants_when_artifact_is_missing(tmp_path: Path) -> None:
    covariates = np.zeros((2, 3, 5), dtype=np.float32)
    meta = {
        "representation_mode": "unclumped",
        "selected_determinant_modifiers": [
            {"canonical_name": "alpha", "curation_score": 0.7, "dag_score": 0.3},
            {"canonical_name": "beta", "curation_score": 0.6, "dag_score": 0.2},
        ],
        "covariate_names": ["obs_diagnosed", "obs_art", "obs_suppression", "det::behavior::alpha", "det::policy::beta"],
        "transition_hook_masks": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 1, 1]],
        "coupling_covariate_names": [],
        "network_family_indices": {},
    }

    trimmed, trimmed_meta = _apply_scaffold_dominance_gate(
        run_dir=tmp_path,
        covariates=covariates,
        covariate_meta=meta,
    )

    assert trimmed.shape[-1] == 3
    assert trimmed_meta["selected_determinant_modifiers"] == []
    assert trimmed_meta["scaffold_dominance_summary"]["verdict"] == "scaffold_only_enforced"


def test_scaffold_dominance_gate_keeps_only_topk_when_representation_beats_scaffold(tmp_path: Path) -> None:
    phase3_dir = tmp_path / "phase3_frozen_backtest_tournament"
    phase3_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_rows": [
            {
                "representation": "unclumped",
                "model_mean_absolute_error": 0.040,
                "simple_compartmental_mean_absolute_error": 0.050,
                "carry_forward_mean_absolute_error": 0.052,
            },
            {
                "representation": "unclumped",
                "model_mean_absolute_error": 0.041,
                "simple_compartmental_mean_absolute_error": 0.050,
                "carry_forward_mean_absolute_error": 0.053,
            },
        ]
    }
    (phase3_dir / "representation_tournament.json").write_text(json.dumps(payload), encoding="utf-8")

    covariates = np.zeros((2, 3, 8), dtype=np.float32)
    meta = {
        "representation_mode": "unclumped",
        "selected_determinant_modifiers": [
            {"canonical_name": "a", "curation_score": 0.1, "dag_score": 0.1},
            {"canonical_name": "b", "curation_score": 0.4, "dag_score": 0.3},
            {"canonical_name": "c", "curation_score": 0.9, "dag_score": 0.2},
            {"canonical_name": "d", "curation_score": 0.6, "dag_score": 0.1},
            {"canonical_name": "e", "curation_score": 0.3, "dag_score": 0.2},
        ],
        "covariate_names": [
            "obs_diagnosed",
            "obs_art",
            "obs_suppression",
            "det::behavior::a",
            "det::policy::b",
            "det::population::c",
            "det::logistics::d",
            "det::economics::e",
        ],
        "transition_hook_masks": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 1, 1], [1, 1, 0, 0, 0], [0, 1, 0, 1, 1], [0, 1, 0, 1, 1]],
        "coupling_covariate_names": [],
        "network_family_indices": {},
    }

    trimmed, trimmed_meta = _apply_scaffold_dominance_gate(
        run_dir=tmp_path,
        covariates=covariates,
        covariate_meta=meta,
    )

    assert trimmed.shape[-1] == 7
    assert len(trimmed_meta["selected_determinant_modifiers"]) == 4
    assert trimmed_meta["scaffold_dominance_summary"]["verdict"] == "passed_topk_trimmed"


def test_scaffold_dominance_gate_uses_partial_rolling_origin_split_rows_when_summary_is_missing(tmp_path: Path) -> None:
    tournament_dir = tmp_path / "phase3_frozen_backtest_tournament"
    tournament_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trial_rows": [
            {
                "representation": "unclumped",
                "model_mean_absolute_error": 0.040,
                "simple_compartmental_mean_absolute_error": 0.050,
                "carry_forward_mean_absolute_error": 0.052,
            }
        ]
    }
    (tournament_dir / "representation_tournament.json").write_text(json.dumps(payload), encoding="utf-8")

    covariates = np.zeros((2, 3, 5), dtype=np.float32)
    meta = {
        "representation_mode": "unclumped",
        "selected_determinant_modifiers": [
            {"canonical_name": "alpha", "curation_score": 0.7, "dag_score": 0.3},
            {"canonical_name": "beta", "curation_score": 0.6, "dag_score": 0.2},
        ],
        "covariate_names": ["obs_diagnosed", "obs_art", "obs_suppression", "det::behavior::alpha", "det::policy::beta"],
        "transition_hook_masks": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 1, 0, 0, 0], [0, 1, 0, 1, 1]],
        "coupling_covariate_names": [],
        "network_family_indices": {},
    }

    trimmed, trimmed_meta = _apply_scaffold_dominance_gate(
        run_dir=tmp_path,
        covariates=covariates,
        covariate_meta=meta,
    )

    assert trimmed.shape[-1] == 5
    assert trimmed_meta["scaffold_dominance_summary"]["source_kind"] == "broad_frozen_history_rescue_core"
    assert trimmed_meta["scaffold_dominance_summary"]["passes_scaffold_gate"] is True


def test_peak_search_target_series_and_group_templates() -> None:
    peak_cfg = get_disease_plugin("hiv").prior_hyperparameters["phase3"]["peak_search"]
    states = np.asarray(
        [
            [[0.40, 0.20, 0.20, 0.15, 0.05], [0.35, 0.21, 0.21, 0.18, 0.05]],
            [[0.38, 0.21, 0.22, 0.14, 0.05], [0.34, 0.22, 0.23, 0.16, 0.05]],
        ],
        dtype=np.float32,
    )
    gap = _target_series(states, ["Philippines", "NCR"], "documented_suppression_gap")
    templates = _group_transition_templates(
        [
            {"group_name": "behavior_stigma"},
            {"group_name": "biology_suppression"},
            {"group_name": "mobility_logistics"},
        ],
        peak_cfg,
    )

    assert gap.shape == (2,)
    assert np.all(gap >= 0.0)
    assert templates.shape == (3, 5)
    assert float(templates[0, 0]) > 0.0
    assert float(templates[1, 2]) > 0.0
    assert float(templates[2, 1]) > 0.0


def test_peak_search_simulation_tracks_mass_without_shape_collapse() -> None:
    initial_state = np.asarray([[0.40, 0.20, 0.20, 0.15, 0.05]], dtype=np.float32)
    transition_probs = np.full((1, 3, 5), 0.05, dtype=np.float32)

    future_states, mass_violation = _simulate_future_states(initial_state, transition_probs)

    assert future_states.shape == (1, 3, 5)
    assert np.allclose(future_states.sum(axis=-1), 1.0, atol=1e-5)
    assert mass_violation >= 0.0


def test_torch_anchor_penalties_accept_single_month_override_arrays() -> None:
    if torch is None:
        return

    diagnosed = torch.tensor([[0.61]], dtype=torch.float32)
    art = torch.tensor([[0.39]], dtype=torch.float32)
    suppression = torch.tensor([[0.21]], dtype=torch.float32)
    testing = torch.tensor([[0.22]], dtype=torch.float32)
    official_penalty, official_detail = _national_anchor_penalty_torch(
        diagnosed_pred=diagnosed,
        art_pred=art,
        suppression_pred=suppression,
        province_axis=["Philippines"],
        month_axis=["2025-12"],
        device=diagnosed.device,
        curve_overrides={
            "diagnosed_stock": [0.61],
            "art_stock": [0.39],
            "documented_suppression": [0.21],
            "third95": [0.53],
            "weight": [1.0],
        },
    )
    harp_penalty, harp_detail = _harp_program_penalty_torch(
        diagnosed_pred=diagnosed,
        art_pred=art,
        suppression_pred=suppression,
        testing_pred=testing,
        province_axis=["Philippines"],
        month_axis=["2025-12"],
        device=diagnosed.device,
        curve_overrides={
            "diagnosed_stock": [0.61],
            "art_stock": [0.39],
            "testing_coverage": [0.22],
            "documented_suppression": [0.21],
            "viral_load_tested_among_art": [0.56],
            "suppressed_among_art": [0.53],
            "weight": [1.0],
        },
    )

    assert float(official_penalty) >= 0.0
    assert float(harp_penalty) >= 0.0
    assert "national_anchor_penalty" in official_detail
    assert "harp_program_penalty" in harp_detail


def test_phase3_cli_parser_accepts_peak_search_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "phase3",
            "peak-search",
            "--run-id",
            "example-run",
            "--plugin",
            "hiv",
            "--profile",
            "hiv_rescue_v2",
            "--phase3-inference",
            "torch_map",
            "--representation",
            "hybrid_temporal_multiscale",
            "--target",
            "diagnosed_stock",
            "--horizon-months",
            "60",
        ]
    )

    assert args.phase3_command == "peak-search"
    assert args.representation == "hybrid_temporal_multiscale"
    assert args.target == "diagnosed_stock"
    assert args.horizon_months == 60


def test_peak_search_promotion_gate_gracefully_handles_backend_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        peak_search_module,
        "_prepare_frozen_backtest_inputs",
        lambda **_: {"backtest_config": {"train_years": [2017, 2018, 2019], "holdout_years": [2020, 2021]}},
    )

    def _raise_backend_error(**_: object) -> dict[str, object]:
        raise RuntimeError("requested inference family 'torch_map' but resolved 'numpy_map'")

    monkeypatch.setattr(peak_search_module, "_run_single_trial", _raise_backend_error)
    result = _promotion_gate(
        run_id="example-run",
        plugin_id="hiv",
        profile="hiv_rescue_v2",
        inference_family="torch_map",
        representation="hybrid_temporal_multiscale",
        tolerance=0.0025,
    )

    assert result["passed"] is False
    assert result["reason"] == "backend_unavailable_for_requested_inference_family"
    assert "numpy_map" in str(result["details"])


def test_phase3_temporal_artifacts_are_emitted(rescue_v2_run_dir: Path) -> None:
    phase3_dir = rescue_v2_run_dir / "phase3"
    temporal = read_json(phase3_dir / "temporal_hazard_components.json", default={})
    shock = read_json(phase3_dir / "shock_regime_summary.json", default={})
    mixed = read_json(phase3_dir / "mixed_frequency_observation_summary.json", default={})

    assert temporal.get("scaffold_source") == "simple_compartmental_scaffold"
    assert temporal.get("slow_basis_shape", []) != []
    assert shock.get("regime_names", []) != []
    assert mixed.get("rows", []) != []
    assert (phase3_dir / "transition_probabilities.npz").exists()


def test_phase3_temporal_and_peak_search_modules_have_no_numeric_config_fallbacks() -> None:
    temporal_text = Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase3\temporal_scaffold.py").read_text(encoding="utf-8")
    peak_text = Path(r"D:\EpiGraph_PH\src\epigraph_ph\phase3\peak_search.py").read_text(encoding="utf-8")
    assert '.get("slow_knot_months", ' not in temporal_text
    assert '.get("medium_block_months", ' not in temporal_text
    assert ".get(\"determinant_bound_scale\", " not in peak_text
