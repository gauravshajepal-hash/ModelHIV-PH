from __future__ import annotations

import numpy as np
import pytest

from epigraph_ph.phase2.latent_temporal_graph import (
    _build_sample_arrays,
    _temporal_block_permutation_indices,
    estimate_latent_temporal_scale_graph,
)
from epigraph_ph.phase2.multiscale_dag import _normalize_retained_factor_rows


def test_latent_temporal_graph_recovers_simple_lagged_chain() -> None:
    rng = np.random.default_rng(7)
    unit_count = 8
    month_count = 48
    block_axis = ["testing_engagement", "care_access_continuity", "suppression_capacity"]
    tensor = np.zeros((unit_count, month_count, len(block_axis)), dtype=np.float32)
    for unit_idx in range(unit_count):
        for month_idx in range(1, month_count):
            prev = tensor[unit_idx, month_idx - 1]
            noise = rng.normal(0.0, 0.08, size=len(block_axis)).astype(np.float32)
            tensor[unit_idx, month_idx, 0] = 0.35 * prev[0] + noise[0]
            tensor[unit_idx, month_idx, 1] = 0.25 * prev[1] + 0.95 * prev[0] + noise[1]
            tensor[unit_idx, month_idx, 2] = 0.20 * prev[2] + 0.90 * prev[1] + noise[2]

    bundle, blanket = estimate_latent_temporal_scale_graph(
        scale_name="province",
        state_tensor=tensor,
        block_axis=block_axis,
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block={
            "testing_engagement": ["testing_rate"],
            "care_access_continuity": ["linkage_to_care"],
            "suppression_capacity": ["viral_suppression_rate"],
        },
        cfg={
            "max_lag": 1,
            "ridge_penalty": 0.05,
            "sparse_penalty": 0.04,
            "low_rank_penalty": 0.03,
            "decomposition_steps": 140,
            "rank_threshold": 0.03,
            "edge_threshold": 0.06,
            "hidden_driver_threshold": 0.05,
            "stability_threshold": 0.4,
            "bootstrap_draws": 18,
            "bootstrap_block_length": 6,
            "min_effective_samples": 24,
            "rng_seed": 11,
        },
        phi_by_block={
            "testing_engagement": 0.45,
            "care_access_continuity": 0.40,
            "suppression_capacity": 0.35,
        },
    )

    assert bundle["status"] == "completed"
    edges = {(row["source"], row["target"]): row for row in bundle["combined_operator_rows"]}
    hidden_edges = {(row["source"], row["target"]): row for row in bundle["hidden_driver_rows"]}
    assert ("testing_engagement", "care_access_continuity") in edges
    assert ("care_access_continuity", "suppression_capacity") in edges
    assert int(edges[("testing_engagement", "care_access_continuity")]["lag"]) == 1
    assert edges[("testing_engagement", "care_access_continuity")]["weight"] > 0.0
    assert edges[("care_access_continuity", "suppression_capacity")]["weight"] > 0.0
    assert isinstance(bundle["hidden_driver_rows"], list)
    assert isinstance(bundle["hidden_driver_fallback_rows"], list)
    assert bundle["hidden_driver_fallback_used"] in {True, False}
    assert blanket["phase3_member_canonical_names"] == blanket["direct_phase3_member_canonical_names"]
    assert isinstance(blanket["hidden_phase3_member_canonical_names"], list)


def test_latent_temporal_graph_returns_unavailable_for_too_few_samples() -> None:
    tensor = np.zeros((1, 2, 3), dtype=np.float32)
    bundle, blanket = estimate_latent_temporal_scale_graph(
        scale_name="national",
        state_tensor=tensor,
        block_axis=["a", "b", "c"],
        target_block_ids=["b"],
        indicator_names_by_block={},
        cfg={
            "max_lag": 1,
            "ridge_penalty": 0.1,
            "sparse_penalty": 0.08,
            "low_rank_penalty": 0.05,
            "edge_threshold": 0.05,
            "hidden_driver_threshold": 0.05,
            "stability_threshold": 0.5,
            "bootstrap_draws": 4,
            "bootstrap_block_length": 2,
            "min_effective_samples": 16,
            "rng_seed": 3,
        },
        phi_by_block={"a": 0.5, "b": 0.5, "c": 0.5},
    )

    assert bundle["status"] == "unavailable"
    assert bundle["reason"] == "insufficient_samples"
    assert blanket["blanket_block_ids"] == []


def test_latent_temporal_graph_recovers_multi_lag_edge_and_uses_uncertainty_weights() -> None:
    rng = np.random.default_rng(9)
    unit_count = 6
    month_count = 54
    block_axis = ["testing_engagement", "care_access_continuity", "suppression_capacity"]
    tensor = np.zeros((unit_count, month_count, len(block_axis)), dtype=np.float32)
    uncertainty = np.full_like(tensor, 0.30, dtype=np.float32)
    uncertainty[: unit_count // 2, :, :] = 0.10
    for unit_idx in range(unit_count):
        for month_idx in range(2, month_count):
            prev = tensor[unit_idx, month_idx - 1]
            prev2 = tensor[unit_idx, month_idx - 2]
            noise = rng.normal(0.0, 0.07, size=len(block_axis)).astype(np.float32)
            tensor[unit_idx, month_idx, 0] = 0.35 * prev[0] + noise[0]
            tensor[unit_idx, month_idx, 1] = 0.20 * prev[1] + 0.95 * prev2[0] + noise[1]
            tensor[unit_idx, month_idx, 2] = 0.20 * prev[2] + 0.85 * prev[1] + noise[2]

    bundle, blanket = estimate_latent_temporal_scale_graph(
        scale_name="province",
        state_tensor=tensor,
        uncertainty_tensor=uncertainty,
        block_axis=block_axis,
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block={
            "testing_engagement": ["testing_rate"],
            "care_access_continuity": ["linkage_to_care"],
            "suppression_capacity": ["viral_suppression_rate"],
        },
        cfg={
            "max_lag": 2,
            "ridge_penalty": 0.05,
            "sparse_penalty": 0.04,
            "low_rank_penalty": 0.03,
            "decomposition_steps": 140,
            "rank_threshold": 0.03,
            "edge_threshold": 0.06,
            "hidden_driver_threshold": 0.05,
            "stability_threshold": 0.35,
            "bootstrap_draws": 14,
            "bootstrap_block_length": 6,
            "min_effective_samples": 24,
            "rng_seed": 17,
        },
        phi_by_block={
            "testing_engagement": 0.45,
            "care_access_continuity": 0.35,
            "suppression_capacity": 0.30,
        },
    )

    edges = {(row["source"], row["target"], int(row["lag"])): row for row in bundle["combined_operator_rows"]}
    assert bundle["status"] == "completed"
    assert bundle["uncertainty_available"] is True
    assert bundle["sample_weight_summary"]["max"] > bundle["sample_weight_summary"]["min"]
    assert ("testing_engagement", "care_access_continuity", 2) in edges
    assert ("care_access_continuity", "suppression_capacity", 1) in edges
    assert isinstance(blanket["blanket_block_ids"], list)


def test_latent_temporal_graph_falsification_keeps_direct_false_positives_low() -> None:
    rng = np.random.default_rng(13)
    unit_count = 8
    month_count = 48
    block_axis = ["testing_engagement", "care_access_continuity", "suppression_capacity"]
    tensor = np.zeros((unit_count, month_count, len(block_axis)), dtype=np.float32)
    for unit_idx in range(unit_count):
        hidden = 0.0
        for month_idx in range(1, month_count):
            prev = tensor[unit_idx, month_idx - 1]
            noise = rng.normal(0.0, 0.07, size=len(block_axis)).astype(np.float32)
            hidden = 0.60 * hidden + float(rng.normal(0.0, 0.10))
            tensor[unit_idx, month_idx, 0] = 0.50 * prev[0] + 0.35 * hidden + noise[0]
            tensor[unit_idx, month_idx, 1] = 0.45 * prev[1] + 0.30 * hidden + noise[1]
            tensor[unit_idx, month_idx, 2] = 0.40 * prev[2] + 0.25 * hidden + noise[2]

    bundle, _ = estimate_latent_temporal_scale_graph(
        scale_name="province",
        state_tensor=tensor,
        block_axis=block_axis,
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block={},
        cfg={
            "max_lag": 2,
            "ridge_penalty": 0.05,
            "sparse_penalty": 0.05,
            "low_rank_penalty": 0.03,
            "decomposition_steps": 120,
            "rank_threshold": 0.03,
            "edge_threshold": 0.07,
            "hidden_driver_threshold": 0.05,
            "stability_threshold": 0.45,
            "bootstrap_draws": 12,
            "bootstrap_block_length": 6,
            "min_effective_samples": 24,
            "rng_seed": 19,
        },
        phi_by_block={
            "testing_engagement": 0.50,
            "care_access_continuity": 0.45,
            "suppression_capacity": 0.40,
        },
    )

    assert bundle["status"] == "completed"
    assert bundle["edge_count"] <= 2
    assert bundle["estimated_hidden_rank"] >= 1


def test_temporal_block_null_permutation_preserves_unit_membership() -> None:
    sample_pairs = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    ]
    rng = np.random.default_rng(23)
    mapping = _temporal_block_permutation_indices(sample_pairs, block_length=2, rng=rng)

    assert sorted(mapping.tolist()) == list(range(len(sample_pairs)))
    for row_idx, mapped_idx in enumerate(mapping.tolist()):
        assert sample_pairs[row_idx][0] == sample_pairs[int(mapped_idx)][0]


def test_build_sample_arrays_respects_calendar_gaps() -> None:
    tensor = np.arange(1 * 4 * 1, dtype=np.float32).reshape(1, 4, 1)

    sample_arrays = _build_sample_arrays(
        state_tensor=tensor,
        block_axis=["testing_engagement"],
        phi_by_block={"testing_engagement": 0.25},
        max_lag=1,
        month_axis=["2025-01", "2025-03", "2025-04", "2025-06"],
    )

    assert sample_arrays["sample_pairs"] == [(0, 2)]
    assert sample_arrays["effective_sample_count"] == 1


def test_latent_temporal_graph_exports_sparse_direct_rows_and_deoverlaps_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase2.latent_temporal_graph as ltg

    tensor = np.zeros((1, 12, 2), dtype=np.float32)

    monkeypatch.setattr(
        ltg,
        "_select_temporal_hyperparameters",
        lambda **_kwargs: {
            "selection_failed": False,
            "score": 1.0,
            "max_lag": 1,
            "sparse_penalty": 0.05,
            "low_rank_penalty": 0.05,
            "selection_metric": "weighted_bic",
            "weighted_validation_loss": 0.5,
            "bic_penalty": 0.5,
            "complexity": 2,
            "nnz_sparse": 1,
            "hidden_rank": 1,
        },
    )
    monkeypatch.setattr(
        ltg,
        "_fit_sparse_plus_low_rank_matrix",
        lambda **_kwargs: (
            np.asarray([[0.0, 0.60], [0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.25], [-0.08, 0.0]], dtype=np.float32),
            {"objective": 1.0, "converged": True, "iterations": 5, "objective_trace_summary": {}, "parameter_delta": 0.0, "objective_improvement": 0.0, "complexity": {}},
        ),
    )
    monkeypatch.setattr(
        ltg,
        "_permutation_null_thresholds",
        lambda **_kwargs: {
            "direct_edge_threshold": 0.10,
            "combined_edge_threshold": 0.10,
            "hidden_edge_threshold": 0.06,
            "rank_threshold": 0.10,
            "null_quantile": 0.95,
            "permutation_count": 4,
            "null_block_length": 2,
            "null_mode": "unit_respecting_temporal_block_permutation",
        },
    )
    monkeypatch.setattr(
        ltg,
        "_stability_from_bootstrap",
        lambda **_kwargs: (np.ones((2, 2), dtype=np.float32), 1.0),
    )

    bundle, _ = estimate_latent_temporal_scale_graph(
        scale_name="province",
        state_tensor=tensor,
        block_axis=["testing_engagement", "care_access_continuity"],
        target_block_ids=["care_access_continuity"],
        indicator_names_by_block={},
        cfg={"min_effective_samples": 1, "max_lag": 1},
        phi_by_block={"testing_engagement": 0.2, "care_access_continuity": 0.2},
    )

    direct_keys = {(row["source"], row["target"], int(row["lag"])) for row in bundle["edges"]}
    hidden_keys = {(row["source"], row["target"], int(row["lag"])) for row in bundle["hidden_driver_rows"]}

    assert bundle["direct_surface_kind"] == "sparse_supported_combined_temporal_operator"
    assert direct_keys == {("testing_engagement", "care_access_continuity", 1)}
    assert hidden_keys == {("care_access_continuity", "testing_engagement", 1)}
    assert direct_keys.isdisjoint(hidden_keys)


def test_latent_temporal_graph_fails_closed_when_selection_has_no_finite_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    import epigraph_ph.phase2.latent_temporal_graph as ltg

    monkeypatch.setattr(
        ltg,
        "_select_temporal_hyperparameters",
        lambda **_kwargs: {
            "selection_failed": True,
            "reason": "no_finite_weighted_bic_candidate",
            "score": float("inf"),
            "selection_metric": "weighted_bic",
            "max_lag": 1,
        },
    )

    bundle, blanket = estimate_latent_temporal_scale_graph(
        scale_name="national",
        state_tensor=np.zeros((1, 24, 2), dtype=np.float32),
        block_axis=["a", "b"],
        target_block_ids=["b"],
        indicator_names_by_block={},
        cfg={"max_lag": 1},
        phi_by_block={"a": 0.2, "b": 0.2},
    )

    assert bundle["status"] == "unavailable"
    assert bundle["reason"] == "no_finite_weighted_bic_candidate"
    assert blanket["blanket_block_ids"] == []


def test_multiscale_factor_normalization_handles_factor_id_dialect_mismatch() -> None:
    factor_lookup = {
        "network_factor_008": {
            "factor_id": "network_factor_008",
            "factor_name": "service_single_point_failure_score",
            "factor_class": "network_feature",
            "block_name": "policy_implementation",
            "network_feature_family": "percolation_fragility",
            "member_canonical_names": [],
        },
        "factor_0000": {
            "factor_id": "factor_0000",
            "factor_name": "service_delivery_infrastructure_factor_0000",
            "factor_class": "mesoscopic_factor",
            "block_name": "service_delivery_infrastructure",
            "interpretability_label": "service_delivery_infrastructure",
            "member_canonical_names": ["alive_on_art", "art_share", "tested_for_viral_load", "tested_share"],
        },
    }
    retained_rows = [
        {
            "factor_id": "phase15_pool_alias_a",
            "factor_name": "service_single_point_failure_score",
            "factor_class": "network_feature",
            "block_name": "policy_implementation",
            "network_feature_family": "percolation_fragility",
        },
        {
            "factor_id": "phase15_pool_alias_b",
            "factor_name": "service_delivery_infrastructure_factor_0000",
            "factor_class": "mesoscopic_factor",
            "block_name": "service_delivery_infrastructure",
            "member_canonical_names": ["tested_share", "art_share", "tested_for_viral_load", "alive_on_art"],
        },
    ]

    normalized_rows, summary = _normalize_retained_factor_rows(
        retained_factor_rows=retained_rows,
        factor_lookup=factor_lookup,
    )

    assert summary["input_count"] == 2
    assert summary["alias_match_count"] == 2
    assert summary["unmatched_count"] == 0
    assert [row["factor_id"] for row in normalized_rows] == ["network_factor_008", "factor_0000"]
    assert normalized_rows[0]["input_factor_id"] == "phase15_pool_alias_a"
    assert normalized_rows[1]["input_factor_id"] == "phase15_pool_alias_b"
