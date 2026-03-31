from __future__ import annotations

from pathlib import Path

from epigraph_ph.runtime import read_json


def _assert_gold_standard_package(run_dir: Path, phase_name: str, *, mode: str, truth_claim_level: str) -> tuple[dict, list[dict], dict]:
    phase_dir = run_dir / phase_name
    manifest = read_json(phase_dir / "gold_standard_manifest.json", default={})
    checks = read_json(phase_dir / "gold_standard_checks.json", default=[])
    summary = read_json(phase_dir / "gold_standard_summary.json", default={})

    assert manifest.get("phase_name") == phase_name
    assert manifest.get("gold_standard_mode") == mode
    assert manifest.get("truth_claim_level") == truth_claim_level
    assert manifest.get("standards", [])
    assert checks
    assert summary.get("phase_name") == phase_name
    assert summary.get("gold_standard_mode") == mode
    assert summary.get("truth_claim_level") == truth_claim_level
    assert summary.get("standard_count", 0) >= 1

    check_names = {row.get("name") for row in checks}
    assert "gold_standard_profile_declared" in check_names
    return manifest, checks, summary


def test_phase0_gold_standard_package(phase0_registry_run_dir: Path) -> None:
    _, checks, summary = _assert_gold_standard_package(
        phase0_registry_run_dir,
        "phase0",
        mode="authoritative_external",
        truth_claim_level="process_quality",
    )
    check_names = {row.get("name") for row in checks}
    assert "prisma_traceability_present" in check_names
    assert "uncertainty_propagated" in check_names
    assert summary.get("literature_silo_count", 0) >= 10


def test_phase1_gold_standard_package(legacy_full_run_dir: Path) -> None:
    _, checks, summary = _assert_gold_standard_package(
        legacy_full_run_dir,
        "phase1",
        mode="authoritative_external",
        truth_claim_level="measurement_quality",
    )
    check_names = {row.get("name") for row in checks}
    assert "measurement_uncertainty_fields_present" in check_names
    assert "robust_scaling_documented" in check_names
    assert summary.get("normalized_row_count", 0) >= 1


def test_phase15_gold_standard_package(rescue_v2_run_dir: Path) -> None:
    _, checks, summary = _assert_gold_standard_package(
        rescue_v2_run_dir,
        "phase15",
        mode="benchmark_family",
        truth_claim_level="mesoscopic_benchmark",
    )
    check_names = {row.get("name") for row in checks}
    assert "permutation_null_gap_reported" in check_names
    assert "source_dropout_robustness_reported" in check_names
    assert summary.get("factor_count", 0) >= 1


def test_phase2_gold_standard_package(legacy_full_run_dir: Path) -> None:
    _, checks, summary = _assert_gold_standard_package(
        legacy_full_run_dir,
        "phase2",
        mode="benchmark_family",
        truth_claim_level="benchmark_only",
    )
    check_names = {row.get("name") for row in checks}
    assert "acyclicity_exact" in check_names
    assert "tier_mask_respected" in check_names
    assert "permutation_or_bootstrap_benchmark_present" in check_names
    assert summary.get("edge_count", 0) >= 0


def test_phase3_gold_standard_package(rescue_v2_run_dir: Path) -> None:
    _, checks, summary = _assert_gold_standard_package(
        rescue_v2_run_dir,
        "phase3",
        mode="hybrid_authoritative",
        truth_claim_level="strongest_truth_claim",
    )
    check_names = {row.get("name") for row in checks}
    assert "who_unaids_reference_points_declared" in check_names
    assert "harp_program_points_declared" in check_names
    assert "historical_harp_backtest_present" in check_names
    assert "incumbent_comparator_present" in check_names
    assert summary.get("official_reference_comparison_count", 0) >= 1


def test_phase4_gold_standard_package(legacy_full_run_dir: Path) -> None:
    manifest, checks, summary = _assert_gold_standard_package(
        legacy_full_run_dir,
        "phase4",
        mode="benchmark_family",
        truth_claim_level="decision_benchmark",
    )
    check_names = {row.get("name") for row in checks}
    assert "stochastic_rollout_present" in check_names
    assert "node_graph_runtime_invariant_present" in check_names
    assert "node_graph" in manifest.get("related_layers", [])
    assert summary.get("frontier_count", 0) >= 1
