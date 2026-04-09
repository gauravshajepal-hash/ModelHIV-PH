from __future__ import annotations

from pathlib import Path

from epigraph_ph.core.disease_plugin import get_disease_plugin
from epigraph_ph.phase3._lineage.national_reset_core import (
    fit_national_uda_baseline,
    fit_national_uda_delay_aux,
    fit_national_uda_vl_observation_process,
    quarter_sort_key,
)
from epigraph_ph.phase3._lineage.national_reset_pipeline import (
    _deferred_factor_screen,
    _load_legacy_benchmark_reference,
    build_national_reset_observation_table_payload,
)


def test_quarter_sort_key_orders_quarters() -> None:
    assert sorted(["2025-Q4", "2024-Q4", "2025-Q1"], key=quarter_sort_key) == ["2024-Q4", "2025-Q1", "2025-Q4"]


def test_nr00_observation_table_from_live_archive_has_expected_shape() -> None:
    archive_dir = Path(r"D:\EpiGraph_PH\artifacts\runs\audit-phase0-reuse-s00-20260331\harp_archive")

    payload = build_national_reset_observation_table_payload(
        archive_dir=archive_dir,
        archive_run_id="audit-phase0-reuse-s00-20260331",
        start_quarter="2022-Q1",
    )

    rows = payload["rows"]
    assert rows[0]["quarter"] == "2022-Q1"
    assert rows[-1]["quarter"] == "2025-Q4"
    assert len({row["quarter"] for row in rows}) == len(rows)
    assert sum(1 for row in rows if row["diagnosed_plhiv"] is not None) >= 12
    assert sum(1 for row in rows if row["alive_on_art"] is not None) >= 12
    assert sum(1 for row in rows if row["new_diagnosed_cases_period"] is not None) >= 8
    assert "advanced_hiv_cases_period" in payload["summary"]["missingness_report"]
    assert "median_cd4_at_enrollment" in payload["summary"]["missingness_report"]
    assert any(row["source_ids"] for row in rows)
    assert payload["summary"]["quality_filter_counts"]["median_cd4_at_enrollment"] >= 1
    bad_cd4_quarters = [
        row["quarter"]
        for row in rows
        if row["metric_provenance"]["median_cd4_at_enrollment"]["quality_filter"]
    ]
    assert "2022-Q4" in bad_cd4_quarters


def test_nr01_live_archive_baseline_beats_count_space_baselines(tmp_path) -> None:
    archive_dir = Path(r"D:\EpiGraph_PH\artifacts\runs\audit-phase0-reuse-s00-20260331\harp_archive")
    payload = build_national_reset_observation_table_payload(
        archive_dir=archive_dir,
        archive_run_id="audit-phase0-reuse-s00-20260331",
        start_quarter="2022-Q1",
    )
    legacy_reference = _load_legacy_benchmark_reference()
    plugin = get_disease_plugin("hiv")
    phase3_cfg = dict(((plugin.constraint_settings or {}).get("phase3", {}) or {}))
    national_reset_cfg = dict((phase3_cfg.get("national_reset", {}) or {}))
    benchmark_gates = dict((national_reset_cfg.get("benchmark_gates", {}) or {}))
    diagnosis_flow_gate = float(benchmark_gates["legacy_diagnosis_flow_improvement_pct"])

    result = fit_national_uda_baseline(
        observation_rows=payload["rows"],
        estimated_plhiv_by_quarter=payload["estimated_plhiv_by_quarter"],
        diagnosis_flow_targets=payload["diagnosis_flow_targets_by_quarter"],
        artifact_dir=tmp_path,
        holdout_years=[2025],
        legacy_reference_metrics=legacy_reference,
    )

    baseline = result["baseline_comparison"]
    diagnosis = result["diagnosis_flow_evaluation"]
    decision = result["decision"]
    assert baseline["model_mean_absolute_error"] < baseline["carry_forward_mean_absolute_error"]
    assert baseline["model_mean_absolute_error"] < baseline["simple_compartmental_mean_absolute_error"]
    assert baseline["model_mean_absolute_error"] < legacy_reference["legacy_carry_forward_mae"]
    assert diagnosis["mean_absolute_error"] < legacy_reference["legacy_diagnosis_flow_mae"]
    assert diagnosis["improvement_vs_baseline_pct"] >= diagnosis_flow_gate
    assert decision["keep"] is True


def test_nr02_and_nr03_live_archive_paths_produce_artifacts(tmp_path) -> None:
    archive_dir = Path(r"D:\EpiGraph_PH\artifacts\runs\audit-phase0-reuse-s00-20260331\harp_archive")
    payload = build_national_reset_observation_table_payload(
        archive_dir=archive_dir,
        archive_run_id="audit-phase0-reuse-s00-20260331",
        start_quarter="2022-Q1",
    )
    legacy_reference = _load_legacy_benchmark_reference()
    nr01_dir = tmp_path / "nr01"
    nr01_dir.mkdir()
    nr01 = fit_national_uda_baseline(
        observation_rows=payload["rows"],
        estimated_plhiv_by_quarter=payload["estimated_plhiv_by_quarter"],
        diagnosis_flow_targets=payload["diagnosis_flow_targets_by_quarter"],
        artifact_dir=nr01_dir,
        holdout_years=[2025],
        legacy_reference_metrics=legacy_reference,
    )

    nr02_dir = tmp_path / "nr02"
    nr02_dir.mkdir()
    nr02 = fit_national_uda_delay_aux(
        observation_rows=payload["rows"],
        estimated_plhiv_by_quarter=payload["estimated_plhiv_by_quarter"],
        diagnosis_flow_targets=payload["diagnosis_flow_targets_by_quarter"],
        artifact_dir=nr02_dir,
        holdout_years=[2025],
        reference_metrics={
            "reference_model_mean_absolute_error": float(nr01["baseline_comparison"]["model_mean_absolute_error"]),
            "reference_diagnosis_flow_mean_absolute_error": float(nr01["diagnosis_flow_evaluation"]["mean_absolute_error"]),
        },
    )
    assert "delay_aux_summary" in nr02
    assert (nr02_dir / "delay_aux_summary.json").exists()
    assert nr02["delay_aux_summary"]["auxiliary_signals_are_weak"] is True

    nr03_dir = tmp_path / "nr03"
    nr03_dir.mkdir()
    nr03 = fit_national_uda_vl_observation_process(
        observation_rows=payload["rows"],
        upstream_result=nr02,
        artifact_dir=nr03_dir,
        holdout_years=[2025],
    )
    assert "vl_observation_process" in nr03
    assert (nr03_dir / "vl_observation_process.json").exists()
    assert nr03["vl_observation_process"]["service_rows"]


def test_nr05_phase2_screen_rejects_current_blanket_for_dx01() -> None:
    screen = _deferred_factor_screen("audit-phase0-reuse-s00-20260331")

    assert screen["summary"]["blanket_factor_count"] >= 1
    factor_rows = {str(row["factor_id"]): row for row in screen["factor_rows"]}
    assert "factor_0003" in factor_rows
    assert factor_rows["factor_0003"]["screening_status"] == "reject_observation_adjacent"
    assert "network_factor_003" in factor_rows
    assert factor_rows["network_factor_003"]["screening_status"] == "defer_to_dx04_network_operator"
    assert screen["summary"]["promotable_factor_count"] == 0
