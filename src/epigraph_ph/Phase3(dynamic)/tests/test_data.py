from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phase3_dynamic.data import MISSING_DATA_LADDER, STATE_NAMES, build_blocked_time_dataset, build_early_partial_target_rows, build_observation_rows, default_epigraph_root, partial_observation_splits, rolling_origin_splits
from phase3_dynamic.phase2 import build_direct_prior_features, build_hidden_driver_features, load_phase2_structural_inputs, resolve_phase2_determinant_robustness_report, resolve_phase2_structural_source_run_id
from phase3_dynamic.priors import PHASE2_TRANSITION_PRIOR_MAP
from phase3_dynamic.runtime import ensure_dir, write_json


def _latest_archive_run_id() -> str:
    artifacts_dir = default_epigraph_root() / "artifacts" / "runs"
    candidates = sorted(
        [
            path.name
            for path in artifacts_dir.iterdir()
            if path.is_dir() and (path / "harp_archive" / "historical_metric_rows.json").exists()
        ]
    )
    if not candidates:
        pytest.skip("No live harp_archive run is available for Phase3(dynamic) data tests.")
    preferred = sorted([name for name in candidates if name.startswith("harp-archive-wdi-standard-")])
    if preferred:
        return str(preferred[-1])
    return str(candidates[-1])


def _phase2_run_id_or_skip() -> str:
    artifacts_dir = default_epigraph_root() / "artifacts" / "runs"
    candidates = sorted(
        [
            path.name
            for path in artifacts_dir.iterdir()
            if path.is_dir() and (path / "phase2" / "phase2_structural_payload.json").exists()
        ]
    )
    if not candidates:
        pytest.skip("No phase2_structural_payload.json run is available in this workspace.")
    return str(candidates[-1])


def _phase2_run_id_with_direct_features_or_skip() -> str:
    artifacts_dir = default_epigraph_root() / "artifacts" / "runs"
    candidates = sorted(
        [
            path.name
            for path in artifacts_dir.iterdir()
            if path.is_dir() and (path / "phase2" / "phase2_structural_payload.json").exists()
        ],
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = load_phase2_structural_inputs(default_epigraph_root(), candidate)
            features = build_direct_prior_features(payload, PHASE2_TRANSITION_PRIOR_MAP)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        if sum(len(rows) for rows in features.values()) > 0:
            return str(candidate)
    pytest.skip("No Phase 2 payload with direct-prior features is available in this workspace.")


def _phase2_run_id_with_hidden_features_or_skip() -> str:
    artifacts_dir = default_epigraph_root() / "artifacts" / "runs"
    candidates = sorted(
        [
            path.name
            for path in artifacts_dir.iterdir()
            if path.is_dir() and (path / "phase2" / "phase2_structural_payload.json").exists()
        ],
        reverse=True,
    )
    for candidate in candidates:
        try:
            payload = load_phase2_structural_inputs(default_epigraph_root(), candidate)
            features = build_hidden_driver_features(payload, PHASE2_TRANSITION_PRIOR_MAP)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        if sum(len(rows) for rows in features.values()) > 0:
            return str(candidate)
    pytest.skip("No Phase 2 payload with hidden-driver features is available in this workspace.")


def test_live_archive_builds_observation_rows() -> None:
    rows = build_observation_rows(default_epigraph_root(), _latest_archive_run_id())
    years = sorted({int(str(row["quarter"]).split("-Q", 1)[0]) for row in rows})
    assert years
    assert years[0] >= 2010
    assert years[-1] >= years[0]
    assert any(row.get("new_diagnosed_cases_period") is not None for row in rows)


def test_live_archive_observation_rows_include_missing_data_ladder() -> None:
    rows = build_observation_rows(default_epigraph_root(), _latest_archive_run_id())
    first = rows[0]
    assert set(first["metric_provenance"]).issuperset({"diagnosed_plhiv", "alive_on_art", "new_diagnosed_cases_period"})
    assert str(first["metric_provenance"]["diagnosed_plhiv"]["tier"]) in set(MISSING_DATA_LADDER)
    assert str(first["row_provenance_tier"]) in set(MISSING_DATA_LADDER)


def test_live_archive_builds_blocked_time_splits() -> None:
    rows = build_observation_rows(default_epigraph_root(), _latest_archive_run_id())
    years = sorted({int(str(row["quarter"]).split("-Q", 1)[0]) for row in rows})
    start_year = years[0]
    end_year = years[-1]
    min_train_years = max(2, min(5, len(years) - 1))
    splits = rolling_origin_splits(rows, start_year=start_year, end_year=end_year, min_train_years=min_train_years, horizon_years=1)
    assert splits
    dataset = build_blocked_time_dataset(rows, splits[0]["holdout_years"])
    assert dataset.train_transition_rows
    assert dataset.provenance_summary["ladder"] == list(MISSING_DATA_LADDER)
    assert int(dataset.provenance_summary["train_observation_rows"]["row_count"]) == len(dataset.train_rows)
    assert int(dataset.provenance_summary["holdout_state_rows"]["row_count"]) == len(dataset.holdout_state_rows)
    assert dataset.provenance_summary["state_parameter_contract"] == "train_rows_only"
    assert dataset.provenance_summary["metric_scale_contract"] == "train_rows_only"


def test_blocked_time_dataset_train_state_and_scales_ignore_holdout_values() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 50.0,
            "virally_suppressed": 25.0,
            "estimated_plhiv": 200.0,
            "new_diagnosed_cases_period": 10.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 110.0,
            "alive_on_art": 55.0,
            "virally_suppressed": 30.0,
            "estimated_plhiv": 220.0,
            "new_diagnosed_cases_period": 12.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 60.0,
            "virally_suppressed": None,
            "estimated_plhiv": None,
            "new_diagnosed_cases_period": 14.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
    ]
    baseline = build_blocked_time_dataset(rows, [2022])
    perturbed_rows = [dict(row) for row in rows]
    perturbed_rows[-1] = {
        **perturbed_rows[-1],
        "diagnosed_plhiv": 1000000.0,
        "alive_on_art": 500000.0,
        "virally_suppressed": 499000.0,
        "estimated_plhiv": 2000000.0,
    }

    perturbed = build_blocked_time_dataset(perturbed_rows, [2022])

    assert baseline.train_state_rows == perturbed.train_state_rows
    assert baseline.train_transition_rows == perturbed.train_transition_rows
    assert baseline.metric_scales == perturbed.metric_scales
    assert max(baseline.metric_scales.values()) < 1000.0


def test_backhalf_state_space_splits_vl_testing_and_reengagement() -> None:
    rows = [
        {
            "quarter": "2020-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 60.0,
            "tested_for_viral_load": 40.0,
            "virally_suppressed": 25.0,
            "estimated_plhiv": 160.0,
            "new_diagnosed_cases_period": 10.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
        {
            "quarter": "2021-Q1",
            "diagnosed_plhiv": 120.0,
            "alive_on_art": 90.0,
            "tested_for_viral_load": 70.0,
            "virally_suppressed": 50.0,
            "estimated_plhiv": 170.0,
            "new_diagnosed_cases_period": 12.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 130.0,
            "alive_on_art": 95.0,
            "tested_for_viral_load": 75.0,
            "virally_suppressed": 60.0,
            "estimated_plhiv": 175.0,
            "new_diagnosed_cases_period": 13.0,
            "metric_provenance": {},
            "row_provenance_tier": "exact_observed",
        },
    ]

    dataset = build_blocked_time_dataset(rows, [2022])
    state = dataset.train_state_rows[-1]["state_values"]

    assert tuple(STATE_NAMES) == ("U", "D", "A", "T", "V", "L", "R")
    assert state["A"] + state["T"] + state["V"] + state["R"] == 90.0
    assert state["T"] == 20.0
    assert state["V"] == 50.0
    assert "A_to_T" in dataset.train_transition_rows[-1]["hazards"]
    assert "T_to_V" in dataset.train_transition_rows[-1]["hazards"]
    assert "L_to_R" in dataset.train_transition_rows[-1]["hazards"]


def test_phase2_structural_loader_reads_frozen_artifacts() -> None:
    payload = load_phase2_structural_inputs(default_epigraph_root(), _phase2_run_id_or_skip())
    assert payload.quarter_axis
    assert payload.block_axis
    assert payload.national_quarter_tensor.shape[1] == len(payload.quarter_axis)


def test_phase2_structural_loader_resolves_windows_artifact_paths(tmp_path: Path) -> None:
    epigraph_root = tmp_path / "epigraph"
    phase2_dir = epigraph_root / "artifacts" / "runs" / "win-source" / "phase2"
    phase15_dir = epigraph_root / "artifacts" / "runs" / "win-source" / "phase15"
    ensure_dir(phase2_dir)
    ensure_dir(phase15_dir)

    np.savez(
        phase15_dir / "phase15_v2_national_state_tensor.npz",
        values=np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32),
    )
    np.savez(
        phase2_dir / "phase2_national_hidden_mode_scores.npz",
        values=np.asarray([[[0.1], [0.2], [0.3]]], dtype=np.float32),
    )
    write_json(
        phase2_dir / "phase2_structural_payload.json",
        {
            "month_axis": ["2010-01", "2010-02", "2010-03"],
            "block_axis": ["care_access_continuity"],
            "artifact_paths": {
                "phase15_national_state_tensor": "D:\\EpiGraph_PH\\artifacts\\runs\\win-source\\phase15\\phase15_v2_national_state_tensor.npz",
                "hidden_mode_score_tensor": "D:\\EpiGraph_PH\\artifacts\\runs\\win-source\\phase2\\phase2_national_hidden_mode_scores.npz",
            },
            "hidden_mode_summary": {
                "available": True,
                "tensor_path": "D:\\EpiGraph_PH\\artifacts\\runs\\win-source\\phase2\\phase2_national_hidden_mode_scores.npz",
            },
            "direct_temporal_edge_rows": [],
            "hidden_driver_rows": [{"source": "care_access_continuity", "target": "care_access_continuity", "lag": 1, "weight": 0.2, "stability": 1.0, "support_count": 1}],
            "multiscale_support_rows": [],
        },
    )

    payload = load_phase2_structural_inputs(epigraph_root, "win-source")

    assert payload.quarter_axis == ["2010-Q1"]
    assert payload.national_quarter_tensor.shape == (1, 1, 1)
    assert payload.hidden_mode_quarter_tensor.shape == (1, 1, 1)
    assert float(payload.hidden_mode_quarter_tensor[0, 0, 0]) == pytest.approx(0.2)


def test_phase2_source_resolver_keeps_observation_source_separate(tmp_path: Path) -> None:
    epigraph_root = tmp_path / "epigraph"
    observation_dir = epigraph_root / "artifacts" / "runs" / "expanded-harp-source" / "harp_archive"
    phase2_dir = epigraph_root / "artifacts" / "runs" / "tr-v3-phase2-testing-prevention-rebuild-20260419-s01-monthly" / "phase2"
    ensure_dir(observation_dir)
    ensure_dir(phase2_dir)
    write_json(observation_dir / "historical_metric_rows.json", [])
    write_json(phase2_dir / "phase2_structural_payload.json", {"month_axis": [], "block_axis": []})

    resolved, diagnostics = resolve_phase2_structural_source_run_id(
        epigraph_root,
        observation_source_run_id="expanded-harp-source",
    )

    assert resolved == "tr-v3-phase2-testing-prevention-rebuild-20260419-s01-monthly"
    assert str(diagnostics["resolution"]).startswith("latest_matching_prefix:")


def test_phase2_robustness_resolver_prefers_broad_contract(tmp_path: Path) -> None:
    epigraph_root = tmp_path / "epigraph"
    narrow = epigraph_root / "artifacts" / "runs" / "phase2-source" / "phase2" / "determinant_robustness"
    broad = epigraph_root / "artifacts" / "runs" / "phase2-source" / "phase2" / "determinant_robustness_broad"
    ensure_dir(narrow)
    ensure_dir(broad)
    write_json(narrow / "phase2_determinant_robustness_report.json", {"contract": "narrow"})
    write_json(broad / "phase2_determinant_robustness_report.json", {"contract": "broad"})

    report, diagnostics = resolve_phase2_determinant_robustness_report(epigraph_root, "phase2-source")

    assert report == {"contract": "broad"}
    assert diagnostics["path"].endswith("determinant_robustness_broad/phase2_determinant_robustness_report.json")


def test_direct_prior_feature_builder_finds_supported_rows() -> None:
    payload = load_phase2_structural_inputs(default_epigraph_root(), _phase2_run_id_with_direct_features_or_skip())
    features = build_direct_prior_features(payload, PHASE2_TRANSITION_PRIOR_MAP)
    assert sum(len(rows) for rows in features.values()) > 0
    assert any(feature.transition == transition for transition, rows in features.items() for feature in rows)


def test_hidden_driver_feature_builder_finds_supported_rows() -> None:
    payload = load_phase2_structural_inputs(default_epigraph_root(), _phase2_run_id_with_hidden_features_or_skip())
    features = build_hidden_driver_features(payload, PHASE2_TRANSITION_PRIOR_MAP)
    assert sum(len(rows) for rows in features.values()) > 0
    assert any(feature.transition == "U_to_D" for feature in features.get("U_to_D", []))

def test_early_partial_target_rows_cover_2010_2016() -> None:
    rows = build_early_partial_target_rows(default_epigraph_root(), _latest_archive_run_id(), start_year=2010, end_year=2016)
    years = sorted({int(row['year']) for row in rows})
    assert years[0] == 2010
    assert years[-1] == 2016
    assert rows
    assert str(rows[0]["missing_data_tier"]) in set(MISSING_DATA_LADDER)


def test_partial_observation_splits_build() -> None:
    rows = build_early_partial_target_rows(default_epigraph_root(), _latest_archive_run_id(), start_year=2010, end_year=2016)
    splits = partial_observation_splits(rows, start_year=2010, end_year=2016, min_train_years=3, horizon_years=1)
    assert splits
    assert splits[0]['train_end_year'] == 2012
    assert splits[0]['holdout_year'] == 2013
