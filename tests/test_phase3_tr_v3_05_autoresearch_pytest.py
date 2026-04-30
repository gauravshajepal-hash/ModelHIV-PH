from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import numpy as np

from epigraph_ph.phase3.tr_v3_05_autoresearch import (
    build_annual_anchor_rows,
    build_control_channels,
    build_quarterly_dataset,
    build_quarterly_observation_rows,
    quarter_gap,
    transition_rows,
)
from epigraph_ph.runtime import write_json


def _write_archive(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    archive_dir = tmp_path / "artifacts" / "runs" / "demo" / "harp_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    write_json(archive_dir / "historical_metric_rows.json", rows)
    return tmp_path


def test_build_quarterly_and_annual_rows_from_archive() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"tr_v3_05_archive_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2017-03", "value": 100.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2017-03", "value": 60.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "tested_for_viral_load", "time": "2017-03", "value": 30.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "virally_suppressed", "time": "2017-03", "value": 24.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "estimated_plhiv", "time": "2017-03", "value": 150.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2017-01", "value": 4.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2017-02", "value": 5.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2017-03", "value": 6.0, "source_quality_tier": "official"},
            {
                "region": "National",
                "metric_name": "new_diagnosed_cases_period",
                "time": "2017-06",
                "period_start": "2017-06",
                "period_end": "2017-06",
                "series_kind": "quarterly_snapshot",
                "temporal_precision": "quarterly_snapshot",
                "value": 21.0,
                "source_quality_tier": "official",
            },
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2017-06", "value": 108.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2017-06", "value": 64.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "annual_new_infections", "time": "2017-12", "value": 12.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "annual_aids_deaths", "time": "2017-12", "value": 2.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "estimated_plhiv", "time": "2017-12", "value": 150.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "art_coverage_percent", "time": "2017-12", "value": 40.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "population_total", "time": "2017-12", "value": 108119693.0, "source_quality_tier": "official"},
        ],
    )

    quarterly_rows = build_quarterly_observation_rows("demo", repo=repo)
    annual_rows = build_annual_anchor_rows("demo", repo=repo)

    assert quarterly_rows == [
        {
            "quarter": "2017-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 60.0,
            "new_diagnosed_cases_period": 15.0,
            "tested_for_viral_load": 30.0,
            "virally_suppressed": 24.0,
            "estimated_plhiv": 150.0,
        },
        {
            "quarter": "2017-Q2",
            "diagnosed_plhiv": 108.0,
            "alive_on_art": 64.0,
            "new_diagnosed_cases_period": 21.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": None,
        }
    ]
    assert {row["metric_name"] for row in annual_rows} == {"annual_new_infections", "annual_aids_deaths", "estimated_plhiv", "art_coverage_percent", "population_total"}
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_quarterly_rows_sums_single_month_period_rows_into_quarter() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"tr_v3_05_archive_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2022-12", "value": 102931.0, "source_quality_tier": "official"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-12", "value": 63221.0, "source_quality_tier": "official"},
            {
                "region": "National",
                "metric_name": "new_diagnosed_cases_period",
                "time": "2022-10",
                "period_start": "2022-10",
                "period_end": "2022-10",
                "series_kind": "monthly_snapshot",
                "temporal_precision": "monthly_snapshot",
                "value": 1383.0,
                "source_quality_tier": "official",
            },
            {
                "region": "National",
                "metric_name": "new_diagnosed_cases_period",
                "time": "2022-11",
                "period_start": "2022-11",
                "period_end": "2022-11",
                "series_kind": "monthly_snapshot",
                "temporal_precision": "monthly_snapshot",
                "value": 1517.0,
                "source_quality_tier": "official",
            },
            {
                "region": "National",
                "metric_name": "new_diagnosed_cases_period",
                "time": "2022-12",
                "period_start": "2022-12",
                "period_end": "2022-12",
                "series_kind": "monthly_snapshot",
                "temporal_precision": "monthly_snapshot",
                "value": 1378.0,
                "source_quality_tier": "official",
            },
        ],
    )

    quarterly_rows = build_quarterly_observation_rows("demo", repo=repo)

    assert quarterly_rows == [
        {
            "quarter": "2022-Q4",
            "diagnosed_plhiv": 102931.0,
            "alive_on_art": 63221.0,
            "new_diagnosed_cases_period": 4278.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": None,
        }
    ]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_quarterly_rows_prefers_quarter_end_snapshot_over_earlier_month_bridge_rows() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"tr_v3_05_archive_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2022-01", "value": 96800.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92},
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2022-03", "value": 99200.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92},
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-01", "value": 56982.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92},
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-03", "value": 58746.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2022-01", "value": 875.0, "source_quality_tier": "official_doh_archive"},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2022-02", "value": 1054.0, "source_quality_tier": "official_doh_archive"},
            {"region": "National", "metric_name": "new_diagnosed_cases_monthly", "time": "2022-03", "value": 1539.0, "source_quality_tier": "official_doh_archive"},
        ],
    )

    quarterly_rows = build_quarterly_observation_rows("demo", repo=repo)

    assert quarterly_rows == [
        {
            "quarter": "2022-Q1",
            "diagnosed_plhiv": 99200.0,
            "alive_on_art": 58746.0,
            "new_diagnosed_cases_period": 3468.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": None,
        }
    ]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_control_channels_returns_finite_scores() -> None:
    rows = [
        {"diagnosed_plhiv": 100.0, "alive_on_art": 50.0, "estimated_plhiv": 180.0, "tested_for_viral_load": 20.0, "virally_suppressed": 15.0},
        {"diagnosed_plhiv": 110.0, "alive_on_art": 60.0, "estimated_plhiv": 190.0, "tested_for_viral_load": 25.0, "virally_suppressed": 18.0},
        {"diagnosed_plhiv": 125.0, "alive_on_art": 72.0, "estimated_plhiv": 205.0, "tested_for_viral_load": 31.0, "virally_suppressed": 24.0},
        {"diagnosed_plhiv": 133.0, "alive_on_art": 79.0, "estimated_plhiv": 214.0, "tested_for_viral_load": 35.0, "virally_suppressed": 27.0},
    ]

    controls = build_control_channels(rows)

    assert set(controls) == {"A", "C", "R"}
    for values in controls.values():
        assert len(values) == len(rows)
        assert np.isfinite(np.asarray(values, dtype=np.float64)).all()


def test_build_quarterly_dataset_excludes_sparse_train_transition_gaps() -> None:
    observation_rows = [
        {
            "quarter": "2018-Q1",
            "diagnosed_plhiv": 100.0,
            "alive_on_art": 60.0,
            "new_diagnosed_cases_period": 12.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": 150.0,
        },
        {
            "quarter": "2018-Q4",
            "diagnosed_plhiv": 130.0,
            "alive_on_art": 76.0,
            "new_diagnosed_cases_period": 18.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": 180.0,
        },
        {
            "quarter": "2019-Q1",
            "diagnosed_plhiv": 136.0,
            "alive_on_art": 80.0,
            "new_diagnosed_cases_period": 7.0,
            "tested_for_viral_load": None,
            "virally_suppressed": None,
            "estimated_plhiv": 186.0,
        },
    ]

    dataset = build_quarterly_dataset(observation_rows, holdout_years=[2019])

    assert quarter_gap("2018-Q1", "2018-Q4") == 3
    assert [row["quarter"] for row in dataset.train_rows] == ["2018-Q1", "2018-Q4"]
    assert dataset.train_transition_rows == []


def test_transition_rows_attach_support_flags_from_metric_tiers() -> None:
    observation_rows = [
        {
            "quarter": "2018-Q1",
            "diagnosed_plhiv": 100.0,
            "diagnosed_plhiv_tier": "bridge_observed",
            "alive_on_art": 60.0,
            "alive_on_art_tier": "bridge_observed",
            "new_diagnosed_cases_period": 12.0,
            "new_diagnosed_cases_period_tier": "bridge_observed",
            "virally_suppressed": None,
            "virally_suppressed_tier": "rule_based_extrapolated",
            "tested_for_viral_load": None,
            "tested_for_viral_load_tier": "rule_based_extrapolated",
            "estimated_plhiv": 150.0,
            "estimated_plhiv_tier": "rule_based_extrapolated",
        },
        {
            "quarter": "2018-Q2",
            "diagnosed_plhiv": 112.0,
            "diagnosed_plhiv_tier": "bridge_observed",
            "alive_on_art": 68.0,
            "alive_on_art_tier": "bridge_observed",
            "new_diagnosed_cases_period": 14.0,
            "new_diagnosed_cases_period_tier": "bridge_observed",
            "virally_suppressed": None,
            "virally_suppressed_tier": "rule_based_extrapolated",
            "tested_for_viral_load": None,
            "tested_for_viral_load_tier": "rule_based_extrapolated",
            "estimated_plhiv": 160.0,
            "estimated_plhiv_tier": "rule_based_extrapolated",
        },
    ]

    rows = transition_rows(
        [
            {
                **state_row,
            }
            for state_row in build_quarterly_dataset(observation_rows, holdout_years=[2018]).train_state_rows
            + build_quarterly_dataset(observation_rows, holdout_years=[2018]).holdout_state_rows
        ],
        eps=float(np.finfo(np.float32).eps),
    )

    assert len(rows) == 1
    flags = dict(rows[0]["support_flags"])
    sources = dict(rows[0]["support_sources"])
    assert flags["U_to_D"] is True
    assert flags["D_to_A"] is True
    assert flags["A_to_V"] is False
    assert flags["A_to_L"] is True
    assert flags["L_to_A"] is True
    assert sources["U_to_D"] == "observed_diagnosis_flow"
