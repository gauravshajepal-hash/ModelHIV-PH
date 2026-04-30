from __future__ import annotations

from epigraph_ph.phase3 import tr_v3_probabilistic_batch as prob
from epigraph_ph.phase3 import tr_v3_probabilistic_extension_batch as batch


def test_exact_scale_search_returns_grid_aligned_metric_scales() -> None:
    calibration_points = {}
    for metric_name in prob.PRIMARY_METRICS:
        calibration_points[metric_name] = [
                {
                    "split_idx": idx // 4,
                    "quarter": f"{2018 + (idx // 4)}-Q{(idx % 4) + 1}",
                "tier": "exact_observed",
                "target": 100.0 + float(idx) * 5.0,
                "prediction": 100.0 + float(idx) * 5.0 + (20.0 if idx % 3 == 0 else 0.0),
                "residual": 20.0 if idx % 3 == 0 else 0.0,
            }
            for idx in range(12)
        ]
    scales = batch._exact_scale_search(calibration_points)
    assert set(scales) == set(prob.PRIMARY_METRICS)
    assert all(float(value) in batch.EXACT_SCALE_GRID for value in scales.values())


def test_exact_era_scale_search_returns_metric_stratifications() -> None:
    calibration_points = {}
    for metric_name in prob.PRIMARY_METRICS:
        calibration_points[metric_name] = [
            {
                "split_idx": idx // 4,
                "quarter": f"{2019 + (idx // 4)}-Q{(idx % 4) + 1}",
                "tier": "exact_observed",
                "target": 100.0 + float(idx) * 10.0,
                "prediction": 100.0 + float(idx) * 10.0 + (5.0 if idx < 8 else 25.0),
                "residual": 5.0 if idx < 8 else 25.0,
            }
            for idx in range(16)
        ]
    stratification = batch._exact_era_scale_search(calibration_points)
    assert set(stratification) == set(prob.PRIMARY_METRICS)
    for payload in stratification.values():
        assert "cut_year" in payload
        assert "scale_by_era" in payload
        assert all(float(value) in batch.EXACT_SCALE_GRID for value in dict(payload["scale_by_era"]).values())


def test_exact_asymmetric_search_returns_bias_and_scales() -> None:
    calibration_points = {}
    for metric_name in prob.PRIMARY_METRICS:
        calibration_points[metric_name] = [
            {
                "split_idx": idx // 4,
                "quarter": f"{2020 + (idx // 4)}-Q{(idx % 4) + 1}",
                "tier": "exact_observed",
                "target": 100.0 + float(idx) * 10.0,
                "prediction": 90.0 + float(idx) * 10.0 + (10.0 if idx % 3 == 0 else 0.0),
                "residual": (90.0 + float(idx) * 10.0 + (10.0 if idx % 3 == 0 else 0.0)) - (100.0 + float(idx) * 10.0),
            }
            for idx in range(16)
        ]
    asymmetric = batch._exact_asymmetric_search(calibration_points)
    assert set(asymmetric) == set(prob.PRIMARY_METRICS)
    for payload in asymmetric.values():
        assert "center_shift" in payload
        assert "lower_scale" in payload
        assert "upper_scale" in payload
        assert float(payload["lower_scale"]) in batch.ASYM_SCALE_GRID
        assert float(payload["upper_scale"]) in batch.ASYM_SCALE_GRID


def test_quarter_regime_rows_include_baseline_probabilities() -> None:
    points = []
    for quarter_idx in range(10):
        quarter = f"{2018 + quarter_idx}-Q1"
        residual = 2.0 if quarter_idx < 8 else 25.0
        for metric_name in prob.PRIMARY_METRICS:
            points.append(
                {
                    "quarter": quarter,
                    "metric_name": metric_name,
                    "residual": residual,
                    "target": 100.0,
                }
            )
    rows = batch._quarter_regime_rows(points)
    assert rows
    assert all("candidate_probability" in row for row in rows)
    assert all("prevalence_probability" in row for row in rows)
    assert all("persistence_probability" in row for row in rows)
    assert all(0.0 <= float(row["candidate_probability"]) <= 1.0 for row in rows)


def test_markdown_report_mentions_requested_experiments() -> None:
    payload = {
        "generated_at": "2026-04-14T00:00:00+00:00",
        "archive_run_id": "archive-sample",
        "baseline_exact": {"primary_normalized_wis": 1.0},
        "baseline_dense": {"primary_normalized_wis": 0.8},
        "exact_uq01x": {
            "primary_normalized_wis": 0.9,
            "coverage_gap": 0.1,
            "lockbox": {"primary_normalized_wis": 0.7},
            "scale_by_metric": {"diagnosed_plhiv": 1.5},
        },
        "exact_uq01y": {
            "primary_normalized_wis": 0.88,
            "coverage_gap": 0.09,
            "lockbox": {"primary_normalized_wis": 0.68},
            "stratification_by_metric": {"diagnosed_plhiv": {"cut_year": 2024, "scale_by_era": {"early": 2.0, "late": 1.5}}},
        },
        "exact_uq01z": {
            "primary_normalized_wis": 0.87,
            "coverage_gap": 0.08,
            "lockbox": {"primary_normalized_wis": 0.66},
            "asymmetric_by_metric": {"diagnosed_plhiv": {"center_shift": 25.0, "lower_scale": 1.0, "upper_scale": 1.5}},
        },
        "dense_uq03": {
            "rolling": {"primary_normalized_wis": 0.75, "coverage_gap": 0.05},
            "lockbox": {"primary_normalized_wis": 0.7},
            "risk": {"candidate_brier": 0.2, "prevalence_brier": 0.21, "persistence_brier": 0.22},
        },
        "dense_uq04": {
            "rolling": {"primary_normalized_wis": 0.74, "coverage_gap": 0.04},
            "lockbox": {"primary_normalized_wis": 0.69},
            "risk": {"candidate_brier": 0.19, "prevalence_brier": 0.21, "persistence_brier": 0.22},
        },
        "dense_uq05": {"rolling": {"primary_normalized_wis": 0.73, "coverage_gap": 0.03}, "lockbox": {"primary_normalized_wis": 0.68}},
        "dense_plat01": {"rolling": {"primary_normalized_wis": 0.72, "coverage_gap": 0.02}, "lockbox": {"primary_normalized_wis": 0.67}},
        "decisions": {
            "exact_uq01x": "keep",
            "exact_uq01y": "keep",
            "exact_uq01z": "keep",
            "dense_uq03": "revert",
            "dense_uq04": "keep",
            "dense_uq05": "revert",
            "dense_plat01": "revert",
        },
    }
    markdown = batch._markdown_report(payload)
    assert "EXP-UQ-01X-exact" in markdown
    assert "EXP-UQ-01Y-exact" in markdown
    assert "EXP-UQ-01Z-exact" in markdown
    assert "EXP-UQ-03-dense" in markdown
    assert "EXP-UQ-04-dense" in markdown
    assert "EXP-UQ-05" in markdown
    assert "EXP-PLAT-01" in markdown
