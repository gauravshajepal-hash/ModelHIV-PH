from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from epigraph_ph.phase1.latent_observability import build_latent_observability_audit
from epigraph_ph.phase15.latent_measurements import build_archive_derived_indicator_rows
from epigraph_ph.phase15.national_factor_model import build_national_measurement_spec
from epigraph_ph.phase15.province_factor_graph import build_province_factor_graph_scaffold


def test_archive_derived_indicator_rows_promote_care_and_suppression_blocks(tmp_path: Path) -> None:
    archive_dir = tmp_path / "harp_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "harp_program_points.json").write_text(
        json.dumps(
            {
                "points": [
                    {
                        "effective_month": "2024-12",
                        "temporal_precision": "quarterly_snapshot",
                        "diagnosed": 135026.0,
                        "on_art": 90854.0,
                        "viral_load_tested": 41860.0,
                        "suppressed": 36723.0,
                        "source_url": "local_archive",
                        "label": "2024 October - December",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    rows = build_archive_derived_indicator_rows(run_dir=tmp_path, plugin_id="hiv")
    audit = build_latent_observability_audit(normalized_rows=rows, parameter_catalog=[], plugin_id="hiv")
    measurement_spec = build_national_measurement_spec(
        normalized_rows=rows,
        observability_audit=audit,
        month_axis=["2024-12"],
        plugin_id="hiv",
    )

    canonical_names = {row["canonical_name"] for row in rows}
    retained_blocks = {row["block_id"] for row in measurement_spec["retained_blocks"]}

    assert {"art_uptake_rate", "retention_adherence", "suppression_outcomes", "viral_suppression_rate"} <= canonical_names
    assert {"care_access_continuity", "suppression_capacity"} <= retained_blocks
    assert all(row["measurement_role"] == "direct_indicator" for row in rows)


def test_province_factor_graph_scaffold_shrinks_to_local_support() -> None:
    axis_catalogs = {
        "province": ["Cebu", "Bohol"],
        "month": ["2025-01", "2025-02"],
        "canonical_name": ["testing_rate", "prevention_coverage"],
    }
    standardized_tensor = np.asarray(
        [
            [[0.2, 1.4], [0.2, 1.2]],
            [[0.2, 0.1], [0.2, 0.0]],
        ],
        dtype=np.float32,
    )
    national_scaffold = {
        "loadings": {
            "rows": [
                {
                    "block_id": "testing_engagement",
                    "display_name": "Testing Engagement",
                    "canonical_name": "testing_rate",
                    "loading": 0.75,
                },
                {
                    "block_id": "testing_engagement",
                    "display_name": "Testing Engagement",
                    "canonical_name": "prevention_coverage",
                    "loading": 0.25,
                },
            ]
        },
        "states": {
            "rows": [
                {
                    "block_id": "testing_engagement",
                    "state_values": [0.1, 0.1],
                }
            ]
        },
    }
    normalized_rows = [
        {
            "canonical_name": "prevention_coverage",
            "candidate_block": "testing_engagement",
            "measurement_role": "direct_indicator",
            "province": "Cebu",
            "geo": "Cebu",
            "geo_resolution": "province",
            "region": "region_vii",
            "time": "2025",
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
        }
    ]

    payload = build_province_factor_graph_scaffold(
        standardized_tensor=standardized_tensor,
        axis_catalogs=axis_catalogs,
        normalized_rows=normalized_rows,
        national_scaffold=national_scaffold,
        region_labels=["region_vii", "region_vii"],
        plugin_id="hiv",
    )

    state_tensor = np.asarray(payload["state_tensor"], dtype=np.float32)
    deviation_rows = list(payload["loading_deviations"]["rows"])
    cebu_state = state_tensor[0, :, 0]
    bohol_state = state_tensor[1, :, 0]

    assert state_tensor.shape == (2, 2, 1)
    assert not np.allclose(cebu_state, bohol_state)
    assert not np.allclose(cebu_state, np.asarray([0.1, 0.1], dtype=np.float32))
    cebu_prevention = next(
        row
        for row in deviation_rows
        if row["province"] == "Cebu" and row["canonical_name"] == "prevention_coverage"
    )
    assert cebu_prevention["observed_month_count"] >= 1
    assert cebu_prevention["effective_loading"] > 0.0


def test_province_factor_graph_scaffold_exposes_regional_support_mass() -> None:
    axis_catalogs = {
        "province": ["Cebu", "Bohol"],
        "month": ["2025-01"],
        "canonical_name": ["economic_access_constraint"],
    }
    standardized_tensor = np.asarray(
        [
            [[0.0]],
            [[0.0]],
        ],
        dtype=np.float32,
    )
    national_scaffold = {
        "loadings": {
            "rows": [
                {
                    "block_id": "structural_barrier_pressure",
                    "display_name": "Structural Barrier Pressure",
                    "canonical_name": "economic_access_constraint",
                    "loading": 1.0,
                },
            ]
        },
        "states": {
            "rows": [
                {
                    "block_id": "structural_barrier_pressure",
                    "state_values": [0.2],
                }
            ]
        },
    }
    normalized_rows = [
        {
            "canonical_name": "economic_access_constraint",
            "candidate_block": "structural_barrier_pressure",
            "measurement_role": "direct_indicator",
            "geo": "Central Visayas",
            "geo_resolution": "region",
            "region": "region_vii",
            "time": "2025",
            "evidence_weight": 1.0,
            "quality_weight": 1.0,
            "model_numeric_value": 35.0,
        }
    ]

    payload = build_province_factor_graph_scaffold(
        standardized_tensor=standardized_tensor,
        axis_catalogs=axis_catalogs,
        normalized_rows=normalized_rows,
        national_scaffold=national_scaffold,
        region_labels=["region_vii", "region_vii"],
        plugin_id="hiv",
    )

    uncertainty_rows = list(payload["uncertainty"]["rows"])
    cebu_uncertainty = next(row for row in uncertainty_rows if row["province"] == "Cebu")

    assert cebu_uncertainty["local_support_mass_values"] == [0.0]
    assert cebu_uncertainty["regional_support_mass_values"][0] > 0.0
    assert cebu_uncertainty["regional_indicator_count_values"] == [1]
