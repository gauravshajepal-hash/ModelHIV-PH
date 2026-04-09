from __future__ import annotations

import json
from pathlib import Path

from epigraph_ph.validate.extraction_quality_audit import (
    _candidate_key,
    _candidate_sanity_audit,
    _counter_diff,
    _harp_source_adjudication_rows,
    _harp_discrepancy_rows,
    _harp_invariant_violations,
    _harp_time_mismatch_rows,
    build_extraction_quality_audit,
)


def test_counter_diff_detects_missing_and_extra_rows() -> None:
    expected = [
        {
            "source_id": "a",
            "canonical_name": "poverty_rate",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_x",
            "value": 18.1,
        }
    ]
    emitted = [
        {
            "source_id": "a",
            "canonical_name": "poverty_rate",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_x",
            "value": 18.1,
        },
        {
            "source_id": "a",
            "canonical_name": "education",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_x",
            "value": 81.2,
        },
    ]
    diff = _counter_diff(expected, emitted)
    assert diff["missing_count"] == 0
    assert diff["extra_count"] == 1
    assert diff["extra_examples"][0]["canonical_name"] == "education"


def test_harp_helpers_flag_mismatch_time_and_invariants() -> None:
    seed = {2024: {"alive_on_art": 90568.0}}
    observed = {2024: {"alive_on_art": 90854.0}}
    discrepancies = _harp_discrepancy_rows(seed=seed, observed=observed, label="panel")
    assert discrepancies == [
        {
            "label": "panel",
            "year": 2024,
            "metric_name": "alive_on_art",
            "expected_value": 90568.0,
            "observed_value": 90854.0,
            "difference": 286.0,
        }
    ]

    time_mismatches = _harp_time_mismatch_rows(
        [
            {"year": 2024, "time": "2025-01"},
            {"year": 2025, "time": "bad-time"},
        ]
    )
    assert len(time_mismatches) == 2
    assert time_mismatches[0]["reason"] == "year_prefix_mismatch"
    assert time_mismatches[1]["reason"] == "invalid_time"

    violations = _harp_invariant_violations(
        [
            {
                "year": 2024,
                "estimated_plhiv": 200.0,
                "diagnosed_plhiv": 210.0,
                "alive_on_art": 220.0,
                "tested_for_viral_load": 50.0,
                "virally_suppressed": 60.0,
            }
        ]
    )
    rules = {row["rule"] for row in violations}
    assert rules == {"diagnosed_le_estimated", "art_le_diagnosed", "suppressed_le_tested"}


def test_candidate_sanity_audit_flags_duplicate_conflicts_and_bounds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    extracted_dir = run_dir / "phase0" / "extracted"
    extracted_dir.mkdir(parents=True)
    rows = [
        {
            "candidate_id": "c1",
            "source_id": "source-a",
            "canonical_name": "poverty_rate",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_world_bank_api",
            "measurement_type": "rate",
            "unit": "percent",
            "value": 18.1,
        },
        {
            "candidate_id": "c2",
            "source_id": "source-a",
            "canonical_name": "poverty_rate",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_world_bank_api",
            "measurement_type": "rate",
            "unit": "percent",
            "value": 19.1,
        },
        {
            "candidate_id": "c3",
            "source_id": "source-b",
            "canonical_name": "philhealth_coverage",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "2022",
            "extraction_method": "structured_philhealth_open_portal_json",
            "measurement_type": "rate",
            "unit": "percent",
            "value": 140.0,
        },
        {
            "candidate_id": "c4",
            "source_id": "source-c",
            "canonical_name": "alive_on_art",
            "geo": "Philippines",
            "region": "national",
            "province": "",
            "time": "bad",
            "extraction_method": "manual",
            "measurement_type": "count",
            "unit": "people",
            "value": -1.0,
        },
    ]
    (extracted_dir / "canonical_parameter_candidates.json").write_text(json.dumps(rows), encoding="utf-8")
    audit = _candidate_sanity_audit(run_dir=run_dir)
    assert audit["conflicting_duplicate_key_count"] == 1
    assert audit["bounded_percent_outlier_count"] == 1
    assert audit["negative_count_row_count"] == 1
    assert audit["time_format_issue_count"] == 1
    assert not audit["passed"]


def test_harp_source_adjudication_rows_classify_quarterly_model_and_override() -> None:
    metric_rows = [
        {
            "year": 2024,
            "time": "2024-01",
            "metric_name": "estimated_plhiv",
            "value": 215400.0,
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "source_label": "Official",
            "source_url": "user_attached_doh_slide_2026_03_31",
            "measurement_class": "model_estimate",
            "series_kind": "annual_snapshot",
            "source_quality_tier": "official_user_provided_slide",
            "evidence_confidence": 1.0,
        },
        {
            "year": 2024,
            "time": "2024-03",
            "metric_name": "estimated_plhiv",
            "value": 210000.0,
            "source_id": "q1_report",
            "source_label": "Q1",
            "measurement_class": "model_estimate",
            "series_kind": "quarterly_snapshot",
            "temporal_precision": "quarterly_snapshot",
            "source_quality_tier": "official_doh_archive",
            "evidence_confidence": 0.92,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 216900.0,
            "source_id": "core_team_2025",
            "source_label": "Core team",
            "measurement_class": "model_estimate",
            "series_kind": "annual_snapshot",
            "evidence_confidence": 0.99,
        },
        {
            "year": 2024,
            "time": "2024-01",
            "metric_name": "alive_on_art",
            "value": 90568.0,
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "source_label": "Official",
            "source_url": "user_attached_doh_slide_2026_03_31",
            "measurement_class": "program_observed_harp",
            "series_kind": "annual_snapshot",
            "source_quality_tier": "official_user_provided_slide",
            "evidence_confidence": 1.0,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "alive_on_art",
            "value": 90854.0,
            "source_id": "core_team_2025",
            "source_label": "Core team",
            "measurement_class": "program_observed_harp",
            "series_kind": "annual_snapshot",
            "evidence_confidence": 0.99,
        },
    ]

    rows = _harp_source_adjudication_rows(metric_rows)
    categories = {
        (row["metric_name"], row["alternative_source_id"]): row["adjudication_category"]
        for row in rows
    }

    assert categories[("estimated_plhiv", "q1_report")] == "quarterly_snapshot"
    assert categories[("estimated_plhiv", "core_team_2025")] == "model_update"
    assert categories[("alive_on_art", "core_team_2025")] == "official_override"


def test_build_extraction_quality_audit_writes_adjudication_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis_dir = run_dir / "analysis"
    (run_dir / "phase0" / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "phase0" / "raw" / "source_manifest.json").write_text("[]", encoding="utf-8")
    (run_dir / "phase0" / "extracted").mkdir(parents=True, exist_ok=True)
    (run_dir / "phase0" / "extracted" / "canonical_parameter_candidates.json").write_text("[]", encoding="utf-8")
    (run_dir / "phase0" / "extracted" / "philhealth_portal_candidate_rows.json").write_text("[]", encoding="utf-8")
    (run_dir / "phase0" / "extracted" / "philhealth_portal_metric_summary.json").write_text("{}", encoding="utf-8")
    (run_dir / "harp_archive").mkdir(parents=True, exist_ok=True)
    (run_dir / "harp_archive" / "historical_harp_panel.json").write_text(json.dumps({"rows": []}), encoding="utf-8")
    (run_dir / "harp_archive" / "harp_program_points.json").write_text(json.dumps({"points": []}), encoding="utf-8")
    metric_rows = [
        {
            "year": 2024,
            "time": "2024-01",
            "metric_name": "estimated_plhiv",
            "value": 215400.0,
            "source_id": "doh_official_cascade_ground_truth_2018_2025",
            "source_label": "Official",
            "source_url": "user_attached_doh_slide_2026_03_31",
            "measurement_class": "model_estimate",
            "series_kind": "annual_snapshot",
            "source_quality_tier": "official_user_provided_slide",
            "evidence_confidence": 1.0,
        },
        {
            "year": 2024,
            "time": "2024-12",
            "metric_name": "estimated_plhiv",
            "value": 216900.0,
            "source_id": "core_team_2025",
            "source_label": "Core team",
            "measurement_class": "model_estimate",
            "series_kind": "annual_snapshot",
            "evidence_confidence": 0.99,
        },
    ]
    (run_dir / "harp_archive" / "historical_metric_rows.json").write_text(json.dumps(metric_rows), encoding="utf-8")

    audit = build_extraction_quality_audit(run_dir=run_dir, plugin_id="hiv")

    paths = audit["harp_quality"]["source_adjudication_table_paths"]
    assert Path(paths["json"]).exists()
    assert Path(paths["csv"]).exists()
    adjudication_payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    assert adjudication_payload["rows"][0]["adjudication_category"] == "model_update"
