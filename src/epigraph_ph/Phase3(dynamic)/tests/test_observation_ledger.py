from __future__ import annotations

import json

from phase3_dynamic.data import build_observation_rows
from phase3_dynamic.observation_ledger import build_observation_role_ledger


def _write_archive(root, run_id: str, rows: list[dict[str, object]]) -> None:
    archive_dir = root / "artifacts" / "runs" / run_id / "harp_archive"
    archive_dir.mkdir(parents=True)
    (archive_dir / "historical_metric_rows.json").write_text(json.dumps(rows), encoding="utf-8")


def test_observation_role_ledger_classifies_roles_and_support(tmp_path) -> None:
    project_root = tmp_path / "EpiGraph_PH"
    baseline_rows = [
        {
            "source_id": "baseline-diagnosed",
            "region": "national",
            "metric_name": "diagnosed_plhiv",
            "time": "2020-03-01",
            "value": 100.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "official_local_corpus",
            "measurement_class": "program_observed",
            "source_tier": "official",
            "extraction_method": "structured",
        },
        {
            "source_id": "baseline-art",
            "region": "national",
            "metric_name": "alive_on_art",
            "time": "2020-03-01",
            "value": 80.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "official_local_corpus",
            "measurement_class": "program_observed",
            "source_tier": "official",
            "extraction_method": "structured",
        },
        {
            "source_id": "baseline-diagnosed-q4",
            "region": "national",
            "metric_name": "diagnosed_plhiv",
            "time": "2020-12-01",
            "value": 120.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "official_local_corpus",
            "measurement_class": "program_observed",
            "source_tier": "official",
            "extraction_method": "structured",
        },
        {
            "source_id": "baseline-art-q4",
            "region": "national",
            "metric_name": "alive_on_art",
            "time": "2020-12-01",
            "value": 90.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "official_local_corpus",
            "measurement_class": "program_observed",
            "source_tier": "official",
            "extraction_method": "structured",
        },
    ]
    source_rows = baseline_rows + [
        {
            "source_id": "estimated",
            "region": "national",
            "metric_name": "estimated_plhiv",
            "time": "2020-03-01",
            "value": 140.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "model_estimate",
            "measurement_class": "model_estimate",
            "source_tier": "external_reference",
            "extraction_method": "structured",
        },
        {
            "source_id": "incidence",
            "region": "national",
            "metric_name": "annual_new_infections",
            "time": "2020-12-01",
            "value": 22.0,
            "unit": "people",
            "series_kind": "annual_series",
            "source_quality_tier": "reference_only_external_series",
            "measurement_class": "model_estimate",
            "source_tier": "external_reference",
            "source_organization": "UNAIDS",
            "extraction_method": "structured",
        },
        {
            "source_id": "quarantined",
            "region": "national",
            "metric_name": "",
            "time": "2020-03-01",
            "value": 5.0,
            "unit": "people",
            "series_kind": "quarterly_snapshot",
            "source_quality_tier": "official_local_corpus",
            "measurement_class": "program_observed",
            "source_tier": "official",
            "extraction_method": "structured",
        },
    ]
    _write_archive(project_root, "baseline-run", baseline_rows)
    _write_archive(project_root, "source-run", source_rows)

    ledger = build_observation_role_ledger(
        project_root,
        source_run_id="source-run",
        baseline_source_run_id="baseline-run",
    )
    rows = build_observation_rows(
        project_root,
        source_run_id="source-run",
        baseline_source_run_id="baseline-run",
    )

    assert ledger["summary"]["observation_role_counts"]["direct_target"] == 4
    assert ledger["summary"]["observation_role_counts"]["auxiliary_likelihood"] == 1
    assert ledger["summary"]["observation_role_counts"]["validation_only"] == 1
    assert ledger["summary"]["observation_role_counts"]["quarantined"] == 1
    assert ledger["summary"]["training_use_counts"]["primary_training_and_scoring_target"] == 4
    assert ledger["summary"]["training_use_counts"]["held_out_validation_or_diagnostic_only"] == 1

    assert len(rows) == 2
    row_q1 = rows[0]
    row_q4 = rows[1]
    assert row_q1["metric_provenance"]["diagnosed_plhiv"]["support_partition"] == "common_support"
    assert row_q1["metric_provenance"]["estimated_plhiv"]["observation_role"] == "auxiliary_likelihood"
    assert row_q1["metric_provenance"]["estimated_plhiv"]["support_partition"] == "expanded_support"
    assert row_q4["annual_new_infections"] is None
    assert row_q4["metric_provenance"]["annual_new_infections"] is None

    diagnostic_rows = build_observation_rows(
        project_root,
        source_run_id="source-run",
        baseline_source_run_id="baseline-run",
        include_validation_only=True,
    )
    assert diagnostic_rows[1]["metric_provenance"]["annual_new_infections"]["observation_role"] == "validation_only"
