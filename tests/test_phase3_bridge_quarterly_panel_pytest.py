from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from epigraph_ph.phase3.bridge_quarterly_panel import build_bridge_quarterly_panel_rows
from epigraph_ph.runtime import write_json


def _write_archive(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    archive_dir = tmp_path / "artifacts" / "runs" / "demo" / "harp_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    write_json(archive_dir / "historical_metric_rows_harp_only.json", rows)
    return tmp_path


def test_bridge_quarterly_panel_derives_diagnosed_from_cumulative_cases_minus_deaths() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"bridge_panel_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-03", "value": 58746.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "march_2022"},
            {"region": "National", "metric_name": "diagnosed_cases_cumulative", "time": "2022-03", "value": 99200.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "march_2022"},
            {"region": "National", "metric_name": "deaths_reported_cumulative", "time": "2022-03", "value": 5400.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "march_2022"},
            {"region": "National", "metric_name": "new_diagnosed_cases_period", "time": "2022-01", "period_start": "2022-01", "period_end": "2022-01", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 875.0, "source_quality_tier": "official_doh_archive", "source_id": "jan_2022"},
            {"region": "National", "metric_name": "new_diagnosed_cases_period", "time": "2022-02", "period_start": "2022-02", "period_end": "2022-02", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 1054.0, "source_quality_tier": "official_doh_archive", "source_id": "feb_2022"},
            {"region": "National", "metric_name": "new_diagnosed_cases_period", "time": "2022-03", "period_start": "2022-03", "period_end": "2022-03", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 1539.0, "source_quality_tier": "official_doh_archive", "source_id": "mar_2022"},
            {"region": "National", "metric_name": "deaths_reported_period", "time": "2022-01", "period_start": "2022-01", "period_end": "2022-01", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 33.0, "source_quality_tier": "official_doh_archive", "source_id": "jan_2022"},
            {"region": "National", "metric_name": "deaths_reported_period", "time": "2022-02", "period_start": "2022-02", "period_end": "2022-02", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 45.0, "source_quality_tier": "official_doh_archive", "source_id": "feb_2022"},
            {"region": "National", "metric_name": "deaths_reported_period", "time": "2022-03", "period_start": "2022-03", "period_end": "2022-03", "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "value": 66.0, "source_quality_tier": "official_doh_archive", "source_id": "mar_2022"},
        ],
    )

    payload = build_bridge_quarterly_panel_rows("demo", repo=repo)
    row = next(item for item in payload["rows"] if item["quarter"] == "2022-Q1")

    assert row["diagnosed_plhiv"] == 93800.0
    assert row["diagnosed_plhiv_source_kind"] == "bridge_cumulative_minus_deaths"
    assert row["alive_on_art"] == 58746.0
    assert row["alive_on_art_source_kind"] == "monthly_bridge"
    assert row["new_diagnosed_cases_period"] == 3468.0
    assert row["deaths_reported_period"] == 144.0
    assert row["stock_anchor_complete"] is True
    assert "2022-Q1" in payload["summary"]["bridge_added_complete_stock_quarters"]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_bridge_quarterly_panel_prefers_exact_snapshot_when_available() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"bridge_panel_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2023-03", "value": 110000.0, "source_quality_tier": "official", "evidence_confidence": 0.95, "source_id": "q1_exact"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2023-03", "value": 67194.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_bridge"},
            {"region": "National", "metric_name": "diagnosed_cases_cumulative", "time": "2023-03", "value": 114008.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_bridge"},
            {"region": "National", "metric_name": "deaths_reported_cumulative", "time": "2023-03", "value": 6200.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_bridge"},
        ],
    )

    payload = build_bridge_quarterly_panel_rows("demo", repo=repo)
    row = next(item for item in payload["rows"] if item["quarter"] == "2023-Q1")

    assert row["diagnosed_plhiv"] == 110000.0
    assert row["diagnosed_plhiv_source_kind"] == "exact_snapshot"
    assert row["alive_on_art"] == 67194.0
    assert row["alive_on_art_source_kind"] == "monthly_bridge"
    assert row["anchor_class"] == "mixed"
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_bridge_quarterly_panel_uses_direct_monthly_diagnosed_stock_before_derived_bridge() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"bridge_panel_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2022-03", "value": 94123.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.95, "source_id": "mar_diag"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-03", "value": 58746.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_art"},
            {"region": "National", "metric_name": "diagnosed_cases_cumulative", "time": "2022-03", "value": 99200.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_cum"},
            {"region": "National", "metric_name": "deaths_reported_cumulative", "time": "2022-03", "value": 5400.0, "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.92, "source_id": "mar_deaths"},
        ],
    )

    payload = build_bridge_quarterly_panel_rows("demo", repo=repo)
    row = next(item for item in payload["rows"] if item["quarter"] == "2022-Q1")

    assert row["diagnosed_plhiv"] == 94123.0
    assert row["diagnosed_plhiv_source_kind"] == "monthly_bridge"
    assert row["alive_on_art"] == 58746.0
    assert row["alive_on_art_source_kind"] == "monthly_bridge"
    assert "2022-Q1" in payload["summary"]["diagnosed_direct_bridge_quarters"]
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_bridge_quarterly_panel_uses_latest_month_in_quarter_when_quarter_end_month_missing() -> None:
    tmp_path = Path("D:/EpiGraph_PH/artifacts/test_tmp") / f"bridge_panel_{uuid.uuid4().hex}"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    repo = _write_archive(
        tmp_path,
        [
            {"region": "National", "metric_name": "diagnosed_plhiv", "time": "2022-05", "value": 96152.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.94, "source_id": "may_diag"},
            {"region": "National", "metric_name": "alive_on_art", "time": "2022-05", "value": 61234.0, "series_kind": "monthly_snapshot", "temporal_precision": "monthly_snapshot", "source_quality_tier": "official_doh_archive", "evidence_confidence": 0.94, "source_id": "may_art"},
        ],
    )

    payload = build_bridge_quarterly_panel_rows("demo", repo=repo)
    row = next(item for item in payload["rows"] if item["quarter"] == "2022-Q2")

    assert row["diagnosed_plhiv"] == 96152.0
    assert row["diagnosed_plhiv_source_kind"] == "monthly_bridge_latest_in_quarter"
    assert row["alive_on_art"] == 61234.0
    assert row["alive_on_art_source_kind"] == "monthly_bridge_latest_in_quarter"
    assert row["stock_anchor_complete"] is True
    shutil.rmtree(tmp_path, ignore_errors=True)
