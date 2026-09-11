from pathlib import Path
import hashlib
import json

import pytest

from phase3_dynamic.r97_hasp_report_vintages import (
    agreed_rows, asof_vintage, choose_at_origin, layout_consensus, parse_rows, predict_period, report_period,
)
from phase3_dynamic.r96_monthly_diagnosis_state import FAMILIES


def test_zero_placeholders_after_report_end_are_not_observations():
    text = "2025 | 1810 1690 1592 1610 1630 1739 0 0 0 0 0 0 1679"
    result = parse_rows(text, "2025-06")
    assert result[2025]["counts"] == [1810, 1690, 1592, 1610, 1630, 1739]
    assert result[2025]["printed_average"] == 1679
    assert not parse_rows(text.replace("0 0 0 0 0 0", "10 0 0 0 0 0"), "2025-06")


def test_legacy_table_without_avg_is_supported_but_ocr_disagreement_is_blocked():
    text = "2023 1,445 1,291 2,076 1,237 1,250 1,518 1,554 1,572 1,569 1,501 1,277 952"
    a = parse_rows(text, "2024-09")
    b = parse_rows(text.replace("952", "950"), "2024-09")
    assert a[2023]["printed_average"] is None
    assert len(a[2023]["counts"]) == 12
    assert not agreed_rows(a, b)


def test_period_normalizes_multiple_separators():
    assert report_period(Path("HASP-REPORT-2025_-Q3_signed-1.pdf")) == "2025-Q3"
    assert report_period(Path("HASP-Q3-2024-1.pdf")) == "2024-Q3"


def test_avg_header_prevents_accepting_truncated_average_row():
    text = "JAN FEB MAR AVG\n2026 10 20 30"
    assert not parse_rows(text, "2026-03")
    assert parse_rows(text + " 20", "2026-03")[2026]["counts"] == [10, 20, 30]


def test_cross_layout_requires_unique_cross_segmentation_agreement():
    a = parse_rows("2026 10 20 30 20", "2026-03")
    b = parse_rows("2026 11 19 30 20", "2026-03")
    assert layout_consensus({3: [{}, a], 6: [a]}) == a
    assert not layout_consensus({3: [a], 6: [b]})
    assert not layout_consensus({3: [a, b], 6: [a, b]})


def _catalog():
    return [{"period": "2025-Q3", "status": "accepted", "end_month": "2025-09",
             "mirror_posted_at": "2025-12-01T09:59:53Z", "values": list(range(100, 133)), "source_sha256": "old"},
            {"period": "2025-Q4", "status": "accepted", "end_month": "2025-12",
             "mirror_posted_at": "2026-02-26T12:00:00Z", "values": list(range(100, 136)), "source_sha256": "new"}]


def test_publication_time_not_event_end_controls_availability():
    catalog = _catalog()
    assert asof_vintage(catalog, "2026-01-01")["period"] == "2025-Q3"
    assert asof_vintage(catalog, "2026-02-26T08:00:00Z")["period"] == "2025-Q3"
    assert asof_vintage(catalog, "2026-02-26T12:00:00Z")["period"] == "2025-Q4"
    with pytest.raises(ValueError, match="eligible"):
        asof_vintage(catalog, "2025-11-01")


def test_unobserved_gap_is_propagated_and_no_newer_vintage_backfills_training():
    catalog = _catalog()
    v = asof_vintage(catalog, "2026-01-01")
    prediction = predict_period(v, "last_month", "2026-01")
    assert prediction["unobserved_gap_months"] == 3
    assert prediction["prediction"] == 3 * 132
    catalog[1]["values"] = [100000] * 36
    assert predict_period(asof_vintage(catalog, "2026-01-01"), "last_month", "2026-01") == prediction
    with pytest.raises(ValueError, match="overlaps"):
        predict_period(v, "last_month", "2025-09")


def test_selection_cannot_see_unpublished_target_errors():
    rows = [{"family": family, "absolute_error": i, "target_posted_at": "2026-08-26T12:00:00Z"}
            for i, family in enumerate(reversed(FAMILIES))]
    assert choose_at_origin(rows, "2026-04-01") == "quarter_carry"
    assert choose_at_origin(rows, "2026-08-26T08:00:00Z") == "quarter_carry"
    assert choose_at_origin(rows, "2026-08-26T13:00:00Z") == "local_level_drift"
    with pytest.raises(ValueError, match="every family"):
        choose_at_origin(rows[:-1], "2026-09-01")


def test_calendar_holes_are_not_silently_reinterpreted_as_contiguous_history():
    vintage = _catalog()[0]
    vintage["values"] = vintage["values"][:-1]
    with pytest.raises(ValueError, match="calendar"):
        predict_period(vintage, "quarter_carry", "2026-01")


def test_registry_never_promotes_unscored_future_forecasts():
    from phase3_dynamic.r53_publication_claim_registry import _report_vintage_claim

    assert _report_vintage_claim({}, Path("missing"))["claim_status"] == "blocked"
    row = _report_vintage_claim({"gate": {"status": "diagnostic_and_prospective_lock_only"},
                                "prospective_Q4": {"status": "prospective_forecasts_frozen_unscored"}}, Path("missing"))
    assert row["claim_status"] == "diagnostic_only"


def test_frozen_archive_row_hashes_and_report_consistency():
    root = Path(__file__).resolve().parents[4]
    report = json.loads((root / "docs/phase3_r97_results_20260911.json").read_text())
    ledger = json.loads((root / "docs/phase3_r97_observation_ledger_20260911.json").read_text())
    assert len(ledger) == report["monthly_vintage_rows"]
    catalog = {r["period"]: r for r in report["catalog"]}
    for row in ledger:
        expected_hash = row["row_hash"]
        unhashed = {k: v for k, v in row.items() if k != "row_hash"}
        assert hashlib.sha256(json.dumps(unhashed, sort_keys=True).encode()).hexdigest() == expected_hash
        source = catalog[row["report_period"]]
        assert source["status"] == "accepted"
        index = (int(row["month"][:4]) - 2023) * 12 + int(row["month"][5:]) - 1
        assert row["value"] == source["values"][index]
        assert row["month"] <= source["end_month"]
    lock = report["prospective_Q4"]
    assert lock["generated_at"][:10] < lock["target_start"]
    assert lock["observed_outcome"] is None
    assert len([r for r in lock["monthly_candidates"] if r["selected"]]) == 1
    assert report["gate"]["champion"] is None


def test_frozen_replay_uses_the_correct_vintage_and_is_numerically_reproducible():
    root = Path(__file__).resolve().parents[4]
    report = json.loads((root / "docs/phase3_r97_results_20260911.json").read_text())
    for row in report["availability_replays"]:
        vintage = asof_vintage(report["catalog"], row["issued_at"])
        assert vintage["period"] == row["training_report_period"]
        result = predict_period(vintage, row["family"], row["issued_at"][:7])
        assert result["prediction"] == pytest.approx(row["prediction"], rel=1e-6)
