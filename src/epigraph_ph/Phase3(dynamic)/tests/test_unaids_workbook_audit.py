from pathlib import Path
import json
import hashlib

import pytest

from phase3_dynamic.unaids_workbook_audit import audit_workbook, parse_display, METRICS


@pytest.mark.parametrize("raw,numeric,status", [("31 000",31000,"display_value"), ("4.5 m",4500000,"display_value"),
                                               ("<200",None,"left_censored"), ("...",None,"missing"),
                                               ("unresolved",None,"unparsed")])
def test_no_precision_or_point_estimate_invented(raw, numeric, status):
    actual = parse_display(raw)
    assert actual["numeric"] == numeric
    assert actual["status"] == status


def test_vintage_comes_from_contents_not_filename_and_duplicate_sheet_not_double_counted(tmp_path):
    import openpyxl

    path = tmp_path / "2026-latest-estimates.xlsx"
    w = openpyxl.Workbook()
    s = w.active
    s.title = "HIV2025Estimates_ByYear"
    for r in [["HIV estimates"], ["1990-2024"], ["Source: UNAIDS 2025 estimates"], [],
              [None,None,None,next(iter(METRICS)),None,None], [None,None,None,"Estimate","Low","High"],
              [2024,"PHL","Philippines",2300,1500,3200]]:
        s.append(r)
    w.copy_worksheet(s).title = "HIV2025Estimates_ByArea"
    w.save(path)
    w.close()
    result = audit_workbook(path, requested_release=2026, source_url="https://example.org/2026")
    assert result["status"] == "requested_vintage_not_present"
    assert result["actual_releases"] == [2025]
    assert result["max_philippines_year"] == 2024
    assert len(result["core_annual_rows"]) == 1
    assert result["core_annual_rows"][0]["observation_role"] == "validation_only"
    assert not result["is_official_model_forecast_benchmark"]
    assert not result["is_new_external_validation"]


def test_downloaded_workbook_audit_source_code_and_role_lock():
    root = Path(__file__).resolve().parents[4]
    report = json.loads((root / "docs/unaids_workbook_vintage_audit_20260911.json").read_text())
    source = root / "src/epigraph_ph/Phase3(dynamic)/src/phase3_dynamic/unaids_workbook_audit.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == report["audit_code_sha256"]
    assert report["actual_releases"] == [2025]
    assert report["max_philippines_year"] == 2024
    assert len(report["core_annual_rows"]) == 105
    for row in report["core_annual_rows"]:
        assert row["observation_role"] == "validation_only"
        assert row["allowed_use"] == ["estimate_agreement_diagnostic"]
        assert not row["is_forecast"]
        data = {k: v for k, v in row.items() if k != "row_hash"}
        assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == row["row_hash"]
