from __future__ import annotations

import zipfile
from pathlib import Path

from epigraph_ph.phase0.evidence_artifacts import build_candidate_evidence_rows
from epigraph_ph.phase0.structured_numeric_sources import (
    build_philhealth_portal_artifacts,
    _collect_local_fies_structured_rows,
    _collect_local_psa_poverty_rows,
    _collect_philhealth_portal_rows,
    _parse_philhealth_leave_benefit_tables,
    build_structured_numeric_candidates,
)
from epigraph_ph.phase1.latent_observability import build_latent_observability_audit
from epigraph_ph.phase15.national_factor_model import build_national_measurement_spec


_PHILHEALTH_LEAVE_BENEFIT_SAMPLE = """
2024
Region Regular Casual Total
Head Office 183,559,977 16,337,562 199,897,539
NCR 53,757,821 28,418,008 82,175,829
CAR 35,389,479 6,081,711 41,471,190
I 25,103,365 12,253,079 37,356,444
II 31,228,670 5,944,000 37,172,670
III 36,626,574 18,638,200 55,264,774
IV-A 50,837,067 22,812,827 73,649,894
IV-B 19,408,408 12,070,399 31,478,807
V 23,981,068 11,237,952 35,219,020
VI 23,835,135 14,583,368 38,418,503
VII 25,473,093 8,802,110 34,275,203
VIII 24,008,667 5,463,228 29,471,895
IX 20,435,451 5,689,166 26,124,617
X 15,008,542 5,605,391 20,613,933
XI 16,505,187 8,175,752 24,680,939
XII 16,228,297 7,885,414 24,113,711
BARMM 26,687,305 7,399,460 34,086,765
CARAGA 11,531,313 6,815,838 18,347,151
Total 639,605,419 204,213,465 843,818,884
2023
Region Regular Casual Total
Head Office 176,982,487 15,577,760 192,560,247
NCR 52,448,664 26,988,901 79,437,565
CAR 32,658,554 5,929,276 38,587,830
I 29,988,559 11,153,305 41,141,864
II 28,966,812 6,126,667 35,093,479
III 35,136,109 20,134,585 55,270,694
IV-A 48,468,068 22,512,960 70,981,028
IV-B 17,595,448 11,814,399 29,409,847
V 23,073,433 10,945,966 34,019,399
VI 19,917,016 12,426,651 32,343,667
VII 24,394,134 9,306,641 33,700,775
VIII 23,847,723 6,320,960 30,168,683
IX 22,106,855 5,888,962 27,995,817
X 14,287,022 6,413,386 20,700,408
XI 14,447,154 6,841,220 21,288,374
XII 14,297,456 8,050,748 22,348,204
BARMM 10,969,078 7,013,715 17,982,793
CARAGA 23,748,674 7,084,895 30,833,569
Total 613,333,246 200,530,997 813,864,243
"""

_FIES_YEARBOOK_TABLE_2_2_SAMPLE = """
TABLE 2.2 Number of Families, Average Annual Income and Expenditure by Region: 2006, 2009, 2012, and 2015
2006 17,403 173 147 26 2012 21,426 235 193 31
NCR 2,362 311 258 53 2,917 379 325 54
CAR 303 192 151 42 3 75 257 188 44
I-Ilocos Region 947 142 124 19 1,105 204 159 35
VII-Central Visayas 1,293 144 124 21 1,577 209 164 45
ARMM 534 89 75 14 5 57 130 114 16
2009 18,452 206 176 31 2015 22,730 267 215 52
NCR 2,461 356 309 47 3,019 425 349 76
CAR 322 219 174 44 402 282 209 73
I-Ilocos Region 1,005 186 152 35 1,170 238 182 56
VII-Central Visayas 1,374 184 152 32 1,672 239 193 46
ARMM 572 113 98 15 616 139 111 28
Note: Details may not add up to totals due to rounding.
Source: Philippine Statistics Authority
"""


class _MockResponse:
    def __init__(self, *, json_payload=None, text_payload: str = "", status_code: int = 200):
        self._json_payload = json_payload
        self._text_payload = text_payload
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http_{self.status_code}")

    def json(self):
        return self._json_payload

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._text_payload.splitlines():
            yield line if decode_unicode else line.encode("utf-8")

    def iter_content(self, chunk_size: int = 1024):
        payload = self._text_payload.encode("utf-8")
        for idx in range(0, len(payload), chunk_size):
            yield payload[idx : idx + chunk_size]


def _source_rows() -> dict[str, dict[str, object]]:
    return {
        "wdi-poverty": {
            "source_id": "wdi-poverty",
            "platform": "world_bank_wdi",
            "title": "WDI Poverty Headcount Ratio Philippines",
            "source_tier": "tier3_structured_repository",
            "query_geo_focus": "philippines",
            "year": 2024,
            "url": "https://api.worldbank.org/v2/country/PHL/indicator/SI.POV.NAHC?format=json",
        },
        "wdi-education": {
            "source_id": "wdi-education",
            "platform": "world_bank_wdi",
            "title": "WDI Lower Secondary Completion Rate Philippines",
            "source_tier": "tier3_structured_repository",
            "query_geo_focus": "philippines",
            "year": 2024,
            "url": "https://api.worldbank.org/v2/country/PHL/indicator/SE.SEC.CMPT.LO.ZS?format=json",
        },
        "wdi-seed": {
            "source_id": "wdi-seed",
            "platform": "world_bank_wdi",
            "title": "World Bank World Development Indicators",
            "source_tier": "tier3_structured_repository",
            "query_geo_focus": "philippines",
            "year": 2024,
            "url": "https://data.worldbank.org/",
        },
        "google-2020": {
            "source_id": "google-2020",
            "platform": "google_mobility",
            "title": "Google Community Mobility Reports Philippines 2020",
            "source_tier": "tier3_structured_repository",
            "query_geo_focus": "philippines",
            "year": 2020,
            "url": "https://www.google.com/covid19/mobility/",
        },
        "google-2021": {
            "source_id": "google-2021",
            "platform": "google_mobility",
            "title": "Google Community Mobility Reports Philippines 2021",
            "source_tier": "tier3_structured_repository",
            "query_geo_focus": "philippines",
            "year": 2021,
            "url": "https://www.google.com/covid19/mobility/",
        },
        "yafs-2021": {
            "source_id": "yafs-2021",
            "platform": "yafs",
            "title": "YAFS 5 Key Indicators and Regional Tables",
            "source_tier": "tier2_official_survey",
            "query_geo_focus": "philippines",
            "year": 2021,
            "url": "https://www.uppi.upd.edu.ph/yafs5",
        },
        "fies-yearbook": {
            "source_id": "fies-yearbook",
            "platform": "fies",
            "title": "2018 Philippines Statistical Yearbook Table 2.2 Number of Families Average Annual Income and Expenditure by Region",
            "source_tier": "tier2_official_survey",
            "query_geo_focus": "philippines",
            "year": 2015,
            "url": "https://psa.gov.ph/statistics/income-expenditure/fies",
        },
        "philhealth-2024": {
            "source_id": "philhealth-2024",
            "platform": "philhealth",
            "title": "PhilHealth Annual Report 2024",
            "source_tier": "tier1_official_anchor",
            "query_geo_focus": "philippines",
            "year": 2024,
            "url": "https://www.philhealth.gov.ph/about_us/annual_report/ar2024.pdf",
        },
        "philhealth-open-portal": {
            "source_id": "philhealth-open-portal",
            "platform": "philhealth_open_portal",
            "title": "PhilHealth Open Portal",
            "source_tier": "tier1_official_anchor",
            "query_geo_focus": "philippines",
            "year": 2025,
            "url": "https://philhealth.open.gov.ph/",
        },
        "psa-poverty": {
            "source_id": "psa-poverty",
            "platform": "psa_poverty",
            "title": "PSA Poverty Tables",
            "source_tier": "tier1_official_anchor",
            "query_geo_focus": "philippines",
            "year": 2024,
            "url": "",
        },
    }


def _mock_requests_get(url: str, *args, **kwargs):
    if "SI.POV.NAHC" in url:
        return _MockResponse(
            json_payload=[
                {"page": 1},
                [
                    {"date": "2022", "value": 18.1},
                    {"date": "2021", "value": 18.3},
                    {"date": "2009", "value": 26.0},
                ],
            ]
        )
    if "SE.SEC.CMPT.LO.ZS" in url:
        return _MockResponse(
            json_payload=[
                {"page": 1},
                [
                    {"date": "2022", "value": 82.4},
                    {"date": "2021", "value": 81.2},
                ],
            ]
        )
    if "FP.CPI.TOTL.ZG" in url:
        return _MockResponse(
            json_payload=[
                {"page": 1},
                [
                    {"date": "2022", "value": 5.8},
                    {"date": "2021", "value": 3.9},
                ],
            ]
        )
    if "Global_Mobility_Report.csv" in url:
        return _MockResponse(
            text_payload=(
                "country_region_code,country_region,sub_region_1,sub_region_2,metro_area,iso_3166_2_code,census_fips_code,place_id,date,"
                "retail_and_recreation_percent_change_from_baseline,grocery_and_pharmacy_percent_change_from_baseline,"
                "parks_percent_change_from_baseline,transit_stations_percent_change_from_baseline,"
                "workplaces_percent_change_from_baseline,residential_percent_change_from_baseline\n"
                "PH,Philippines,,,,,,place-ph,2020-01-10,-5,0,0,-7,-4,3\n"
                "PH,Philippines,,,,,,place-ph,2020-01-20,-3,0,0,-5,-2,2\n"
                "PH,Philippines,,,,,,place-ph,2020-02-03,2,0,0,1,3,-1\n"
                "PH,Philippines,Metro Manila,,,,,place-mm,2020-02-07,6,0,0,4,5,-2\n"
                "PH,Philippines,Central Visayas,,,,,place-cv,2020-02-10,8,0,0,3,4,-1\n"
            )
        )
    if "philhealth.open.gov.ph/data/coverage.json" in url:
        return _MockResponse(
            json_payload={
                "overview": {
                    "totalBeneficiaries": 106_235_882,
                    "registeredMembers": 61_519_044,
                    "registeredDependents": 44_716_838,
                    "populationCovered": 113_863_084,
                    "coverageRate": 100,
                    "registrationRate": 93,
                }
            }
        )
    if "philhealth.open.gov.ph/data/financials.json" in url:
        return _MockResponse(
            json_payload={
                "annualReports": [
                    {
                        "year": "2025-H1",
                        "coverageRate": 100,
                        "beneficiaries": 106_235_882,
                        "populationCovered": 113_863_084,
                        "claimsPaid": 139_310_045_787,
                        "averageProcessingDays": 25,
                        "breakdown": {
                            "yakap": {
                                "registrations": 31_789_799,
                                "firstEncounters": 10_030_102,
                                "accreditedClinics": 3_476,
                            }
                        },
                    },
                    {
                        "year": 2023,
                        "coverageRate": 100,
                        "totalBeneficiaries": 108_505_167,
                        "populationCovered": 112_892_781,
                        "claimsPaid": 122_383_003_091,
                        "healthcareFacilities": {
                            "hospitals": {"total": 1_879},
                            "otherFacilities": {"total": 10_025},
                            "specificFacilities": {
                                "konsultaProviders": {"total": 2_611},
                                "hivAidsCenters": {"total": 199},
                            },
                            "programCoverage": {
                                "konsulta": {"coveragePercentage": 81},
                                "mcp": {"coveragePercentage": 67},
                                "tbDots": {"coveragePercentage": 60},
                            }
                        },
                        "philhealthInfrastructure": {"facilitiesWithCARES": 1_102},
                    },
                ],
                "keyMetrics": {
                    "2024": {"claimsProcessingTime": 60},
                    "2023": {"claimsProcessingTime": 60},
                },
            }
        )
    if "philhealth.open.gov.ph/data/statistics-charts-2025.json" in url:
        return _MockResponse(
            json_payload={
                "philhealth_transparency_data_2025": {
                    "accreditation": {
                        "health_care_providers_institutions": {"grand_total": 13_042}
                    }
                }
            }
        )
    raise AssertionError(f"unexpected URL {url}")


def _write_minimal_xlsx(path: Path) -> None:
    workbook_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>
"""
    sheet_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    <row r="1"><c r="A1" t="inlineStr"><is><t>Title</t></is></c></row>
    <row r="2"><c r="A2" t="inlineStr"><is><t>Header</t></is></c></row>
    <row r="3"><c r="A3" t="inlineStr"><is><t>Spacer</t></is></c></row>
    <row r="4"><c r="A4" t="inlineStr"><is><t>ID</t></is></c></row>
    <row r="5"><c r="B5" t="inlineStr"><is><t>Central Visayas</t></is></c></row>
    <row r="6">
      <c r="A6"><v>72217</v></c>
      <c r="B6" t="inlineStr"><is><t>Cebu</t></is></c>
      <c r="C6" t="inlineStr"><is><t>Cebu City</t></is></c>
      <c r="D6"><v>12.5</v></c>
      <c r="E6"><v>10.1</v></c>
      <c r="F6"><v>8.4</v></c>
    </row>
  </sheetData>
</worksheet>
"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def test_structured_numeric_candidates_promote_structural_and_mobility_blocks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("epigraph_ph.phase0.structured_numeric_sources.requests.get", _mock_requests_get)
    monkeypatch.setattr(
        "epigraph_ph.phase0.structured_numeric_sources._LOCAL_STRUCTURED_DOC_ROOT",
        Path("D:/EpiGraph_PH/docs/Pdf"),
    )
    philhealth_stub = tmp_path / "ar2024.pdf"
    philhealth_stub.write_bytes(b"%PDF-1.4\n%%EOF\n")
    monkeypatch.setattr(
        "epigraph_ph.phase0.structured_numeric_sources._existing_local_philhealth_report_path",
        lambda file_name, cache_path: philhealth_stub,
    )
    monkeypatch.setattr(
        "epigraph_ph.phase0.structured_numeric_sources._extract_pdf_text",
        lambda pdf_path: _PHILHEALTH_LEAVE_BENEFIT_SAMPLE,
    )
    monkeypatch.setattr(
        "epigraph_ph.phase0.structured_numeric_sources._extract_statistical_yearbook_table_2_2_text",
        lambda pdf_path: _FIES_YEARBOOK_TABLE_2_2_SAMPLE,
    )

    payload = build_structured_numeric_candidates(
        raw_dir=tmp_path,
        source_rows=_source_rows(),
        plugin_id="hiv",
    )
    evidence_rows = build_candidate_evidence_rows(validated_candidates=payload["candidate_rows"], plugin_id="hiv")
    audit = build_latent_observability_audit(normalized_rows=evidence_rows, parameter_catalog=[], plugin_id="hiv")
    measurement_spec = build_national_measurement_spec(
        normalized_rows=evidence_rows,
        observability_audit=audit,
        month_axis=["2020-01", "2020-02", "2021", "2022"],
        plugin_id="hiv",
    )

    canonical_names = {row["canonical_name"] for row in payload["candidate_rows"]}
    retained_blocks = {row["block_id"] for row in measurement_spec["retained_blocks"]}
    mobility_region_rows = [
        row
        for row in payload["candidate_rows"]
        if row["canonical_name"] in {"mobility_network_mixing", "congestion_travel_time"} and row.get("region") not in {"", "national"}
    ]
    structural_region_rows = [
        row
        for row in payload["candidate_rows"]
        if row["canonical_name"] in {"education", "economic_access_constraint"} and row.get("region") not in {"", "national"}
    ]
    audit_lookup = {row["canonical_name"]: row for row in audit["rows"]}

    assert {
        "poverty_rate",
        "education",
        "cash_instability",
        "economic_access_constraint",
        "household_expenditure_burden",
        "mobility_network_mixing",
        "congestion_travel_time",
        "health_system_reach",
        "policy_implementation_weakness",
    } <= canonical_names
    assert {"structural_barrier_pressure", "mobility_exposure_pressure"} <= retained_blocks
    assert all(bool(row["is_direct_measurement"]) for row in payload["candidate_rows"])
    assert {row["region"] for row in mobility_region_rows} >= {"ncr", "region_vii"}
    assert len(structural_region_rows) >= 20
    assert audit_lookup["mobility_network_mixing"]["regional_support_count"] > 0
    assert audit_lookup["congestion_travel_time"]["regional_support_count"] > 0
    assert audit_lookup["education"]["regional_support_count"] > 0
    assert audit_lookup["economic_access_constraint"]["regional_support_count"] > 0
    assert audit_lookup["household_expenditure_burden"]["regional_support_count"] > 0
    assert audit_lookup["health_system_reach"]["regional_support_count"] > 0
    assert audit_lookup["policy_implementation_weakness"]["regional_support_count"] > 0


def test_parse_philhealth_leave_benefit_tables_extracts_region_year_values() -> None:
    rows = _parse_philhealth_leave_benefit_tables(_PHILHEALTH_LEAVE_BENEFIT_SAMPLE)
    lookup = {(row["year"], row["region"]): row for row in rows}

    assert len(rows) == 34
    assert lookup[(2024, "ncr")]["geo"] == "National Capital Region"
    assert lookup[(2024, "ncr")]["total_value"] == 82175829.0
    assert lookup[(2024, "region_iv_a")]["casual_value"] == 22812827.0
    assert lookup[(2023, "region_xiii")]["regular_value"] == 23748674.0
    assert (2024, "national") not in lookup


def test_collect_philhealth_portal_rows_extracts_capacity_and_governance_proxies(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("epigraph_ph.phase0.structured_numeric_sources.requests.get", _mock_requests_get)

    rows, collectors = _collect_philhealth_portal_rows(
        cache_dir=tmp_path,
        source_rows=_source_rows(),
    )
    lookup = {(row["canonical_name"], row["time"]): row for row in rows}
    service_rows = [row for row in rows if str(row["canonical_name"]).startswith("service_delivery_reach")]
    clinic_rows = [row for row in rows if str(row["canonical_name"]).startswith("clinics_per_capita")]

    assert ("philhealth_coverage", "2023") in lookup
    assert lookup[("philhealth_coverage", "2023")]["value"] == 100.0
    assert ("health_expenditure", "2025-06") in lookup
    assert ("policy_implementation_weakness", "2024") in lookup
    assert lookup[("policy_implementation_weakness", "2024")]["unit"] == "days"
    assert any(row["time"] == "2025-06" and row["unit"] == "facilities_per_100k" for row in clinic_rows)
    assert any(row["time"] == "2025-06" and row["value"] > 10.0 for row in clinic_rows)
    assert any(row["time"] == "2023" and row["value"] == 81.0 for row in service_rows)
    assert len(service_rows) >= 5
    assert len(clinic_rows) >= 3
    assert any(row["time"] == "2025-06" and row["value"] > 30.0 for row in service_rows)
    assert all(row["status"] == "ok" for row in collectors)


def test_build_philhealth_portal_artifacts_groups_rows_by_source_and_year(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("epigraph_ph.phase0.structured_numeric_sources.requests.get", _mock_requests_get)

    rows, collectors = _collect_philhealth_portal_rows(
        cache_dir=tmp_path,
        source_rows=_source_rows(),
    )
    artifacts = build_philhealth_portal_artifacts(candidate_rows=rows, collector_rows=collectors)
    summary = artifacts["summary"]
    source_rows = summary["sources"]
    year_rows = summary["years"]
    canonical_rows = {row["canonical_name"]: row for row in summary["canonical_metrics"]}

    assert artifacts["candidate_rows"]
    assert len(source_rows) == 2
    assert {row["source_title"] for row in source_rows} == {
        "PhilHealth Open Portal Financial Data",
        "PhilHealth Open Portal Statistics and Charts 2025",
    }
    assert any(row["year"] == "2023" and any(name.startswith("service_delivery_reach") for name in row["canonical_names"]) for row in year_rows)
    assert any(row["year"] == "2025" and any(name.startswith("clinics_per_capita") for name in row["canonical_names"]) for row in year_rows)
    assert canonical_rows["policy_implementation_weakness"]["years"] == ["2023", "2024", "2025"]
    assert "days" in canonical_rows["policy_implementation_weakness"]["units"]
    service_delivery_rows = [row for row in summary["canonical_metrics"] if str(row["canonical_name"]).startswith("service_delivery_reach")]
    assert service_delivery_rows
    assert max(float(row["value_max"]) for row in service_delivery_rows) > min(float(row["value_min"]) for row in service_delivery_rows)


def test_collect_local_psa_poverty_rows_extracts_city_and_province_support(tmp_path: Path) -> None:
    doc_root = tmp_path / "docs"
    doc_root.mkdir()

    (doc_root / "190710_poverty-statistics(Poverty_City_Mun_2009,2012,2015).csv").write_text(
        (
            "Region,Province  ,Municipality/City  ,Poverty Incidence_2009,Poverty Incidence_2012,"
            "Poverty Incidence_2015,Coefficient of Variation_2009,Coefficient of Variation_2012,Coefficient of Variation_2015\n"
            "Central Visayas,Cebu,Cebu City,14.2,11.4,9.5,10.0,9.1,8.7\n"
        ),
        encoding="utf-8",
    )
    (doc_root / "200604_updated-annual-per-capita-poverty-threshold-poverty-incidence-and-magnitude-of-poor-fami(By Province).csv").write_text(
        (
            "Region,Province,Poverty Incidence among Families Estimates (%) 2015u,Poverty Incidence among Families Estimates (%) 2018u ,"
            "Poverty Incidence among Families Coefficient of Variation 2015u ,Poverty Incidence among Families Coefficient of Variation 2018u \n"
            "Central Visayas,Cebu,16.0,12.0,8.0,7.0\n"
        ),
        encoding="utf-8",
    )
    (doc_root / "200305_annual-per-capita-poverty-threshold-poverty-incidence-among-families_by-region-and-provi(By Province).csv").write_text(
        (
            "Region,Region code,Province,Province code,Notes,2015*_Annual Per Capita Poverty Threshold \\n(in Pesos),"
            "2018_Annual Per Capita Poverty Threshold \\n(in Pesos),2015* Estimate_Poverty Incidence among Families (%),"
            "2018 Estimate_Poverty Incidence among Families (%),2015*_CV,2018_CV\n"
            "Central Visayas,PH070000000,Cebu,PH072200000,,23000,27000,16.0,12.0,8.0,7.0\n"
        ),
        encoding="utf-8",
    )
    _write_minimal_xlsx(doc_root / "2_2023 SAE_with PSGC_noHUC_06Feb2026.xlsx")

    rows, collectors = _collect_local_psa_poverty_rows(
        doc_root=doc_root,
        source_rows=_source_rows(),
    )

    city_rows = [row for row in rows if row["geo"] == "Cebu City"]
    province_rows = [row for row in rows if row["geo"] == "Cebu"]
    access_rows = [row for row in rows if row["canonical_name"] == "economic_access_constraint"]
    time_labels = {row["time"] for row in rows}

    assert collectors and all(row["status"] == "ok" for row in collectors)
    assert len(city_rows) >= 6
    assert len(province_rows) >= 4
    assert {row["time"] for row in access_rows} == {"2015", "2018"}
    assert all(row["unit"] == "php_per_person_per_year" for row in access_rows)
    assert {"2009", "2012", "2015", "2018", "2021", "2023"} <= time_labels
    assert {"poverty_rate", "economic_access_constraint"} == {row["canonical_name"] for row in rows}
    assert {row["region"] for row in rows} == {"region_vii"}


def test_collect_local_fies_yearbook_rows_extracts_household_expenditure_burden(tmp_path: Path, monkeypatch) -> None:
    doc_root = tmp_path / "docs"
    doc_root.mkdir()
    pdf_path = doc_root / "2018 Philippines Statistical Yearbook (2019).pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%%EOF\n")

    monkeypatch.setattr(
        "epigraph_ph.phase0.structured_numeric_sources._extract_statistical_yearbook_table_2_2_text",
        lambda path: _FIES_YEARBOOK_TABLE_2_2_SAMPLE if Path(path) == pdf_path else "",
    )

    rows, collectors = _collect_local_fies_structured_rows(
        doc_root=doc_root,
        source_rows=_source_rows(),
    )
    lookup = {(row["geo"], row["time"]): row for row in rows}

    assert collectors and all(row["status"] == "ok" for row in collectors)
    assert len(rows) == 20
    assert all(row["canonical_name"] == "household_expenditure_burden" for row in rows)
    assert lookup[("National Capital Region", "2015")]["value"] == 82.117647
    assert lookup[("Central Visayas", "2012")]["value"] == 78.4689
    assert lookup[("Autonomous Region in Muslim Mindanao", "2006")]["value"] == 84.269663


def test_structured_numeric_candidates_reuse_cache_without_network(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("epigraph_ph.phase0.structured_numeric_sources.requests.get", _mock_requests_get)
    first = build_structured_numeric_candidates(
        raw_dir=tmp_path,
        source_rows=_source_rows(),
        plugin_id="hiv",
    )

    def _raise_requests_get(url: str, *args, **kwargs):
        raise RuntimeError(f"network disabled for {url}")

    monkeypatch.setattr("epigraph_ph.phase0.structured_numeric_sources.requests.get", _raise_requests_get)
    second = build_structured_numeric_candidates(
        raw_dir=tmp_path,
        source_rows=_source_rows(),
        plugin_id="hiv",
    )

    assert len(first["candidate_rows"]) == len(second["candidate_rows"])
    collector_rows = list(second["summary"]["collectors"])
    assert any(row["collector"] == "world_bank_wdi" and row["cache_used"] for row in collector_rows)
    assert any(row["collector"] == "google_mobility" and row["cache_used"] for row in collector_rows)
