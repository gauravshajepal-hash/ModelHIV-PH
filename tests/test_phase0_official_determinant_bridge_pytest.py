from __future__ import annotations

import json

from epigraph_ph.phase0.citation_evidence_ledger import build_phase3_determinant_evidence_ledger
from epigraph_ph.phase0.official_determinant_bridge import build_official_determinant_bridge_rows


def test_official_determinant_bridge_maps_unaids_and_harp_without_promoting_truth(tmp_path) -> None:
    run_dir = tmp_path / "run"
    harp_dir = run_dir / "harp_archive"
    raw_dir = run_dir / "phase0" / "raw"
    extracted_dir = run_dir / "phase0" / "extracted"
    harp_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    (raw_dir / "source_manifest.json").write_text("[]")
    (extracted_dir / "canonical_parameter_candidates.json").write_text("[]")
    (harp_dir / "multinational_hiv_metric_rows.json").write_text(
        json.dumps(
            [
                {
                    "metric_name": "late_hiv_diagnosis_percent",
                    "source_id": "unaids_late_hiv_diagnosis_all_ages",
                    "source_label": "UNAIDS Late HIV Diagnosis - All Ages With CD4 <200",
                    "source_organization": "UNAIDS",
                    "source_url": "",
                    "time": "2020-01",
                    "year": 2020,
                    "value": 55.0,
                    "unit": "percent",
                    "geo": "Philippines",
                    "region": "national",
                    "measurement_class": "external_reference_unaids",
                }
            ]
        )
    )
    (harp_dir / "wdi_hiv_rows.json").write_text("[]")
    (harp_dir / "observed_program_panel.json").write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "metric_name": "virally_suppressed",
                        "source_id": "doh_official_cascade_ground_truth_2018_2025",
                        "source_label": "DOH 95-95-95 Ground Truth 2018-2025",
                        "source_url": "user_attached_doh_slide_2026_03_31",
                        "time": "2024-12",
                        "year": 2024,
                        "value": 59000,
                        "unit": "count_people",
                        "geo": "Philippines",
                        "region": "national",
                        "measurement_class": "program_observed_harp",
                    }
                ]
            }
        )
    )

    summary = build_official_determinant_bridge_rows(run_dir=run_dir)

    assert summary["candidate_row_count"] == 3
    rows = json.loads((run_dir / "phase0" / "evidence_ledger" / "official_determinant_candidate_rows.json").read_text())
    by_name = {row["canonical_name"]: row for row in rows}
    assert by_name["late_hiv_diagnosis_percent"]["source_url"] == "https://aidsinfo.unaids.org/"
    assert by_name["late_hiv_diagnosis_percent"]["allowed_use_hint"] == "validation_or_auxiliary_only"
    assert by_name["documented_suppression"]["allowed_use_hint"] == "program_observation_head_only"


def test_citation_ledger_includes_official_bridge_allowed_use(tmp_path) -> None:
    run_dir = tmp_path / "run"
    harp_dir = run_dir / "harp_archive"
    raw_dir = run_dir / "phase0" / "raw"
    extracted_dir = run_dir / "phase0" / "extracted"
    harp_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    (raw_dir / "source_manifest.json").write_text("[]")
    (extracted_dir / "canonical_parameter_candidates.json").write_text("[]")
    (harp_dir / "multinational_hiv_metric_rows.json").write_text(
        json.dumps(
            [
                {
                    "metric_name": "prep_people_receiving",
                    "source_id": "unaids_prep_people_receiving",
                    "source_label": "UNAIDS People Receiving PrEP",
                    "source_organization": "UNAIDS",
                    "time": "2024-01",
                    "year": 2024,
                    "value": 36000,
                    "unit": "count_people",
                    "geo": "Philippines",
                    "region": "national",
                    "measurement_class": "external_reference_unaids",
                }
            ]
        )
    )
    (harp_dir / "wdi_hiv_rows.json").write_text("[]")
    (harp_dir / "observed_program_panel.json").write_text(json.dumps({"rows": []}))

    summary = build_phase3_determinant_evidence_ledger(run_dir=run_dir)

    assert summary["official_bridge_candidate_row_count"] == 1
    ledger_rows = [
        json.loads(line)
        for line in (run_dir / "phase0" / "evidence_ledger" / "phase3_determinant_evidence_ledger.jsonl").read_text().splitlines()
    ]
    assert ledger_rows[0]["determinant_name"] == "prep_active_refill"
    assert ledger_rows[0]["source_family"] == "unaids"
    assert ledger_rows[0]["allowed_use"] == "validation_or_auxiliary_only"


def test_official_bridge_aliases_noncanonical_official_rows_without_rescaling(tmp_path) -> None:
    run_dir = tmp_path / "run"
    harp_dir = run_dir / "harp_archive"
    raw_dir = run_dir / "phase0" / "raw"
    extracted_dir = run_dir / "phase0" / "extracted"
    harp_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    (raw_dir / "source_manifest.json").write_text("[]")
    (harp_dir / "multinational_hiv_metric_rows.json").write_text("[]")
    (harp_dir / "wdi_hiv_rows.json").write_text("[]")
    (harp_dir / "observed_program_panel.json").write_text(json.dumps({"rows": []}))
    (extracted_dir / "canonical_parameter_candidates.json").write_text(
        json.dumps(
            [
                {
                    "canonical_name": "congestion_travel_time",
                    "candidate_block": "mobility_exposure_pressure",
                    "source_id": "google_mobility_2020",
                    "source_title": "Google Mobility 2020",
                    "platform": "google_mobility",
                    "source_bank": "phase0_structured_numeric",
                    "is_direct_measurement": True,
                    "value": -35.0,
                    "unit": "percent",
                    "time": "2020-03",
                    "geo": "Philippines",
                    "region": "national",
                    "evidence_span": "transit-station mobility change from baseline",
                },
                {
                    "canonical_name": "service_delivery_reach_konsulta",
                    "candidate_block": "care_access_continuity",
                    "source_id": "philhealth_open_portal",
                    "source_title": "PhilHealth Open Portal",
                    "platform": "philhealth_open_portal",
                    "source_bank": "phase0_structured_numeric",
                    "is_direct_measurement": True,
                    "value": 81.0,
                    "unit": "percent",
                    "time": "2023",
                    "geo": "Philippines",
                    "region": "national",
                    "evidence_span": "Konsulta cities with providers coverage",
                },
                {
                    "canonical_name": "clinics_per_capita_hiv_aids_centers",
                    "candidate_block": "care_access_continuity",
                    "source_id": "philhealth_open_portal",
                    "source_title": "PhilHealth Open Portal",
                    "platform": "philhealth_open_portal",
                    "source_bank": "phase0_structured_numeric",
                    "is_direct_measurement": True,
                    "value": 0.17,
                    "unit": "facilities_per_100k",
                    "time": "2023",
                    "geo": "Philippines",
                    "region": "national",
                    "evidence_span": "HIV/AIDS centers per 100k",
                },
            ]
        )
    )

    summary = build_official_determinant_bridge_rows(run_dir=run_dir)

    assert summary["candidate_row_count"] == 3
    rows = json.loads((run_dir / "phase0" / "evidence_ledger" / "official_determinant_candidate_rows.json").read_text())
    by_name = {row["canonical_name"]: row for row in rows}
    assert by_name["transport_friction"]["value"] == -35.0
    assert by_name["transport_friction"]["value_semantics"] == "transit_station_mobility_change_proxy_no_numeric_transform"
    assert by_name["service_delivery_reach"]["value"] == 81.0
    assert by_name["clinics_per_capita"]["value"] == 0.17


def test_official_bridge_reads_cached_world_bank_determinants(tmp_path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "phase0" / "raw"
    extracted_dir = run_dir / "phase0" / "extracted"
    cache_dir = run_dir / "phase0" / "evidence_ledger" / "official_bridge_cache"
    (run_dir / "harp_archive").mkdir(parents=True)
    raw_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "world_bank_wdi",
                    "platform": "world_bank_wdi",
                    "organization": "World Bank",
                    "title": "World Bank World Development Indicators",
                    "url": "https://data.worldbank.org/",
                }
            ]
        )
    )
    (extracted_dir / "canonical_parameter_candidates.json").write_text("[]")
    for name in ("multinational_hiv_metric_rows.json", "wdi_hiv_rows.json"):
        (run_dir / "harp_archive" / name).write_text("[]")
    (run_dir / "harp_archive" / "observed_program_panel.json").write_text(json.dumps({"rows": []}))
    (cache_dir / "world_bank_en.pop.dnst.json").write_text(json.dumps([{"date": "2024", "value": 394.4}]))
    (cache_dir / "world_bank_sp.urb.totl.in.zs.json").write_text(json.dumps([{"date": "2024", "value": 48.3}]))
    (cache_dir / "world_bank_sh.xpd.chex.pc.cd.json").write_text(json.dumps([{"date": "2024", "value": 187.2}]))

    summary = build_official_determinant_bridge_rows(run_dir=run_dir)

    assert summary["candidate_row_count"] == 3
    assert summary["canonical_name_counts"]["population_density"] == 1
    assert summary["canonical_name_counts"]["urbanization_pressure"] == 1
    assert summary["canonical_name_counts"]["health_expenditure"] == 1
    rows = json.loads((run_dir / "phase0" / "evidence_ledger" / "official_determinant_candidate_rows.json").read_text())
    assert {row["allowed_use_hint"] for row in rows} == {"direct_determinant_covariate_candidate"}
