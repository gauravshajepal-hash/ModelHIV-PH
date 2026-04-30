from __future__ import annotations

import json

from epigraph_ph.phase0.citation_evidence_ledger import build_phase3_determinant_evidence_ledger


def test_citation_evidence_ledger_preserves_official_locators_and_allowed_use(tmp_path) -> None:
    run_dir = tmp_path / "run"
    raw_dir = run_dir / "phase0" / "raw"
    extracted_dir = run_dir / "phase0" / "extracted"
    raw_dir.mkdir(parents=True)
    extracted_dir.mkdir(parents=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(
            [
                {
                    "source_id": "world-bank-poverty",
                    "title": "World Bank WDI Poverty",
                    "platform": "world_bank_wdi",
                    "organization": "World Bank",
                    "source_tier": "tier3_structured_repository",
                    "url": "https://api.worldbank.org/v2/country/PHL/indicator/SI.POV.NAHC",
                },
                {
                    "source_id": "who-guidance",
                    "title": "WHO HIV Strategic Information",
                    "platform": "who",
                    "organization": "WHO",
                    "source_tier": "tier1_official_anchor",
                    "url": "https://www.who.int/publications",
                },
            ]
        )
    )
    (extracted_dir / "canonical_parameter_candidates.json").write_text(
        json.dumps(
            [
                {
                    "canonical_name": "economic_access_constraint",
                    "candidate_block": "structural_barrier_pressure",
                    "source_id": "world-bank-poverty",
                    "source_bank": "phase0_structured_numeric",
                    "source_title": "World Bank WDI Poverty",
                    "measurement_role": "direct_indicator",
                    "is_direct_measurement": True,
                    "model_numeric_value": 0.18,
                    "normalized_unit": "percent",
                    "year": 2021,
                    "geo": "Philippines",
                    "evidence_span": "poverty headcount ratio",
                    "evidence_weight": 0.35,
                },
                {
                    "canonical_name": "testing_uptake",
                    "candidate_block": "testing_prevention_reach",
                    "source_id": "who-guidance",
                    "source_bank": "phase0_chunk_soft_candidates",
                    "source_title": "WHO HIV Strategic Information",
                    "measurement_role": "context_only",
                    "is_direct_measurement": False,
                    "is_anchor_eligible": True,
                    "year": 2022,
                    "geo": "Philippines",
                    "candidate_text": "testing uptake strategic information",
                },
            ]
        )
    )

    summary = build_phase3_determinant_evidence_ledger(run_dir=run_dir)

    assert summary["covered_determinant_count"] == 2
    assert summary["official_source_manifest_summary"]["required_official_family_counts"]["world_bank_wdi"] == 1
    assert summary["official_source_manifest_summary"]["required_official_family_counts"]["who"] == 1
    ledger_path = run_dir / "phase0" / "evidence_ledger" / "phase3_determinant_evidence_ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    by_name = {row["determinant_name"]: row for row in rows}
    assert by_name["economic_access_constraint"]["has_verifiable_locator"] is True
    assert by_name["economic_access_constraint"]["allowed_use"] == "direct_determinant_covariate_candidate"
    assert by_name["testing_uptake"]["allowed_use"] == "anchor_context_only"

