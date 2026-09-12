import hashlib
import json
from pathlib import Path

import pytest

from phase3_dynamic.annual_benchmark_contract import annual_claim_limit, require_training_role


@pytest.mark.parametrize("role", ["validation_only", "quarantined", "prior_context", None])
def test_nontraining_roles_block_even_if_allowed_use_says_train(role):
    row = {"observation_role": role, "allowed_use": ["train_origin_weak_measurement"]}
    with pytest.raises(ValueError):
        require_training_role(row, "annual_new_infections")


def test_explicit_auxiliary_role_allowed_but_metric_specific_role_takes_precedence():
    row = {"observation_role": "auxiliary_likelihood", "allowed_use": ["train_origin_weak_measurement"]}
    require_training_role(row, "annual_new_infections")
    row["metric_provenance"] = {"annual_new_infections": {"observation_role": "validation_only", "allowed_use": "validation_only"}}
    with pytest.raises(ValueError):
        require_training_role(row, "annual_new_infections")
    with pytest.raises(ValueError):
        require_training_role({"observation_role": "auxiliary_likelihood"}, "x")


def test_claim_boundary_preserves_history_without_claiming_calibration_or_official_win():
    original = {"claim_status": "promoted", "blockers": [], "key_metrics": {"error": .1}}
    result = annual_claim_limit(original)
    assert result["claim_status"] == "diagnostic_only"
    assert result["historical_claim_status"] == "promoted"
    assert result["key_metrics"] == original["key_metrics"]
    assert original["claim_status"] == "promoted"
    assert original["blockers"] == []


def test_real_annual_design_role_audit_is_traced_and_performs_no_new_fit():
    root = Path(__file__).resolve().parents[4]
    report = json.loads((root / "docs/annual_benchmark_integrity_20260912.json").read_text())
    assert report["split_count"] == 12
    assert report["violating_metric_split_count"] == 36
    assert not report["fresh_fit_performed"]
    for row in report["metric_split_audits"]:
        assert row["strictly_inadmissible_rows"] == row["legacy_design_rows"]
    for name, digest in report["code_hashes"].items():
        source = root / "src/epigraph_ph/Phase3(dynamic)/src/phase3_dynamic" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def test_effective_registry_narrows_annual_claims_not_r41_parameters():
    root = Path(__file__).resolve().parents[4]
    registry = json.loads((root / "docs/phase3_claim_registry_20260912.json").read_text())
    claims = {r["claim_id"]: r for r in registry["claim_rows"]}
    assert claims["national_r41_research_champion"]["claim_status"] == "promoted"
    for name in ["phase3_r86_annual_calibrated_forecast_grid_ledger", "phase3_r88_guarded_annual_ledger_selector", "phase3_r90_claim_grade_gate"]:
        assert claims[name]["claim_status"] == "diagnostic_only"
        assert claims[name]["historical_claim_status"] != "diagnostic_only"
        assert "annual_training_role_contract_requires_repair" in claims[name]["blockers"]
    assert not registry["registry_gate"]["official_forecast_superiority_established"]
