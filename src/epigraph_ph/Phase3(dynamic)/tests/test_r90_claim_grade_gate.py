from __future__ import annotations

from pathlib import Path

from phase3_dynamic.r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
from phase3_dynamic.r90_claim_grade_gate import (
    _annual_candidate_claim_grade,
    _claim_grade_gate,
    _mechanism_claim_grade,
)


def _artifact(tmp_path: Path) -> Path:
    path = tmp_path / "report.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _annual_report(*, leaked: bool = False) -> dict:
    counts = {metric: 2 for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS}
    role = "direct_target" if leaked else "validation_only"
    return {
        "status": "example_pass",
        "candidate_family": "example_annual_family",
        "example_gate": {
            "status": "example_pass",
            "candidate_mean_norm_error": 0.2,
            "carry_forward_mean_norm_error": 0.4,
            "candidate_interval_coverage": 0.9,
            "carry_forward_interval_coverage": 0.6,
            "scored_counts_by_metric": counts,
            "target_counts_by_metric": counts,
        },
        "score_rows": [
            {
                "metric_name": "annual_new_infections",
                "quarter": "2024-Q4",
                "observation_role": role,
                "allowed_use": "validation_only",
            }
        ],
        "metric_rows": [
            {
                "metric_name": metric,
                "candidate_mean_norm_error": 0.2,
                "carry_forward_mean_norm_error": 0.4,
                "candidate_p90_norm_error": 0.3,
                "carry_forward_p90_norm_error": 0.5,
                "candidate_interval_coverage": 0.9,
                "carry_forward_interval_coverage": 0.6,
            }
            for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
        ],
        "horizon_rows": [
            {
                "horizon_years": horizon,
                "candidate_mean_norm_error": 0.2,
                "carry_forward_mean_norm_error": 0.4,
                "candidate_p90_norm_error": 0.3,
                "carry_forward_p90_norm_error": 0.5,
                "candidate_interval_coverage": 0.9,
                "carry_forward_interval_coverage": 0.6,
            }
            for horizon in (1, 3, 5)
        ],
    }


def test_r90_annual_candidate_claim_grade_passes_when_all_requirements_pass(tmp_path: Path) -> None:
    claim, requirements, leakage = _annual_candidate_claim_grade(
        candidate_id="annual_candidate",
        report=_annual_report(),
        path=_artifact(tmp_path),
        gate_key="example_gate",
        expected_status="example_pass",
        claim_scope="annual",
        allowed_claim_if_ready="allowed",
        claim_limit="limited",
    )

    assert claim["claim_status"] == "claim_grade_ready"
    assert not claim["blockers"]
    assert not leakage
    assert requirements
    assert all(row["passed"] for row in requirements)


def test_r90_annual_candidate_blocks_validation_role_leakage(tmp_path: Path) -> None:
    claim, requirements, leakage = _annual_candidate_claim_grade(
        candidate_id="annual_candidate",
        report=_annual_report(leaked=True),
        path=_artifact(tmp_path),
        gate_key="example_gate",
        expected_status="example_pass",
        claim_scope="annual",
        allowed_claim_if_ready="allowed",
        claim_limit="limited",
    )

    assert claim["claim_status"] == "blocked"
    assert "no_validation_role_leakage" in claim["blockers"]
    assert leakage
    assert any(row["requirement_id"] == "no_validation_role_leakage" and not row["passed"] for row in requirements)


def test_r90_mechanism_claim_blocks_when_r89_is_diagnostic(tmp_path: Path) -> None:
    r89 = {
        "status": "incidence_mortality_mechanism_support_diagnostic_only",
        "candidate_family": "reported_death_bridge_and_incidence_support_gate",
        "incidence_mortality_mechanism_support_gate": {
            "status": "incidence_mortality_mechanism_support_diagnostic_only",
            "blockers": ["direct_incidence_process_support_absent"],
            "direct_incidence_process_support_count": 0,
            "reported_death_bridge_candidate_mean_norm_error": 0.6,
            "reported_death_bridge_carry_forward_mean_norm_error": 0.4,
            "reported_death_bridge_candidate_interval_coverage": 0.4,
            "reported_death_bridge_carry_forward_interval_coverage": 0.6,
        },
        "mortality_score_rows": [
            {
                "metric_name": "annual_aids_deaths",
                "quarter": "2024-Q4",
                "observation_role": "validation_only",
                "allowed_use": "validation_only",
            }
        ],
    }

    claim, requirements, leakage = _mechanism_claim_grade(r89, _artifact(tmp_path))

    assert claim["claim_status"] == "mechanism_claim_blocked"
    assert "direct_incidence_process_support_absent" in claim["blockers"]
    assert not leakage
    assert any(not row["passed"] for row in requirements)


def test_r90_gate_allows_readout_ready_with_mechanisms_blocked() -> None:
    gate = _claim_grade_gate(
        [
            {"candidate_id": "r86_annual_calibrated_ledger", "claim_status": "claim_grade_ready", "allowed_claim": "r86"},
            {"candidate_id": "r88_guarded_annual_ledger", "claim_status": "claim_grade_ready", "allowed_claim": "r88"},
            {
                "candidate_id": "r89_incidence_mortality_mechanism_support",
                "claim_status": "mechanism_claim_blocked",
                "blockers": ["direct_incidence_process_support_absent"],
                "allowed_claim": "blocked",
            },
        ]
    )

    assert gate["status"] == "claim_grade_annual_readout_ready_mechanisms_blocked"
    assert gate["r86_annual_calibrated_ledger_ready"] is True
    assert gate["r88_guarded_annual_ledger_ready"] is True
    assert gate["incidence_mortality_mechanism_ready"] is False
