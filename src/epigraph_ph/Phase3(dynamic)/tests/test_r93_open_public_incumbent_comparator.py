from __future__ import annotations

from pathlib import Path

from phase3_dynamic.r11_sparse_state_space import OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
from phase3_dynamic.r53_publication_claim_registry import _open_public_incumbent_comparator_claim
from phase3_dynamic.r78_public_annual_family_expansion import R78_SELECTED_FAMILY
from phase3_dynamic.r93_open_public_incumbent_comparator import _gate


def _r78_report(*, leaked: bool = False) -> dict:
    role = "direct_target" if leaked else "validation_only"
    return {
        "expanded_public_annual_gate": {"status": "expanded_public_annual_comparator_promoted"},
        "target_rows": [{"metric_name": metric} for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS],
        "family_rows": [
            {
                "candidate_family": R78_SELECTED_FAMILY,
                "candidate_mean_norm_error": 0.16,
                "candidate_interval_coverage": 0.92,
            }
        ],
        "score_rows": [
            {
                "candidate_family": R78_SELECTED_FAMILY,
                "metric_name": metric,
                "observation_role": role,
                "allowed_use": "validation_only",
            }
            for metric in OFFICIAL_ANNUAL_REQUIRED_MODEL_HEADS
        ],
    }


def test_r93_ready_comparator_can_block_model_superiority() -> None:
    gate = _gate(
        _r78_report(),
        {
            "expanded_public_proxy_annual_gate": {
                "status": "annual_model_blocked_by_expanded_public_proxy",
                "score_row_count": 9,
                "model_family": "phase3_model",
                "model_mean_norm_error": 0.24,
                "public_proxy_mean_norm_error": 0.16,
                "model_interval_coverage": 0.75,
                "public_proxy_interval_coverage": 0.92,
            }
        },
    )

    assert gate["status"] == "public_incumbent_comparator_ready_model_blocked"
    assert gate["annual_superiority_status"] == "blocked_by_open_public_incumbent"
    assert not gate["blockers"]


def test_r93_blocks_comparator_when_incumbent_scores_leak_roles() -> None:
    gate = _gate(
        _r78_report(leaked=True),
        {
            "expanded_public_proxy_annual_gate": {
                "status": "annual_model_blocked_by_expanded_public_proxy",
                "score_row_count": 9,
            }
        },
    )

    assert gate["status"] == "public_incumbent_comparator_blocked"
    assert "incumbent_validation_role_leakage" in gate["blockers"]


def test_r93_registry_claim_blocks_annual_superiority(tmp_path: Path) -> None:
    artifact = tmp_path / "r93.json"
    artifact.write_text("{}", encoding="utf-8")
    claim = _open_public_incumbent_comparator_claim(
        {
            "status": "public_incumbent_comparator_ready_model_blocked",
            "candidate_family": "open_public_aem_spectrum_style_annual_incumbent_comparator",
            "open_public_incumbent_comparator_gate": {
                "status": "public_incumbent_comparator_ready_model_blocked",
                "annual_superiority_status": "blocked_by_open_public_incumbent",
                "blockers": [],
            },
        },
        artifact,
    )

    assert claim["claim_status"] == "annual_superiority_blocked_by_open_public_incumbent"
    assert "open public annual incumbent comparator" in claim["allowed_claim"]
