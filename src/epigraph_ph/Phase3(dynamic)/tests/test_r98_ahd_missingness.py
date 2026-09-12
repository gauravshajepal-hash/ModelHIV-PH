from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import minimize

from phase3_dynamic.ahd_observation import (
    AhdStatus, backlog_status_deviance, classification_deviance, mar_deviance,
    naive_binary_deviance, observation_probabilities, profile_deviance,
)
from phase3_dynamic.r98_ahd_missingness import extract_status, eligible_training, fit_status_forecast, score_blocks, FAMILIES


TEXT = """From April to June 2026, there were 2,994 confirmed HIV-positive individuals.
Of the recorded cases for this quarter, 828 (28%) had an advanced HIV disease.
2,166 cases had non-advanced HIV infection. Of these, 1,732 (80%) had no data on immunologic/clinical criteria."""


def row(period="2026-Q1", counts=(1104, 2333, 1196), posted="2026-08-26T08:43:38Z"):
    a, e, m = counts
    return {"period": period, "advanced": a, "known_nonadvanced": e, "unknown": m,
            "total_diagnoses": a+e+m, "observation_role": "auxiliary_likelihood",
            "allowed_use": ["classification_likelihood"], "measurement_semantics": "joint_classification_count",
            "status": "accepted", "row_hash": period, "mirror_posted_at": posted}


def test_extract_explicit_unknown_partition_and_no_missing_as_negative():
    parsed = extract_status(TEXT)
    assert (parsed["advanced"], parsed["known_nonadvanced"], parsed["unknown"]) == (828, 434, 1732)
    assert parsed["bounds"] == pytest.approx((828/2994, 2560/2994))
    assert parsed["complete_case_fraction"] == pytest.approx(828/1262)
    with pytest.raises(ValueError, match="reconcile"):
        extract_status(TEXT.replace("2,994", "2,995"))
    with pytest.raises(ValueError, match="containment"):
        extract_status(TEXT.replace(". Of these,", " and"))


@pytest.mark.parametrize("counts", [(-1, 2, 3), (1.2, 2, 3), (float("nan"), 2, 3), (0, 0, 0)])
def test_invalid_counts_are_rejected(counts):
    with pytest.raises(ValueError):
        AhdStatus(*counts)


def test_sharp_bounds_flat_profile_not_a_point_estimate():
    s = AhdStatus(828, 434, 1732)
    lo, hi = s.bounds
    for p in np.linspace(lo, hi, 31):
        assert profile_deviance(s, p) == pytest.approx(0)
    assert profile_deviance(s, lo/2) > 0
    assert profile_deviance(s, (1+hi)/2) > 0
    assert naive_binary_deviance(s, lo) == pytest.approx(0)
    assert mar_deviance(s, s.complete_case_fraction) == pytest.approx(0)
    assert mar_deviance(s, lo) > 0


@pytest.mark.parametrize("p", [.1, .3, .7, .95])
def test_exact_profile_matches_independent_constrained_numerical_optimization(p):
    s = AhdStatus(10, 20, 30)
    fit = minimize(lambda q: classification_deviance(s, observation_probabilities(p, *q)),
                   [.5, .5], method="Nelder-Mead", bounds=[(0, 1), (0, 1)],
                   options={"xatol": 1e-10, "fatol": 1e-10})
    assert fit.success
    assert fit.fun == pytest.approx(profile_deviance(s, p), abs=1e-7)


def test_complete_and_entirely_missing_extremes():
    s = AhdStatus(10, 20, 0)
    for p in [.1, .3, .7]:
        assert profile_deviance(s, p) == pytest.approx(mar_deviance(s, p))
    missing = AhdStatus(0, 0, 10)
    assert missing.bounds == (0, 1)
    assert missing.complete_case_fraction is None
    assert all(profile_deviance(missing, p) == 0 for p in np.linspace(0, 1, 11))


def test_quarantined_and_validation_only_rows_cannot_enter_likelihood():
    for role in ["quarantined", "validation_only", "prior_context"]:
        r = row()
        r["observation_role"] = role
        with pytest.raises(ValueError):
            fit_status_forecast([r], "pooled_status_mix")


def test_forecasts_ignore_future_targets_and_check_actual_posting_time():
    train = row("2025-Q4", (100, 200, 50), "2026-02-26T10:00:00Z")
    q1, q2 = row(), row("2026-Q2", (828, 434, 1732), "2026-08-26T08:44:51Z")
    assert eligible_training([train, q1, q2], q1, release_aware=True) == []
    assert eligible_training([train, q1, q2], q2, release_aware=True) == [train]
    pred = {f: fit_status_forecast([train], f) for f in FAMILIES}
    changed = deepcopy(q2)
    changed.update(advanced=10000, known_nonadvanced=1, unknown=1, total_diagnoses=10002)
    eligible = eligible_training([train, q1, changed], changed, release_aware=True)
    assert {f: fit_status_forecast(eligible, f) for f in FAMILIES} == pred
    scores, _ = score_blocks([train, q1, q2], release_aware=True)
    assert len(scores) == len(FAMILIES)


def test_no_pseudocount_hides_impossible_outcome():
    s = AhdStatus(1, 0, 0)
    assert np.isinf(classification_deviance(s, [0, 1, 0]))
    with pytest.raises(ValueError):
        observation_probabilities(1.1, 1, 1)


def test_quarterly_backlog_adapter_preserves_process_and_does_not_invent_months():
    from phase3_dynamic.diagnosis_incidence_repair import _simulate_backlog_late_emission

    trajectory = _simulate_backlog_late_emission(incidence_by_month={m: 10 for m in range(3)}, months=[0,1,2],
                 early_to_late_hazard=.1, early_diagnosis_hazard=.2, late_diagnosis_hazard=.3,
                 initial_early_undiagnosed=10, initial_late_undiagnosed=5)
    before = deepcopy(trajectory)
    s = AhdStatus(10, 10, 100)
    assert backlog_status_deviance(trajectory, s, [0,1,2]) >= 0
    assert trajectory == before
    for t in trajectory.values():
        assert t["early_undiagnosed_start"] + t["late_undiagnosed_start"] + t["incident_infections"] == pytest.approx(
            t["early_undiagnosed_end"] + t["late_undiagnosed_end"] + t["new_diagnosed_cases_period"])
    with pytest.raises(ValueError, match="full contiguous"):
        backlog_status_deviance(trajectory, s, [0,2])
    with pytest.raises(ValueError, match="Missing model months"):
        backlog_status_deviance(trajectory, s, [1,2,3])


def test_tracked_run_is_reconciled_reproducible_and_does_not_promote():
    root = Path(__file__).resolve().parents[4]
    report_path = root / "docs/phase3_r98_results_20260911.json"
    report = json.loads(report_path.read_text())
    rows = report["accepted_partitions"]
    assert len(rows) == 6
    assert len(report["quarantined_partitions"]) == 4
    for r in rows:
        s = AhdStatus.from_ledger(r)
        assert sum(r["value"].values()) == s.total
        data = {k: v for k, v in r.items() if k != "row_hash"}
        assert hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest() == r["row_hash"]
    for release_aware, key in [(False, "event_time_scores"), (True, "mirror_time_scores")]:
        computed, _ = score_blocks(rows, release_aware=release_aware)
        assert json.loads(json.dumps(computed)) == report[key]
    assert report["gate"]["champion"] is None
    assert report["gate"]["incidence_training_rows"] == 0
    from phase3_dynamic.r53_publication_claim_registry import _ahd_missingness_claim
    assert _ahd_missingness_claim(report, report_path)["claim_status"] == "diagnostic_only"
    assert _ahd_missingness_claim({}, report_path)["claim_status"] == "blocked"


def test_paper_bundle_and_frozen_forecast_checksums():
    root = Path(__file__).resolve().parents[4]
    manifest = json.loads((root / "docs/phase3_r98_bundle_manifest_20260911.json").read_text())
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest, relative
    frozen = root / "docs/phase3_r97_prospective_Q4_lock_20260911.json"
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() == manifest["frozen_R97_sha256"]
    source = root / "src/epigraph_ph/Phase3(dynamic)/src/phase3_dynamic"
    assert hashlib.sha256((source / "r98_report.py").read_bytes()).hexdigest() == manifest["builder_sha256"]
    run_manifest = json.loads((root / "docs/phase3_r98_run_manifest_20260911.json").read_text())
    for name, digest in run_manifest["code_hashes"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
