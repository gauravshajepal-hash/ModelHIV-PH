from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np
import pytest

from phase3_dynamic.hasp_monthly_table import parse_monthly_diagnosis_table
from phase3_dynamic.r96_monthly_diagnosis_state import (
    Q1_PDF, Q2_PDF, _filter, backtest, choose_family, extract_pdf_text,
    extract_hasp_2026_q1_rows, extract_hasp_2026_q2_rows, fit_local_level,
    forecast, monthly_series, promotion_gate,
)


def test_column_prose_and_partial_year_parse_without_average_becoming_a_month():
    text = """Figure 3. Number of monthly newly diagnosed HIV cases, Jan 2025 - Mar 2026
    prose with 2025 in it. 2025 100 100 100 100 100 100 100 100 100 100 100 100 100
    another column.       2026 1690 1407 1536 1544
    Geographic Distribution
    2026 999 999 999 999
    """
    result = parse_monthly_diagnosis_table(text, end_month="2026-03")
    assert [len(r["counts"]) for r in result] == [12, 3]
    assert sum(result[-1]["counts"]) == 4633
    with pytest.raises(ValueError, match="AVG"):
        parse_monthly_diagnosis_table(text.replace("1536 1544", "1536 2000"), end_month="2026-03")
    with pytest.raises(ValueError, match="missing years"):
        parse_monthly_diagnosis_table(text.replace("2025 100", "broken 100"), end_month="2026-03")


@pytest.mark.skipif(not shutil.which("pdftotext") or not Q1_PDF.exists() or not Q2_PDF.exists(), reason="HASP PDFs/Poppler absent")
def test_real_pdf_complete_extraction_and_vintage_revision():
    q1, _ = extract_hasp_2026_q1_rows(extract_pdf_text(Q1_PDF), Q1_PDF)
    q2, _ = extract_hasp_2026_q2_rows(extract_pdf_text(Q2_PDF), Q2_PDF)
    a, b = monthly_series(q1), monthly_series(q2)
    assert len(a) == 39
    assert len(b) == 42
    assert a[-1]["value"] == 1536
    assert b[38]["value"] == 1533
    assert sum(r["value"] for r in b[-3:]) == 2994
    assert a[-1]["row_hash"] != b[38]["row_hash"]


def _series(values):
    return [{"time_start": f"{2023+i//12}-{i%12+1:02}", "time_end": f"{2023+i//12}-{i%12+1:02}",
             "value": float(v), "metric_id": "new_diagnosed_cases_period", "time_granularity": "monthly",
             "geography": "national", "population": "all", "unit": "people", "measurement_semantics": "flow_count",
             "allowed_use": "direct_target", "observation_role": "direct_target"} for i, v in enumerate(values)]


@pytest.mark.parametrize("role", ["quarantined", "validation_only", "prior_context", "auxiliary_likelihood"])
def test_training_role_fail_closed(role):
    rows = _series([100, 101, 102, 103])
    rows[-1]["observation_role"] = role
    with pytest.raises(ValueError, match="not permitted"):
        monthly_series(rows)


def test_missing_and_duplicate_months_are_not_silently_imputed():
    rows = _series([100, 101, 102, 103])
    with pytest.raises(ValueError, match="contiguous"):
        monthly_series(rows[:1] + rows[2:])
    with pytest.raises(ValueError, match="Duplicate"):
        monthly_series(rows + [rows[0]])


def test_filter_limiting_cases():
    z = np.log1p([100, 120, 150, 180])
    _, _, level, covariance = _filter(z, theta=1.0, drift=0.0)
    assert level == z[-1]
    assert covariance == 0
    _, _, level, covariance = _filter(z, theta=0.0, drift=0.0)
    assert level == pytest.approx(np.mean(z))
    assert covariance == pytest.approx(1 / len(z))


def test_constant_counts_are_exact_and_stochastic_fit_is_finite():
    for family in ("local_level", "local_level_drift"):
        result = forecast([1500.] * 24, family, horizon=12)
        assert result["points"] == pytest.approx([1500.] * 12)
        assert np.isfinite(result["upper95"]).all()
    rng = np.random.default_rng(96)
    values = rng.poisson(np.exp(np.cumsum(rng.normal(0, .03, 36)) + 7)).tolist()
    fitted = fit_local_level(values)
    assert fitted["optimizer_success"]
    assert fitted["process_variance"] >= 0
    assert fitted["measurement_variance"] >= 0
    result = forecast(values, "local_level")
    assert all(0 <= lo <= p <= hi for lo, p, hi in zip(result["lower95"], result["points"], result["upper95"]))


def test_target_mutation_cannot_change_same_origin_prediction_or_selection():
    series = _series([100 + i * 10 for i in range(30)])
    original, original_choices = backtest(series)
    altered = deepcopy(series)
    for row in altered[-3:]:
        row["value"] *= 100
    mutated, mutated_choices = backtest(altered)
    assert [r["prediction"] for r in original] == [r["prediction"] for r in mutated]
    assert [r["family"] for r in original_choices] == [r["family"] for r in mutated_choices]
    assert choose_family(series[:-3]) == choose_family(altered[:-3])
    assert original[-1]["actual"] != mutated[-1]["actual"]


def test_numerical_win_does_not_bypass_vintage_and_prospective_gate():
    history = {"quarter_carry": {"MAE": 2, "p90_absolute_error": 3, "folds": 8},
               "nested_selector": {"MAE": 1, "p90_absolute_error": 2, "folds": 8}}
    comparisons = [{"selected_absolute_error": 1, "carry_absolute_error": 2, "R41_absolute_error": 3}] * 2
    gate = promotion_gate(history, comparisons, {"stock_cone_valid": True})
    assert gate["history_pass"] and gate["challenge_pass"]
    assert gate["champion"] is None
    assert "historical_issue_dates_unverified" in gate["blockers"]


def test_registry_diagnostic_cannot_replace_champion(tmp_path):
    from phase3_dynamic.r53_publication_claim_registry import _monthly_diagnosis_state_claim
    claim = _monthly_diagnosis_state_claim({"gate": {"status": "diagnostic_only", "history_pass": True}}, tmp_path / "report.json")
    assert claim["claim_status"] == "diagnostic_only"
    assert "No R41 replacement" in claim["claim_limit"]
