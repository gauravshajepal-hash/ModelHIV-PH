from __future__ import annotations

from types import SimpleNamespace

from phase3_dynamic.monthly_joint_observation import (
    MonthlyJointContext,
    _month_to_quarter,
    _quarter_alignment_features,
    build_quarterly_joint_observation_features,
)
from phase3_dynamic.monthly_shock import _month_index


def _monthly_value(month: str, **values: float) -> tuple[int, dict[str, float]]:
    return _month_index(month), dict(values)


def _dataset() -> SimpleNamespace:
    return SimpleNamespace(
        train_rows=[
            {
                "quarter": "2020-Q1",
                "diagnosed_plhiv": 100.0,
                "alive_on_art": 70.0,
                "new_diagnosed_cases_period": 30.0,
                "deaths_reported_period": 3.0,
            }
        ],
        train_transition_rows=[
            {
                "quarter": "2020-Q1",
                "flows": {"U_to_D": 30.0},
                "stock_balance": {"attrition_outflow": 3.0},
            }
        ],
    )


def test_month_to_quarter_mapping_handles_year_boundaries() -> None:
    assert _month_to_quarter(_month_index("2020-01")) == "2020-Q1"
    assert _month_to_quarter(_month_index("2020-12")) == "2020-Q4"


def test_quarter_alignment_uses_monthly_and_quarterly_cascade_evidence() -> None:
    monthly_values = dict(
        [
            _monthly_value("2020-01", new_diagnosed_cases_period=5.0, diagnosed_plhiv=80.0, alive_on_art=60.0, deaths_reported_period=1.0),
            _monthly_value("2020-02", new_diagnosed_cases_period=5.0, diagnosed_plhiv=90.0, alive_on_art=65.0, deaths_reported_period=1.0),
            _monthly_value("2020-03", new_diagnosed_cases_period=5.0, diagnosed_plhiv=95.0, alive_on_art=68.0, deaths_reported_period=1.0),
        ]
    )

    features = _quarter_alignment_features(dataset=_dataset(), monthly_values=monthly_values)

    assert "2020-Q1" in features
    assert features["2020-Q1"]["monthly_quarter_support"] == 1.0
    assert "diagnosis_backlog_gap" in features["2020-Q1"]


def test_joint_feature_builder_is_train_origin_safe(monkeypatch) -> None:
    from phase3_dynamic import monthly_joint_observation as mjo

    rows = [
        {"metric_name": "new_diagnosed_cases_period", "value": 10.0, "_month_index": _month_index("2020-01")},
        {"metric_name": "new_diagnosed_cases_period", "value": 10000.0, "_month_index": _month_index("2020-04")},
    ]
    monkeypatch.setattr(mjo, "load_monthly_signal_rows", lambda _context: rows)
    payload = build_quarterly_joint_observation_features(
        MonthlyJointContext(epigraph_root="/tmp", source_run_id="synthetic"),
        dataset=_dataset(),
        train_end_quarter="2020-Q1",
        quarters=["2020-Q1", "2020-Q2"],
    )

    assert payload["train_end_month"] == "2020-03"
    assert payload["quarter_features"]["2020-Q2"]["reporting_intensity"] == 0.0
