from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phase3_dynamic import backhalf_channels
from phase3_dynamic.backhalf_channels import (
    BackHalfChannelContext,
    apply_backhalf_channel_adjustment_to_paths,
    build_backhalf_channel_features,
)


def _monthly_row(month: int, metric_name: str, value: float) -> dict[str, object]:
    return {
        "_month_index": month,
        "metric_name": metric_name,
        "value": value,
        "source_quality_tier": "official",
        "measurement_class": "program_observed",
        "series_kind": "monthly_snapshot",
        "source_id": "synthetic",
    }


def test_backhalf_channel_features_are_train_origin_safe(monkeypatch) -> None:
    rows = [
        _monthly_row(2019 * 12 + 11, "alive_on_art", 65.0),
        _monthly_row(2020 * 12 + 0, "diagnosed_plhiv", 100.0),
        _monthly_row(2020 * 12 + 0, "alive_on_art", 70.0),
        _monthly_row(2020 * 12 + 0, "tested_for_viral_load", 35.0),
        _monthly_row(2020 * 12 + 0, "virally_suppressed", 20.0),
        _monthly_row(2020 * 12 + 1, "diagnosed_plhiv", 110.0),
        _monthly_row(2020 * 12 + 1, "alive_on_art", 75.0),
        _monthly_row(2020 * 12 + 1, "tested_for_viral_load", 40.0),
        _monthly_row(2020 * 12 + 1, "virally_suppressed", 25.0),
        _monthly_row(2020 * 12 + 2, "diagnosed_plhiv", 120.0),
        _monthly_row(2020 * 12 + 2, "alive_on_art", 80.0),
        _monthly_row(2020 * 12 + 2, "tested_for_viral_load", 45.0),
        _monthly_row(2020 * 12 + 2, "virally_suppressed", 30.0),
        _monthly_row(2020 * 12 + 3, "alive_on_art", 9999.0),
        _monthly_row(2020 * 12 + 3, "virally_suppressed", 9999.0),
    ]
    monkeypatch.setattr(backhalf_channels, "load_monthly_signal_rows", lambda _context: rows)

    payload = build_backhalf_channel_features(
        BackHalfChannelContext(Path("/tmp"), "synthetic"),
        train_end_quarter="2020-Q1",
        quarters=["2020-Q1", "2020-Q2"],
    )

    assert payload["schema_version"] == backhalf_channels.BACKHALF_CHANNEL_SCHEMA_VERSION
    assert payload["train_end_month"] == "2020-03"
    assert payload["monthly_row_count"] == len(rows)
    assert payload["quarter_features"]["2020-Q2"]["backhalf_support"] <= payload["quarter_features"]["2020-Q1"]["backhalf_support"]
    assert "A_to_T" in payload["transition_feature_map"]
    assert "T_to_V" in payload["transition_feature_map"]
    assert payload["art_retention_evidence_summary"]["support_class_counts"]["art_stock_balance_proxy"] >= 1


def test_backhalf_channel_features_use_direct_art_process_evidence(monkeypatch) -> None:
    rows = [
        _monthly_row(2023 * 12 + 0, "alive_on_art", 100.0),
        _monthly_row(2022 * 12 + 11, "art_ltfu_cumulative", 10.0),
        _monthly_row(2023 * 12 + 0, "diagnosed_plhiv", 160.0),
        _monthly_row(2023 * 12 + 0, "art_ltfu_cumulative", 14.0),
        _monthly_row(2023 * 12 + 0, "art_deaths_cumulative", 2.0),
        _monthly_row(2023 * 12 + 1, "alive_on_art", 110.0),
        _monthly_row(2023 * 12 + 1, "newly_enrolled_to_treatment", 20.0),
        _monthly_row(2023 * 12 + 1, "art_ltfu_period", 6.0),
        _monthly_row(2023 * 12 + 1, "art_reengagement_period", 3.0),
    ]
    monkeypatch.setattr(backhalf_channels, "load_monthly_signal_rows", lambda _context: rows)

    payload = build_backhalf_channel_features(
        BackHalfChannelContext(Path("/tmp"), "synthetic"),
        train_end_quarter="2023-Q1",
        quarters=["2023-Q1"],
    )

    evidence = payload["art_retention_evidence"]["2023-Q1"]
    features = payload["quarter_features"]["2023-Q1"]
    assert evidence["support_class"] == "direct_process_observed"
    assert evidence["interruption_count"] == 6.0
    assert evidence["reengagement_count"] == 3.0
    assert "art_interruption_evidence_pressure" in payload["transition_feature_map"]["A_to_L"]
    assert "art_reengagement_evidence_pressure" in payload["transition_feature_map"]["L_to_R"]
    assert "latent_reengagement_balance_pressure" in payload["transition_feature_map"]["L_to_R"]
    assert features["art_retention_support"] > 0.0
    assert payload["art_retention_evidence_summary"]["direct_reengagement_quarter_count"] == 1


def test_backhalf_channel_features_label_reengagement_as_proxy_when_restart_rows_absent(monkeypatch) -> None:
    rows = [
        _monthly_row(2022 * 12 + 11, "alive_on_art", 100.0),
        _monthly_row(2022 * 12 + 11, "art_ltfu_cumulative", 50.0),
        _monthly_row(2023 * 12 + 0, "diagnosed_plhiv", 170.0),
        _monthly_row(2023 * 12 + 0, "alive_on_art", 115.0),
        _monthly_row(2023 * 12 + 0, "newly_enrolled_to_treatment", 20.0),
        _monthly_row(2023 * 12 + 0, "art_ltfu_cumulative", 60.0),
    ]
    monkeypatch.setattr(backhalf_channels, "load_monthly_signal_rows", lambda _context: rows)

    payload = build_backhalf_channel_features(
        BackHalfChannelContext(Path("/tmp"), "synthetic"),
        train_end_quarter="2023-Q1",
        quarters=["2023-Q1"],
    )

    evidence = payload["art_retention_evidence"]["2023-Q1"]
    features = payload["quarter_features"]["2023-Q1"]
    assert evidence["latent_reengagement_proxy_count"] == 5.0
    assert evidence["direct_reengagement_count"] == 0.0
    assert evidence["reengagement_evidence"]["publishable_process_claim"] is False
    assert payload["art_retention_evidence_summary"]["direct_reengagement_quarter_count"] == 0
    assert payload["art_retention_evidence_summary"]["proxy_reengagement_quarter_count"] == 1
    assert features["latent_reengagement_balance_pressure"] != 0.0


def test_backhalf_channel_features_apply_reengagement_sensitivity_mode(monkeypatch) -> None:
    rows = [
        _monthly_row(2022 * 12 + 11, "alive_on_art", 100.0),
        _monthly_row(2022 * 12 + 11, "art_ltfu_cumulative", 50.0),
        _monthly_row(2023 * 12 + 0, "diagnosed_plhiv", 170.0),
        _monthly_row(2023 * 12 + 0, "alive_on_art", 115.0),
        _monthly_row(2023 * 12 + 0, "newly_enrolled_to_treatment", 20.0),
        _monthly_row(2023 * 12 + 0, "art_ltfu_cumulative", 60.0),
    ]
    monkeypatch.setattr(backhalf_channels, "load_monthly_signal_rows", lambda _context: rows)

    zero_payload = build_backhalf_channel_features(
        BackHalfChannelContext(Path("/tmp"), "synthetic", reengagement_sensitivity_mode="zero"),
        train_end_quarter="2023-Q1",
        quarters=["2023-Q1"],
    )
    upper_payload = build_backhalf_channel_features(
        BackHalfChannelContext(Path("/tmp"), "synthetic", reengagement_sensitivity_mode="upper_bound_proxy"),
        train_end_quarter="2023-Q1",
        quarters=["2023-Q1"],
    )

    assert zero_payload["art_retention_evidence"]["2023-Q1"]["reengagement_count"] == 0.0
    assert upper_payload["art_retention_evidence"]["2023-Q1"]["reengagement_count"] == 60.0
    assert zero_payload["art_retention_evidence_summary"]["zero_sensitivity_reengagement_quarter_count"] == 1
    assert upper_payload["art_retention_evidence_summary"]["upper_bound_reengagement_quarter_count"] == 1


def test_backhalf_adjustment_changes_backhalf_hazards_only(monkeypatch) -> None:
    def fake_features(_context, *, train_end_quarter: str, quarters: list[str]) -> dict[str, object]:
        return {
            "schema_version": backhalf_channels.BACKHALF_CHANNEL_SCHEMA_VERSION,
            "feature_names": ["art_initiation_pressure"],
            "transition_feature_map": {
                "D_to_A": ["art_initiation_pressure"],
                "A_to_T": ["art_initiation_pressure"],
                "T_to_V": ["art_initiation_pressure"],
                "A_to_L": ["art_initiation_pressure"],
                "T_to_L": ["art_initiation_pressure"],
                "V_to_L": ["art_initiation_pressure"],
                "L_to_R": ["art_initiation_pressure"],
            },
            "quarter_features": {quarter: {"art_initiation_pressure": 1.0} for quarter in quarters},
            "train_end_quarter": train_end_quarter,
        }

    monkeypatch.setattr(backhalf_channels, "build_backhalf_channel_features", fake_features)
    dataset = SimpleNamespace(
        eps=1e-6,
        train_transition_rows=[
            {"quarter": "2020-Q1", "hazards": {"U_to_D": 0.1, "D_to_A": 0.2, "A_to_T": 0.2, "T_to_V": 0.2, "A_to_L": 0.1, "T_to_L": 0.1, "V_to_L": 0.1, "L_to_R": 0.1}},
            {"quarter": "2020-Q2", "hazards": {"U_to_D": 0.1, "D_to_A": 0.7, "A_to_T": 0.7, "T_to_V": 0.7, "A_to_L": 0.5, "T_to_L": 0.5, "V_to_L": 0.5, "L_to_R": 0.5}},
        ],
        holdout_rows=[{"quarter": "2020-Q3"}],
    )
    hazard_paths = {
        "train_hazard_map": {
            "2020-Q1": {"U_to_D": 0.1, "D_to_A": 0.2, "A_to_T": 0.2, "T_to_V": 0.2, "A_to_L": 0.1, "T_to_L": 0.1, "V_to_L": 0.1, "L_to_R": 0.1},
            "2020-Q2": {"U_to_D": 0.1, "D_to_A": 0.2, "A_to_T": 0.2, "T_to_V": 0.2, "A_to_L": 0.1, "T_to_L": 0.1, "V_to_L": 0.1, "L_to_R": 0.1},
        },
        "holdout_hazard_map": {
            "2020-Q3": {"U_to_D": 0.1, "D_to_A": 0.2, "A_to_T": 0.2, "T_to_V": 0.2, "A_to_L": 0.1, "T_to_L": 0.1, "V_to_L": 0.1, "L_to_R": 0.1},
        },
    }

    adjusted = apply_backhalf_channel_adjustment_to_paths(
        context=BackHalfChannelContext(Path("/tmp"), "synthetic"),
        dataset=dataset,
        hazard_paths=hazard_paths,
        incidence_paths={},
    )

    holdout = adjusted["hazard_paths"]["holdout_hazard_map"]["2020-Q3"]
    assert holdout["U_to_D"] == 0.1
    assert holdout["D_to_A"] != 0.2
    assert holdout["A_to_T"] != 0.2
    assert holdout["T_to_V"] != 0.2
    assert holdout["A_to_L"] != 0.1
    assert holdout["L_to_R"] != 0.1
    assert adjusted["diagnostics"]["targets"]["D_to_A"]["channel_role"] == "ART initiation"
