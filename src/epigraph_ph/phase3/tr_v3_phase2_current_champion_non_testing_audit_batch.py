from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np

from epigraph_ph.phase3 import tr_v3_experiment_suite as suite
from epigraph_ph.phase3 import tr_v3_phase2_admissibility_backtest_batch as prior_audit
from epigraph_ph.phase3 import tr_v3_phase2_champion_equivalence_batch as champion_equivalence
from epigraph_ph.phase3 import tr_v3_phase2_champion_testing_demotion_batch as testing_demotion
from epigraph_ph.phase3 import tr_v3_phase2_seeded_champion_batch as seeded
from epigraph_ph.phase3 import tr_v3_phase2_seeded_gate_batch as seeded_gate
from epigraph_ph.runtime import ROOT_DIR, ensure_dir, read_json, write_json


NON_TESTING_SCENARIOS: tuple[str, ...] = ("disruption_recovery", "mobility_spike")
PLACEBO_SCENARIO_MAP: dict[str, str] = {
    "disruption_recovery": "disruption_recovery_placebo",
    "mobility_spike": "mobility_spike_placebo",
}
PLACEBO_SCENARIOS: tuple[str, ...] = tuple(PLACEBO_SCENARIO_MAP.values())
ALL_REPLAY_SCENARIOS: tuple[str, ...] = NON_TESTING_SCENARIOS + PLACEBO_SCENARIOS


def _current_winner_configs() -> dict[str, dict[str, Any]]:
    return champion_equivalence._current_winner_configs()


def _seeded_payload_from_phase2_state(
    *,
    archive_run_id: str,
    phase2_state: seeded.Phase2QuarterState,
    forecast_horizon_quarters: int,
    winner_configs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    annual_rows = seeded.build_annual_anchor_rows(str(archive_run_id))
    kernel = seeded._fit_structural_kernel(phase2_state.quarter_states, phase2_state.block_axis, phase2_state.edge_rows)
    block_scales = seeded._recent_block_scales(phase2_state.quarter_states)
    last_state = np.asarray(phase2_state.quarter_states[-1, :], dtype=np.float64)
    scenario_names = ("status_quo",) + tuple(NON_TESTING_SCENARIOS)
    scenario_paths = {
        scenario_name: seeded._propagate_structural_states(
            kernel=kernel,
            last_state=last_state,
            steps=int(forecast_horizon_quarters),
            shock_matrix=prior_audit._shock_matrix_for_replay(
                scenario_name=scenario_name,
                steps=int(forecast_horizon_quarters),
                block_axis=phase2_state.block_axis,
                block_scales=block_scales,
            ),
        )
        for scenario_name in scenario_names
    }
    history_feature_matrix = seeded._feature_matrix(phase2_state.quarter_states)
    feature_mean = history_feature_matrix.mean(axis=0)
    feature_std = history_feature_matrix.std(axis=0)
    feature_std = np.where(feature_std > 1e-6, feature_std, 1.0)
    quarter_feature_map = {
        quarter: (history_feature_matrix[idx, :] - feature_mean) / feature_std
        for idx, quarter in enumerate(phase2_state.quarter_axis)
    }
    spec_map = {spec.experiment_id: spec for spec in suite.build_experiment_suite_specs()}
    contract_payloads: dict[str, Any] = {}
    terminal_delta_rows: list[dict[str, Any]] = []
    for contract_name, config in winner_configs.items():
        result = seeded._collect_contract_result(
            archive_run_id=str(archive_run_id),
            contract_name=str(config["suite_contract"]),
            experiment_id=str(config["winner_id"]),
            quarterly_start_year=2010,
            quarterly_end_year=2025,
            quarterly_min_train_years=3,
            annual_start_year=2010,
            annual_end_year=2024,
            annual_min_train_years=5,
            horizon_years=1,
        )
        readout = seeded._fit_residual_readout(
            result=result,
            allowed_tiers=set(config["allowed_tiers"]),
            quarter_feature_map=quarter_feature_map,
            feature_names=seeded._feature_names(phase2_state.block_axis),
        )
        observation_rows, _, _ = seeded.champion_forecast._contract_payload(str(archive_run_id), str(config["forecast_contract"]))
        latest_quarter = max((str(row["quarter"]) for row in observation_rows), key=suite.quarter_sort_key)
        forecast_quarters = seeded.champion_forecast._future_quarters(latest_quarter, int(forecast_horizon_quarters))
        forecast_candidate = seeded.champion_forecast._forecast_future_rows(
            observation_rows,
            annual_rows=annual_rows,
            spec=spec_map[str(config["winner_id"])],
            best_candidate=dict(result["best_candidate"]),
            forecast_quarters=forecast_quarters,
        )
        base_rows = [dict(row) for row in list(forecast_candidate.get("prediction_rows") or [])]
        base_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], scenario_paths["status_quo"]]))[1:, :] - feature_mean) / feature_std
        scenario_rows: dict[str, list[dict[str, Any]]] = {"status_quo": [dict(row) for row in base_rows]}
        for scenario_name in NON_TESTING_SCENARIOS:
            scenario_feature_path = (seeded._feature_matrix(np.vstack([last_state[None, :], scenario_paths[scenario_name]]))[1:, :] - feature_mean) / feature_std
            scenario_rows[scenario_name] = seeded._apply_scenario_to_forecast(
                base_rows=base_rows,
                base_features=base_feature_path,
                scenario_features=scenario_feature_path,
                readout=readout,
            )
        for scenario_name in NON_TESTING_SCENARIOS:
            final_row = dict(scenario_rows[scenario_name][-1])
            terminal_delta_rows.append(
                {
                    "contract": contract_name,
                    "scenario": scenario_name,
                    "diagnosed_plhiv_delta": float(final_row.get("diagnosed_plhiv_scenario_delta") or 0.0),
                    "alive_on_art_delta": float(final_row.get("alive_on_art_scenario_delta") or 0.0),
                    "new_diagnosed_cases_period_delta": float(final_row.get("new_diagnosed_cases_period_scenario_delta") or 0.0),
                }
            )
        contract_payloads[contract_name] = {
            "winner_id": str(config["winner_id"]),
            "readout": readout,
            "base_forecast_rows": base_rows,
            "scenario_rows": scenario_rows,
            "quarterly_mean_mae": float(result["quarterly_summary"]["candidate_mean_mae"]),
        }
    return {
        "archive_run_id": str(archive_run_id),
        "phase2_seed": {
            "block_axis": list(phase2_state.block_axis),
            "quarter_axis": list(phase2_state.quarter_axis),
            "edge_rows": [dict(row) for row in phase2_state.edge_rows],
            "edge_count": int(len(phase2_state.edge_rows)),
        },
        "contracts": contract_payloads,
        "terminal_delta_rows": terminal_delta_rows,
    }


def _archive_alignment_rows_filtered(
    baseline_payload: dict[str, Any],
    aligned_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = seeded_gate._archive_alignment_rows(baseline_payload, aligned_payload)
    return [dict(row) for row in rows if str(row.get("scenario") or "") in set(NON_TESTING_SCENARIOS)]


def _placebo_separation_rows(
    summary_rows: list[dict[str, Any]],
    *,
    winner_configs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {(str(row["contract"]), str(row["scenario"])): dict(row) for row in summary_rows}
    rows: list[dict[str, Any]] = []
    for real_scenario, placebo_scenario in PLACEBO_SCENARIO_MAP.items():
        for contract_name in winner_configs:
            real_row = dict(lookup.get((contract_name, real_scenario)) or {})
            placebo_row = dict(lookup.get((contract_name, placebo_scenario)) or {})
            if not real_row or not placebo_row:
                continue
            mae_margin = float(placebo_row["mean_mae_delta"]) - float(real_row["mean_mae_delta"])
            directional_margin = float(real_row["mean_directional_fit_rate"]) - float(placebo_row["mean_directional_fit_rate"])
            rows.append(
                {
                    "contract": contract_name,
                    "scenario": real_scenario,
                    "placebo_scenario": placebo_scenario,
                    "real_mean_mae_delta": float(real_row["mean_mae_delta"]),
                    "placebo_mean_mae_delta": float(placebo_row["mean_mae_delta"]),
                    "mae_margin_vs_placebo": mae_margin,
                    "real_directional_fit_rate": float(real_row["mean_directional_fit_rate"]),
                    "placebo_directional_fit_rate": float(placebo_row["mean_directional_fit_rate"]),
                    "directional_margin_vs_placebo": directional_margin,
                    "separation_keep": bool(mae_margin > 0.0 and directional_margin > 0.0),
                }
            )
    return rows


def _markdown_report(payload: dict[str, Any]) -> str:
    orth_gate = dict(payload.get("orthogonalized_gate") or {})
    lines = [
        "# TR-V3 Phase 2 Current-Champion Non-Testing Audit Batch",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Demotion run: `{payload['demotion_run_id']}`",
        f"- Demoted monthly run: `{payload['demoted_monthly_run_id']}`",
        f"- Exact champion: `{payload['current_champions']['exact_only']}`",
        f"- Dense champion: `{payload['current_champions']['purged_dense']}`",
        "",
        "## Orthogonalized-state admissibility",
        "",
        f"- Active outcome-loaded block count: `{int(orth_gate.get('active_outcome_loaded_block_count') or 0)}`",
        f"- Sign agreement: `{float(orth_gate.get('sign_agreement_rate') or 0.0):.3f}`",
        f"- Mean abs delta-of-delta: `{float(orth_gate.get('mean_abs_delta_of_delta') or 0.0):.3f}`",
        f"- Mean abs ratio: `{float(orth_gate.get('mean_abs_ratio') or 0.0):.3f}`",
        f"- Decision: `{orth_gate.get('decision')}`",
        "",
        "## Blocked replay summary",
        "",
        "| Contract | Scenario | Splits | Mean MAE delta | Directional fit | Improved fraction | Keep |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in list(payload.get("replay_summary_rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | `{int(row['split_count'])}` | "
            f"`{float(row['mean_mae_delta']):.3f}` | `{float(row['mean_directional_fit_rate']):.3f}` | "
            f"`{float(row['mean_improved_fraction']):.3f}` | `{bool(row['replay_keep'])}` |"
        )
    lines.extend(
        [
            "",
            "## Placebo separation",
            "",
            "| Contract | Scenario | Placebo | MAE margin | Directional margin | Keep |",
            "|---|---|---|---:|---:|---|",
        ]
    )
    for row in list(payload.get("placebo_separation_rows") or []):
        lines.append(
            f"| {row['contract']} | {row['scenario']} | {row['placebo_scenario']} | "
            f"`{float(row['mae_margin_vs_placebo']):.3f}` | `{float(row['directional_margin_vs_placebo']):.3f}` | "
            f"`{bool(row['separation_keep'])}` |"
        )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- Replay gate: `{payload['gate_summary']['replay_status']}`",
            f"- Placebo gate: `{payload['gate_summary']['placebo_status']}`",
            f"- Overall: `{payload['gate_summary']['overall_decision']}`",
            "",
            "## Graphs",
            "",
        ]
    )
    for key, value in dict(payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines)


def run_tr_v3_phase2_current_champion_non_testing_audit_batch(
    *,
    run_id: str,
    baseline_monthly_run_id: str = champion_equivalence.DEFAULT_BASELINE_RUN_ID,
    candidate_monthly_run_id: str = champion_equivalence.DEFAULT_CANDIDATE_RUN_ID,
    forecast_horizon_quarters: int = 8,
) -> dict[str, Any]:
    demotion_run_id = f"{run_id}-demotion"
    demotion_payload = testing_demotion.run_tr_v3_phase2_champion_testing_demotion_batch(
        run_id=demotion_run_id,
        baseline_monthly_run_id=baseline_monthly_run_id,
        candidate_monthly_run_id=candidate_monthly_run_id,
    )
    demoted_monthly_run_id = str(demotion_payload["demoted_monthly_run_id"])
    current_winner_configs = _current_winner_configs()

    phase2_state = seeded._load_phase2_quarter_state(demoted_monthly_run_id)
    active_blocks = tuple(str(name) for name in list(phase2_state.block_axis))
    loading_rows = seeded_gate._load_block_loading_rows(demoted_monthly_run_id)
    outcome_loading_rows = seeded_gate._summarize_outcome_circularity_rows(
        loading_rows=loading_rows,
        outcome_canonical_names=set(seeded_gate.OUTCOME_CANONICALS),
    )
    active_outcome_rows = [
        dict(row)
        for row in outcome_loading_rows
        if str(row.get("block_id") or "") in set(active_blocks)
    ]
    orthogonalized_state, orthogonalization_rows = seeded_gate._orthogonalize_phase2_state_against_outcomes(
        phase2_state=phase2_state,
        archive_run_id=demoted_monthly_run_id,
        loading_rows=loading_rows,
    )
    baseline_payload = _seeded_payload_from_phase2_state(
        archive_run_id=demoted_monthly_run_id,
        phase2_state=phase2_state,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        winner_configs=current_winner_configs,
    )
    orthogonalized_payload = _seeded_payload_from_phase2_state(
        archive_run_id=demoted_monthly_run_id,
        phase2_state=orthogonalized_state,
        forecast_horizon_quarters=int(forecast_horizon_quarters),
        winner_configs=current_winner_configs,
    )
    orthogonalized_alignment_rows = _archive_alignment_rows_filtered(baseline_payload, orthogonalized_payload)
    sign_agreement = (
        float(np.mean([1.0 if bool(row["sign_agrees"]) else 0.0 for row in orthogonalized_alignment_rows]))
        if orthogonalized_alignment_rows
        else 0.0
    )
    mean_abs_delta = (
        float(np.mean([abs(float(row["delta_of_delta"])) for row in orthogonalized_alignment_rows]))
        if orthogonalized_alignment_rows
        else 0.0
    )
    abs_ratios = [
        abs(float(row["aligned_terminal_delta"])) / max(abs(float(row["baseline_terminal_delta"])), 1.0)
        for row in orthogonalized_alignment_rows
    ]
    mean_abs_ratio = float(np.mean(abs_ratios)) if abs_ratios else 0.0
    orth_gate = {
        "active_outcome_loaded_block_count": int(sum(1 for row in active_outcome_rows if float(row.get("outcome_loading_share") or 0.0) > 1e-9)),
        "sign_agreement_rate": sign_agreement,
        "mean_abs_delta_of_delta": mean_abs_delta,
        "mean_abs_ratio": mean_abs_ratio,
        "decision": "keep" if sign_agreement >= 0.75 and 0.4 <= mean_abs_ratio <= 2.0 else "revert",
        "loading_rows": active_outcome_rows,
        "orthogonalization_rows": orthogonalization_rows,
        "alignment_rows": orthogonalized_alignment_rows,
    }

    replay_rows: list[dict[str, Any]] = []
    for contract_name, config in current_winner_configs.items():
        replay_rows.extend(
            prior_audit._replay_contract_scenarios(
                archive_run_id=demoted_monthly_run_id,
                contract_name=contract_name,
                config=config,
                phase2_state=phase2_state,
                scenarios=ALL_REPLAY_SCENARIOS,
            )
        )
    replay_summary_rows = prior_audit._summarize_replay_rows(replay_rows)
    placebo_rows = _placebo_separation_rows(replay_summary_rows, winner_configs=current_winner_configs)
    real_summary_rows = [row for row in replay_summary_rows if str(row["scenario"]) in set(NON_TESTING_SCENARIOS)]
    replay_status = (
        "keep"
        if real_summary_rows and all(bool(row["replay_keep"]) for row in real_summary_rows)
        else "revert_replay_instability"
    )
    placebo_status = (
        "keep"
        if placebo_rows and all(bool(row["separation_keep"]) for row in placebo_rows)
        else "revert_placebo_nonseparation"
    )
    overall_decision = (
        "keep_non_testing_sidecar"
        if str(demotion_payload.get("decision") or "") == "keep_testing_as_measurement_sidecar"
        and orth_gate["decision"] == "keep"
        and replay_status == "keep"
        and placebo_status == "keep"
        else "revert_to_champion_only_plus_measurement_sidecars"
    )

    analysis_dir = ensure_dir(ROOT_DIR / "artifacts" / "runs" / run_id / "analysis")
    orth_graph = analysis_dir / "non_testing_orthogonalized_alignment_compare.png"
    prior_audit._plot_orthogonalized_alignment(orthogonalized_alignment_rows, orth_graph)
    replay_mae_graph = analysis_dir / "non_testing_replay_mae_delta_compare.png"
    prior_audit._plot_replay_mae_delta(replay_summary_rows, replay_mae_graph)
    replay_directional_graph = analysis_dir / "non_testing_replay_directional_fit_heatmap.png"
    prior_audit._plot_directional_fit_heatmap(replay_summary_rows, replay_directional_graph)
    placebo_graph = analysis_dir / "non_testing_placebo_separation_heatmap.png"
    prior_audit._plot_placebo_separation(placebo_rows, placebo_graph)
    replay_split_graph = analysis_dir / "non_testing_replay_split_heatmap.png"
    prior_audit._plot_replay_split_heatmap(replay_rows, replay_split_graph)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "demotion_run_id": str(demotion_run_id),
        "demoted_monthly_run_id": str(demoted_monthly_run_id),
        "current_champions": {name: str(config["winner_id"]) for name, config in current_winner_configs.items()},
        "active_block_subset": list(active_blocks),
        "active_scenarios": list(NON_TESTING_SCENARIOS),
        "placebo_scenarios": list(PLACEBO_SCENARIOS),
        "orthogonalized_gate": orth_gate,
        "replay_rows": replay_rows,
        "replay_summary_rows": replay_summary_rows,
        "placebo_separation_rows": placebo_rows,
        "gate_summary": {
            "replay_status": replay_status,
            "placebo_status": placebo_status,
            "overall_decision": overall_decision,
        },
        "artifacts": {
            "demotion_report": str(Path(demotion_run_id) / "analysis" / "tr_v3_phase2_champion_testing_demotion_batch_report.md"),
            "orthogonalized_alignment_graph": orth_graph.name,
            "replay_mae_delta_graph": replay_mae_graph.name,
            "replay_directional_fit_graph": replay_directional_graph.name,
            "placebo_separation_graph": placebo_graph.name,
            "replay_split_graph": replay_split_graph.name,
        },
    }
    write_json(analysis_dir / "tr_v3_phase2_current_champion_non_testing_audit_batch_report.json", payload)
    (analysis_dir / "tr_v3_phase2_current_champion_non_testing_audit_batch_report.md").write_text(_markdown_report(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the remaining non-testing Phase 2 sidecar under the current predictive champions.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-monthly-run-id", default=champion_equivalence.DEFAULT_BASELINE_RUN_ID)
    parser.add_argument("--candidate-monthly-run-id", default=champion_equivalence.DEFAULT_CANDIDATE_RUN_ID)
    parser.add_argument("--forecast-horizon-quarters", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_tr_v3_phase2_current_champion_non_testing_audit_batch(
        run_id=str(args.run_id),
        baseline_monthly_run_id=str(args.baseline_monthly_run_id),
        candidate_monthly_run_id=str(args.candidate_monthly_run_id),
        forecast_horizon_quarters=int(args.forecast_horizon_quarters),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
