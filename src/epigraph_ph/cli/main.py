from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epigraph")
    subparsers = parser.add_subparsers(dest="command")

    phase0 = subparsers.add_parser("phase0")
    phase0_sub = phase0.add_subparsers(dest="phase0_command")
    for name in ("harvest", "parse", "extract", "index", "build", "score-sweep", "semantic-benchmark", "literature-review", "merge-shards", "slice-corpus", "pilot-report"):
        cmd = phase0_sub.add_parser(name)
        cmd.add_argument("--run-id", required=True)
        cmd.add_argument("--plugin", default="hiv")
        if name in {"harvest", "build"}:
            cmd.add_argument("--offline", action="store_true")
            cmd.add_argument("--max-results", type=int, default=10)
            cmd.add_argument("--target-records", type=int, default=200)
            cmd.add_argument("--corpus-mode", default="default")
            cmd.add_argument("--relevance-mode", default="auto")
            cmd.add_argument("--download-budget", type=int, default=25)
            cmd.add_argument("--embed-metadata-payload", action="store_true")
            cmd.add_argument("--metadata-only", action="store_true")
            cmd.add_argument("--query-shard-count", type=int, default=1)
            cmd.add_argument("--query-shard-index", type=int, default=0)
        if name in {"parse", "build"}:
            cmd.add_argument("--enable-chart-extraction", action="store_true")
            cmd.add_argument("--enable-ocr-sidecar", action="store_true")
            cmd.add_argument("--ocr-backend", default="auto", choices=["auto", "disabled", "lighton_local", "lighton_vllm"])
            cmd.add_argument("--force-ocr-sidecar", action="store_true")
            cmd.add_argument("--working-set-size", type=int, default=250)
        if name in {"extract", "build"}:
            cmd.add_argument("--skip-live-normalizer", action="store_true")
        if name in {"score-sweep", "build"}:
            cmd.add_argument("--sweep-json-path", default=None)
            cmd.add_argument("--min-domain-quality", type=float, default=0.2)
        if name == "semantic-benchmark":
            cmd.add_argument("--candidate-json-path", default=None)
            cmd.add_argument("--top-k", type=int, default=10)
        if name == "pilot-report":
            cmd.add_argument("--top-k", type=int, default=10)
        if name == "merge-shards":
            cmd.add_argument("--source-run-ids", nargs="+", required=True)
        if name == "slice-corpus":
            cmd.add_argument("--source-run-id", required=True)
            cmd.add_argument("--shard-count", type=int, required=True)
            cmd.add_argument("--shard-index", type=int, required=True)
            cmd.add_argument("--max-documents", type=int, default=None)

    registry = subparsers.add_parser("registry")
    registry_sub = registry.add_subparsers(dest="registry_command")
    registry_build = registry_sub.add_parser("build")
    registry_build.add_argument("--run-id", required=True)
    registry_build.add_argument("--plugin", default="hiv")
    registry_build.add_argument("--profile", default="legacy")

    harp_archive = subparsers.add_parser("harp-archive")
    harp_archive_sub = harp_archive.add_subparsers(dest="harp_archive_command")
    harp_archive_build = harp_archive_sub.add_parser("build")
    harp_archive_build.add_argument("--run-id", required=True)
    harp_archive_build.add_argument("--plugin", default="hiv")
    harp_archive_build.add_argument("--desktop-seed-dir", default=None)
    harp_archive_build.add_argument("--manual-seed-dir", default=None)
    harp_archive_build.add_argument("--force-refresh", action="store_true")
    harp_archive_wdi = harp_archive_sub.add_parser("extract-wdi-hiv")
    harp_archive_wdi.add_argument("--run-id", required=True)
    harp_archive_wdi.add_argument("--plugin", default="hiv")
    harp_archive_wdi.add_argument("--workbook-path", default=None)
    harp_archive_wdi.add_argument("--country-code", default="PHL")
    harp_archive_wdi.add_argument("--country-name", default="Philippines")
    harp_archive_merge_wdi = harp_archive_sub.add_parser("merge-wdi-hiv")
    harp_archive_merge_wdi.add_argument("--run-id", required=True)
    harp_archive_merge_wdi.add_argument("--plugin", default="hiv")
    harp_archive_merge_wdi.add_argument("--archive-run-id", required=True)
    harp_archive_merge_wdi.add_argument("--wdi-run-id", required=True)

    aidsdatahub = subparsers.add_parser("aidsdatahub")
    aidsdatahub_sub = aidsdatahub.add_subparsers(dest="aidsdatahub_command")
    aidsdatahub_extract = aidsdatahub_sub.add_parser("philippines-extract")
    aidsdatahub_extract.add_argument("--run-id", required=True)
    aidsdatahub_extract.add_argument("--plugin", default="hiv")
    aidsdatahub_extract.add_argument("--force-refresh", action="store_true")
    aidsdatahub_extract.add_argument("--max-resources", type=int, default=None)
    aidsdatahub_extract.add_argument("--use-phase0-ocr-fallback", action="store_true")
    aidsdatahub_extract.add_argument("--phase0-ocr-backend", default="auto")
    aidsdatahub_extract.add_argument("--phase0-ocr-max-pages", type=int, default=5)

    for phase_name in ("phase1", "phase15", "phase2", "phase3", "phase1_5"):
        phase = subparsers.add_parser(phase_name)
        phase_sub = phase.add_subparsers(dest=f"{phase_name}_command")
        build = phase_sub.add_parser("build")
        build.add_argument("--run-id", required=True)
        build.add_argument("--plugin", default="hiv")
        build.add_argument("--profile", default="legacy")
        if phase_name == "phase3":
            from epigraph_ph.phase3.frontier.registry import list_cli_names as list_transition_research_cli_names
            from epigraph_ph.phase3.incidence.registry import list_cli_names as list_incidence_cli_names

            build.add_argument("--top-k-per-block", type=int, default=20)
            build.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi", "jax_nuts"])
            incidence_research = phase_sub.add_parser("incidence-research")
            incidence_research_sub = incidence_research.add_subparsers(dest="phase3_incidence_research_command")
            for cli_name in list_incidence_cli_names():
                experiment = incidence_research_sub.add_parser(cli_name)
                experiment.add_argument("--run-id", required=True)
                experiment.add_argument("--plugin", default="hiv")
                experiment.add_argument("--source-run-id", required=True)
            transition_research = phase_sub.add_parser("transition-research")
            transition_research_sub = transition_research.add_subparsers(dest="phase3_transition_research_command")
            for cli_name in list_transition_research_cli_names():
                experiment = transition_research_sub.add_parser(cli_name)
                experiment.add_argument("--run-id", required=True)
                experiment.add_argument("--plugin", default="hiv")
                experiment.add_argument("--source-run-id", required=True)
                experiment.add_argument("--phase3-result-dir-name", default=None)
            transition_report = phase_sub.add_parser("transition-report")
            transition_report_sub = transition_report.add_subparsers(dest="phase3_transition_report_command")
            rolling_origin = transition_report_sub.add_parser("rolling-origin")
            rolling_origin.add_argument("--run-id", required=True)
            rolling_origin.add_argument("--plugin", default="hiv")
            rolling_origin.add_argument("--source-run-id", required=True)
            rolling_origin.add_argument("--start-year", type=int, default=2010)
            rolling_origin.add_argument("--end-year", type=int, default=2025)
            rolling_origin.add_argument("--min-train-years", type=int, default=5)
            rolling_origin.add_argument("--horizon-years", type=int, default=1)
            early_history_partial = transition_report_sub.add_parser("early-history-partial")
            early_history_partial.add_argument("--run-id", required=True)
            early_history_partial.add_argument("--plugin", default="hiv")
            early_history_partial.add_argument("--source-run-id", required=True)
            early_history_partial.add_argument("--start-year", type=int, default=2010)
            early_history_partial.add_argument("--end-year", type=int, default=2016)
            early_history_partial.add_argument("--min-train-years", type=int, default=3)
            early_history_partial.add_argument("--horizon-years", type=int, default=1)
            benchmark_dashboard = transition_report_sub.add_parser("benchmark-dashboard")
            benchmark_dashboard.add_argument("--run-id", required=True)
            benchmark_dashboard.add_argument("--plugin", default="hiv")
            benchmark_dashboard.add_argument("--source-run-id", required=True)
            benchmark_dashboard.add_argument("--rolling-origin-run-id", default=None)
            benchmark_dashboard.add_argument("--early-history-run-id", default=None)
            bridge_quarterly_panel = phase_sub.add_parser("bridge-quarterly-panel")
            bridge_quarterly_panel.add_argument("--run-id", required=True)
            bridge_quarterly_panel.add_argument("--plugin", default="hiv")
            bridge_quarterly_panel.add_argument("--archive-run-id", required=True)
            repair_search = phase_sub.add_parser("repair-search")
            repair_search.add_argument("--run-id", required=True)
            repair_search.add_argument("--plugin", default="hiv")
            repair_search.add_argument("--archive-run-id", default=None)
            repair_search.add_argument(
                "--contracts",
                nargs="+",
                default=["exact_only", "dense_train_observed_score"],
                choices=["exact_only", "dense_train_observed_score"],
            )
            repair_search.add_argument(
                "--primary-contract",
                default="exact_only",
                choices=["exact_only", "dense_train_observed_score"],
            )
            repair_search.add_argument("--quarterly-start-year", type=int, default=2010)
            repair_search.add_argument("--quarterly-end-year", type=int, default=2025)
            repair_search.add_argument("--quarterly-min-train-years", type=int, default=3)
            repair_search.add_argument("--annual-start-year", type=int, default=2010)
            repair_search.add_argument("--annual-end-year", type=int, default=2024)
            repair_search.add_argument("--annual-min-train-years", type=int, default=5)
            repair_search.add_argument("--horizon-years", type=int, default=1)
            champion_forecast = phase_sub.add_parser("champion-forecast")
            champion_forecast.add_argument("--run-id", required=True)
            champion_forecast.add_argument("--plugin", default="hiv")
            champion_forecast.add_argument("--archive-run-id", default=None)
            champion_forecast.add_argument(
                "--contracts",
                nargs="+",
                default=["exact_only", "dense_train_observed_score"],
                choices=["exact_only", "dense_train_observed_score"],
            )
            champion_forecast.add_argument("--quarterly-start-year", type=int, default=2010)
            champion_forecast.add_argument("--quarterly-end-year", type=int, default=2025)
            champion_forecast.add_argument("--quarterly-min-train-years", type=int, default=3)
            champion_forecast.add_argument("--annual-start-year", type=int, default=2010)
            champion_forecast.add_argument("--annual-end-year", type=int, default=2024)
            champion_forecast.add_argument("--annual-min-train-years", type=int, default=5)
            champion_forecast.add_argument("--horizon-years", type=int, default=1)
            champion_forecast.add_argument("--forecast-horizon-quarters", type=int, default=4)
            champion_forecast.add_argument("--interval-alpha", type=float, default=0.1)
            publishability_batch = phase_sub.add_parser("publishability-batch")
            publishability_batch.add_argument("--run-id", required=True)
            publishability_batch.add_argument("--plugin", default="hiv")
            publishability_batch.add_argument("--archive-run-id", default=None)
            publishability_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            publishability_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            publishability_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            publishability_batch.add_argument("--annual-start-year", type=int, default=2010)
            publishability_batch.add_argument("--annual-end-year", type=int, default=2024)
            publishability_batch.add_argument("--annual-min-train-years", type=int, default=5)
            publishability_batch.add_argument("--horizon-years", type=int, default=1)
            publishability_batch.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
            eval_hardening_batch = phase_sub.add_parser("eval-hardening-batch")
            eval_hardening_batch.add_argument("--run-id", required=True)
            eval_hardening_batch.add_argument("--plugin", default="hiv")
            eval_hardening_batch.add_argument("--archive-run-id", default=None)
            eval_hardening_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            eval_hardening_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            eval_hardening_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            eval_hardening_batch.add_argument("--annual-start-year", type=int, default=2010)
            eval_hardening_batch.add_argument("--annual-end-year", type=int, default=2024)
            eval_hardening_batch.add_argument("--annual-min-train-years", type=int, default=5)
            eval_hardening_batch.add_argument("--horizon-years", type=int, default=1)
            eval_hardening_batch.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
            grasp_falsification_batch = phase_sub.add_parser("grasp-falsification-batch")
            grasp_falsification_batch.add_argument("--run-id", required=True)
            grasp_falsification_batch.add_argument("--plugin", default="hiv")
            grasp_falsification_batch.add_argument("--archive-run-id", default=None)
            grasp_falsification_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            grasp_falsification_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            grasp_falsification_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            grasp_falsification_batch.add_argument("--annual-start-year", type=int, default=2010)
            grasp_falsification_batch.add_argument("--annual-end-year", type=int, default=2024)
            grasp_falsification_batch.add_argument("--annual-min-train-years", type=int, default=5)
            grasp_falsification_batch.add_argument("--horizon-years", type=int, default=1)
            grasp_falsification_batch.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
            probabilistic_batch = phase_sub.add_parser("probabilistic-batch")
            probabilistic_batch.add_argument("--run-id", required=True)
            probabilistic_batch.add_argument("--plugin", default="hiv")
            probabilistic_batch.add_argument("--archive-run-id", default=None)
            probabilistic_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            probabilistic_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            probabilistic_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            probabilistic_batch.add_argument("--annual-start-year", type=int, default=2010)
            probabilistic_batch.add_argument("--annual-end-year", type=int, default=2024)
            probabilistic_batch.add_argument("--annual-min-train-years", type=int, default=5)
            probabilistic_batch.add_argument("--horizon-years", type=int, default=1)
            probabilistic_batch.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
            probabilistic_extension_batch = phase_sub.add_parser("probabilistic-extension-batch")
            probabilistic_extension_batch.add_argument("--run-id", required=True)
            probabilistic_extension_batch.add_argument("--plugin", default="hiv")
            probabilistic_extension_batch.add_argument("--archive-run-id", default=None)
            probabilistic_extension_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            probabilistic_extension_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            probabilistic_extension_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            probabilistic_extension_batch.add_argument("--annual-start-year", type=int, default=2010)
            probabilistic_extension_batch.add_argument("--annual-end-year", type=int, default=2024)
            probabilistic_extension_batch.add_argument("--annual-min-train-years", type=int, default=5)
            probabilistic_extension_batch.add_argument("--horizon-years", type=int, default=1)
            probabilistic_extension_batch.add_argument("--lockbox-holdout-years", nargs="+", type=int, default=[2025])
            phase2_structure_batch = phase_sub.add_parser("phase2-structure-batch")
            phase2_structure_batch.add_argument("--run-id", required=True)
            phase2_structure_batch.add_argument("--plugin", default="hiv")
            phase2_structure_batch.add_argument("--archive-run-id", default=None)
            phase2_structure_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            phase2_structure_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            phase2_structure_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            phase2_structure_batch.add_argument("--annual-start-year", type=int, default=2010)
            phase2_structure_batch.add_argument("--annual-end-year", type=int, default=2024)
            phase2_structure_batch.add_argument("--annual-min-train-years", type=int, default=5)
            phase2_structure_batch.add_argument("--horizon-years", type=int, default=1)
            phase2_true_replay_batch = phase_sub.add_parser("phase2-true-replay-batch")
            phase2_true_replay_batch.add_argument("--run-id", required=True)
            phase2_true_replay_batch.add_argument("--plugin", default="hiv")
            phase2_true_replay_batch.add_argument("--source-run-id", required=True)
            phase2_true_replay_batch.add_argument("--archive-run-id", default=None)
            phase2_true_replay_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            phase2_true_replay_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            phase2_true_replay_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            phase2_true_replay_batch.add_argument("--annual-start-year", type=int, default=2010)
            phase2_true_replay_batch.add_argument("--annual-end-year", type=int, default=2024)
            phase2_true_replay_batch.add_argument("--annual-min-train-years", type=int, default=5)
            phase2_true_replay_batch.add_argument("--horizon-years", type=int, default=1)
            monthly_phase2_lane_batch = phase_sub.add_parser("monthly-phase2-lane-batch")
            monthly_phase2_lane_batch.add_argument("--run-id", required=True)
            monthly_phase2_lane_batch.add_argument("--plugin", default="hiv")
            monthly_phase2_lane_batch.add_argument("--source-run-id", default="phase2-replay-source-20260414-s00")
            monthly_phase2_lane_batch.add_argument("--coverage-run-id", default="")
            monthly_phase2_lane_batch.add_argument("--start-month", default="2010-01")
            harp_review_batch = phase_sub.add_parser("harp-review-batch")
            harp_review_batch.add_argument("--run-id", required=True)
            harp_review_batch.add_argument("--plugin", default="hiv")
            harp_review_batch.add_argument("--archive-run-id", default="phase2-replay-source-20260414-s00")
            monthly_edge_audit_batch = phase_sub.add_parser("monthly-edge-audit-batch")
            monthly_edge_audit_batch.add_argument("--run-id", required=True)
            monthly_edge_audit_batch.add_argument("--baseline-run-id", default="tr-v3-monthly-phase2-lane-20260416-s01")
            monthly_edge_audit_batch.add_argument("--candidate-run-id", default="tr-v3-monthly-phase2-lane-20260418-s01")
            phase2_seeded_champion_batch = phase_sub.add_parser("phase2-seeded-champion-batch")
            phase2_seeded_champion_batch.add_argument("--run-id", required=True)
            phase2_seeded_champion_batch.add_argument("--plugin", default="hiv")
            phase2_seeded_champion_batch.add_argument("--archive-run-id", default=None)
            phase2_seeded_champion_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_seeded_champion_batch.add_argument("--quarterly-start-year", type=int, default=2010)
            phase2_seeded_champion_batch.add_argument("--quarterly-end-year", type=int, default=2025)
            phase2_seeded_champion_batch.add_argument("--quarterly-min-train-years", type=int, default=3)
            phase2_seeded_champion_batch.add_argument("--annual-start-year", type=int, default=2010)
            phase2_seeded_champion_batch.add_argument("--annual-end-year", type=int, default=2024)
            phase2_seeded_champion_batch.add_argument("--annual-min-train-years", type=int, default=5)
            phase2_seeded_champion_batch.add_argument("--horizon-years", type=int, default=1)
            phase2_seeded_champion_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_seeded_champion_batch.add_argument("--readout-source-archive-run-id", default=None)
            phase2_seeded_champion_batch.add_argument("--active-block", action="append", default=None)
            phase2_seeded_gate_batch = phase_sub.add_parser("phase2-seeded-gate-batch")
            phase2_seeded_gate_batch.add_argument("--run-id", required=True)
            phase2_seeded_gate_batch.add_argument("--plugin", default="hiv")
            phase2_seeded_gate_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_seeded_gate_batch.add_argument("--baseline-seeded-run-id", default=None)
            phase2_seeded_gate_batch.add_argument("--aligned-archive-run-id", default=None)
            phase2_seeded_gate_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            monthly_loading_sanity_batch = phase_sub.add_parser("monthly-loading-sanity-batch")
            monthly_loading_sanity_batch.add_argument("--run-id", required=True)
            monthly_loading_sanity_batch.add_argument("--plugin", default="hiv")
            monthly_loading_sanity_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_sidecar_ablation_batch = phase_sub.add_parser("phase2-sidecar-ablation-batch")
            phase2_sidecar_ablation_batch.add_argument("--run-id", required=True)
            phase2_sidecar_ablation_batch.add_argument("--plugin", default="hiv")
            phase2_sidecar_ablation_batch.add_argument("--base-monthly-run-id", default=None)
            phase2_sidecar_ablation_batch.add_argument("--baseline-seeded-run-id", default=None)
            phase2_sidecar_ablation_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_sidecar_ablation_batch.add_argument("--exclude-canonical", action="append", default=None)
            phase2_archive_alignment_batch = phase_sub.add_parser("phase2-archive-alignment-batch")
            phase2_archive_alignment_batch.add_argument("--run-id", required=True)
            phase2_archive_alignment_batch.add_argument("--plugin", default="hiv")
            phase2_archive_alignment_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_archive_alignment_batch.add_argument("--legacy-archive-run-id", default=None)
            phase2_archive_alignment_batch.add_argument("--aligned-archive-run-id", default=None)
            phase2_archive_alignment_batch.add_argument("--readout-source-archive-run-id", default=None)
            phase2_archive_alignment_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_archive_alignment_batch.add_argument("--active-block", action="append", default=None)
            phase2_two_block_kernel_batch = phase_sub.add_parser("phase2-two-block-kernel-batch")
            phase2_two_block_kernel_batch.add_argument("--run-id", required=True)
            phase2_two_block_kernel_batch.add_argument("--plugin", default="hiv")
            phase2_two_block_kernel_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_two_block_kernel_batch.add_argument("--legacy-archive-run-id", default=None)
            phase2_two_block_kernel_batch.add_argument("--aligned-archive-run-id", default=None)
            phase2_two_block_kernel_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_two_block_kernel_batch.add_argument("--active-block", action="append", default=None)
            phase2_admissibility_backtest_batch = phase_sub.add_parser("phase2-admissibility-backtest-batch")
            phase2_admissibility_backtest_batch.add_argument("--run-id", required=True)
            phase2_admissibility_backtest_batch.add_argument("--plugin", default="hiv")
            phase2_admissibility_backtest_batch.add_argument("--monthly-phase2-run-id", default=None)
            phase2_admissibility_backtest_batch.add_argument("--legacy-archive-run-id", default=None)
            phase2_admissibility_backtest_batch.add_argument("--aligned-archive-run-id", default=None)
            phase2_admissibility_backtest_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_admissibility_backtest_batch.add_argument("--active-block", action="append", default=None)
            phase2_testing_prevention_rebuild_batch = phase_sub.add_parser("phase2-testing-prevention-rebuild-batch")
            phase2_testing_prevention_rebuild_batch.add_argument("--run-id", required=True)
            phase2_testing_prevention_rebuild_batch.add_argument("--plugin", default="hiv")
            phase2_testing_prevention_rebuild_batch.add_argument("--baseline-two-block-run-id", default=None)
            phase2_testing_prevention_rebuild_batch.add_argument("--base-monthly-run-id", default=None)
            phase2_testing_prevention_rebuild_batch.add_argument("--legacy-archive-run-id", default=None)
            phase2_testing_prevention_rebuild_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_testing_indicator_ablation_batch = phase_sub.add_parser("phase2-testing-indicator-ablation-batch")
            phase2_testing_indicator_ablation_batch.add_argument("--run-id", required=True)
            phase2_testing_indicator_ablation_batch.add_argument("--plugin", default="hiv")
            phase2_testing_indicator_ablation_batch.add_argument("--baseline-two-block-run-id", default=None)
            phase2_testing_indicator_ablation_batch.add_argument("--base-monthly-run-id", default=None)
            phase2_testing_indicator_ablation_batch.add_argument("--legacy-archive-run-id", default=None)
            phase2_testing_indicator_ablation_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            phase2_coverage_indicator_effect_batch = phase_sub.add_parser("phase2-coverage-indicator-effect-batch")
            phase2_coverage_indicator_effect_batch.add_argument("--run-id", required=True)
            phase2_coverage_indicator_effect_batch.add_argument("--plugin", default="hiv")
            phase2_coverage_indicator_effect_batch.add_argument("--baseline-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_coverage_indicator_effect_batch.add_argument("--source-run-id", default="phase2-replay-source-hiv-anchors-20260418-s00")
            phase2_coverage_indicator_effect_batch.add_argument("--coverage-run-id", default="harp-archive-hiv-data-coverage-20260419-s00")
            phase2_coverage_indicator_effect_batch.add_argument("--start-month", default="2010-01")
            phase2_coverage_group_ablation_batch = phase_sub.add_parser("phase2-coverage-group-ablation-batch")
            phase2_coverage_group_ablation_batch.add_argument("--run-id", required=True)
            phase2_coverage_group_ablation_batch.add_argument("--plugin", default="hiv")
            phase2_coverage_group_ablation_batch.add_argument("--baseline-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_coverage_group_ablation_batch.add_argument("--source-run-id", default="phase2-replay-source-hiv-anchors-20260418-s00")
            phase2_coverage_group_ablation_batch.add_argument("--coverage-run-id", default="harp-archive-hiv-data-coverage-20260419-s00")
            phase2_coverage_group_ablation_batch.add_argument("--start-month", default="2010-01")
            phase2_substrate_equivalence_batch = phase_sub.add_parser("phase2-substrate-equivalence-batch")
            phase2_substrate_equivalence_batch.add_argument("--run-id", required=True)
            phase2_substrate_equivalence_batch.add_argument("--baseline-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_substrate_equivalence_batch.add_argument("--candidate-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260419-s03-coverage-flag")
            phase2_champion_equivalence_batch = phase_sub.add_parser("phase2-champion-equivalence-batch")
            phase2_champion_equivalence_batch.add_argument("--run-id", required=True)
            phase2_champion_equivalence_batch.add_argument("--baseline-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_champion_equivalence_batch.add_argument("--candidate-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260419-s03-coverage-flag")
            phase2_champion_testing_demotion_batch = phase_sub.add_parser("phase2-champion-testing-demotion-batch")
            phase2_champion_testing_demotion_batch.add_argument("--run-id", required=True)
            phase2_champion_testing_demotion_batch.add_argument("--baseline-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_champion_testing_demotion_batch.add_argument("--candidate-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260419-s03-coverage-flag")
            phase2_current_champion_non_testing_audit_batch = phase_sub.add_parser("phase2-current-champion-non-testing-audit-batch")
            phase2_current_champion_non_testing_audit_batch.add_argument("--run-id", required=True)
            phase2_current_champion_non_testing_audit_batch.add_argument("--baseline-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260418-s02")
            phase2_current_champion_non_testing_audit_batch.add_argument("--candidate-monthly-run-id", default="tr-v3-monthly-phase2-lane-20260419-s03-coverage-flag")
            phase2_current_champion_non_testing_audit_batch.add_argument("--forecast-horizon-quarters", type=int, default=8)
            current_champion_expanded_harp_compatibility_batch = phase_sub.add_parser("current-champion-expanded-harp-compatibility-batch")
            current_champion_expanded_harp_compatibility_batch.add_argument("--run-id", required=True)
            current_champion_expanded_harp_compatibility_batch.add_argument("--baseline-archive-run-id", default="")
            current_champion_expanded_harp_compatibility_batch.add_argument("--coverage-archive-run-id", default="harp-archive-hiv-data-coverage-20260419-s00")
            current_champion_expanded_harp_compatibility_batch.add_argument("--forecast-horizon-quarters", type=int, default=4)
            current_champion_r10_neighborhood_batch = phase_sub.add_parser("current-champion-r10-neighborhood-batch")
            current_champion_r10_neighborhood_batch.add_argument("--run-id", required=True)
            current_champion_r10_neighborhood_batch.add_argument("--merged-archive-run-id", default="tr-v3-current-champion-expanded-harp-compatibility-20260419-s00")
            current_champion_r10_neighborhood_batch.add_argument("--baseline-archive-run-id", default="harp-archive-wdi-standard-20260412-s19")
            current_champion_exact_support_partition_batch = phase_sub.add_parser("current-champion-exact-support-partition-batch")
            current_champion_exact_support_partition_batch.add_argument("--run-id", required=True)
            current_champion_exact_support_partition_batch.add_argument("--merged-archive-run-id", default="tr-v3-current-champion-expanded-harp-compatibility-20260419-s00")
            current_champion_exact_support_partition_batch.add_argument("--baseline-archive-run-id", default="harp-archive-wdi-standard-20260412-s19")
            indicator_inventory_batch = phase_sub.add_parser("indicator-inventory-batch")
            indicator_inventory_batch.add_argument("--run-id", required=True)
            indicator_inventory_batch.add_argument("--active-monthly-run-id", default=None)
            indicator_inventory_batch.add_argument("--candidate-monthly-run-id", default=None)
            indicator_triage_batch = phase_sub.add_parser("indicator-triage-batch")
            indicator_triage_batch.add_argument("--run-id", required=True)
            indicator_triage_batch.add_argument("--inventory-run-id", default=None)
            indicator_triage_batch.add_argument("--active-monthly-run-id", default=None)
            backtest = phase_sub.add_parser("frozen-backtest")
            backtest.add_argument("--run-id", required=True)
            backtest.add_argument("--plugin", default="hiv")
            backtest.add_argument("--profile", default="hiv_rescue_v2")
            backtest.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi", "jax_nuts"])
            backtest.add_argument("--train-years", nargs="+", type=int, default=None)
            backtest.add_argument("--holdout-years", nargs="+", type=int, default=None)
            tournament_backtest = phase_sub.add_parser("tournament-frozen-backtest")
            tournament_backtest.add_argument("--run-id", required=True)
            tournament_backtest.add_argument("--plugin", default="hiv")
            tournament_backtest.add_argument("--profile", default="hiv_rescue_v2")
            tournament_backtest.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi", "jax_nuts"])
            tournament_backtest.add_argument("--train-years", nargs="+", type=int, default=None)
            tournament_backtest.add_argument("--holdout-years", nargs="+", type=int, default=None)
            tune_backtest = phase_sub.add_parser("tune-frozen-backtest")
            tune_backtest.add_argument("--run-id", required=True)
            tune_backtest.add_argument("--plugin", default="hiv")
            tune_backtest.add_argument("--profile", default="hiv_rescue_v2")
            tune_backtest.add_argument("--phase3-inference", default="torch_map", choices=["torch_map"])
            tune_backtest.add_argument("--train-years", nargs="+", type=int, default=None)
            tune_backtest.add_argument("--holdout-years", nargs="+", type=int, default=None)
            national_reset_table = phase_sub.add_parser("national-reset-observation-table")
            national_reset_table.add_argument("--run-id", required=True)
            national_reset_table.add_argument("--plugin", default="hiv")
            national_reset_table.add_argument("--archive-run-id", default=None)
            national_reset_table.add_argument("--start-quarter", default=None)
            national_reset_baseline = phase_sub.add_parser("national-reset-baseline")
            national_reset_baseline.add_argument("--run-id", required=True)
            national_reset_baseline.add_argument("--plugin", default="hiv")
            national_reset_baseline.add_argument("--archive-run-id", default=None)
            national_reset_baseline.add_argument("--start-quarter", default=None)
            national_reset_baseline.add_argument("--holdout-years", nargs="+", type=int, default=None)
            national_reset_delay_aux = phase_sub.add_parser("national-reset-delay-aux")
            national_reset_delay_aux.add_argument("--run-id", required=True)
            national_reset_delay_aux.add_argument("--plugin", default="hiv")
            national_reset_delay_aux.add_argument("--archive-run-id", default=None)
            national_reset_delay_aux.add_argument("--start-quarter", default=None)
            national_reset_delay_aux.add_argument("--holdout-years", nargs="+", type=int, default=None)
            national_reset_vl = phase_sub.add_parser("national-reset-vl-observation-process")
            national_reset_vl.add_argument("--run-id", required=True)
            national_reset_vl.add_argument("--plugin", default="hiv")
            national_reset_vl.add_argument("--archive-run-id", default=None)
            national_reset_vl.add_argument("--start-quarter", default=None)
            national_reset_vl.add_argument("--holdout-years", nargs="+", type=int, default=None)
            national_reset_deferred = phase_sub.add_parser("national-reset-deferred-complexity-scan")
            national_reset_deferred.add_argument("--run-id", required=True)
            national_reset_deferred.add_argument("--plugin", default="hiv")
            national_reset_deferred.add_argument("--archive-run-id", default=None)
            national_reset_deferred.add_argument("--start-quarter", default=None)
            peak_search = phase_sub.add_parser("peak-search")
            peak_search.add_argument("--run-id", required=True)
            peak_search.add_argument("--plugin", default="hiv")
            peak_search.add_argument("--profile", default="hiv_rescue_v2")
            peak_search.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi", "jax_nuts"])
            peak_search.add_argument("--representation", default="hybrid_temporal_multiscale")
            peak_search.add_argument("--target", required=True)
            peak_search.add_argument("--horizon-months", type=int, default=60)
    phase4 = subparsers.add_parser("phase4")
    phase4_sub = phase4.add_subparsers(dest="phase4_command")
    for name in ("build", "simulate", "optimize"):
        cmd = phase4_sub.add_parser(name)
        cmd.add_argument("--run-id", required=True)
        cmd.add_argument("--plugin", default="hiv")
        cmd.add_argument("--profile", default="legacy")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "phase0":
        from epigraph_ph.phase0 import (
            run_phase0_build,
            run_phase0_extract,
            run_phase0_harvest,
            run_phase0_index,
            run_phase0_literature_review,
            run_phase0_merge_shards,
            run_phase0_parse,
            run_phase0_score_wide_sweep,
            run_phase0_semantic_benchmark,
            run_phase0_slice_corpus,
        )
        from epigraph_ph.validate.phase0_pilot_report import run_phase0_pilot_report

        if args.phase0_command == "harvest":
            run_phase0_harvest(
                run_id=args.run_id,
                plugin_id=args.plugin,
                offline=args.offline,
                max_results=args.max_results,
                target_records=args.target_records,
                corpus_mode=args.corpus_mode,
                relevance_mode=args.relevance_mode,
                download_budget=args.download_budget,
                embed_metadata_payload=args.embed_metadata_payload,
                metadata_only=args.metadata_only,
                query_shard_count=args.query_shard_count,
                query_shard_index=args.query_shard_index,
            )
            return 0
        if args.phase0_command == "score-sweep":
            run_phase0_score_wide_sweep(
                run_id=args.run_id,
                plugin_id=args.plugin,
                sweep_json_path=args.sweep_json_path,
                min_domain_quality=args.min_domain_quality,
            )
            return 0
        if args.phase0_command == "parse":
            run_phase0_parse(
                run_id=args.run_id,
                plugin_id=args.plugin,
                working_set_size=args.working_set_size,
                enable_chart_extraction=args.enable_chart_extraction,
                enable_ocr_sidecar=args.enable_ocr_sidecar,
                ocr_backend=args.ocr_backend,
                force_ocr_sidecar=args.force_ocr_sidecar,
            )
            return 0
        if args.phase0_command == "extract":
            run_phase0_extract(run_id=args.run_id, plugin_id=args.plugin, skip_live_normalizer=args.skip_live_normalizer)
            return 0
        if args.phase0_command == "index":
            run_phase0_index(run_id=args.run_id, plugin_id=args.plugin)
            return 0
        if args.phase0_command == "build":
            run_phase0_build(
                run_id=args.run_id,
                plugin_id=args.plugin,
                offline=args.offline,
                max_results=args.max_results,
                target_records=args.target_records,
                corpus_mode=args.corpus_mode,
                relevance_mode=args.relevance_mode,
                download_budget=args.download_budget,
                embed_metadata_payload=args.embed_metadata_payload,
                metadata_only=args.metadata_only,
                query_shard_count=args.query_shard_count,
                query_shard_index=args.query_shard_index,
                enable_chart_extraction=args.enable_chart_extraction,
                enable_ocr_sidecar=args.enable_ocr_sidecar,
                ocr_backend=args.ocr_backend,
                force_ocr_sidecar=args.force_ocr_sidecar,
                working_set_size=args.working_set_size,
                skip_live_normalizer=args.skip_live_normalizer,
                min_domain_quality=args.min_domain_quality,
            )
            return 0
        if args.phase0_command == "semantic-benchmark":
            run_phase0_semantic_benchmark(
                run_id=args.run_id,
                plugin_id=args.plugin,
                candidate_json_path=args.candidate_json_path,
                top_k=args.top_k,
            )
            return 0
        if args.phase0_command == "literature-review":
            run_phase0_literature_review(run_id=args.run_id, plugin_id=args.plugin)
            return 0
        if args.phase0_command == "pilot-report":
            run_phase0_pilot_report(run_id=args.run_id, plugin_id=args.plugin, top_k=args.top_k)
            return 0
        if args.phase0_command == "merge-shards":
            run_phase0_merge_shards(run_id=args.run_id, plugin_id=args.plugin, source_run_ids=args.source_run_ids)
            return 0
        if args.phase0_command == "slice-corpus":
            run_phase0_slice_corpus(
                run_id=args.run_id,
                plugin_id=args.plugin,
                source_run_id=args.source_run_id,
                shard_count=args.shard_count,
                shard_index=args.shard_index,
                max_documents=args.max_documents,
            )
            return 0
    if args.command == "registry" and args.registry_command == "build":
        from epigraph_ph.registry.sources import build_source_registry
        from epigraph_ph.registry.subparameters import build_subparameter_registry
        from epigraph_ph.runtime import RunContext

        ctx = RunContext.create(run_id=args.run_id, plugin_id=args.plugin)
        registry_dir = Path(ctx.run_dir) / "registry"
        registry_dir.mkdir(parents=True, exist_ok=True)
        build_source_registry(
            plugin_id=args.plugin,
            output_path=registry_dir / "source_registry.json",
            phase0_run_dir=ctx.run_dir,
        )
        build_subparameter_registry(
            plugin_id=args.plugin,
            output_path=registry_dir / "subparameter_registry.json",
            phase0_run_dir=ctx.run_dir,
        )
        return 0
    if args.command == "harp-archive" and args.harp_archive_command == "build":
        from epigraph_ph.harp_archive import run_harp_archive_build

        run_harp_archive_build(
            run_id=args.run_id,
            plugin_id=args.plugin,
            desktop_seed_dir=args.desktop_seed_dir,
            manual_seed_dir=args.manual_seed_dir,
            force_refresh=args.force_refresh,
        )
        return 0
    if args.command == "harp-archive" and args.harp_archive_command == "extract-wdi-hiv":
        from epigraph_ph.harp_archive import run_harp_archive_wdi_hiv_extract

        run_harp_archive_wdi_hiv_extract(
            run_id=args.run_id,
            plugin_id=args.plugin,
            workbook_path=args.workbook_path,
            country_code=args.country_code,
            country_name=args.country_name,
        )
        return 0
    if args.command == "harp-archive" and args.harp_archive_command == "merge-wdi-hiv":
        from epigraph_ph.harp_archive import run_harp_archive_merge_wdi_hiv

        run_harp_archive_merge_wdi_hiv(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            wdi_run_id=args.wdi_run_id,
        )
        return 0
    if args.command == "aidsdatahub" and args.aidsdatahub_command == "philippines-extract":
        from epigraph_ph.aidsdatahub import run_aidsdatahub_philippines_extract

        run_aidsdatahub_philippines_extract(
            run_id=args.run_id,
            plugin_id=args.plugin,
            refresh=args.force_refresh,
            max_resources=args.max_resources,
            use_phase0_ocr_fallback=args.use_phase0_ocr_fallback,
            phase0_ocr_backend=args.phase0_ocr_backend,
            phase0_ocr_max_pages=args.phase0_ocr_max_pages,
        )
        return 0
    if args.command == "phase1" and args.phase1_command == "build":
        from epigraph_ph.phase1 import run_phase1_build

        run_phase1_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase15" and args.phase15_command == "build":
        from epigraph_ph.phase15 import run_phase15_build

        run_phase15_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase1_5" and args.phase1_5_command == "build":
        from epigraph_ph.phase15 import run_phase15_build

        run_phase15_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase2" and args.phase2_command == "build":
        from epigraph_ph.phase2 import run_phase2_build

        run_phase2_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase3" and args.phase3_command == "build":
        from epigraph_ph.phase3 import run_phase3_build

        run_phase3_build(
            run_id=args.run_id,
            plugin_id=args.plugin,
            top_k_per_block=args.top_k_per_block,
            profile=args.profile,
            inference_family=args.phase3_inference,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "transition-research":
        from epigraph_ph.phase3.frontier.cli import run_phase3_transition_research

        run_phase3_transition_research(
            run_id=args.run_id,
            plugin_id=args.plugin,
            source_run_id=args.source_run_id,
            cli_experiment_name=args.phase3_transition_research_command,
            phase3_result_dir_name=args.phase3_result_dir_name,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "transition-report":
        from epigraph_ph.phase3.frontier.cli import run_phase3_transition_report

        run_phase3_transition_report(
            run_id=args.run_id,
            plugin_id=args.plugin,
            source_run_id=args.source_run_id,
            cli_report_name=args.phase3_transition_report_command,
            start_year=getattr(args, "start_year", None),
            end_year=getattr(args, "end_year", None),
            min_train_years=getattr(args, "min_train_years", None),
            horizon_years=getattr(args, "horizon_years", None),
            rolling_origin_run_id=getattr(args, "rolling_origin_run_id", None),
            early_history_run_id=getattr(args, "early_history_run_id", None),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "bridge-quarterly-panel":
        from epigraph_ph.phase3 import run_phase3_bridge_quarterly_panel

        run_phase3_bridge_quarterly_panel(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "repair-search":
        from epigraph_ph.phase3 import run_phase3_repair_search

        run_phase3_repair_search(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            contracts=list(args.contracts),
            primary_contract=args.primary_contract,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "champion-forecast":
        from epigraph_ph.phase3 import run_phase3_champion_forecast

        run_phase3_champion_forecast(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            contracts=list(args.contracts),
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            interval_alpha=args.interval_alpha,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "publishability-batch":
        from epigraph_ph.phase3 import run_phase3_publishability_batch

        run_phase3_publishability_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            lockbox_holdout_years=list(args.lockbox_holdout_years),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "eval-hardening-batch":
        from epigraph_ph.phase3 import run_phase3_eval_hardening_batch

        run_phase3_eval_hardening_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            lockbox_holdout_years=list(args.lockbox_holdout_years),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "grasp-falsification-batch":
        from epigraph_ph.phase3 import run_phase3_grasp_falsification_batch

        run_phase3_grasp_falsification_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            lockbox_holdout_years=list(args.lockbox_holdout_years),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "probabilistic-batch":
        from epigraph_ph.phase3 import run_phase3_probabilistic_batch

        run_phase3_probabilistic_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            lockbox_holdout_years=list(args.lockbox_holdout_years),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "probabilistic-extension-batch":
        from epigraph_ph.phase3 import run_phase3_probabilistic_extension_batch

        run_phase3_probabilistic_extension_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            lockbox_holdout_years=list(args.lockbox_holdout_years),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-structure-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_structure_batch

        run_phase3_phase2_structure_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-true-replay-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_true_replay_batch

        run_phase3_phase2_true_replay_batch(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            archive_run_id=args.archive_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "monthly-phase2-lane-batch":
        from epigraph_ph.phase3 import run_phase3_monthly_phase2_lane_batch

        run_phase3_monthly_phase2_lane_batch(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            coverage_run_id=args.coverage_run_id,
            plugin_id=args.plugin,
            start_month=args.start_month,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "harp-review-batch":
        from epigraph_ph.phase3 import run_phase3_harp_review_batch

        run_phase3_harp_review_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            plugin_id=args.plugin,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "monthly-edge-audit-batch":
        from epigraph_ph.phase3 import run_phase3_monthly_edge_audit_batch

        run_phase3_monthly_edge_audit_batch(
            run_id=args.run_id,
            baseline_run_id=args.baseline_run_id,
            candidate_run_id=args.candidate_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-seeded-champion-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_seeded_champion_batch

        run_phase3_phase2_seeded_champion_batch(
            run_id=args.run_id,
            archive_run_id=args.archive_run_id,
            readout_source_archive_run_id=args.readout_source_archive_run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
            quarterly_start_year=args.quarterly_start_year,
            quarterly_end_year=args.quarterly_end_year,
            quarterly_min_train_years=args.quarterly_min_train_years,
            annual_start_year=args.annual_start_year,
            annual_end_year=args.annual_end_year,
            annual_min_train_years=args.annual_min_train_years,
            horizon_years=args.horizon_years,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-seeded-gate-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_seeded_gate_batch

        run_phase3_phase2_seeded_gate_batch(
            run_id=args.run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
            baseline_seeded_run_id=args.baseline_seeded_run_id,
            aligned_archive_run_id=args.aligned_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "monthly-loading-sanity-batch":
        from epigraph_ph.phase3 import run_phase3_monthly_loading_sanity_batch

        run_phase3_monthly_loading_sanity_batch(
            run_id=args.run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-sidecar-ablation-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_sidecar_ablation_batch

        run_phase3_phase2_sidecar_ablation_batch(
            run_id=args.run_id,
            base_monthly_run_id=args.base_monthly_run_id,
            baseline_seeded_run_id=args.baseline_seeded_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            additional_excluded_canonicals=tuple(str(value) for value in list(args.exclude_canonical or [])),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-archive-alignment-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_archive_alignment_batch

        run_phase3_phase2_archive_alignment_batch(
            run_id=args.run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
            legacy_archive_run_id=args.legacy_archive_run_id,
            aligned_archive_run_id=args.aligned_archive_run_id,
            readout_source_archive_run_id=args.readout_source_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-two-block-kernel-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_two_block_kernel_batch

        run_phase3_phase2_two_block_kernel_batch(
            run_id=args.run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
            legacy_archive_run_id=args.legacy_archive_run_id,
            aligned_archive_run_id=args.aligned_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-admissibility-backtest-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_admissibility_backtest_batch

        run_phase3_phase2_admissibility_backtest_batch(
            run_id=args.run_id,
            monthly_phase2_run_id=args.monthly_phase2_run_id,
            legacy_archive_run_id=args.legacy_archive_run_id,
            aligned_archive_run_id=args.aligned_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
            active_block_subset=tuple(str(value) for value in list(args.active_block or [])),
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-testing-prevention-rebuild-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_testing_prevention_rebuild_batch

        run_phase3_phase2_testing_prevention_rebuild_batch(
            run_id=args.run_id,
            baseline_two_block_run_id=args.baseline_two_block_run_id,
            base_monthly_run_id=args.base_monthly_run_id,
            legacy_archive_run_id=args.legacy_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-testing-indicator-ablation-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_testing_indicator_ablation_batch

        run_phase3_phase2_testing_indicator_ablation_batch(
            run_id=args.run_id,
            baseline_two_block_run_id=args.baseline_two_block_run_id,
            base_monthly_run_id=args.base_monthly_run_id,
            legacy_archive_run_id=args.legacy_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-coverage-indicator-effect-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_coverage_indicator_effect_batch

        run_phase3_phase2_coverage_indicator_effect_batch(
            run_id=args.run_id,
            baseline_run_id=args.baseline_run_id,
            source_run_id=args.source_run_id,
            coverage_run_id=args.coverage_run_id,
            plugin_id=args.plugin,
            start_month=args.start_month,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-coverage-group-ablation-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_coverage_group_ablation_batch

        run_phase3_phase2_coverage_group_ablation_batch(
            run_id=args.run_id,
            baseline_run_id=args.baseline_run_id,
            source_run_id=args.source_run_id,
            coverage_run_id=args.coverage_run_id,
            plugin_id=args.plugin,
            start_month=args.start_month,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-substrate-equivalence-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_substrate_equivalence_batch

        run_phase3_phase2_substrate_equivalence_batch(
            run_id=args.run_id,
            baseline_monthly_run_id=args.baseline_monthly_run_id,
            candidate_monthly_run_id=args.candidate_monthly_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-champion-equivalence-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_champion_equivalence_batch

        run_phase3_phase2_champion_equivalence_batch(
            run_id=args.run_id,
            baseline_monthly_run_id=args.baseline_monthly_run_id,
            candidate_monthly_run_id=args.candidate_monthly_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-champion-testing-demotion-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_champion_testing_demotion_batch

        run_phase3_phase2_champion_testing_demotion_batch(
            run_id=args.run_id,
            baseline_monthly_run_id=args.baseline_monthly_run_id,
            candidate_monthly_run_id=args.candidate_monthly_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "phase2-current-champion-non-testing-audit-batch":
        from epigraph_ph.phase3 import run_phase3_phase2_current_champion_non_testing_audit_batch

        run_phase3_phase2_current_champion_non_testing_audit_batch(
            run_id=args.run_id,
            baseline_monthly_run_id=args.baseline_monthly_run_id,
            candidate_monthly_run_id=args.candidate_monthly_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "current-champion-expanded-harp-compatibility-batch":
        from epigraph_ph.phase3 import run_phase3_current_champion_expanded_harp_compatibility_batch

        run_phase3_current_champion_expanded_harp_compatibility_batch(
            run_id=args.run_id,
            baseline_archive_run_id=args.baseline_archive_run_id or None,
            coverage_archive_run_id=args.coverage_archive_run_id,
            forecast_horizon_quarters=args.forecast_horizon_quarters,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "current-champion-r10-neighborhood-batch":
        from epigraph_ph.phase3 import run_phase3_current_champion_r10_neighborhood_batch

        run_phase3_current_champion_r10_neighborhood_batch(
            run_id=args.run_id,
            merged_archive_run_id=args.merged_archive_run_id,
            baseline_archive_run_id=args.baseline_archive_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "current-champion-exact-support-partition-batch":
        from epigraph_ph.phase3 import run_phase3_current_champion_exact_support_partition_batch

        run_phase3_current_champion_exact_support_partition_batch(
            run_id=args.run_id,
            merged_archive_run_id=args.merged_archive_run_id,
            baseline_archive_run_id=args.baseline_archive_run_id,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "indicator-inventory-batch":
        from epigraph_ph.phase3 import run_phase3_indicator_inventory_batch

        run_phase3_indicator_inventory_batch(
            run_id=args.run_id,
            active_monthly_run_id=args.active_monthly_run_id or "tr-v3-phase2-sidecar-ablation-20260419-s00-monthly",
            candidate_monthly_run_id=args.candidate_monthly_run_id or "tr-v3-phase2-testing-prevention-rebuild-20260419-s01-monthly",
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "indicator-triage-batch":
        from epigraph_ph.phase3 import run_phase3_indicator_triage_batch

        run_phase3_indicator_triage_batch(
            run_id=args.run_id,
            inventory_run_id=args.inventory_run_id or "tr-v3-indicator-inventory-20260419-s00",
            active_monthly_run_id=args.active_monthly_run_id or "tr-v3-phase2-sidecar-ablation-20260419-s00-monthly",
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "incidence-research":
        from epigraph_ph.phase3.incidence.cli import run_phase3_incidence_research

        run_phase3_incidence_research(
            run_id=args.run_id,
            plugin_id=args.plugin,
            source_run_id=args.source_run_id,
            cli_experiment_name=args.phase3_incidence_research_command,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "frozen-backtest":
        from epigraph_ph.phase3 import run_phase3_frozen_backtest

        run_phase3_frozen_backtest(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
            train_years=args.train_years,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "tune-frozen-backtest":
        from epigraph_ph.phase3 import run_phase3_frozen_backtest_tuning

        run_phase3_frozen_backtest_tuning(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
            train_years=args.train_years,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "tournament-frozen-backtest":
        from epigraph_ph.phase3 import run_phase3_frozen_backtest_tournament

        run_phase3_frozen_backtest_tournament(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
            train_years=args.train_years,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "national-reset-observation-table":
        from epigraph_ph.phase3 import run_phase3_national_reset_observation_table

        run_phase3_national_reset_observation_table(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            start_quarter=args.start_quarter,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "national-reset-baseline":
        from epigraph_ph.phase3 import run_phase3_national_reset_baseline

        run_phase3_national_reset_baseline(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            start_quarter=args.start_quarter,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "national-reset-delay-aux":
        from epigraph_ph.phase3 import run_phase3_national_reset_delay_aux

        run_phase3_national_reset_delay_aux(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            start_quarter=args.start_quarter,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "national-reset-vl-observation-process":
        from epigraph_ph.phase3 import run_phase3_national_reset_vl_observation_process

        run_phase3_national_reset_vl_observation_process(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            start_quarter=args.start_quarter,
            holdout_years=args.holdout_years,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "national-reset-deferred-complexity-scan":
        from epigraph_ph.phase3 import run_phase3_national_reset_deferred_complexity_scan

        run_phase3_national_reset_deferred_complexity_scan(
            run_id=args.run_id,
            plugin_id=args.plugin,
            archive_run_id=args.archive_run_id,
            start_quarter=args.start_quarter,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "peak-search":
        from epigraph_ph.phase3 import run_phase3_peak_search

        run_phase3_peak_search(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
            representation=args.representation,
            target=args.target,
            horizon_months=args.horizon_months,
        )
        return 0
    if args.command == "phase4" and args.phase4_command == "build":
        from epigraph_ph.phase4 import run_phase4_build

        run_phase4_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase4" and args.phase4_command == "simulate":
        from epigraph_ph.phase4 import run_phase4_simulate

        run_phase4_simulate(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase4" and args.phase4_command == "optimize":
        from epigraph_ph.phase4 import run_phase4_optimize

        run_phase4_optimize(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

