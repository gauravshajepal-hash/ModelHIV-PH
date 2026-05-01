from __future__ import annotations

import argparse
from pathlib import Path

from .data import default_epigraph_root, sandbox_repo_root
from .experiment_tracker import write_experiment_registry
from .failure_anatomy import write_failure_anatomy_report
from .loop import run_tr_v3_00_loop, run_tr_v3_01_loop, run_tr_v3_02_loop, run_tr_v3_02b_loop, run_tr_v3_03_loop, run_tr_v3_04_loop, run_tr_v3_04b_loop, run_tr_v3_04b_early_history_partial, run_tr_v3_04c_loop, run_tr_v3_04c_early_history_partial, run_tr_v3_04d_loop, run_tr_v3_04d_early_history_partial
from .lineage import write_project_lineage_manifest
from .revision_program import (
    run_publication_revision_program,
    run_rev_00_observation_role_ledger,
    run_rev_01_conserved_national_ssm,
    run_rev_02_diagnosis_delay,
    run_rev_03_art_ltfu_vl,
    run_rev_04_prep_persistence,
    run_rev_05_determinant_shrinkage,
    run_rev_06_kp_lite_overlay,
    run_rev_07_region_hierarchy,
)
from .r11_sparse_state_space import run_r11_first_batch, run_r12_reference_branch
from .r13_priority_experiments import run_r13_priority_experiment_queue
from .scientific_contracts import write_project_contract_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(prog="phase3-dynamic")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _common_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--run-id", required=True)
        target.add_argument("--source-run-id", default="smoke-latent-blocks")
        target.add_argument("--epigraph-root")
        target.add_argument("--start-year", type=int, default=2017)
        target.add_argument("--end-year", type=int, default=2025)
        target.add_argument("--min-train-years", type=int, default=5)
        target.add_argument("--horizon-years", type=int, default=1)

    _common_args(subparsers.add_parser("tr-v3-00-loop"))
    _common_args(subparsers.add_parser("tr-v3-01-loop"))
    _common_args(subparsers.add_parser("tr-v3-02-loop"))
    _common_args(subparsers.add_parser("tr-v3-02b-loop"))
    _common_args(subparsers.add_parser("tr-v3-03-loop"))
    _common_args(subparsers.add_parser("tr-v3-04-loop"))
    _common_args(subparsers.add_parser("tr-v3-04b-loop"))
    _common_args(subparsers.add_parser("tr-v3-04b-early-history-partial"))
    _common_args(subparsers.add_parser("tr-v3-04c-loop"))
    _common_args(subparsers.add_parser("tr-v3-04c-early-history-partial"))
    _common_args(subparsers.add_parser("tr-v3-04d-loop"))
    _common_args(subparsers.add_parser("tr-v3-04d-early-history-partial"))
    _common_args(subparsers.add_parser("rev-01-conserved-national-ssm"))
    _common_args(subparsers.add_parser("rev-02-diagnosis-delay"))
    _common_args(subparsers.add_parser("rev-03-art-ltfu-vl"))
    rev05_parser = subparsers.add_parser("rev-05-determinant-shrinkage")
    _common_args(rev05_parser)
    rev05_parser.add_argument("--phase2-source-run-id")
    rev00_parser = subparsers.add_parser("rev-00-observation-role-ledger")
    rev00_parser.add_argument("--run-id", required=True)
    rev00_parser.add_argument("--source-run-id")
    rev00_parser.add_argument("--baseline-source-run-id")
    rev04_parser = subparsers.add_parser("rev-04-prep-persistence")
    rev04_parser.add_argument("--run-id", required=True)
    rev04_parser.add_argument("--source-run-id")
    rev04_parser.add_argument("--baseline-source-run-id")
    rev06_parser = subparsers.add_parser("rev-06-kp-lite-overlay")
    rev06_parser.add_argument("--run-id", required=True)
    rev06_parser.add_argument("--source-run-id")
    rev06_parser.add_argument("--baseline-source-run-id")
    rev07_parser = subparsers.add_parser("rev-07-region-hierarchy")
    rev07_parser.add_argument("--run-id", required=True)
    rev07_parser.add_argument("--source-run-id")
    rev07_parser.add_argument("--baseline-source-run-id")
    rev_program_parser = subparsers.add_parser("run-publication-revision-program")
    rev_program_parser.add_argument("--run-id-prefix", required=True)
    rev_program_parser.add_argument("--source-run-id")
    rev_program_parser.add_argument("--baseline-source-run-id")
    rev_program_parser.add_argument("--phase2-source-run-id")
    rev_program_parser.add_argument("--start-year", type=int, default=2010)
    rev_program_parser.add_argument("--end-year", type=int, default=2025)
    rev_program_parser.add_argument("--min-train-years", type=int, default=5)
    rev_program_parser.add_argument("--horizon-years", type=int, default=1)
    tracker_parser = subparsers.add_parser("track-experiments")
    tracker_parser.add_argument("--output-dir")
    contract_parser = subparsers.add_parser("write-project-contracts")
    contract_parser.add_argument("--epigraph-root")
    lineage_parser = subparsers.add_parser("write-project-lineage-manifest")
    lineage_parser.add_argument("--epigraph-root")
    failure_parser = subparsers.add_parser("write-failure-anatomy-report")
    failure_parser.add_argument("--epigraph-root")
    failure_parser.add_argument("--output-stem", default="phase3_failure_anatomy")
    failure_parser.add_argument("--top-years", type=int, default=3)
    failure_parser.add_argument("--report-path", action="append", required=True)
    scenario_parser = subparsers.add_parser("scenario-lab")
    scenario_parser.add_argument("--run-id", required=True)
    scenario_parser.add_argument("--source-run-id")
    scenario_parser.add_argument("--baseline-source-run-id")
    scenario_parser.add_argument("--phase2-source-run-id")
    scenario_parser.add_argument("--reference-report-path")
    scenario_parser.add_argument("--scenario-start-year", type=int, default=2026)
    scenario_parser.add_argument("--scenario-end-year", type=int, default=2035)
    scenario_parser.add_argument("--draw-count", type=int)
    scenario_parser.add_argument("--seed", type=int)
    hybrid_parser = subparsers.add_parser("hybrid-champion-search")
    hybrid_parser.add_argument("--run-id", required=True)
    hybrid_parser.add_argument("--source-run-id")
    hybrid_parser.add_argument("--baseline-source-run-id")
    hybrid_parser.add_argument("--reference-report-path")
    hybrid_parser.add_argument("--start-year", type=int, default=2010)
    hybrid_parser.add_argument("--end-year", type=int, default=2025)
    hybrid_parser.add_argument("--min-train-years", type=int, default=5)
    hybrid_parser.add_argument("--horizon-years", type=int, default=1)
    hybrid_parser.add_argument("--scenario-start-year", type=int, default=2026)
    hybrid_parser.add_argument("--scenario-end-year", type=int, default=2035)
    r11_parser = subparsers.add_parser("r11-first-batch")
    r11_parser.add_argument("--run-id", required=True)
    r11_parser.add_argument("--source-run-id")
    r11_parser.add_argument("--baseline-source-run-id")
    r11_parser.add_argument("--epigraph-root")
    r11_parser.add_argument("--start-year", type=int, default=2010)
    r11_parser.add_argument("--end-year", type=int, default=2025)
    r11_parser.add_argument("--min-train-years", type=int, default=5)
    r11_parser.add_argument("--horizon-years", type=int, default=1)
    r12_parser = subparsers.add_parser("r12-reference-branch")
    r12_parser.add_argument("--run-id", required=True)
    r12_parser.add_argument("--source-run-id")
    r12_parser.add_argument("--baseline-source-run-id")
    r12_parser.add_argument("--epigraph-root")
    r12_parser.add_argument("--start-year", type=int, default=2010)
    r12_parser.add_argument("--end-year", type=int, default=2025)
    r12_parser.add_argument("--min-train-years", type=int, default=5)
    r13_parser = subparsers.add_parser("r13-priority-experiment-queue")
    r13_parser.add_argument("--run-id", required=True)
    r13_parser.add_argument("--source-run-id")
    r13_parser.add_argument("--baseline-source-run-id")
    r13_parser.add_argument("--epigraph-root")
    r13_parser.add_argument("--start-year", type=int, default=2010)
    r13_parser.add_argument("--end-year", type=int, default=2025)
    r13_parser.add_argument("--min-train-years", type=int, default=5)
    r13_parser.add_argument("--max-experiments", type=int)

    args = parser.parse_args()
    if args.command == "track-experiments":
        output_dir = Path(args.output_dir) if args.output_dir else None
        write_experiment_registry(sandbox_root=sandbox_repo_root(), output_dir=output_dir)
        return
    if args.command == "write-project-contracts":
        epigraph_root = Path(args.epigraph_root) if args.epigraph_root else default_epigraph_root()
        write_project_contract_artifacts(project_root=epigraph_root, phase3_dynamic_root=sandbox_repo_root())
        return
    if args.command == "write-project-lineage-manifest":
        epigraph_root = Path(args.epigraph_root) if args.epigraph_root else default_epigraph_root()
        write_project_lineage_manifest(project_root=epigraph_root, phase3_dynamic_root=sandbox_repo_root())
        return
    if args.command == "write-failure-anatomy-report":
        epigraph_root = Path(args.epigraph_root) if args.epigraph_root else default_epigraph_root()
        write_failure_anatomy_report(
            report_paths=[Path(path) for path in args.report_path],
            output_stem=str(args.output_stem),
            epigraph_root=epigraph_root,
            top_years=int(args.top_years),
        )
        return
    if args.command == "scenario-lab":
        from .scenario_lab import run_scenario_lab

        run_scenario_lab(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            phase2_source_run_id=args.phase2_source_run_id,
            reference_report_path=args.reference_report_path,
            scenario_start_year=args.scenario_start_year,
            scenario_end_year=args.scenario_end_year,
            draw_count=args.draw_count,
            seed=args.seed,
        )
        return
    if args.command == "hybrid-champion-search":
        from .hybrid_champion import run_hybrid_champion_search

        run_hybrid_champion_search(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            reference_report_path=args.reference_report_path,
            start_year=args.start_year,
            end_year=args.end_year,
            min_train_years=args.min_train_years,
            horizon_years=args.horizon_years,
            scenario_start_year=args.scenario_start_year,
            scenario_end_year=args.scenario_end_year,
        )
        return
    if args.command == "r11-first-batch":
        run_r11_first_batch(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            epigraph_root=Path(args.epigraph_root) if args.epigraph_root else None,
            start_year=args.start_year,
            end_year=args.end_year,
            min_train_years=args.min_train_years,
            horizon_years=args.horizon_years,
        )
        return
    if args.command == "r12-reference-branch":
        run_r12_reference_branch(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            epigraph_root=Path(args.epigraph_root) if args.epigraph_root else None,
            start_year=args.start_year,
            end_year=args.end_year,
            min_train_years=args.min_train_years,
        )
        return
    if args.command == "r13-priority-experiment-queue":
        run_r13_priority_experiment_queue(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            epigraph_root=Path(args.epigraph_root) if args.epigraph_root else None,
            start_year=args.start_year,
            end_year=args.end_year,
            min_train_years=args.min_train_years,
            max_experiments=args.max_experiments,
        )
        return
    if args.command == "rev-00-observation-role-ledger":
        run_rev_00_observation_role_ledger(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
        )
        return
    if args.command == "rev-04-prep-persistence":
        run_rev_04_prep_persistence(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
        )
        return
    if args.command == "rev-06-kp-lite-overlay":
        run_rev_06_kp_lite_overlay(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
        )
        return
    if args.command == "rev-07-region-hierarchy":
        run_rev_07_region_hierarchy(
            run_id=args.run_id,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
        )
        return
    if args.command == "run-publication-revision-program":
        run_publication_revision_program(
            run_id_prefix=args.run_id_prefix,
            source_run_id=args.source_run_id,
            baseline_source_run_id=args.baseline_source_run_id,
            phase2_source_run_id=args.phase2_source_run_id,
            start_year=args.start_year,
            end_year=args.end_year,
            min_train_years=args.min_train_years,
            horizon_years=args.horizon_years,
        )
        return

    kwargs = {"run_id": args.run_id, "source_run_id": args.source_run_id, "epigraph_root": Path(args.epigraph_root) if args.epigraph_root else None, "start_year": args.start_year, "end_year": args.end_year, "min_train_years": args.min_train_years, "horizon_years": args.horizon_years}
    revision_kwargs = {
        "run_id": args.run_id,
        "source_run_id": args.source_run_id,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "min_train_years": args.min_train_years,
        "horizon_years": args.horizon_years,
    }
    if args.command == "tr-v3-00-loop":
        run_tr_v3_00_loop(**kwargs)
    elif args.command == "tr-v3-01-loop":
        run_tr_v3_01_loop(**kwargs)
    elif args.command == "tr-v3-02-loop":
        run_tr_v3_02_loop(**kwargs)
    elif args.command == "tr-v3-02b-loop":
        run_tr_v3_02b_loop(**kwargs)
    elif args.command == "tr-v3-03-loop":
        run_tr_v3_03_loop(**kwargs)
    elif args.command == "tr-v3-04-loop":
        run_tr_v3_04_loop(**kwargs)
    elif args.command == "tr-v3-04b-loop":
        run_tr_v3_04b_loop(**kwargs)
    elif args.command == "tr-v3-04b-early-history-partial":
        run_tr_v3_04b_early_history_partial(**kwargs)
    elif args.command == "tr-v3-04c-loop":
        run_tr_v3_04c_loop(**kwargs)
    elif args.command == "tr-v3-04c-early-history-partial":
        run_tr_v3_04c_early_history_partial(**kwargs)
    elif args.command == "tr-v3-04d-loop":
        run_tr_v3_04d_loop(**kwargs)
    elif args.command == "tr-v3-04d-early-history-partial":
        run_tr_v3_04d_early_history_partial(**kwargs)
    elif args.command == "rev-01-conserved-national-ssm":
        run_rev_01_conserved_national_ssm(**revision_kwargs)
    elif args.command == "rev-02-diagnosis-delay":
        run_rev_02_diagnosis_delay(**revision_kwargs)
    elif args.command == "rev-03-art-ltfu-vl":
        run_rev_03_art_ltfu_vl(**revision_kwargs)
    elif args.command == "rev-05-determinant-shrinkage":
        run_rev_05_determinant_shrinkage(**revision_kwargs, phase2_source_run_id=args.phase2_source_run_id)


if __name__ == "__main__":
    main()
