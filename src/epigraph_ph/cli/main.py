from __future__ import annotations

import argparse
from pathlib import Path

from epigraph_ph.harp_archive import run_harp_archive_build
from epigraph_ph.phase0 import (
    run_phase0_build,
    run_phase0_extract,
    run_phase0_harvest,
    run_phase0_index,
    run_phase0_literature_review,
    run_phase0_parse,
    run_phase0_score_wide_sweep,
    run_phase0_semantic_benchmark,
)
from epigraph_ph.phase1 import run_phase1_build
from epigraph_ph.phase15 import run_phase15_build
from epigraph_ph.phase2 import run_phase2_build
from epigraph_ph.phase3 import run_phase3_build, run_phase3_frozen_backtest, run_phase3_frozen_backtest_tuning
from epigraph_ph.phase4 import run_phase4_build, run_phase4_optimize, run_phase4_simulate
from epigraph_ph.registry.sources import build_source_registry
from epigraph_ph.registry.subparameters import build_subparameter_registry
from epigraph_ph.runtime import RunContext


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="epigraph")
    subparsers = parser.add_subparsers(dest="command")

    phase0 = subparsers.add_parser("phase0")
    phase0_sub = phase0.add_subparsers(dest="phase0_command")
    for name in ("harvest", "parse", "extract", "index", "build", "score-sweep", "semantic-benchmark", "literature-review"):
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
        if name in {"parse", "build"}:
            cmd.add_argument("--enable-chart-extraction", action="store_true")
            cmd.add_argument("--enable-ocr-sidecar", action="store_true")
            cmd.add_argument("--ocr-backend", default="auto", choices=["auto", "disabled", "lighton_local", "lighton_vllm"])
            cmd.add_argument("--working-set-size", type=int, default=250)
        if name in {"extract", "build"}:
            cmd.add_argument("--skip-live-normalizer", action="store_true")
        if name in {"score-sweep", "build"}:
            cmd.add_argument("--sweep-json-path", default=None)
            cmd.add_argument("--min-domain-quality", type=float, default=0.2)
        if name == "semantic-benchmark":
            cmd.add_argument("--candidate-json-path", default=None)
            cmd.add_argument("--top-k", type=int, default=10)

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

    for phase_name in ("phase1", "phase15", "phase2", "phase3"):
        phase = subparsers.add_parser(phase_name)
        phase_sub = phase.add_subparsers(dest=f"{phase_name}_command")
        build = phase_sub.add_parser("build")
        build.add_argument("--run-id", required=True)
        build.add_argument("--plugin", default="hiv")
        build.add_argument("--profile", default="legacy")
        if phase_name == "phase3":
            build.add_argument("--top-k-per-block", type=int, default=20)
            build.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi"])
            backtest = phase_sub.add_parser("frozen-backtest")
            backtest.add_argument("--run-id", required=True)
            backtest.add_argument("--plugin", default="hiv")
            backtest.add_argument("--profile", default="hiv_rescue_v2")
            backtest.add_argument("--phase3-inference", default="torch_map", choices=["torch_map", "jax_svi"])
            tune_backtest = phase_sub.add_parser("tune-frozen-backtest")
            tune_backtest.add_argument("--run-id", required=True)
            tune_backtest.add_argument("--plugin", default="hiv")
            tune_backtest.add_argument("--profile", default="hiv_rescue_v2")
            tune_backtest.add_argument("--phase3-inference", default="torch_map", choices=["torch_map"])
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
                enable_chart_extraction=args.enable_chart_extraction,
                enable_ocr_sidecar=args.enable_ocr_sidecar,
                ocr_backend=args.ocr_backend,
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
    if args.command == "registry" and args.registry_command == "build":
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
        run_harp_archive_build(
            run_id=args.run_id,
            plugin_id=args.plugin,
            desktop_seed_dir=args.desktop_seed_dir,
            manual_seed_dir=args.manual_seed_dir,
        )
        return 0
    if args.command == "phase1" and args.phase1_command == "build":
        run_phase1_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase15" and args.phase15_command == "build":
        run_phase15_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase2" and args.phase2_command == "build":
        run_phase2_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase3" and args.phase3_command == "build":
        run_phase3_build(
            run_id=args.run_id,
            plugin_id=args.plugin,
            top_k_per_block=args.top_k_per_block,
            profile=args.profile,
            inference_family=args.phase3_inference,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "frozen-backtest":
        run_phase3_frozen_backtest(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
        )
        return 0
    if args.command == "phase3" and args.phase3_command == "tune-frozen-backtest":
        run_phase3_frozen_backtest_tuning(
            run_id=args.run_id,
            plugin_id=args.plugin,
            profile=args.profile,
            inference_family=args.phase3_inference,
        )
        return 0
    if args.command == "phase4" and args.phase4_command == "build":
        run_phase4_build(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase4" and args.phase4_command == "simulate":
        run_phase4_simulate(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0
    if args.command == "phase4" and args.phase4_command == "optimize":
        run_phase4_optimize(run_id=args.run_id, plugin_id=args.plugin, profile=args.profile)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
