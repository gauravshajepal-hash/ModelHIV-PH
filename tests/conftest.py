from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from epigraph_ph.phase0.pipeline import run_phase0_build
from epigraph_ph.phase1.pipeline import run_phase1_build
from epigraph_ph.phase15.pipeline import run_phase15_build
from epigraph_ph.phase2.pipeline import run_phase2_build
from epigraph_ph.phase3.pipeline import run_phase3_build, run_phase3_frozen_backtest
from epigraph_ph.phase4.pipeline import run_phase4_build
from epigraph_ph.registry.sources import build_source_registry
from epigraph_ph.registry.subparameters import build_subparameter_registry
from epigraph_ph.runtime import ROOT_DIR, detect_backends, ensure_dir


def _reset_run_dir(run_id: str) -> Path:
    run_dir = ROOT_DIR / "artifacts" / "runs" / run_id
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir, ignore_errors=True)
    return run_dir


def _build_phase0_registry_run(*, run_id: str, target_records: int = 40, working_set_size: int = 30) -> Path:
    run_dir = _reset_run_dir(run_id)
    run_phase0_build(
        run_id=run_id,
        plugin_id="hiv",
        offline=True,
        corpus_mode="default",
        target_records=target_records,
        working_set_size=working_set_size,
        skip_live_normalizer=True,
    )
    registry_dir = ensure_dir(run_dir / "registry")
    build_source_registry(plugin_id="hiv", output_path=registry_dir / "source_registry.json", phase0_run_dir=run_dir)
    build_subparameter_registry(plugin_id="hiv", output_path=registry_dir / "subparameter_registry.json", phase0_run_dir=run_dir)
    return run_dir


def _build_legacy_run(run_id: str) -> Path:
    run_dir = _build_phase0_registry_run(run_id=run_id, target_records=42, working_set_size=30)
    run_phase1_build(run_id=run_id, plugin_id="hiv")
    run_phase2_build(run_id=run_id, plugin_id="hiv")
    run_phase3_build(run_id=run_id, plugin_id="hiv", top_k_per_block=10)
    run_phase4_build(run_id=run_id, plugin_id="hiv")
    return run_dir


def _build_rescue_v1_run(run_id: str, *, inference_family: str = "torch_map", target_records: int = 40, working_set_size: int = 30) -> Path:
    run_dir = _build_phase0_registry_run(run_id=run_id, target_records=target_records, working_set_size=working_set_size)
    run_phase1_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v1")
    run_phase2_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v1")
    run_phase3_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v1", inference_family=inference_family)
    run_phase4_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v1")
    return run_dir


def _build_rescue_v2_run(run_id: str, *, inference_family: str = "torch_map", target_records: int = 36, working_set_size: int = 24) -> Path:
    run_dir = _build_phase0_registry_run(run_id=run_id, target_records=target_records, working_set_size=working_set_size)
    run_phase1_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2")
    run_phase15_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2")
    run_phase2_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2")
    run_phase3_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2", inference_family=inference_family)
    run_phase4_build(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2")
    return run_dir


def _build_rescue_v2_backtest(run_id: str, *, inference_family: str = "torch_map") -> Path:
    run_dir = _build_rescue_v2_run(run_id=run_id, inference_family=inference_family)
    run_phase3_frozen_backtest(run_id=run_id, plugin_id="hiv", profile="hiv_rescue_v2", inference_family=inference_family)
    return run_dir


@pytest.fixture(scope="session")
def phase0_registry_run_dir() -> Path:
    return _build_phase0_registry_run(run_id="pytest-phase0-registry")


@pytest.fixture(scope="session")
def legacy_full_run_dir() -> Path:
    return _build_legacy_run("pytest-legacy-full")


@pytest.fixture(scope="session")
def rescue_v1_run_dir() -> Path:
    return _build_rescue_v1_run("pytest-rescue-v1-full")


@pytest.fixture(scope="session")
def rescue_v2_run_dir() -> Path:
    return _build_rescue_v2_run("pytest-rescue-v2-full")


@pytest.fixture(scope="session")
def rescue_v2_backtest_run_dir() -> Path:
    return _build_rescue_v2_backtest("pytest-rescue-v2-backtest")


@pytest.fixture(scope="session")
def rescue_v1_jax_run_dir() -> Path:
    backends = detect_backends()
    if not backends["jax"].available:
        pytest.skip("JAX backend unavailable in this environment")
    return _build_rescue_v1_run(
        "pytest-rescue-v1-jax",
        inference_family="jax_svi",
        target_records=30,
        working_set_size=25,
    )
