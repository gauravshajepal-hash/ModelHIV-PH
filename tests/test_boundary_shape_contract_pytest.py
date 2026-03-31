from __future__ import annotations

from pathlib import Path

from epigraph_ph.runtime import read_json


def _assert_boundary_package(phase_dir: Path) -> None:
    manifest = read_json(phase_dir / "boundary_shape_manifest.json", default={})
    checks = read_json(phase_dir / "boundary_shape_checks.json", default=[])
    summary = read_json(phase_dir / "boundary_shape_summary.json", default={})

    assert manifest.get("phase_name")
    assert manifest.get("boundary_count", 0) >= 1
    assert checks
    assert summary.get("overall_passed") is True
    assert summary.get("failed_check_count") == 0


def test_boundary_shape_packages_exist_for_phase0_to_phase4(legacy_full_run_dir: Path, rescue_v2_run_dir: Path) -> None:
    _assert_boundary_package(legacy_full_run_dir / "phase0")
    _assert_boundary_package(legacy_full_run_dir / "phase1")
    _assert_boundary_package(rescue_v2_run_dir / "phase15")
    _assert_boundary_package(legacy_full_run_dir / "phase2")
    _assert_boundary_package(legacy_full_run_dir / "phase3")
    _assert_boundary_package(legacy_full_run_dir / "phase4")
    _assert_boundary_package(rescue_v2_run_dir / "phase3")
    _assert_boundary_package(rescue_v2_run_dir / "phase4")
