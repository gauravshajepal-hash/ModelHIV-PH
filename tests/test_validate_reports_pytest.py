from __future__ import annotations

from pathlib import Path

from epigraph_ph.validate import build_deep_pipeline_audit, build_gold_standard_report, run_module_test_report


def test_deep_pipeline_audit_emits_reports(rescue_v2_run_dir: Path) -> None:
    report = build_deep_pipeline_audit(run_dir=rescue_v2_run_dir)
    artifacts = report.get("artifacts", {})
    assert Path(str(artifacts.get("report_json"))).exists()
    assert Path(str(artifacts.get("report_md"))).exists()
    assert "phase_rows" in report
    assert any(str(row.get("phase")) == "phase3" for row in report.get("phase_rows", []))


def test_gold_standard_report_smoke(legacy_full_run_dir: Path) -> None:
    report = build_gold_standard_report(run_dir=legacy_full_run_dir)
    artifacts = report.get("artifacts", {})
    assert Path(str(artifacts.get("gold_standard_report_json"))).exists()
    assert Path(str(artifacts.get("gold_standard_report_md"))).exists()
    assert report.get("phase_rows")


def test_module_test_report_emits_outputs(tmp_path: Path) -> None:
    report = run_module_test_report(
        root_dir=Path(__file__).resolve().parents[1],
        output_dir=tmp_path / "module-tests",
        module_groups=[("runtime", ["tests/test_runtime_interop_pytest.py"])],
    )
    artifacts = report.get("artifacts", {})
    assert Path(str(artifacts.get("report_json"))).exists()
    assert Path(str(artifacts.get("report_md"))).exists()
    assert report.get("rows")
    assert any(str(row.get("module")) == "runtime" for row in report.get("rows", []))
