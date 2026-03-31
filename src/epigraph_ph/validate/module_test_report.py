from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from epigraph_ph.runtime import ensure_dir, write_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


MODULE_TEST_GROUPS: list[tuple[str, list[str]]] = [
    ("phase0", ["tests/test_phase0_pipeline_pytest.py", "tests/test_registry_pytest.py"]),
    ("phase1", ["tests/test_phase1_pipeline_pytest.py"]),
    ("phase15", ["tests/test_phase15_pipeline_pytest.py"]),
    ("phase2", ["tests/test_phase2_pipeline_pytest.py"]),
    ("phase3", ["tests/test_phase3_pipeline_pytest.py"]),
    ("phase4", ["tests/test_phase4_pipeline_pytest.py"]),
    ("contracts", ["tests/test_boundary_shape_contract_pytest.py", "tests/test_gold_standard_contract_pytest.py", "tests/test_runtime_interop_pytest.py"]),
]


def _parse_pytest_summary(text: str) -> dict[str, Any]:
    summary = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "warnings": 0,
    }
    lower = text.lower()
    match = re.search(r"(\d+)\s+passed", lower)
    if match:
        summary["passed"] = int(match.group(1))
    match = re.search(r"(\d+)\s+failed", lower)
    if match:
        summary["failed"] = int(match.group(1))
    match = re.search(r"(\d+)\s+skipped", lower)
    if match:
        summary["skipped"] = int(match.group(1))
    match = re.search(r"(\d+)\s+warnings?", lower)
    if match:
        summary["warnings"] = int(match.group(1))
    return summary


def _plot_test_matrix(rows: list[dict[str, Any]], output_path: Path) -> str:
    if plt is None or not rows:
        return ""
    labels = [row["module"] for row in rows]
    durations = [float(row["duration_seconds"]) for row in rows]
    passed = [int(row["passed"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].bar(labels, durations, color="#457b9d")
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Module Test Duration")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(labels, passed, color="#2e8b57")
    axes[1].set_ylabel("Passed tests")
    axes[1].set_title("Module Test Pass Counts")
    axes[1].tick_params(axis="x", rotation=25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def run_module_test_report(
    *,
    root_dir: str | Path,
    output_dir: str | Path | None = None,
    module_groups: list[tuple[str, list[str]]] | None = None,
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    analysis_dir = ensure_dir(Path(output_dir) if output_dir is not None else root_dir / "artifacts" / "analysis" / "module_tests")

    rows: list[dict[str, Any]] = []
    for module_name, targets in (module_groups or MODULE_TEST_GROUPS):
        command = [sys.executable, "-m", "pytest", *targets, "-q"]
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=root_dir, capture_output=True, text=True)
        duration = time.perf_counter() - start
        combined_output = f"{completed.stdout}\n{completed.stderr}".strip()
        summary = _parse_pytest_summary(combined_output)
        rows.append(
            {
                "module": module_name,
                "targets": targets,
                "return_code": int(completed.returncode),
                "duration_seconds": round(duration, 3),
                **summary,
                "output_path": str(analysis_dir / f"{module_name}_pytest_output.txt"),
            }
        )
        (analysis_dir / f"{module_name}_pytest_output.txt").write_text(combined_output, encoding="utf-8")

    plot_path = _plot_test_matrix(rows, analysis_dir / "module_test_matrix.png")
    markdown_lines = [
        "# Module Test Report",
        "",
        "| module | passed | failed | skipped | warnings | duration_seconds | return_code |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown_lines.append(
            f"| {row['module']} | {row['passed']} | {row['failed']} | {row['skipped']} | {row['warnings']} | {row['duration_seconds']:.3f} | {row['return_code']} |"
        )

    report = {
        "root_dir": str(root_dir),
        "analysis_dir": str(analysis_dir),
        "rows": rows,
        "artifacts": {
            "matrix_plot": plot_path,
            "report_json": str(analysis_dir / "module_test_report.json"),
            "report_md": str(analysis_dir / "module_test_report.md"),
        },
    }
    write_json(analysis_dir / "module_test_report.json", report)
    (analysis_dir / "module_test_report.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return report
