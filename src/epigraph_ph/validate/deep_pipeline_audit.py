from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from epigraph_ph.runtime import ensure_dir, load_tensor_artifact, read_json, write_json

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None


PHASES = ["phase0", "phase1", "phase2", "phase3"]
REQUIRED_COLLECTORS = {
    "who",
    "unaids",
    "doh_philippines",
    "un",
    "ndhs",
    "yafs",
    "fies",
    "philgis_boundary_proxy",
    "transport_network_proxies",
    "google_mobility",
    "world_bank_wdi",
    "philhealth_reports",
    "doh_facility_stats",
}


def _run_nvidia_smi() -> dict[str, float | str]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return {"available": False}
    line = next((row.strip() for row in completed.stdout.splitlines() if row.strip()), "")
    if not line:
        return {"available": False}
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return {"available": False}
    return {
        "available": True,
        "name": parts[0],
        "memory_total_mb": float(parts[1]),
        "memory_free_mb": float(parts[2]),
        "memory_used_mb": max(0.0, float(parts[1]) - float(parts[2])),
    }


def _system_memory_summary() -> dict[str, Any]:
    if psutil is None:
        return {"available": False}
    vm = psutil.virtual_memory()
    gpu = _run_nvidia_smi()
    return {
        "available": True,
        "ram_total_gb": round(float(vm.total) / 1024**3, 3),
        "ram_available_gb": round(float(vm.available) / 1024**3, 3),
        "ram_used_percent": round(float(vm.percent), 3),
        "gpu": gpu,
    }


def _repo_line_count_summary(root_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in (root_dir / "src" / "epigraph_ph").rglob("*.py"):
        try:
            lines = sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if lines > 900:
            rows.append({"path": str(path), "lines": lines})
    rows.sort(key=lambda item: int(item["lines"]), reverse=True)
    return {
        "oversized_files": rows,
        "files_over_1000": [row for row in rows if int(row["lines"]) > 1000],
    }


def _has_cycle(adjacency: np.ndarray, *, threshold: float = 1e-6) -> bool:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        return True
    n = int(adjacency.shape[0])
    graph = [[dst for dst in range(n) if dst != src and abs(float(adjacency[src, dst])) > threshold] for src in range(n)]
    state = [0] * n

    def visit(node: int) -> bool:
        if state[node] == 1:
            return True
        if state[node] == 2:
            return False
        state[node] = 1
        for nxt in graph[node]:
            if visit(nxt):
                return True
        state[node] = 2
        return False

    return any(visit(node) for node in range(n) if state[node] == 0)


def _plot_bars(rows: list[dict[str, Any]], *, label_key: str, value_key: str, title: str, output_path: Path, color: str = "#2a6f97") -> str:
    if plt is None or not rows:
        return ""
    labels = [str(row[label_key]) for row in rows]
    values = [float(row[value_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(labels, values, color=color)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _plot_memory_summary(memory: dict[str, Any], output_path: Path) -> str:
    if plt is None or not memory.get("available"):
        return ""
    rows = [("RAM used", float(memory.get("ram_used_percent", 0.0))), ("RAM free", max(0.0, 100.0 - float(memory.get("ram_used_percent", 0.0))))]
    gpu = dict(memory.get("gpu") or {})
    if gpu.get("available"):
        total = float(gpu.get("memory_total_mb", 0.0))
        used = float(gpu.get("memory_used_mb", 0.0))
        rows.extend(
            [
                ("GPU used", (used / total * 100.0) if total else 0.0),
                ("GPU free", (float(gpu.get("memory_free_mb", 0.0)) / total * 100.0) if total else 0.0),
            ]
        )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = [item[0] for item in rows]
    values = [item[1] for item in rows]
    ax.bar(labels, values, color=["#c44e52", "#2e8b57", "#8c564b", "#17becf"][: len(rows)])
    ax.set_ylim(0.0, 100.0)
    ax.set_ylabel("Percent")
    ax.set_title("Current Memory Budget")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return str(output_path)


def _phase0_audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = run_dir / "phase0"
    analysis_dir = phase_dir / "analysis"
    source_rows = read_json(phase_dir / "raw" / "source_manifest.json", default=[])
    collectors = read_json(phase_dir / "raw" / "collector_manifest.json", default=[])
    alignment = read_json(phase_dir / "extracted" / "alignment_summary.json", default={})
    schema = read_json(analysis_dir / "schema_validation_summary.json", default={})
    ocr_manifest = read_json(phase_dir / "parsed" / "ocr_sidecar_manifest.json", default=[])
    issues = [
        {
            "name": "literature_floor_2010",
            "passed": all(int(str(month)[:4]) >= 2010 for month in alignment.get("month_axis", []) if str(month)[:4].isdigit()),
        },
        {
            "name": "official_collectors_present",
            "passed": REQUIRED_COLLECTORS.issubset({str(row.get("collector_id") or "") for row in collectors}),
        },
        {
            "name": "generic_canonical_fraction_below_0_25",
            "passed": float(schema.get("generic_canonical_fraction", 1.0)) <= 0.25,
        },
        {
            "name": "ocr_manifest_recorded",
            "passed": (phase_dir / "parsed" / "ocr_sidecar_manifest.json").exists(),
        },
        {
            "name": "source_manifest_nonempty",
            "passed": bool(source_rows),
        },
    ]
    summary = {
        "source_count": len(source_rows),
        "collector_count": len(collectors),
        "generic_canonical_fraction": float(schema.get("generic_canonical_fraction", np.nan)),
        "ocr_row_count": len(ocr_manifest),
        "alignment_month_count": len(alignment.get("month_axis", [])),
    }
    return issues, summary


def _phase1_audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = run_dir / "phase1"
    standardized = load_tensor_artifact(phase_dir / "standardized_tensor.npz")
    denominators = load_tensor_artifact(phase_dir / "denominator_tensor.npz")
    missing_mask = load_tensor_artifact(phase_dir / "missing_mask.npz")
    quality = load_tensor_artifact(phase_dir / "quality_weight_tensor.npz")
    report = read_json(phase_dir / "normalization_report.json", default={})
    issues = [
        {"name": "phase1_tensor_shapes_match", "passed": standardized.shape == denominators.shape == missing_mask.shape == quality.shape},
        {"name": "phase1_tensors_finite", "passed": bool(np.isfinite(standardized).all() and np.isfinite(denominators).all() and np.isfinite(missing_mask).all() and np.isfinite(quality).all())},
        {"name": "phase1_denominators_not_all_ones", "passed": bool(np.any(denominators != 1.0))},
        {"name": "phase1_missing_mask_binary", "passed": set(np.unique(missing_mask)).issubset({0.0, 1.0})},
    ]
    summary = {
        "shape": list(standardized.shape),
        "missing_mask_fraction": float(report.get("missing_mask_fraction", np.nan)),
        "transform_mode": str(report.get("preprocess_meta", {}).get("transform", {}).get("transform_mode") or ""),
        "scaling_mode": str(report.get("preprocess_meta", {}).get("scaling", {}).get("scaling_mode") or ""),
    }
    return issues, summary


def _phase2_audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = run_dir / "phase2"
    dag = load_tensor_artifact(phase_dir / "dag_adjacency.npz")
    tier_mask = load_tensor_artifact(phase_dir / "tier_mask.npz")
    lag_mask = load_tensor_artifact(phase_dir / "lag_mask.npz")
    mix_report = read_json(phase_dir / "feature_matrix_mix_report.json", default={})
    time_cv = read_json(phase_dir / "time_stratified_cv_report.json", default={})
    collinearity = read_json(phase_dir / "collinearity_report.json", default={})
    blanket = read_json(phase_dir / "markov_blanket.json", default={})
    issues = [
        {"name": "phase2_adjacency_square", "passed": dag.ndim == 2 and dag.shape[0] == dag.shape[1]},
        {"name": "phase2_cycle_free", "passed": not _has_cycle(dag)},
        {"name": "phase2_tier_mask_respected", "passed": bool(np.all(np.abs(dag[tier_mask == 0]) <= 1e-6))},
        {"name": "phase2_lag_mask_respected", "passed": bool(np.all(np.abs(dag[lag_mask == 0]) <= 1e-6))},
        {"name": "phase2_markov_blanket_nonempty", "passed": bool(blanket.get("blanket_nodes", []))},
        {"name": "phase2_time_cv_available", "passed": bool(time_cv.get("available"))},
        {"name": "phase2_collinearity_available", "passed": bool(collinearity.get("available"))},
    ]
    summary = {
        "dag_shape": list(dag.shape),
        "edge_count": int(np.sum(np.abs(dag) > 1e-6)),
        "soft_feature_count": int(mix_report.get("soft_feature_count", 0)),
        "numeric_feature_count": int(mix_report.get("numeric_feature_count", 0)),
        "mean_test_reconstruction_mse": float(time_cv.get("mean_test_reconstruction_mse", np.nan)),
        "condition_number": float(collinearity.get("condition_number", np.nan)),
    }
    return issues, summary


def _phase3_audit(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    phase_dir = run_dir / "phase3"
    state_estimates = load_tensor_artifact(phase_dir / "state_estimates.npz")
    forecast_states = load_tensor_artifact(phase_dir / "forecast_states.npz")
    validation = read_json(phase_dir / "validation_artifact.json", default={})
    chain = read_json(phase_dir / "chain_diagnostics.json", default={})
    calibration = read_json(phase_dir / "posterior_calibration.json", default={})
    fit = read_json(phase_dir / "fit_artifact.json", default={})
    calibration_metrics = [row for row in list(calibration.get("metrics", [])) if bool(row.get("available"))]
    calibration_mean_picp = (
        float(np.mean([float(row.get("picp", np.nan)) for row in calibration_metrics]))
        if calibration_metrics
        else np.nan
    )
    issues = [
        {
            "name": "phase3_state_axes_match",
            "passed": state_estimates.ndim == 3
            and forecast_states.ndim == 3
            and state_estimates.shape[0] == forecast_states.shape[0]
            and state_estimates.shape[-1] == forecast_states.shape[-1],
        },
        {"name": "phase3_states_finite", "passed": bool(np.isfinite(state_estimates).all() and np.isfinite(forecast_states).all())},
        {"name": "phase3_states_nonnegative", "passed": bool(np.all(state_estimates >= 0.0) and np.all(forecast_states >= 0.0))},
        {"name": "phase3_state_mass_reasonable", "passed": bool(np.all(state_estimates.sum(axis=-1) <= 1.0001) and float(state_estimates.sum(axis=-1).mean()) >= 0.90)},
        {"name": "phase3_calibration_available", "passed": bool(calibration.get("available"))},
        {"name": "phase3_chain_diagnostics_present", "passed": "available" in chain or "rhat_max" in chain},
    ]
    summary = {
        "state_shape": list(state_estimates.shape),
        "mean_state_mass": float(state_estimates.sum(axis=-1).mean()) if state_estimates.size else np.nan,
        "inference_family": str(validation.get("inference_family") or fit.get("inference_family") or ""),
        "claim_eligible": bool(validation.get("claim_eligible")),
        "rhat_max": float(chain.get("rhat_max", np.nan)) if chain.get("rhat_max") is not None else np.nan,
        "divergence_count": int(chain.get("divergence_count", 0)) if chain.get("divergence_count") is not None else 0,
        "picp": calibration_mean_picp,
    }
    return issues, summary


def build_deep_pipeline_audit(*, run_dir: str | Path, output_dir: str | Path | None = None, root_dir: str | Path | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    root_dir = Path(root_dir) if root_dir is not None else run_dir.parents[2]
    analysis_dir = ensure_dir(Path(output_dir) if output_dir is not None else run_dir / "analysis")

    phase_builders = {
        "phase0": _phase0_audit,
        "phase1": _phase1_audit,
        "phase2": _phase2_audit,
        "phase3": _phase3_audit,
    }
    phase_rows: list[dict[str, Any]] = []
    phase_details: dict[str, Any] = {}
    for phase_name in PHASES:
        phase_dir = run_dir / phase_name
        if not phase_dir.exists():
            continue
        checks, summary = phase_builders[phase_name](run_dir)
        passed = sum(1 for check in checks if bool(check.get("passed")))
        failed = len(checks) - passed
        phase_rows.append(
            {
                "phase": phase_name,
                "passed": passed,
                "failed": failed,
                "pass_ratio": passed / max(1, len(checks)),
            }
        )
        phase_details[phase_name] = {
            "checks": checks,
            "summary": summary,
        }

    memory_summary = _system_memory_summary()
    bloat_summary = _repo_line_count_summary(root_dir)

    phase_plot = _plot_bars(
        phase_rows,
        label_key="phase",
        value_key="pass_ratio",
        title="Phase 0-3 Audit Pass Ratio",
        output_path=analysis_dir / "deep_audit_phase_pass_ratio.png",
        color="#2e8b57",
    )
    bloat_plot = _plot_bars(
        list(bloat_summary.get("oversized_files", []))[:12],
        label_key="path",
        value_key="lines",
        title="Oversized Python Files",
        output_path=analysis_dir / "deep_audit_code_bloat.png",
        color="#c44e52",
    )
    memory_plot = _plot_memory_summary(memory_summary, analysis_dir / "deep_audit_memory_budget.png")

    markdown_lines = [
        "# Deep Pipeline Audit",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "## Phase summary",
        "",
    ]
    for row in phase_rows:
        markdown_lines.append(f"- `{row['phase']}`: `{row['passed']}` passed, `{row['failed']}` failed, ratio=`{row['pass_ratio']:.3f}`")
        details = phase_details.get(row["phase"], {})
        for check in details.get("checks", []):
            status = "pass" if check.get("passed") else "fail"
            markdown_lines.append(f"  - `{status}` `{check.get('name')}`")
    markdown_lines.extend(["", "## Code bloat", ""])
    for row in bloat_summary.get("oversized_files", [])[:15]:
        markdown_lines.append(f"- `{row['path']}`: `{row['lines']}` lines")
    markdown_lines.extend(["", "## Memory budget", ""])
    if memory_summary.get("available"):
        markdown_lines.append(
            f"- RAM: `{memory_summary.get('ram_available_gb')}` GB free of `{memory_summary.get('ram_total_gb')}` GB"
        )
        gpu = dict(memory_summary.get("gpu") or {})
        if gpu.get("available"):
            markdown_lines.append(
                f"- GPU: `{gpu.get('name')}` free `{gpu.get('memory_free_mb')}` MB of `{gpu.get('memory_total_mb')}` MB"
            )

    report = {
        "run_dir": str(run_dir),
        "analysis_dir": str(analysis_dir),
        "phase_rows": phase_rows,
        "phase_details": phase_details,
        "memory_summary": memory_summary,
        "bloat_summary": bloat_summary,
        "artifacts": {
            "phase_pass_ratio_plot": phase_plot,
            "code_bloat_plot": bloat_plot,
            "memory_budget_plot": memory_plot,
            "report_json": str(analysis_dir / "deep_pipeline_audit.json"),
            "report_md": str(analysis_dir / "deep_pipeline_audit.md"),
        },
    }
    write_json(analysis_dir / "deep_pipeline_audit.json", report)
    (analysis_dir / "deep_pipeline_audit.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    return report
