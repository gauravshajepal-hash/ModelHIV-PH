from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .data import sandbox_repo_root
from .r11_sparse_state_space import _finite_float, _generated_at, _sha256
from .r75_bulk_unaids_annual_challenge import (
    R75_RUN_ID,
    _score_summary_by_fields,
    _write_csv,
)
from .r76_public_domain_annual_comparator import (
    PUBLIC_SELECTED_FAMILY,
    R76_RUN_ID,
)
from .runtime import ensure_dir, read_json, write_json


R77_SCHEMA_VERSION = "phase3_dynamic.r77_public_proxy_annual_gate.v1"
R77_RUN_ID = "p3d-r77-public-proxy-annual-gate-20260507-s00"
R75_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R75_RUN_ID
    / "analysis"
    / "r75_bulk_unaids_annual_challenge_report.json"
)
R76_DEFAULT_REPORT = (
    sandbox_repo_root()
    / "artifacts"
    / "runs"
    / R76_RUN_ID
    / "analysis"
    / "r76_public_domain_annual_comparator_report.json"
)


def _best_r75_family(r75: dict[str, Any]) -> str | None:
    gate = dict(r75.get("bulk_unaids_annual_gate") or {})
    family = str(gate.get("best_candidate_family") or "")
    if family:
        return family
    family_rows = [
        dict(row)
        for row in list(r75.get("family_rows") or [])
        if _finite_float(row.get("candidate_mean_norm_error")) is not None
    ]
    family_rows.sort(key=lambda row: (float(row["candidate_mean_norm_error"]), str(row.get("candidate_family") or "")))
    return None if not family_rows else str(family_rows[0].get("candidate_family") or "")


def _score_key(row: dict[str, Any]) -> tuple[int, int, str, str]:
    return (
        int(row.get("horizon_years") or 0),
        int(row.get("train_end_year") or 0),
        str(row.get("quarter") or ""),
        str(row.get("metric_name") or ""),
    )


def _matched_proxy_rows(
    r75: dict[str, Any],
    r76: dict[str, Any],
    *,
    r75_family: str | None = None,
    r76_family: str = PUBLIC_SELECTED_FAMILY,
) -> list[dict[str, Any]]:
    model_family = r75_family or _best_r75_family(r75)
    if not model_family:
        return []
    r75_rows = [
        dict(row)
        for row in list(r75.get("score_rows") or [])
        if str(row.get("candidate_family") or "") == str(model_family)
    ]
    r76_by_key = {
        _score_key(dict(row)): dict(row)
        for row in list(r76.get("score_rows") or [])
        if str(row.get("candidate_family") or "") == str(r76_family)
    }
    rows: list[dict[str, Any]] = []
    for model in r75_rows:
        proxy = r76_by_key.get(_score_key(model))
        if proxy is None:
            continue
        model_error = _finite_float(model.get("candidate_norm_error"))
        proxy_error = _finite_float(proxy.get("candidate_norm_error"))
        model_interval_miss = _finite_float(model.get("candidate_interval_miss_norm"))
        proxy_interval_miss = _finite_float(proxy.get("candidate_interval_miss_norm"))
        rows.append(
            {
                "horizon_years": int(model.get("horizon_years") or 0),
                "train_end_year": int(model.get("train_end_year") or 0),
                "holdout_years": list(model.get("holdout_years") or []),
                "quarter": str(model.get("quarter") or ""),
                "year": int(model.get("year") or 0),
                "metric_name": str(model.get("metric_name") or ""),
                "model_family": str(model_family),
                "public_proxy_family": str(r76_family),
                "target_value": _finite_float(model.get("target_value")),
                "target_lower": _finite_float(model.get("target_lower")),
                "target_upper": _finite_float(model.get("target_upper")),
                "model_value": _finite_float(model.get("candidate_value")),
                "public_proxy_value": _finite_float(proxy.get("candidate_value")),
                "carry_forward_value": _finite_float(model.get("carry_forward_value")),
                "scale": _finite_float(model.get("scale")),
                "model_norm_error": model_error,
                "public_proxy_norm_error": proxy_error,
                "model_minus_public_proxy_norm_error": None
                if model_error is None or proxy_error is None
                else float(model_error - proxy_error),
                "model_interval_miss_norm": model_interval_miss,
                "public_proxy_interval_miss_norm": proxy_interval_miss,
                "model_minus_public_proxy_interval_miss_norm": None
                if model_interval_miss is None or proxy_interval_miss is None
                else float(model_interval_miss - proxy_interval_miss),
                "model_interval_covered": model.get("candidate_interval_covered"),
                "public_proxy_interval_covered": proxy.get("candidate_interval_covered"),
                "observation_role": str(model.get("observation_role") or ""),
                "allowed_use": str(model.get("allowed_use") or ""),
                "support_partition": str(model.get("support_partition") or ""),
            }
        )
    return rows


def _comparison_summary_by_fields(rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(dict(row))
    output: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        model_errors = [
            float(row["model_norm_error"])
            for row in values
            if _finite_float(row.get("model_norm_error")) is not None
        ]
        proxy_errors = [
            float(row["public_proxy_norm_error"])
            for row in values
            if _finite_float(row.get("public_proxy_norm_error")) is not None
        ]
        deltas = [
            float(row["model_minus_public_proxy_norm_error"])
            for row in values
            if _finite_float(row.get("model_minus_public_proxy_norm_error")) is not None
        ]
        model_covered = [
            bool(row["model_interval_covered"])
            for row in values
            if row.get("model_interval_covered") is not None
        ]
        proxy_covered = [
            bool(row["public_proxy_interval_covered"])
            for row in values
            if row.get("public_proxy_interval_covered") is not None
        ]
        summary = {field: key[index] for index, field in enumerate(group_fields)}
        model_mean = None if not model_errors else float(np.mean(np.asarray(model_errors, dtype=np.float64)))
        proxy_mean = None if not proxy_errors else float(np.mean(np.asarray(proxy_errors, dtype=np.float64)))
        summary.update(
            {
                "entry_count": len(values),
                "model_mean_norm_error": model_mean,
                "public_proxy_mean_norm_error": proxy_mean,
                "model_minus_public_proxy_mean_norm_error": None
                if model_mean is None or proxy_mean is None
                else float(model_mean - proxy_mean),
                "model_p90_norm_error": None
                if not model_errors
                else float(np.quantile(np.asarray(model_errors, dtype=np.float64), 0.9)),
                "public_proxy_p90_norm_error": None
                if not proxy_errors
                else float(np.quantile(np.asarray(proxy_errors, dtype=np.float64), 0.9)),
                "delta_p90_norm_error": None
                if not deltas
                else float(np.quantile(np.asarray(deltas, dtype=np.float64), 0.9)),
                "model_interval_coverage": None
                if not model_covered
                else float(sum(model_covered) / len(model_covered)),
                "public_proxy_interval_coverage": None
                if not proxy_covered
                else float(sum(proxy_covered) / len(proxy_covered)),
            }
        )
        output.append(summary)
    return output


def _gate(matched_rows: list[dict[str, Any]], family_rows: list[dict[str, Any]]) -> dict[str, Any]:
    blockers: list[str] = []
    if not matched_rows:
        blockers.append("no_matched_annual_proxy_scores")
    leakage_rows = [
        row
        for row in matched_rows
        if str(row.get("observation_role") or "") != "validation_only"
        or str(row.get("allowed_use") or "") != "validation_only"
    ]
    if leakage_rows:
        blockers.append("annual_proxy_gate_role_leakage")
    overall = next((dict(row) for row in family_rows if str(row.get("model_family") or "") != ""), {})
    model_mean = _finite_float(overall.get("model_mean_norm_error"))
    proxy_mean = _finite_float(overall.get("public_proxy_mean_norm_error"))
    model_coverage = _finite_float(overall.get("model_interval_coverage"))
    proxy_coverage = _finite_float(overall.get("public_proxy_interval_coverage"))
    if model_mean is None or proxy_mean is None:
        blockers.append("model_or_public_proxy_not_evaluable")
    elif model_mean > proxy_mean:
        blockers.append("model_worse_than_public_annual_proxy")
    if model_coverage is not None and proxy_coverage is not None and model_coverage < proxy_coverage:
        blockers.append("model_interval_coverage_worse_than_public_annual_proxy")
    return {
        "status": "annual_model_beats_public_proxy" if not blockers else "annual_model_blocked_by_public_proxy",
        "blockers": blockers,
        "score_row_count": len(matched_rows),
        "model_family": overall.get("model_family"),
        "public_proxy_family": overall.get("public_proxy_family") or PUBLIC_SELECTED_FAMILY,
        "model_mean_norm_error": model_mean,
        "public_proxy_mean_norm_error": proxy_mean,
        "model_minus_public_proxy_mean_norm_error": None
        if model_mean is None or proxy_mean is None
        else float(model_mean - proxy_mean),
        "model_interval_coverage": model_coverage,
        "public_proxy_interval_coverage": proxy_coverage,
        "contract": (
            "R77 treats the R76 train-origin public-domain annual proxy as the open incumbent annual benchmark. "
            "Phase 3 annual superiority claims are blocked unless the model's annual head matches the same "
            "held-out UNAIDS-style annual targets with no higher mean normalized error and no worse interval coverage."
        ),
    }


def run_r77_public_proxy_annual_gate(
    *,
    run_id: str = R77_RUN_ID,
    r75_report_path: Path | None = None,
    r76_report_path: Path | None = None,
) -> dict[str, Any]:
    analysis_dir = ensure_dir(sandbox_repo_root() / "artifacts" / "runs" / str(run_id) / "analysis")
    r75_path = R75_DEFAULT_REPORT if r75_report_path is None else Path(r75_report_path)
    r76_path = R76_DEFAULT_REPORT if r76_report_path is None else Path(r76_report_path)
    r75 = dict(read_json(r75_path, default={}) or {}) if r75_path.exists() else {}
    r76 = dict(read_json(r76_path, default={}) or {}) if r76_path.exists() else {}
    matched_rows = _matched_proxy_rows(r75, r76)
    metric_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family", "metric_name"))
    horizon_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family", "horizon_years"))
    family_rows = _comparison_summary_by_fields(matched_rows, group_fields=("model_family", "public_proxy_family"))
    gate = _gate(matched_rows, family_rows)
    report_path = analysis_dir / "r77_public_proxy_annual_gate_report.json"
    markdown_path = analysis_dir / "r77_public_proxy_annual_gate_report.md"
    matched_csv = analysis_dir / "r77_matched_score_rows.csv"
    family_csv = analysis_dir / "r77_family_rows.csv"
    metric_csv = analysis_dir / "r77_metric_rows.csv"
    horizon_csv = analysis_dir / "r77_horizon_rows.csv"
    report = {
        "schema_version": R77_SCHEMA_VERSION,
        "generated_at": _generated_at(),
        "run_id": run_id,
        "public_proxy_annual_gate": gate,
        "matched_score_rows": matched_rows,
        "family_rows": family_rows,
        "metric_rows": metric_rows,
        "horizon_rows": horizon_rows,
        "source_artifacts": {
            "r75": {"path": r75_path.as_posix(), "sha256": _sha256(r75_path) if r75_path.exists() else None},
            "r76": {"path": r76_path.as_posix(), "sha256": _sha256(r76_path) if r76_path.exists() else None},
        },
        "artifact_paths": {
            "json": report_path.as_posix(),
            "markdown": markdown_path.as_posix(),
            "matched_score_rows_csv": matched_csv.as_posix(),
            "family_rows_csv": family_csv.as_posix(),
            "metric_rows_csv": metric_csv.as_posix(),
            "horizon_rows_csv": horizon_csv.as_posix(),
        },
    }
    _write_csv(matched_csv, matched_rows)
    _write_csv(family_csv, family_rows)
    _write_csv(metric_csv, metric_rows)
    _write_csv(horizon_csv, horizon_rows)
    _write_markdown(markdown_path, report)
    write_json(report_path, report)
    return report


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    gate = dict(report.get("public_proxy_annual_gate") or {})
    lines = [
        "# Phase 3 R77 Public-Proxy Annual Gate",
        "",
        f"Generated: {report.get('generated_at')}",
        "",
        "## Gate",
        "",
        f"- Status: `{gate.get('status')}`",
        f"- Model family: `{gate.get('model_family')}`",
        f"- Public proxy family: `{gate.get('public_proxy_family')}`",
        f"- Model mean normalized error: `{gate.get('model_mean_norm_error')}`",
        f"- Public proxy mean normalized error: `{gate.get('public_proxy_mean_norm_error')}`",
        f"- Delta model minus public proxy: `{gate.get('model_minus_public_proxy_mean_norm_error')}`",
        f"- Model interval coverage: `{gate.get('model_interval_coverage')}`",
        f"- Public proxy interval coverage: `{gate.get('public_proxy_interval_coverage')}`",
        f"- Blockers: `{', '.join(gate.get('blockers') or []) or 'none'}`",
        "",
        "## Metric Comparison",
        "",
        "| Metric | Entries | Model mean | Public proxy mean | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report.get("metric_rows") or []:
        lines.append(
            f"| `{row.get('metric_name')}` | {int(row.get('entry_count') or 0)} | "
            f"{float(row.get('model_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('public_proxy_mean_norm_error') or 0.0):.6f} | "
            f"{float(row.get('model_minus_public_proxy_mean_norm_error') or 0.0):.6f} |"
        )
    lines.extend(["", "## Contract", "", str(gate.get("contract") or ""), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run R77 public-proxy annual gate.")
    parser.add_argument("--run-id", default=R77_RUN_ID)
    parser.add_argument("--r75-report-path", default=None)
    parser.add_argument("--r76-report-path", default=None)
    args = parser.parse_args()
    run_r77_public_proxy_annual_gate(
        run_id=str(args.run_id),
        r75_report_path=None if args.r75_report_path is None else Path(args.r75_report_path),
        r76_report_path=None if args.r76_report_path is None else Path(args.r76_report_path),
    )


if __name__ == "__main__":
    _main()
