from __future__ import annotations

from collections import defaultdict
from typing import Any


def _document_group_key(document_row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(document_row.get("platform") or "unknown"),
        str(document_row.get("query_silo") or ""),
        str(document_row.get("source_tier") or ""),
    )


def _document_sort_key(document_row: dict[str, Any]) -> tuple[int, int, int, str]:
    anchor_rank = 1 if bool(document_row.get("is_anchor_eligible")) else 0
    pdf_rank = 1 if str(document_row.get("snapshot_type") or "").lower() == "pdf" else 0
    year_value = int(document_row.get("year") or 0)
    return (-anchor_rank, -pdf_rank, -year_value, str(document_row.get("document_id") or ""))


def select_document_shard(
    documents: list[dict[str, Any]],
    *,
    shard_count: int,
    shard_index: int,
    max_documents: int | None = None,
) -> list[dict[str, Any]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError("shard_index must be within [0, shard_count)")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in documents:
        grouped[_document_group_key(row)].append(dict(row))
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for group_key in sorted(grouped):
        rows = grouped[group_key]
        rows.sort(key=_document_sort_key)
        for offset, row in enumerate(rows):
            assigned[offset % shard_count].append(row)
    selected = assigned[shard_index]
    selected.sort(key=_document_sort_key)
    if max_documents is not None:
        selected = selected[: max(0, int(max_documents))]
    return selected


def build_slice_payload(
    *,
    source_rows: list[dict[str, Any]],
    document_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    query_rows: list[dict[str, Any]],
    shard_count: int,
    shard_index: int,
    max_documents: int | None = None,
) -> dict[str, Any]:
    selected_documents = select_document_shard(
        document_rows,
        shard_count=shard_count,
        shard_index=shard_index,
        max_documents=max_documents,
    )
    selected_source_ids = {str(row.get("source_id") or "") for row in selected_documents}
    selected_source_rows = [dict(row) for row in source_rows if str(row.get("source_id") or "") in selected_source_ids]
    selected_sweep_rows = [
        dict(row)
        for row in sweep_rows
        if str(row.get("record_id") or row.get("source_id") or "") in selected_source_ids
    ]
    query_signatures = {
        (
            str(row.get("query") or ""),
            str(row.get("query_domain") or ""),
            str(row.get("query_lane") or ""),
            str(row.get("query_geo_focus") or ""),
            str(row.get("query_silo") or ""),
        )
        for row in selected_source_rows
    }
    selected_query_rows = [
        dict(row)
        for row in query_rows
        if (
            str(row.get("query") or ""),
            str(row.get("query_domain") or ""),
            str(row.get("query_lane") or ""),
            str(row.get("query_geo_focus") or ""),
            str(row.get("query_silo") or ""),
        )
        in query_signatures
    ]
    platform_counts: dict[str, int] = defaultdict(int)
    for row in selected_documents:
        platform_counts[str(row.get("platform") or "unknown")] += 1
    return {
        "source_rows": selected_source_rows,
        "document_rows": selected_documents,
        "sweep_rows": selected_sweep_rows,
        "query_rows": selected_query_rows,
        "summary": {
            "shard_count": int(shard_count),
            "shard_index": int(shard_index),
            "source_count": len(selected_source_rows),
            "document_count": len(selected_documents),
            "sweep_record_count": len(selected_sweep_rows),
            "query_count": len(selected_query_rows),
            "platform_counts": dict(sorted(platform_counts.items())),
        },
    }
