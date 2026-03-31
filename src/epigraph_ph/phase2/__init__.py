from __future__ import annotations

from .pipeline import run_phase2_build
from .shard_summary import merge_phase2_shard_factor_summaries


def run_phase2_merge_shard_summaries(
    *,
    run_id: str,
    plugin_id: str,
    source_run_ids: list[str],
    bridge_edge_budget_per_block_pair: int = 4,
):
    return merge_phase2_shard_factor_summaries(
        run_id=run_id,
        plugin_id=plugin_id,
        source_run_ids=source_run_ids,
        bridge_edge_budget_per_block_pair=bridge_edge_budget_per_block_pair,
    )


__all__ = ["run_phase2_build", "run_phase2_merge_shard_summaries"]
