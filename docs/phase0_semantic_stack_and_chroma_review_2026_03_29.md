# Phase 0 Semantic Stack and Chroma Review

## Decision

Keep:

- DuckDB
- FAISS
- Parquet

Upgrade:

- default Phase 0 embedding from `hashed_local` to local `sentence-transformers/all-MiniLM-L6-v2`

Do not adopt Chroma as the default retrieval layer for semantic quality reasons.

## What changed in code

- Phase 0 now defaults to a real local embedder when available:
  - `sentence-transformers/all-MiniLM-L6-v2`
- JSON row tables now get Parquet sidecars automatically.
- Query banks now include explicit determinant silos for:
  - cash instability
  - labor migration
  - housing precarity
  - education
  - social capital
  - policy implementation weakness
  - transport friction
  - remoteness
  - congestion / travel time
  - sexual risk / collective risk behavior

## Fresh upgrade artifact

Bounded Phase 0 rebuild:

- [index_manifest.json](/D:/EpiGraph_PH/artifacts/runs/phase0-embed-upgrade-20260329/phase0/index/index_manifest.json)
- [alignment_summary.json](/D:/EpiGraph_PH/artifacts/runs/phase0-embed-upgrade-20260329/phase0/extracted/alignment_summary.json)

Observed:

- embedding backend: `sentence_transformers`
- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- fallback used: `false`
- FAISS selected: `true`
- Chroma available: `true`

## Quality benchmark

Large corpus benchmark on the current all-province run:

- [semantic_quality_benchmark.json](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/phase0/analysis/semantic_quality_benchmark.json)
- [semantic_quality_benchmark.md](/D:/EpiGraph_PH/artifacts/runs/bounded-v2-replay-allprovinces-harparchive-20260329/phase0/analysis/semantic_quality_benchmark.md)

Mean metrics at `top_k = 10`:

- `hashed_local_faiss`
  - ndcg@10: `0.3946`
  - avg relevance@10: `0.3278`
- `local_embedder_faiss`
  - ndcg@10: `0.7291`
  - avg relevance@10: `0.83`
- `local_embedder_chroma`
  - ndcg@10: `0.5686`
  - avg relevance@10: `0.6653`

Interpretation:

- replacing hashed retrieval with the local embedder produced the main quality jump
- on the same embedding vectors, FAISS beat Chroma on this larger corpus
- Chroma did not add semantic quality by itself

## Small fresh-corpus check

Fresh bounded Phase 0 benchmark:

- [semantic_quality_benchmark.json](/D:/EpiGraph_PH/artifacts/runs/phase0-embed-upgrade-20260329/phase0/analysis/semantic_quality_benchmark.json)

Mean metrics at `top_k = 10`:

- `hashed_local_faiss`
  - ndcg@10: `0.8319`
  - avg relevance@10: `0.7244`
- `local_embedder_faiss`
  - ndcg@10: `0.8394`
  - avg relevance@10: `0.8903`
- `local_embedder_chroma`
  - ndcg@10: `0.8394`
  - avg relevance@10: `0.8903`

Interpretation:

- on a smaller cleaner corpus, FAISS and Chroma were effectively identical when fed the same embeddings
- this supports the stronger conclusion:
  - Chroma is a storage / collection / filtering layer
  - it is not the main source of semantic intelligence

## Broad-silo check

Bounded broad-corpus benchmark with the new determinant silos present:

- [semantic_quality_benchmark.json](/D:/EpiGraph_PH/artifacts/runs/phase0-embed-silos-broad-20260329/phase0/analysis/semantic_quality_benchmark.json)

Mean metrics at `top_k = 10`:

- `hashed_local_faiss`
  - ndcg@10: `0.791`
  - avg relevance@10: `0.7094`
- `local_embedder_faiss`
  - ndcg@10: `0.8546`
  - avg relevance@10: `0.9737`
- `local_embedder_chroma`
  - ndcg@10: `0.8546`
  - avg relevance@10: `0.9737`
- `local_embedder_chroma_filtered`
  - ndcg@10: `1.0`
  - avg relevance@10: `1.192`

Interpretation:

- the local embedder still explains the main semantic gain over hashed retrieval
- raw Chroma and raw FAISS are again effectively identical on the same embeddings
- Chroma only adds value here when the corpus carries useful, explicit silo metadata and the query can use that metadata as a filter

That means:

- **default pipeline**: keep DuckDB + Parquet + FAISS
- **optional research mode**: Chroma can be useful for metadata-filtered exploratory retrieval once Phase 0 silo labels are rich enough

## Why Chroma is not the default

The benchmark result is the main reason:

- same embeddings, no semantic gain over FAISS
- sometimes worse on the larger corpus

Chroma may still be useful later for:

- collection management
- metadata filtering
- retrieval debugging
- interactive exploration

But for the research pipeline default:

- DuckDB handles tabular persistence
- Parquet handles reproducible stage tables
- FAISS handles vector retrieval
- the local embedder determines semantic quality

That stack is simpler and better supported by the current evidence.
