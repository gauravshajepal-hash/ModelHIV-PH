# Phase 2 Conversation Extract

Date: 2026-04-08

This file extracts and organizes the Phase 2 related discussion from the prior conversation history. It is not a verbatim transcript. It is a faithful chronological reconstruction of the Phase 2 thread: what was proposed, what math was described, what was implemented, what later drifted, and why the current restore was requested.

## 1. Early Phase 2 explanation

Phase 2 was originally explained as a graph-learning stage over the normalized Phase 1 surface.

The described math was:

- Build a feature matrix `X[c,j]` over province-month cells `c` and canonical variables `j`.
- Blend numeric and soft evidence into a single matrix.
- Use mutual information, correlation filters, and partial correlation skeletons to reduce the candidate edge set.
- Apply structural masks from tiering and lag plausibility.
- Fit a NOTEARS-style DAG over the remaining candidates.
- Project the resulting weighted graph to an exact DAG.
- Extract Markov blankets around target-relevant variables.

In plain English, the early Phase 2 idea was:

1. turn evidence into a cleaned feature matrix,
2. prune obviously bad edges,
3. fit a sparse same-time directed graph,
4. use that graph to decide what matters for Phase 3.

This was the old same-time observable DAG framing.

## 2. Concern raised after Phase 1 and Phase 15 became more serious

Once Phase 1 moved toward learned observation noise and Phase 15 moved toward a structured latent model, the question became:

If Phase 15 now produces better latent block states with temporal structure and uncertainty, should Phase 2 still be a same-time DAG over a blended observable matrix?

The answer discussed in the conversation was: probably not.

The reasoning was:

- Phase 1 is now about evidence normalization and measurement reliability.
- Phase 15 is now about latent state inference with temporal smoothing and mixed-frequency structure.
- Therefore Phase 2 should probably operate on the latent states or latent innovations, not on raw blended observables.

That was the start of the redesign away from NOTEARS on observable features.

## 3. Literature search for a better Phase 2

Phase 2 was then reconsidered using ideas from:

- statistics and econometrics,
- astronomy,
- physics data assimilation,
- computer science / latent state-space modeling.

The main literature-driven conclusion was:

Phase 2 should become a multiscale latent temporal graph rather than a same-time feature graph.

The specific shift was:

- remove own-block persistence first,
- then fit lagged cross-block effects,
- and separate direct sparse effects from hidden shared driver structure.

The mathematical target became:

`epsilon[u,t,b] = z[u,t,b] - phi[b] * z[u,t-1,b]`

then

`epsilon_t ~= lagged_direct_effects + hidden_shared_effects`

where:

- `z[u,t,b]` is the Phase 15 latent state for unit `u`, time `t`, block `b`,
- `phi[b]` is the self-persistence for block `b`,
- `epsilon` is the innovation after removing own momentum.

In plain English:

Phase 2 should stop asking "which variables move together now?" and instead ask "after removing each block's own persistence, which other latent blocks at which lag explain the remaining movement?"

## 4. First redesigned Phase 2 concept

The redesigned Phase 2 was described as:

- province / region / national scale graph estimation,
- over Phase 15 latent states,
- using temporal lags,
- with uncertainty-aware weighting,
- and synthetic recovery / falsification tests.

The core design was:

1. compute latent innovations,
2. build lagged design matrices,
3. fit direct sparse edges,
4. fit low-rank hidden-driver structure separately,
5. keep only stable edges under bootstrap,
6. merge evidence across scales.

This directly tied Phase 2 to Phase 15 instead of leaving it as an unrelated same-time observable DAG.

## 5. Sparse-plus-low-rank temporal graph proposal

The Phase 2 design then became sharper:

Fit a sparse-plus-low-rank temporal operator on the latent innovations.

The discussed math was:

`Y = X (S + L) + error`

with objective

`min_{S,L} (1/2n)||Y - X(S+L)||_F^2 + lambda_s ||S||_1 + lambda_l ||L||_*`

where:

- `S` is the sparse direct lagged graph,
- `L` is the low-rank hidden shared driver structure,
- `||S||_1` promotes sparse direct edges,
- `||L||_*` promotes a small number of hidden shared modes.

In English:

- `S` answers: which block-lag pairs look like direct temporal influences?
- `L` answers: which co-movements are better explained by shared hidden drift rather than direct edges?

This was a much more serious causal-temporal interpretation than the old NOTEARS path.

## 6. Multi-lag and uncertainty-aware upgrade

Phase 2 was later pushed further:

- true multi-lag instead of single-lag,
- uncertainty-aware weighting using Phase 15 posterior uncertainty,
- synthetic recovery and falsification tests,
- explicit lag-labeled edges.

The math discussed was:

1. remove persistence:
   `epsilon[u,t,b] = z[u,t,b] - phi[b] z[u,t-1,b]`

2. build lagged design:
   `X = [z_{t-1}, z_{t-2}, ..., z_{t-L}]`

3. fit:
   `min_{S,L} (1/2n)||Y - X(S+L)||_F^2 + lambda_s ||S||_1 + lambda_l ||L||_*`

4. weight samples by inverse uncertainty:
   noisier latent months contribute less.

In plain English:

Phase 2 became:

- innovation modeling,
- multi-lag,
- uncertainty-aware,
- direct effects and hidden effects kept separate.

## 7. How Phase 2 was supposed to feed Phase 3

After the Phase 2 redesign, the discussion turned to how Phase 3 should use it.

The proposed use was not:

- "Phase 2 gives truth"

but:

- "Phase 2 gives structural priors and eligibility gates."

The suggested Phase 3 integration was:

- sparse direct Phase 2 edges become priors on hazard-transition effects,
- low-rank hidden Phase 2 structure becomes separate latent shock channels,
- blanket membership becomes an eligibility / shrinkage mechanism,
- uncertainty and stability in Phase 2 control prior strength in Phase 3.

So Phase 2 was no longer just a feature selector. It became a structure-learning layer whose outputs would constrain or inform Phase 3 transition equations.

## 8. Council and memo perspective on Phase 2

Later council review judged the updated Phase 2 as:

- much better than the old same-time observable DAG,
- materially aligned with the newer Phase 15 math,
- but still not mechanistic truth.

The council view was:

- direct edges from Phase 2 are temporal hypotheses, not established mechanisms,
- hidden-driver structure should be treated separately,
- uncertainty must be explicit,
- synthetic falsification matters.

The main recommendation was to keep Phase 2 as:

- latent,
- temporal,
- multiscale,
- uncertainty-aware,

and avoid collapsing back to a generic same-time NOTEARS feature graph.

## 9. Drift discovered in the actual repository

Later in the conversation, the repository was audited and it was discovered that the current tracked `phase2` source no longer matched the Phase 2 described above.

What the audit found:

- tests, plugin config, trust audit, and downstream Phase 3 code still expected:
  - `latent_temporal_graph.py`
  - `multiscale_dag.py`
  - `latent_temporal_graph_bundle.json`
  - `multiscale_dag_bundle.json`
- but the tracked source had drifted to:
  - same-time NOTEARS DAG code,
  - block graph builder,
  - shard summary merge,
  - rescue-v2 tournament logic.

In plain English:

The repo entered a split-brain state.

One part of the repo still believed Phase 2 was the latent-temporal/multiscale design.
Another part had reverted or moved to a different NOTEARS/block-graph architecture.

That inconsistency was the direct reason the restore was requested.

## 10. Decision: restore latent-temporal / multiscale Phase 2

The requested direction then became explicit:

- restore the latent-temporal-graph/multiscale Phase 2 path,
- remove the current NOTEARS/block-graph Phase 2,
- eliminate confusion and make the code match the later mathematical direction.

The intended restored contract was:

- `multiscale_dag.py`
- `latent_temporal_graph.py`
- a real Phase 2 pipeline that emits:
  - `multiscale_dag_bundle.json`
  - `multiscale_phase3_target_blankets.json`
  - `latent_temporal_graph_bundle.json`
  - `latent_temporal_phase3_target_blankets.json`
  - `latent_temporal_graph_validation.json`

Plus a small compatibility surface for older Phase 3 / Phase 4 consumers.

## 11. Restore work that was then carried out

The Phase 2 restore effort did the following:

### 11.1 Removed the NOTEARS/block-graph Phase 2 source

The following code paths were identified as belonging to the old divergent Phase 2 surface and were removed from the live source:

- `block_graph_builder.py`
- `rescue_profile.py`
- `shard_summary.py`

The CLI imports for shard-summary merging were also removed.

### 11.2 Reintroduced the restored Phase 2 structure

The restored structure consisted of:

- `latent_temporal_graph.py`
- `multiscale_dag.py`
- `phase3_compat.py`
- a rewritten `pipeline.py`

The new `pipeline.py` was designed to:

1. read Phase 15 outputs,
2. build multiscale DAG outputs,
3. build latent temporal graph outputs,
4. emit validation artifacts,
5. emit a compatibility layer for older Phase 3 / Phase 4 paths.

### 11.3 Added compatibility bridging

Because parts of Phase 3 and Phase 4 still consumed older filenames and shapes, a compatibility builder was added.

This compatibility layer reconstructs artifacts such as:

- `candidate_profiles.json`
- `curated_candidate_blocks.json`
- `markov_blanket.json`
- `edge_scores.json`
- `ranked_linkages.json`
- `phase3_target_blankets.json`
- retained factor sets
- `core_feature_tensor.npz`

The purpose was:

- keep the restored Phase 2 math,
- while not instantly breaking older downstream code.

## 12. Estimator failure discovered during restore

When the restored `latent_temporal_graph.py` was tested against its synthetic unit tests, the first implementation failed.

The failure mode was:

- true direct lagged edges were being pushed almost entirely into the low-rank component,
- so the direct edge list came out empty,
- even though the hidden-driver rows showed the expected relationships.

That revealed an important design lesson:

If the sparse-plus-low-rank decomposition is not carefully structured, the low-rank term can steal real direct signal.

## 13. Fix to the estimator

The restore then changed the decomposition strategy.

Instead of letting the low-rank term absorb strong direct structure during iterative joint updates, the estimator switched to a coefficient-first decomposition:

1. fit a weighted ridge coefficient matrix,
2. extract the sparse direct component by thresholding the coefficient matrix,
3. fit the low-rank component on the residual coefficient matrix,
4. keep self-edges excluded,
5. keep bootstrap stability.

In plain English:

- fit the full lagged map first,
- keep strong direct effects in the direct graph,
- only let the low-rank term absorb what is left over.

This restored the expected behavior on:

- direct recovery tests,
- multi-lag recovery tests,
- falsification tests.

## 14. Real cached run check

After the estimator fix, the restored Phase 2 was run against a real cached Phase 15 output (`smoke-latent-blocks`).

This revealed two additional practical issues:

### 14.1 Manifest/backend-status bug

The restored pipeline was writing `Phase0BackendStatus` with the wrong field (`enabled` instead of `selected`).

This was fixed.

### 14.2 Missing import bug

The restored pipeline used `read_json(...)` late in the build but did not import it.

This was fixed.

### 14.3 Output contamination from old Phase 2 files

Because the run directory already contained the old NOTEARS/block-graph artifacts, rebuilding Phase 2 into the same run left those stale files on disk.

That created on-disk confusion even though the live source had been restored.

The conclusion from that discussion was:

The restored Phase 2 build should clean the Phase 2 output directory before writing the new contract so the restored latent-temporal/multiscale path becomes the only visible Phase 2 artifact surface.

## 15. Factor-ID mismatch discovered during restore

During the real cached run, another reconciliation issue appeared:

- `factor_promotion_pool.json` and `multiscale_factor_catalog.json` were exposing different factor-id dialects.
- That caused the multiscale DAG selection to come through empty even when valid multiscale factors existed.

The restore then added a fallback:

- if the merged promotion-pool selection does not intersect the multiscale factor catalog,
- fall back to selecting retained factors directly from the multiscale catalog,
- prioritizing hard-passing factors with transition hooks or meaningful targets.

In plain English:

When Phase 15 exposes two factor namespaces, the restored Phase 2 should prefer the namespace that actually exists in the multiscale tensors.

## 16. Final Phase 2 design direction from the conversation

The final Phase 2 direction, as established by the conversation, is:

### What Phase 2 should be

- latent-state based,
- temporal,
- multiscale,
- uncertainty-aware,
- direct sparse and hidden low-rank structure kept separate,
- synthetically validated,
- usable as a structural prior layer for Phase 3.

### What Phase 2 should not be

- a same-time NOTEARS DAG over a blended observable matrix,
- a block-graph or shard-survivor architecture that contradicts the rest of the repo,
- a source tree that disagrees with tests, config, trust audit, and downstream consumers.

### What Phase 2 should emit

Core restored artifacts:

- `multiscale_dag_bundle.json`
- `multiscale_phase3_target_blankets.json`
- `latent_temporal_graph_bundle.json`
- `latent_temporal_phase3_target_blankets.json`
- `latent_temporal_graph_validation.json`

Compatibility artifacts:

- `candidate_profiles.json`
- `curated_candidate_blocks.json`
- `markov_blanket.json`
- `edge_scores.json`
- `ranked_linkages.json`
- `phase3_target_blankets.json`
- retained factor sets
- `core_feature_tensor.npz`

### What Phase 3 is expected to do with Phase 2

- use direct lagged edges as structured priors or eligibility channels,
- use hidden-driver structure as latent shock channels,
- use blanket membership as a gating / shrinkage device,
- use uncertainty and bootstrap stability as prior-strength controls.

## 17. Current interpretation for a reader

If you want the short version of the entire Phase 2 conversation:

1. Phase 2 started as a same-time observable DAG.
2. Once Phase 1 and Phase 15 got mathematically stronger, that old design stopped making sense.
3. Literature and internal reasoning pushed Phase 2 toward a latent temporal graph over Phase 15 states.
4. That newer design later drifted out of the tracked source, while tests and downstream code still expected it.
5. The restore effort was therefore not a cosmetic refactor. It was an attempt to realign the code with the later mathematical plan.
6. The restored design keeps:
   - multiscale factor graph outputs,
   - latent temporal block graph outputs,
   - compatibility bridges for legacy consumers.
7. The lasting Phase 2 principle is:

`Phase 2 should model temporal latent structure, not just same-time feature association.`

