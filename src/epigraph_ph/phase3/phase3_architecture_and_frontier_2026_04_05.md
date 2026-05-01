# Phase 3 Architecture And Research Frontier

Date: 2026-04-05

Scope:
- explain what the current `src/epigraph_ph/phase3` folder is doing
- explain which Phase 3 regime is currently winning
- explain why `transition_research` should become the main research frontier
- explain what should remain in `rescue_core`
- explain how Phase 2 should enter the next Phase 3 mathematical model

This note is intentionally explicit. The goal is not only to record the current code state, but to make the folder structure itself defensible.

## Executive Judgment

The current `phase3` folder contains more than one mathematical program.

Those programs are not equivalent:

1. `rescue_core.py` is the broad multiyear, multilevel, general-purpose Phase 3 model.
2. `transition_research/` is the current winning forecasting branch family.
3. `incidence_research/` is a dependent extension layered on a kept transition branch.
4. `national_reset_core.py` and `national_reset_pipeline.py` are precursor infrastructure that helped produce the transition family.
5. `peak_search.py` is not the main model. It is an evaluation and search utility around candidate temporal structures and backtests.

The correct strategic conclusion is:

- do not delete `rescue_core`
- do not keep treating `rescue_core` as the leading research frontier
- do treat `transition_research` as the leading Phase 3 research frontier
- do keep `incidence_research` as a dependent branch family, not as the core forecasting path
- do extract reusable infrastructure from `rescue_core` into shared top-level Phase 3 utilities instead of copying large amounts of code into `transition_research`

In short:

`rescue_core` should become the broad benchmark and infrastructure reservoir.

`transition_research` should become the primary Phase 3 experimental core.

`incidence_research` should remain a downstream branch that inherits from a kept transition winner.

## What The Current Folder Actually Contains

The current top-level layout is:

- `rescue_core.py`
- `pipeline.py`
- `mixed_frequency.py`
- `evaluation_regimes.py`
- `broad_backtest_support.py`
- `national_reset_core.py`
- `national_reset_pipeline.py`
- `peak_search.py`
- `temporal_scaffold.py`
- `transition_research/`
- `incidence_research/`

This is not redundancy by accident. It reflects a real historical progression of ideas.

### 1. `rescue_core.py`

This file is the broadest and most ambitious Phase 3 model currently in the repo.

It tries to do all of the following at once:

- build a multilevel HIV cascade latent state
- model province, subgroup, and monthly structure
- incorporate a wide determinant surface
- handle mixed-frequency observations
- compare against broad frozen-history baselines

Mathematically, this is the most general branch.

Operationally, however, it is not the current winner on the kept broad benchmark regime.

So `rescue_core.py` is scientifically valuable, but it is not currently the best forecasting frontier.

Its main roles should now be:

- broad benchmark model
- source of reusable infrastructure
- long-horizon target for later reintegration

It should not be the default place for the next research increment.

### 2. `pipeline.py`

This file is the Phase 3 orchestration layer.

It should remain the place where all Phase 3 regimes are surfaced and evaluated. It is the correct place for:

- frozen-history rescue-core backtests
- tournament selection across broad representations
- transition research entry points
- incidence research entry points

The current top-level pipeline is not the problem. The issue is which mathematical branch should be considered the research frontier.

### 3. `mixed_frequency.py`

This is one of the most reusable top-level assets in Phase 3.

It encodes observation aggregation logic across different time scales. That is general infrastructure. It should not belong only to `rescue_core`.

This file should remain top-level because both:

- broad `rescue_core`
- future `transition_research v2`

need explicit mixed-frequency observation operators.

This is one of the first files that should be reused by the transition frontier.

### 4. `evaluation_regimes.py`

This file is correct and should stay.

It cleanly separates the three kept regimes:

- broad multiyear frozen-history rescue-core
- quarter-level national transition-research branches
- diagnosis-locked incidence branch inheritance

This is important because much of the earlier confusion came from mixing incompatible evaluation families.

This file is one of the architectural fixes that should remain stable.

### 5. `broad_backtest_support.py`

This file exists to support the kept broad frozen-history evaluation family.

That is correct.

The broad benchmark is still scientifically necessary even if it is not the current winner, because:

- it prevents over-claiming narrow branch wins
- it measures whether new transition-family ideas can eventually generalize
- it is the correct stress test for the more ambitious Phase 3 program

This file should remain top-level and should not be folded into `transition_research`.

### 6. `national_reset_core.py` and `national_reset_pipeline.py`

These files are predecessor infrastructure for the winning transition line.

They are not the current winning frontier by themselves, but they are historically and mathematically important because they helped define the national quarter-level state reconstruction regime that transition research later improved.

They still make sense in the current folder because:

- they anchor the national quarter-level `U/D/A/...` style reconstruction logic
- the transition branch explicitly uses that lineage
- they remain useful as simplified reference models and diagnostics

They should not be deleted casually.

However, they should be treated as support and lineage modules, not as the primary experimental future.

### 7. `peak_search.py`

This file is not the main model. It is a search/evaluation helper.

Its role is still legitimate because the winning transition branch family relies on temporal structure selection and out-of-sample checks around peak-like dynamics.

So `peak_search.py` should remain as tooling, but it should not be confused with the main forecasting model.

### 8. `transition_research/`

This folder is now the most important folder in Phase 3.

It contains the branch family that currently produces the strongest kept forecasting results in its own valid evaluation regime.

This folder is not just "more experiments." It contains a different modeling philosophy:

- quarter-level national state evolution
- anchored holdout forecasting
- explicit transition hazards
- decomposition into trend/shock/residual channels
- support-aware driver fusion
- explicit detector gating
- narrow but better-identified overlays such as age-conditioned diagnosis modification

This is currently the strongest research frontier in Phase 3.

### 9. `incidence_research/`

This folder is not independent of the winning transition line.

Its kept branch inherits a locked transition winner and then adds explicit pre-`U` incidence accounting.

So this folder should remain, but conceptually it should be treated as:

- a dependent branch
- an incidence-specific extension
- not the main baseline forecasting core

This distinction matters because otherwise it becomes easy to overstate incidence wins that are really inherited transition wins.

## Which Regimes Are Currently Valid

The kept evaluation regimes are:

1. broad multiyear frozen-history rescue-core
2. quarter-level national transition-research branches
3. diagnosis-locked incidence branch inheritance

That separation is correct and should remain explicit.

The important point is that these regimes answer different questions.

### Broad Frozen-History Rescue-Core

This is the hardest and broadest test.

It asks:

"Can the broad multilevel mechanistic Phase 3 program beat simple baselines under broad multiyear frozen-history forecasting?"

At the moment, the answer is no.

This means:

- the broad ambition is not yet delivering the best predictive behavior
- the broad regime should remain the honesty benchmark
- it should not currently dominate the research direction

### Quarter-Level National Transition Research

This is the current winning branch family.

It asks:

"Can a tighter, better-identified national quarter-level mechanistic transition model beat carry-forward and simple compartmental baselines?"

At the moment, the answer is yes, for a substantial part of the kept branch family.

This means:

- this branch has earned promotion to primary Phase 3 research frontier
- new mathematical ideas should be tested here first

### Diagnosis-Locked Incidence Branch Inheritance

This asks:

"If the transition path is locked to a kept winner, can we expose an explicit incidence layer and audit compatibility?"

This is scientifically useful.

But because it is inheritance-based, it should not be mistaken for the main forecasting path.

## Why Transition Research Is Winning

The transition family is winning because it is doing less, but doing the right less.

It is more identified.

It is narrower.

It imposes fewer weakly supported moving parts.

It respects empirical support more directly.

The winning lineage is approximately:

`MECH-01A -> MECH-01E -> DECOMP-01E/F -> PEAK-01F -> AGE-01B -> INC-01D`

The forecasting winner is not `INC-01D`.
The forecasting frontier is the `PEAK/AGE/DECOMP` branch family, with `AGE-01B` as the most compelling next base because it inherits `PEAK-01F` and improves slightly further.

### Why `MECH-01A` mattered

`MECH-01A` established the national `U/D/A/V/L` baseline on top of the front-half forecast lineage.

This created a quarter-level mechanistic baseline that was easier to reason about than the full broad rescue-core stack.

### Why `MECH-01E` mattered

`MECH-01E` anchored downstream residual corrections while keeping the diagnosis path locked.

This is important mathematically because it reduced one major source of instability:

- do not let every transition move freely at once
- lock the strongest path
- improve only the downstream parts that remain misspecified

That is a better identified update than broad simultaneous modifier fitting.

### Why `DECOMP-01E/F` mattered

The decomposition line separated hazard behavior into:

- trend
- shock
- residual

This is one of the key mathematical ideas that should survive into the future Phase 3 frontier.

Why it works:

- it separates persistent structure from transient structure
- it prevents all variation from being dumped into one undifferentiated coefficient layer
- it matches what we already learned in Phase 2: direct effects and hidden shared structure should not be conflated

### Why `PEAK-01F` mattered

`PEAK-01F` adds region-plus-KP-modifier detector gating around supported peak windows.

This is not just "more features." It is selective activation.

That matters because it avoids a common failure mode:

- a driver appears useful only in a narrow temporal regime
- broad models smear it everywhere
- performance degrades

`PEAK-01F` instead says:

- only activate the fused driver structure where the detector says there is support
- otherwise fall back to the kept branch baseline

That is a scientifically disciplined use of auxiliary structure.

### Why `AGE-01B` matters

`AGE-01B` is the current best candidate for the next base frontier.

It does something very important mathematically:

- it adds a narrow, support-aware youth diagnosis modifier
- only on top of a kept winning branch
- with leave-one-out gating
- without reopening unsupported downstream structure

This is the right experimental philosophy.

It is not trying to solve everything at once.
It is changing a single empirically plausible channel and checking if that helps.

That is exactly how the next Phase 3 research layer should proceed.

## Why Rescue Core Is Not The Right Frontier Right Now

`rescue_core.py` is still valuable, but it has several structural disadvantages relative to the winning transition line.

### 1. It is too broad relative to current support

The broad rescue-core program is trying to model:

- many geographies
- many subgroups
- many timescales
- many determinants
- a broad observation surface

But the repo still has limited truly decisive support in many of those dimensions.

This increases flexibility faster than identifiability.

### 2. Phase 2 enters too weakly

At present, broad rescue-core still uses Phase 2 mostly as:

`selected determinants -> transition modifiers`

That is weaker than what the current Phase 2 mathematics now provides.

Phase 2 is no longer just a feature selector.
It now produces:

- sparse direct lagged temporal structure
- low-rank hidden shared driver structure
- multiscale support/stability summaries

Those objects deserve a structural role, not just a modifier-covariate role.

### 3. It is harder to diagnose

Because rescue-core is broad, when it underperforms it is harder to tell whether the problem is:

- observation handling
- latent trajectory misspecification
- determinant overfitting
- subgroup prior mismatch
- transition coupling
- pooling imbalance

The transition family is currently much easier to diagnose and therefore much easier to improve.

## What Should Remain In Rescue Core

The answer is not "abandon rescue-core."

The answer is to redefine its role.

`rescue_core` should keep three jobs.

### Job 1: broad benchmark

It should remain the broad frozen-history benchmark family.

This is the honesty constraint on the whole project.

### Job 2: infrastructure reservoir

It contains useful components that should be extracted into reusable Phase 3 utilities:

- mixed-frequency observation logic
- anchor packaging and reference arrays
- evaluation/reporting logic
- possibly some subgroup prior packaging

Those parts should be refactored into shared top-level modules, not copied by hand.

### Job 3: future broad reintegration target

If a new transition-based frontier consistently wins, then later the broad rescue-core program can be rebuilt around that stronger hazard logic.

That is the correct direction of inheritance:

not "make transition research look more like rescue-core"

but

"rebuild broad rescue-core later around what transition research proved works."

## What Should Become The New Phase 3 Frontier

The next research frontier should be:

`transition_research v2`

built on top of the current winning transition branch.

The best starting base is:

`AGE-01B`

because it is:

- already downstream of `PEAK-01F`
- already stronger than the simple baselines in its regime
- already using support-aware gating logic
- still simple enough to extend without losing identifiability

## How Phase 2 Should Enter The New Phase 3 Math

This is the key mathematical point.

The new Phase 3 model should not merely add Phase 2 outputs as additional covariates.

That would underuse what Phase 2 now represents.

Phase 2 currently gives two distinct things:

1. sparse direct lagged temporal graph structure
2. low-rank hidden shared driver structure

Those should enter Phase 3 differently.

### State evolution

Let the quarter-level national state be:

`x_t = (U_t, D_t, A_t, V_t, L_t)`

and let transitions be:

- `U_to_D`
- `D_to_A`
- `A_to_V`
- `A_to_L`
- `L_to_A`

Then the system evolves as:

`x_{t+1} = F(x_t, h_t)`

where `h_t` is the vector of transition hazards for quarter `t`.

### Hazard model with Phase 2 direct edges

For transition `r`, define:

`logit h_r(t) = alpha_r(t) + base_r(t) + sum_{b,l} Gamma[r,b,l] z_b(t-l)`

where:

- `z_b(t-l)` is a Phase 15 latent block state aggregated to the national quarter scale
- `Gamma[r,b,l]` is the contribution of latent block `b` at lag `l` to transition `r`

This is how sparse direct Phase 2 edges should enter:

- not as generic modifiers
- but as structured lagged hazard priors

### Hazard model with hidden shared shocks

Now add low-rank shared structure:

`u_m(t) = rho_m u_m(t-1) + xi_m(t)`

and

`logit h_r(t) = ... + sum_m Lambda[r,m] u_m(t)`

This is how the low-rank Phase 2 component should enter:

- as latent shared shock channels
- separate from the sparse direct graph

That separation is important.

It prevents the model from confusing:

- direct effect of block A on transition r

with

- common hidden force moving several hazards together

### Phase 2 support should control prior strength

The graph should not be treated as exact truth.

Instead:

`Gamma[r,b,l] ~ Normal(mu_phase2[r,b,l], sigma_phase2[r,b,l]^2)`

where:

- `mu_phase2` comes from the signed/estimated Phase 2 relation
- `sigma_phase2` is small when support/stability is high
- `sigma_phase2` is large when support/stability is weak

This is the mathematically safe use of Phase 2:

- it informs the prior
- it does not hard-code the answer

### Mixed-frequency observations should remain explicit

The observation layer should be:

`y_i = H_i(x_{1:T}) + eps_i`

where `H_i` can represent:

- quarterly national stock observation
- annual aggregate anchor
- regime-specific derived measurement

This is why `mixed_frequency.py` should become a shared dependency of the transition frontier.

## Folder-Level Strategic Recommendations

This section is the direct answer to "why the current folder?"

### Keep `phase3/` top-level modules for shared infrastructure

Keep at the top level:

- `pipeline.py`
- `evaluation_regimes.py`
- `broad_backtest_support.py`
- `mixed_frequency.py`
- `temporal_scaffold.py`

Reason:
these are not branch-specific ideas. They are Phase 3 platform infrastructure.

### Keep `rescue_core.py`, but demote its strategic role

Keep it because:

- it is still the broad benchmark
- it contains useful reusable machinery
- it remains the long-run reintegration target

Demote it because:

- it is not currently the winning frontier
- it should not dictate the next research increment

### Keep `transition_research/`, and promote it

This folder should become the main research frontier.

Reason:
- it is where the current wins live
- it has a cleaner hazard-based experimental structure
- it is better aligned with how Phase 2 should enter Phase 3

### Keep `incidence_research/`, but keep it downstream

Reason:
- it is useful for explicit incidence accounting
- but it currently inherits a kept transition winner
- so it should be treated as dependent, not foundational

### Keep `national_reset_*`, but classify them as lineage/support

Reason:
- they are historically part of the winning lineage
- they are useful as simplified reference infrastructure
- but they are not the next frontier branch

### Keep `peak_search.py` as tooling

Reason:
- the peak-gating logic proved useful
- but the file is a search/helper layer, not the main model core

## Concrete Refactor Direction

The refactor should be:

### Phase 3 platform

Shared top-level Phase 3 platform:

- observation operators
- evaluation regimes
- benchmark support
- temporal basis helpers
- maybe anchor packaging

### Transition frontier

Primary modeling frontier:

- `transition_research/`
- add a `v2` branch or equivalent new experiment family here

### Incidence extension

Dependent extension:

- `incidence_research/`
- explicitly inherit from the kept transition winner

### Broad benchmark

Broad benchmark:

- `rescue_core.py`
- keep broad frozen-history evaluation here

That is the clean architecture.

## Proposed Next Experiment Ladder

### TR-V2-00

Exact locked reproduction of `AGE-01B`.

Purpose:
- establish an immutable winning baseline for the next research layer

### TR-V2-01

Add Phase 2 sparse direct edges as lagged hazard priors only.

Purpose:
- test whether direct graph structure improves the winning branch without destabilizing it

### TR-V2-02

Add low-rank hidden shared shock terms as separate latent hazard channels.

Purpose:
- capture common unobserved regime motion without polluting direct hazard coefficients

### TR-V2-03

Ablation study:

- no Phase 2
- direct edges only
- hidden shocks only
- both direct edges and hidden shocks
- each with and without peak gating

Purpose:
- determine whether the current improvement is mostly from peak gating, mostly from graph-informed hazard structure, or from the combination

### TR-V2-04

Promote from national quarter-level to region-pooled partial pooling.

Purpose:
- widen the winning branch only after the national version is stable

## Final Conclusion

The current Phase 3 folder makes sense, but only if it is interpreted correctly.

It is not a single model.
It is a platform containing:

- one broad benchmark family
- one currently winning research family
- one dependent incidence extension
- one precursor national-reset lineage
- several useful shared utilities

The key strategic error would be to keep pushing the broad `rescue_core` branch as though it were already the winning research frontier.

The key strategic correction is:

- make `transition_research` the main Phase 3 frontier
- extract shared infrastructure from `rescue_core`
- later rebuild the broad branch around the winning transition logic if and only if those improvements survive broader evaluation

The right inheritance direction is therefore:

`winning transition branch -> future broad Phase 3 rebuild`

not

`broad rescue_core -> transition branch`.

That is the scientifically and architecturally correct interpretation of the current `phase3` folder.
