# TR-V3 Paper Comparison Memo

Date: 2026-04-12

Primary comparison target:
- [phase3_tr_v3_autoresearch_design_2026_04_10.md](/D:/EpiGraph_PH/src/epigraph_ph/phase3/phase3_tr_v3_autoresearch_design_2026_04_10.md)

Reviewed papers:
- [Lane et al., 2018 preprint: Building Key Populations HIV Cascades in Data-Scarce Environments](https://www.biorxiv.org/content/10.1101/452417v1.full)
- [Romero-Severson et al., 2016 preprint: Inference of direction, diversity, and frequency of HIV-1 transmission using approximate Bayesian computation](https://www.biorxiv.org/content/10.1101/071050v1.full)
- [Dale and Guo, 2017 preprint / 2018 PLOS ONE paper](https://www.biorxiv.org/content/10.1101/219832v1.full)
- [Mittler et al., 2017 preprint / 2020 PLOS Computational Biology paper](https://www.biorxiv.org/content/10.1101/207126v1.full)
- [Johnson et al., 2018 preprint: MicroCOSM](https://www.biorxiv.org/content/10.1101/310763v1.full)
- [How (Not) to Hybridize Neural and Mechanistic Models for Epidemiological Forecasting](https://arxiv.org/html/2602.06323v1)

## 1. Honesty note on what was actually accessible

The older bioRxiv web pages were blocked by Cloudflare in this environment, but you later provided local PDF copies under:

- `C:\Users\gaura\OneDrive\Desktop\Bioarxiv`

So the evidence level for this memo is now much better than in the first draft.

This revised memo uses:

- `full text reviewed`
  - local PDFs for:
    - `452417v1`
    - `071050v1`
    - `207126v1`
    - `219832v1`
    - `310763v1`
  - plus the accessible published versions of:
    - [Dale and Guo, PLOS ONE 2018](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200126)
    - [Mittler et al., PLOS Computational Biology 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007561)
  - and [arXiv 2602.06323](https://arxiv.org/html/2602.06323v1)

One extra local PDF was present:

- `186833v1.full.pdf`

I did not include that paper in the comparison because it was not in your requested list.

## 2. Executive judgment

The short answer is:

- these papers are useful
- but only some of them are useful for the immediate `TR-V3` forecasting loop
- most of the real value is conceptual and architectural, not copy-and-paste mathematics

My honest ranking for immediate usefulness to `TR-V3` is:

| Paper | Immediate usefulness to TR-V3 | Why |
|---|---|---|
| `2602.06323` hybrid neural-mechanistic paper | high | best paper here on partial observability, non-stationarity, structured control branches, and bounded parameters |
| Dale and Guo 2018 | medium-high | directly relevant to unobserved HIV states, conservative Bayesian estimation, and coarse stochastic epidemic fitting |
| Lane et al. 2018 KP cascades | medium-high | very useful for data-scarce evidence triangulation, extrapolation governance, and missing-data handling, but not a forecasting model |
| Mittler et al. 2020 youth-focused TasP ABM | medium | very useful for understanding infection pressure, treatment feedback, age/network structure; too heavy for the core blocked-time forecaster |
| Johnson et al. 2018 MicroCOSM | medium-long-term | conceptually important for social and structural drivers; too data-hungry and too large for the current autoresearch loop |
| Romero-Severson et al. 2016 ABC phylogenetics | low for now | interesting mathematically, but not useful without sequence / phylogenetic data |

The best overall conclusion is:

- our current `TR-V3` memo is already pointed in the right direction
- these papers mostly validate the direction
- they do not justify a major reset
- they do justify sharpening three areas:
  - infection pressure `N(t)`
  - missing-data governance and explicit source-tiered imputation
  - stronger identifiability discipline before adding richer mortality and leakage

## 3. Where the current TR-V3 memo already stands

The current design memo already contains the important pieces that matter most:

- a stronger dynamic hazard backbone
- a learned observation model
- an explicit shock layer
- Phase 2 direct priors
- optional hidden-driver channels
- `TR-V3-05a` hazard-side mechanism controls
- `TR-V3-05b` open leaky incidence-flow model
- an explicit identifiability ledger
- deferral of explicit `S(t)` until denominator data are stronger

That is already more disciplined than many papers in this stack, because it combines:

- mechanistic structure
- partial observability awareness
- blocked-time benchmarking
- explicit keep-or-revert gates

So this literature review is not telling us to throw the plan away.

It is telling us:

- keep the plan
- tighten the parts that deal with infection generation, missing data, and identifiability
- do not import large extra model classes unless they clearly fit the current data contract

## 4. Cross-paper comparison table

| Paper | Main object | What it really gives us | What it does not give us |
|---|---|---|---|
| KP cascades in data-scarce environments | participatory cascade construction | disciplined data triangulation and transparent extrapolation in sparse settings | not a forecasting or dynamic state model |
| ABC transmission inference | phylogenetic simulation-based inference | a template for likelihood-free inference when direct likelihoods are hard | not useful without HIV sequence trees |
| Bayesian SDE HIV dynamics | coarse stochastic HIV compartment inference | a bridge between annual observed proportions and latent epidemic parameters | not enough detail for full `U/D/A/V/L` forecasting |
| Youth-focused TasP ABM | network / agent-based simulation | strong support for infection pressure depending on age/network structure and suppression feedback | not a practical core model for our quarterly blocked-time benchmark |
| MicroCOSM | rich social-structural microsimulation | strong support that access, geography, education, behavior, and service access matter | too large and data-hungry for current autoresearch loop |
| EpiNode hybrid paper | hybrid time-series / mechanistic forecasting | strongest case for explicit non-stationarity controls and bounded dynamic parameters under partial observability | HIV-specific seasonality is not justified by this paper |

## 5. Paper-by-paper analysis

## 5.1 Lane et al. 2018: KP cascades in data-scarce environments

Access level:
- full text reviewed from local PDF

### What the paper is about

This paper is not a dynamic epidemic model.

It is about building HIV treatment cascades for key populations in settings where direct data are patchy, incomplete, and concentrated in only a few places. The paper describes a participatory process in South Africa where stakeholders reviewed surveillance data, program knowledge, and extrapolation rules to create national or subnational cascade estimates.

In plain English:

- they had too little direct data
- they did not pretend the missing data were solved by a fancy model alone
- they used a formal triangulation and consensus process to decide what numbers were defensible

### Mathematics, explained simply

This is still not a heavy mathematical paper, but it is much more concrete in full text than the abstract suggested.

The real structure is a four-phase estimation process:

1. estimate population size from several methods embedded in surveillance studies
2. build preliminary cascades from those estimates
3. run a modified Delphi-style stakeholder consensus process
4. extrapolate to all districts using agreed rules

The practical mathematical idea is:

- do not estimate one number from one method and call it truth
- build a plausible range from several estimators
- use the median or consensus-supported value
- then extrapolate using factors that are explicitly agreed, documented, and auditable

In plain English:

- this is structured estimation under uncertainty
- not just "expert opinion"
- and not just "let the model guess"

The paper also makes a point that is very relevant for us:

- when some indicators are missing, you can borrow from adjacent evidence layers
- but you have to disclose that the missing indicator is being extrapolated rather than observed

That is exactly the kind of issue we have been dealing with in bridge panels and mixed archive construction.

### What maps well to TR-V3

This paper is useful for the part of our work that deals with sparse evidence, especially:

- bridge quarterly panel construction
- missing stock anchors
- KP-related or subnational gaps
- explicit provenance for imputed values

It is also directly supportive of a missing-data ladder where:

1. exact local observations are best
2. nearby survey or program evidence is second-best
3. district extrapolation is allowed if the rule is transparent
4. consensus review is part of the method, not an embarrassment to hide

It supports a strategy like:

1. exact observed values
2. bridge values from closely adjacent source documents
3. extrapolated values using explicit rules
4. model-based latent imputations only after the above are exhausted

That fits what we have been moving toward in the archive and bridge-panel work.

### What does not map well

This paper does not tell us how to build:

- a blocked-time forecasting model
- a dynamic hazard model
- an incidence inflow model
- a leakage or mortality model

So it should not drive `TR-V3-05a` or `05b` directly.

### Concrete value to our memo

This paper strengthens the non-model part of the design memo:

- source tiering
- transparent imputation
- stakeholder-auditable extrapolation rules

It is especially relevant if we decide to make missing-data handling explicit in the memo.

### Bottom-line judgment

Beneficial:
- yes, more than I thought in the first draft

Main use:
- evidence governance
- transparent extrapolation
- data-aware imputation rules

Should change the core mechanistic equations?
- no

## 5.2 Romero-Severson et al. 2016: approximate Bayesian computation for HIV transmission inference

Access level:
- full text reviewed from local PDF

### What the paper is about

This paper uses approximate Bayesian computation, or `ABC`, to infer:

- who infected whom
- whether transmission was one-off or ongoing
- how diverse the founding viral population was

using HIV phylogenetic information.

### Mathematics, explained simply

`ABC` is a way to do Bayesian inference when the exact likelihood is too hard to write down.

The simple version is:

1. simulate many possible histories from the model
2. compute a few summary numbers from each simulated dataset
3. compute the same summary numbers from the real data
4. keep the simulations that look similar

Instead of asking:

- "what is the exact probability of this data under parameter theta?"

it asks:

- "if I simulate with parameter theta, do I get data that looks enough like what I saw?"

In this paper, the summary numbers come from phylogenetic tree shape, branch lengths, and genetic diversity.

### What maps well to TR-V3

After reading the full paper, the transferable part is still narrow, but clearer.

The core method is:

1. write down alternative transmission scenarios
2. simulate phylogenetic trees under each scenario
3. compute summary statistics from those simulated trees
4. compare them with the observed posterior tree sample
5. use accepted simulations to approximate posterior support and Bayes factors

That general pattern is useful:

- compare structured generative scenarios
- simulate from each
- score them using summary statistics

So the paper is still relevant as a methodological pattern for:

- simulation-based model comparison
- synthetic recovery testing
- situations where the full likelihood is too ugly or too expensive

But the actual content remains mismatched to our data.

### What does not map well

Right now this paper is still not operationally useful because our pipeline does not have:

- sequence data
- phylogenetic trees
- donor-recipient pair structure
- within-host evolutionary models

Also, the paper is focused on pair-level inference, not national surveillance dynamics.

Their objects are:

- who infected whom
- whether transmission was single-event or ongoing
- how many lineages founded infection

Our objects are:

- national stock-flow dynamics
- diagnosis and treatment transitions
- infection pressure under mixed surveillance and program data

### Concrete value to our memo

The paper should not change the current roadmap.

The only things worth carrying forward are:

- simulation-based comparison of mechanism variants
- summary-statistic-based recovery tests
- explicit distinction between:
  - "model fits because it simulated the right summary behavior"
  - and
  - "model fits because the exact likelihood was well identified"

### Bottom-line judgment

Beneficial:
- mostly no for now

Worth keeping in the literature memo?
- yes, as a future method class for simulation-based inference

Should shape the next code path?
- no

## 5.3 Dale and Guo 2018: Bayesian estimation for a stochastic HIV SDE model

Access level:
- full text reviewed via [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0200126)

### What the paper is about

This paper is much closer to our problem.

It studies HIV epidemic dynamics when some of the important population is not directly observed, especially the undiagnosed infected group. The authors:

1. estimate yearly proportional changes in diagnosed and undiagnosed populations using hierarchical Bayesian statistics
2. then use those estimates to constrain a stochastic differential equation system
3. then compare explanations such as:
   - susceptible exhaustion
   - lack of access to care
   - ART usage

### Mathematics, explained simply

The first part is a Bayesian smoothing model on yearly proportions.

They use:

\[
x_t \sim \text{Binomial}(n_t, p_t)
\]

Plain English:

- if the total infected population in year `t` is `n_t`
- and the true undiagnosed share is `p_t`
- then the observed count `x_t` behaves like a noisy draw from that share

They then put a prior on the next year's share centered on:

\[
q \, p_{t-1}
\]

Plain English:

- this year's share is expected to be last year's share times a yearly change factor `q`

Then they estimate `q` from data.

After that, they plug those estimated yearly change factors into a stochastic epidemic system:

\[
dU = (\text{transmission} - \text{diagnosis} - \text{death})\,dt + \text{noise}
\]

\[
dD = (\text{diagnosis} - \text{death})\,dt + \text{noise}
\]

Plain English:

- `U` is undiagnosed infected people
- `D` is diagnosed infected people
- each group changes because of inflows, outflows, and noise

This is a coarse stochastic compartment model.

### What I think is genuinely useful

This paper strongly supports three things in our memo:

1. **Open infected-population modeling before full `S(t)`**
   - They do not need a fully observed susceptible stock to do something useful.
   - That matches our current decision to defer explicit `S(t)` until denominator data improve.

2. **Conservative annual latent estimation before richer quarterly dynamics**
   - Their yearly `q`-estimation logic is a sensible template for building annual constraints before trying to fit everything quarterly.

3. **Scenario comparison under missing observability**
   - Their "access to care" and "ART" scenarios show the value of testing mechanism hypotheses rather than estimating every block freely.

### Where it overlaps with our current memo

It overlaps most with `TR-V3-05b`:

- explicit infection-related inflow logic
- diagnosis dynamics
- stochasticity
- access-to-care interpretation
- annual-to-dynamic calibration under weak observability

It also supports the identifiability ledger:

- their model only works because it stays coarse
- once the model becomes more expressive, identifiability gets harder very quickly

### What is weaker than our current memo

Their model is still much simpler than what we now need.

Weaknesses relative to our current design:

- it only has very coarse infected compartments
- no treatment cascade states like `A`, `V`, and `L`
- weak observation model
- crude handling of susceptible exhaustion
- no blocked-time forecasting benchmark
- no mixed-frequency observation design

So we should not copy their exact equations as our main model.

### Best use for us

Use this paper as support for:

- annual latent calibration of infection / diagnosis balance
- conservative priors for coarse annual dynamics
- scenario testing for access-to-care and ART effects

Do not use it as the final architecture.

### Bottom-line judgment

Beneficial:
- yes

How:
- as support for `TR-V3-05b` and for annual latent calibration

What not to do:
- do not regress to a two-state diagnosed/undiagnosed model as the main family

## 5.4 Mittler et al. 2020: youth-focused treatment-as-prevention in an agent-based network model

Access level:
- full text reviewed via [PLOS Computational Biology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007561)

### What the paper is about

This paper asks:

- if treatment resources are limited, is it better to focus linkage and viral suppression on younger people?

They use a stochastic agent-based network model in a generalized heterosexual epidemic.

That means:

- each person is modeled individually
- people form and break partnerships over time
- infection moves through those partnership networks
- treatment changes viral load and therefore changes future transmission

### Mathematics, explained simply

This is not one neat low-dimensional equation.

It is a microsimulation:

- people are nodes
- sexual relationships are edges
- the network changes over time
- infection probability depends on age, sex, viral load, condom use, relationship structure, and treatment

The mathematically important point is:

- transmission depends on who is connected to whom
- not just on total counts

That matters because it creates feedback:

- more suppression in a strategically important group reduces future transmission in their network neighborhood

### What is highly relevant for TR-V3

This paper gives strong conceptual support for the missing infection-pressure term in our memo.

The main lesson is:

- diagnosis pressure is not enough
- care pressure is not enough
- infection generation depends on network structure and who is suppressed

This strongly supports:

- adding latent infection pressure `N(t)`
- letting suppression feed back into infection generation
- treating age/network structure as part of transmission pressure

It also supports a point you raised earlier:

- seasonality is probably not the right decomposition lens for HIV
- network composition and treatment coverage matter much more

### Where it does not fit

This paper is not a good direct blueprint for our core national blocked-time forecast model because:

- it is a large ABM
- it is tuned to a generalized heterosexual setting
- it intentionally leaves out MSM, sex workers, and PWID
- it is policy-simulation heavy rather than observation-model heavy
- it is not built for our current mixed stock/flow quarterly archive

That means:

- it is good evidence for mechanism choice
- it is not a good immediate implementation target

### Best use for us

Use it to justify:

- explicit infection inflow into `U`
- infection pressure driven by network/risk structure
- suppression feedback into transmission

Do not use it to justify:

- replacing `TR-V3` with a full ABM

### Bottom-line judgment

Beneficial:
- yes, conceptually

Best role:
- support for the `N(t)` infection-pressure block and later scenario work

Not advisable:
- using it as the immediate autoresearch model family

## 5.5 Johnson et al. 2018: MicroCOSM

Access level:
- full text reviewed from local PDF

### What the paper is about

MicroCOSM is a very rich agent-based microsimulation for HIV in South Africa.

Even from the abstract alone, the model is clearly large:

- births and deaths
- demographic characteristics
- education
- urban/rural location
- healthcare access
- condom preference
- PrEP
- circumcision
- testing history
- ART uptake
- sexual preference
- sex worker-client contacts
- short-term and long-term relationships

This is not just an epidemic curve model.
It is a social-structural simulator.

### Mathematics, explained simply

Like the youth-focused TasP paper, this is a big rule-based simulator over people and relationships.

In simple terms:

- each person carries many attributes
- those attributes change through time
- relationships form and dissolve
- prevention and treatment access change behavior and risk
- infection spreads through those contacts

So the model is mathematically rich, but the richness comes from many interacting rules and state updates, not from one compact equation.

### What is useful for TR-V3

The full text makes the model's scope much clearer.

MicroCOSM is valuable because it shows how many factors really sit upstream of HIV transmission and care outcomes:

- geography
- education
- migration
- healthcare access
- testing history
- ART uptake
- relationship type
- sexual preference
- concurrency
- condom behavior
- incarceration

That does not mean we should import all of them.

It means the paper gives strong support for treating our latent channels as compressed summaries of a much larger underlying social system.

That maps cleanly onto:

- `N(t)` infection / network pressure
- `A(t)` ascertainment pressure
- `C(t)` care-system pressure
- `R(t)` reporting artifact

It also strengthens one technical point:

- the paper explicitly argues that simpler deterministic, frequency-dependent models can understate the contribution of high-risk groups

That is directly relevant to our concern that a diagnosis-only or stock-only model can miss the real transmission engine.

### What does not fit

This paper is still too large and too data-hungry for the immediate Phase 3 loop.

The full text makes that even more obvious, not less.

Problems for direct adoption:

- huge state space
- many socio-economic and demographic submodels
- many calibration targets
- individual-level weekly simulation
- long historical backcast from 1985
- no clean match to our blocked-time quarterly benchmark

### Best use for us

Use MicroCOSM as:

- conceptual justification for richer latent channels
- support for future scenario engines
- support for the idea that missing data should sometimes be handled by structured priors over social drivers, not only by time-series smoothing

Do not use it as:

- the next forecasting model to implement
- the calibration template for the current autoresearch loop

### Bottom-line judgment

Beneficial:
- yes, but mostly as a long-term conceptual donor

Immediate implementation value:
- low to medium

## 5.6 arXiv 2602.06323: How (Not) to Hybridize Neural and Mechanistic Models for Epidemiological Forecasting

Access level:
- full text reviewed via [arXiv HTML](https://arxiv.org/html/2602.06323v1)

### What the paper is about

This paper studies why many neural-mechanistic hybrid epidemic models fail.

Its strongest contribution is not the final model itself.

Its strongest contribution is its diagnosis of failure:

- partial observability breaks naive latent hybrids
- optimization tricks alone do not fix missing information
- physics-informed losses can still fail when supervision is sparse
- continuous-time neural hybrids can still miss multi-wave structure if the forcing signal is not explicit

Then the authors propose `EpiNode`:

- decompose the observed infection signal
- use separate latent branches for those decomposed pieces
- decode bounded time-varying epidemic parameters
- run those parameters through a mechanistic epidemic system

### Mathematics, explained simply

Their signal decomposition starts from:

\[
I(t) = T(t) + S(t) + R(t)
\]

Plain English:

- observed infections are split into:
  - slow trend
  - seasonal component
  - irregular residual component

Then each component goes through its own latent dynamic branch.

They also optionally use time-delay embedding:

\[
[x(t), x(t-\tau), x(t-2\tau), \ldots]
\]

Plain English:

- instead of giving the model only "the value right now"
- give it a short memory of recent values too

Finally they decode bounded epidemiological parameters.

That means instead of letting a neural net output impossible rates, they force rates into allowed ranges:

\[
\theta(t) = \theta_{\min} + (\theta_{\max} - \theta_{\min}) \sigma(z(t))
\]

Plain English:

- the raw model output `z(t)` can be any number
- the sigmoid squeezes it to `[0,1]`
- then the parameter is rescaled into a meaningful allowed interval

### Why this paper matters to us

This is the most directly useful paper in the set for the immediate architecture discussion.

It strongly supports:

1. **partial observability as a first-class design problem**
2. **explicit non-stationarity controls**
3. **small structured branches instead of one monolithic latent correction**
4. **bounded dynamic parameters**
5. **peak / turning-point evaluation rather than only short-horizon fit**

All of those map directly to the current `TR-V3` memo.

### Where we should adapt, not copy

This is also the paper most likely to mislead us if copied too literally.

Why:

- it is mostly framed around observed infection series
- it uses trend / seasonality / residual decomposition
- it works with SIRS-style epidemic dynamics

For our HIV setting, that is only partly transferable.

The key mismatch is seasonality.

After reading the HIV papers in this set, the case for biological seasonality in HIV looks weak. For HIV, the more plausible structured drivers are:

- network / infection pressure
- diagnosis / testing pressure
- care-system performance
- reporting and registry artifact

So the correct translation of EpiNode into our setting is not:

- `trend + seasonality + residual`

It is:

- `N(t)` infection pressure
- `A(t)` ascertainment pressure
- `C(t)` care pressure
- `R(t)` reporting artifact

That is the right adaptation.

### Best use for us

This paper should directly influence:

- `TR-V3-05a`: hazard-side mechanism controls
- bounded hazard parameterization
- explicit train-only control construction
- future branch-specific lag structure

It is less relevant for:

- evidence triangulation
- KP cascade imputation
- social-structural micro-policy simulation

### Bottom-line judgment

Beneficial:
- strongly yes

But:
- only if adapted to HIV-specific mechanism channels
- not copied literally as trend/seasonal/residual

## 6. What these papers say about seasonality

This is important because it answers a recurring design question directly.

After reading this set, my view is:

- default biological seasonality is not well-supported for our HIV Phase 3 model

Why:

- the HIV papers here emphasize diagnosis, access to care, treatment, network structure, age mixing, social drivers, and suppression feedback
- they do not argue that HIV dynamics are primarily driven by biological seasonal forcing in the way respiratory or vector-borne epidemics often are

So:

- do not make seasonality a default component of `TR-V3`
- if any periodic structure appears, it should first be interpreted as:
  - reporting cadence
  - clinic operations
  - campaign timing
  - administrative batching

That means it belongs in `R(t)` before it belongs in `N(t)`.

## 7. What these papers say about infection pressure

This is where the literature is most aligned with the current gap we identified.

Across the set, especially:

- Mittler et al.
- MicroCOSM
- Dale and Guo

the message is clear:

- diagnosis is not the start of the epidemic process
- infections have to be generated upstream
- treatment and suppression feed back into future infection generation

That supports the direction already in the memo:

\[
U_{t+1} = U_t + \iota(t) - f_{UD}(t) - \cdots
\]

where `\iota(t)` is new infection inflow into `U`.

The literature-based refinement is:

- infection pressure should be treated as a separate latent mechanism
- not folded into diagnosis pressure

So the papers strengthen the case for:

- adding `N(t)` explicitly
- letting `N(t)` drive incidence inflow
- letting suppression reduce effective transmission

## 8. What these papers say about identifiability

The strongest identifiability lessons across the set are:

1. partial observability is not a small nuisance; it is the central problem
2. richer model classes increase non-identifiability unless data support also grows
3. structured priors and explicit mechanism separation help more than generic flexibility

This supports the current memo strongly:

- identifiability ledger
- staged calibration
- deferring explicit `S(t)`
- rejecting models that improve fit only by moving mass between poorly identified blocks

This is also why I do not think we should add:

- richer mortality blocks
- more leakage paths
- or a full susceptible compartment

until denominator and mortality data are materially stronger.

## 9. Missing data and imputation: what the papers add

You asked explicitly to remember that missing data can be handled by a data-aware imputation strategy.

That is correct, and the literature here supports it, but not in the naive sense of "fill missing values with a model."

The best paper for this is the KP cascades paper.

The right hierarchy is:

1. **exact observed**
   - direct source value, exact period, exact metric

2. **bridge observed**
   - derived from nearby reports or adjacent monthly/quarterly releases with explicit provenance

3. **rule-based extrapolated**
   - source-tiered extrapolation with documented assumptions

4. **latent model-imputed**
   - generated by the mechanistic model under explicit uncertainty

5. **never silently merged**
   - the archive should always record which level the number came from

In plain English:

- imputation is fine
- pretending imputed data are exact is not fine

That should probably be stated more explicitly in the Phase 3 design stack.

## 10. Recommended changes to the TR-V3 memo after this review

## Keep as-is

- `TR-V3-05a` as hazard-side mechanism controls
- `TR-V3-05b` as open leaky incidence-flow model
- explicit identifiability ledger
- defer explicit `S(t)` until denominator data improve
- reject generic seasonality as a default HIV component

## Strengthen

### 10.1 Make `N(t)` more explicit

The literature supports turning infection pressure into a first-class object.

That means the memo should continue to treat:

- `N(t)` as infection / network pressure
- `A(t)` as ascertainment pressure
- `C(t)` as care pressure
- `R(t)` as reporting artifact

and not blur them together.

### 10.2 Add a more explicit missing-data ladder

The papers support a clearer contract for missing data:

- exact
- bridge
- extrapolated
- latent-imputed

with provenance tags.

### 10.3 Separate the core forecaster from future scenario engines

The network and microsimulation papers are useful, but they should not be forced into the core blocked-time forecaster.

Better split:

- core forecaster: `TR-V3`
- future scenario engine: possible later ABM / microsimulation layer

That is cleaner and more honest.

## Do not add yet

- full `S(t)` compartment as a core state
- generic seasonal branch
- full social-structural microsimulation
- ABC / SBI machinery without the data to support it

## 11. Final take / defer / reject table

| Paper | Take now | Defer | Reject |
|---|---|---|---|
| KP cascades paper | source-tiered triangulation and extrapolation discipline | stakeholder calibration layer for sparse KP data | using it as a dynamic model |
| ABC phylogenetics paper | only the high-level idea of likelihood-free inference | simulation-based inference if molecular data ever arrive | direct use in current Phase 3 |
| Dale and Guo 2018 | annual latent calibration logic, access-to-care scenario discipline, open infected-system thinking | fuller stochastic calibration once annual anchors improve | collapsing `TR-V3` back to a coarse two-state model |
| Youth-focused TasP ABM | infection pressure, suppression feedback, network logic | later scenario/policy simulation | replacing the core benchmark model with a full ABM |
| MicroCOSM | social-structural interpretation of `N/A/C/R` | long-run structural policy simulator | near-term direct implementation in the autoresearch loop |
| arXiv 2602.06323 | explicit non-stationarity controls, bounded parameters, partial-observability discipline | branch-specific lag structures and more advanced control construction | default HIV seasonality and full neural ODE copying |

## 12. Final bottom line

These papers do help us, but not all in the same way.

The honest synthesis is:

- the current `TR-V3` design memo is broadly correct
- the biggest thing still missing is a stronger, cleaner infection-pressure story
- the literature supports adding that
- the literature does not support copying large agent-based or neural architectures directly into the next loop

If I reduce the entire review to one sentence, it is this:

> Keep the current TR-V3 direction, strengthen `N(t)` and missing-data governance, keep `S(t)` deferred until denominator data are stronger, and do not mistake HIV for a disease where generic seasonality is the main missing mechanism.
