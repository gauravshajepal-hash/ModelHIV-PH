# Philippines HIV Incumbent Competitive Context

This note summarizes the two local reference decks:

- `The Philippine HIV_STI Surveillance.pdf`
- `2025 PH HIV Estimates_Core team_for WHO.pdf`

The purpose is to treat them as the incumbent stack that our model must understand, benchmark against, and eventually outperform on a clearly defined task.

This is not a literature review of all HIV estimation work. It is a competitor-context brief built from the supplied Philippines surveillance and estimates decks, interpreted against the current `hiv_rescue_v1` and `hiv_rescue_v2` architecture.

## 1. Executive Read

The two decks describe two different but tightly coupled systems:

1. a surveillance substrate
2. an estimation engine

The surveillance deck is the measurement and reporting backbone:

- HARP
- HTS / ART / mortality forms
- IHBSS
- LaBBS
- SESS
- CFBS
- PrEP surveillance
- KP size estimation
- OHASIS / one information system

The WHO/core-team deck is the incumbent national estimation workflow on top of that substrate:

- triangulate surveillance and program data
- feed AEM and Spectrum
- fit subnational KP transmission models
- align population totals to Spectrum
- update coverage assumptions
- vet results with experts and country stakeholders
- submit GAM estimates

The important competitive conclusion is:

> our real competition is not a single model.
> it is a country process that combines surveillance operations, special surveys, expert vetting, and a semi-structured national estimation workflow.

If we want to beat it, we cannot only build a better latent-state engine.
We must beat it on:

- evidence synthesis
- calibration discipline
- subnational consistency
- explicit assumption handling
- operational usefulness

## 2. What Each Deck Actually Is

## 2.1 Surveillance deck

`The Philippine HIV_STI Surveillance.pdf` is not a forecasting model deck.
It is the map of the surveillance ecosystem that supplies the data used by estimates and program management.

Its main content is:

- legal and policy mandate for HIV and STI monitoring
- reporting obligations for laboratories and blood banks
- milestone history of the HIV / STI surveillance system
- the surveillance framework and system components
- IHBSS role, history, methods, and 2022 cycle design
- population size estimation methods integrated into IHBSS

In practical terms, this deck defines the observation surface that any serious Philippines HIV model must respect.

## 2.2 Estimates deck

`2025 PH HIV Estimates_Core team_for WHO.pdf` is the incumbent national estimation workflow.

It explains:

- why estimates are needed
- what data streams are used
- how data are analyzed and triangulated
- how AEM and Spectrum are combined
- how 2025 inputs were updated
- what coverage definitions were used
- how ART and PrEP assumptions affect projections
- how sensitive estimates are to key operational definitions like 30-day versus 90-day LTFU

This is the deck that most directly competes with our model.

## 3. Incumbent Stack Decomposed

## 3.1 Layer 1: Surveillance and reporting system

From the surveillance deck, the Philippines HIV/STI information base is built from:

- HARP: diagnosis, ART, viral suppression and ADR, TB/pregnancy/OI, mortality
- HTS form, ART form, Form D
- IHBSS
- HIV IMPACS
- CFBS surveillance
- PrEP surveillance
- SESS
- LaBBS
- Enhanced gonococcal antimicrobial surveillance
- viral hepatitis surveillance
- KP size estimates
- AEM/Spectrum PLHIV estimates
- OHASIS / one HIV/AIDS-STI information system

This means the incumbent system already has a strong observation ladder, even if it is not phrased that way mathematically.

## 3.2 Layer 2: Periodic high-value survey calibration

IHBSS is the major survey calibration instrument in the decks.

Key characteristics called out in the surveillance deck:

- routine HIV survey among key populations
- behavioral interview plus serologic testing
- conducted every 3 to 4 years
- targeted at high-burden cities / sentinel sites
- used for prevalence, behavioral factors, intervention outcomes, planning, and estimate development
- 2022 cycle uses RDS for all key populations
- PSE is integrated into the survey workflow

This matters because the incumbent model is not trying to estimate everything from passive routine data alone.
It periodically re-anchors behavior, prevalence, and key population size with a special survey instrument.

## 3.3 Layer 3: National estimation engine

From the estimates deck, the incumbent workflow is:

1. analyze surveillance and program data
2. triangulate them with IHBSS, HARP, and other studies
3. feed them into AEM + Spectrum
4. validate and vet the results with experts and stakeholders
5. submit and disseminate program-facing estimates

The key modeling split is:

- AEM:
  - KP-specific
  - transmission between key populations
  - 7 subnational models
  - outputs for ages 15+
- Spectrum:
  - MTCT and broader age structure
  - outputs all ages including children
  - PMTCT support

This is a hybrid stack, not a single monolithic model.

## 4. Main Quantities and Assumptions in the Incumbent Workflow

The estimates deck exposes several assumptions that are central to the competitor model.

## 4.1 Core 2024 results reported in the deck

The deck reports:

- estimated PLHIV in 2024: `216,900`
- annual new infections in 2024: `29,800`
- annual AIDS deaths in 2024: `2,100`
- care cascade as of December 2024:
  - estimated PLHIV: `216,900`
  - diagnosed PLHIV: `135,026`
  - alive on ART: `90,854`
  - tested for viral load: `43,534`
  - virally suppressed: `41,164`

Displayed cascade percentages:

- first 95 proxy: `62%`
- second 95 proxy: `67%`
- viral-load-tested among ART: `48%`
- suppressed among viral-load-tested: `95%`

This is already one critical difference from a cleaner mathematical cascade:
the public cascade shown in the deck is partly constrained by viral-load testing coverage, not just a latent suppression state.

## 4.2 Input updates for the 2025 round

The deck highlights these updates:

- population aligned to WPP 2024 using a Baseline Combiner with population adjustment
- behavioral and prevalence inputs retained from the 2022 IHBSS
- diagnosis and ART data updated using HARP through December 2024
- PrEP coverage updated
- PMTCT coverage updated for 2024
- ART effect set to `0.9`

This matters because the incumbent stack is explicit about when inputs are fresh and when they are retained from older survey rounds.

## 4.3 Prevention coverage definition

The deck defines prevention coverage as a package:

- received HIV information
- had access to condoms
- tested for HIV

It also states that the definition and inputs for program coverage were based on the 2022 IHBSS.

An example shown in the deck:

- national MSM & TGW prevention coverage: `26%`
- Category B: `12%`
- Category C: `8%`

This shows the incumbent model uses composite operational coverage definitions rather than single raw indicators.

## 4.4 PrEP coverage and effective use

The deck makes PrEP assumptions explicit.

Coverage formula:

- coverage depends on people enrolled to PrEP
- months stayed on PrEP
- people needed to cover

People needed to cover is based on:

- MSM PSE
- estimated PLHIV
- percent high-risk MSM
- UNAIDS target coverage recommendation

The deck also exposes a critical proxy:

- number of months stayed on PrEP is proxied by bottles dispensed

It also exposes a limitation:

- `1 bottle = 1 month of protection`
- this may not hold for event-driven PrEP users

The deck computes effective use from observed effectiveness relative to assumed efficacy and reports an example effective-use value of `85.64%`.

This is important because it reveals where the incumbent model already uses pragmatic proxies rather than direct truth.

## 4.5 ART coverage definition

The deck defines ART coverage as:

- proportion of estimated PLHIV currently on treatment
- specifically, PLHIV who accessed ARVs or refill within `90 days` from expected pill run-out

This is not a trivial implementation detail.
The deck later shows a major sensitivity to this choice.

## 4.6 Sensitivity to 30-day versus 90-day LTFU

One slide explicitly shows:

- if 30-day LTFU were used instead of 90-day
- there would be `31,700` additional adults living with HIV
- and `7,900` additional new infections for 2024

That is a direct competitor signal.

It means the incumbent model is highly sensitive to operational retention definitions.
Any challenger model that ignores this sensitivity will look scientifically naive.

## 4.7 Projection logic

The deck distinguishes:

- GAM estimates submission for observed years
- baseline intervention
- forward HIV projection

And it compares:

- baseline with no intervention scale-up
- ART-only scale-up
- ART + PrEP scale-up

Example targets shown:

- ART-only scenario targets `90%` ART coverage
- ART + PrEP uses `50%` PrEP target in high-incidence areas and `15%` in lower-incidence areas

So the incumbent stack already produces scenario outputs, not just retrospective estimates.

## 5. What the Incumbent Stack Is Optimized For

The decks imply a very clear optimization target.

The incumbent stack is optimized for:

- national reporting credibility
- GAM / UNAIDS compatibility
- program planning and grant applications
- ARV forecasting
- high-level cascade monitoring
- stable country process under imperfect data

It is not primarily optimized for:

- end-to-end mechanistic transparency at the province x KP x age x sex level
- daily or monthly high-resolution operational forecasting
- discovery of broad non-HIV determinant fields
- formal bias propagation through all candidate variables
- dynamic mesoscopic factor discovery across economics, logistics, stigma, mobility, and information flow

That is the opening for our architecture.

## 6. Strengths of the Incumbent We Must Respect

These are the parts we cannot dismiss.

## 6.1 It is already an evidence-synthesis process

The deck explicitly says:

- exact counts are impossible
- no single data source is complete
- data must be synthesized and triangulated

That is the correct epistemic stance for Philippines HIV.

## 6.2 It is anchored to the actual surveillance regime

The incumbent uses:

- HARP
- IHBSS
- KP size estimation
- program coverage inputs
- subnational models
- formal vetting

So it is deeply tied to the real country information system.

## 6.3 It handles the cascade in a policy-operational way

The displayed cascade is not purely latent.
It reflects:

- diagnosis
- ART
- viral-load testing
- virologic suppression

That gives it operational credibility with stakeholders.

## 6.4 It makes assumptions explicit

Examples from the deck:

- 90-day ART definition
- 30-day versus 90-day sensitivity
- 0.9 ART effect
- PrEP effective-use logic
- 1 bottle equals 1 month assumption
- retention of 2022 IHBSS behavioral inputs

This explicitness is a strength.

## 6.5 It has an institutional vetting loop

The deck includes expert validation and stakeholder vetting.

That matters because country trust is not won by predictive accuracy alone.

## 7. Weaknesses and Attack Surfaces

This is where our model can compete.

## 7.1 Survey latency

The estimates deck is still relying on behavioral and prevalence inputs retained from the 2022 IHBSS for the 2025 round.

That is understandable, but it creates inertia.

It means the system may be slow to absorb:

- rapid behavioral shifts
- digital-platform mediated partner dynamics
- mobility shocks
- service disruption changes
- local stigma/information changes
- subnational divergence after the survey round

## 7.2 Operational assumptions are doing large hidden work

The 30-day vs 90-day comparison shows that an operational definition can move estimates dramatically.

That means some of the incumbent outputs are sensitive to:

- registry definitions
- refill classification rules
- cohort accounting choices

Our model can compete if it exposes these sensitivities as first-class uncertainty objects instead of fixed hidden assumptions.

## 7.3 Subnational structure is still fairly coarse

The deck states AEM has `7` subnational models.

That is useful, but it is still much coarser than a province-aware mechanistic model with explicit subgroup structure.

This is one of the clearest places our architecture can be stronger, provided we stay calibrated.

## 7.4 Weak integration of broad determinant fields

The incumbent stack uses behavior, prevalence, treatment, prevention, and KP size inputs.

What it does not appear to do in a systematic, explicit way is integrate broad upstream domains such as:

- logistics friction
- transport disruption
- remoteness and care-access geometry
- service-network fragility
- information propagation and stigma diffusion
- economic shock
- digital behavior proxies

That does not mean they are irrelevant to the real epidemic.
It means the incumbent process has limited room to formalize them.

## 7.5 Limited explicit bias modeling

The decks acknowledge triangulation, but they do not present a formal bias layer for:

- source reliability
- measurement error classes
- sampling bias classes
- proxy penalties
- missingness propagation

That is a major opening for `hiv_rescue_v2`.

## 8. What This Means for Our Model

The competitive lesson is not:

> replace AEM/Spectrum with a giant unconstrained ML system

The competitive lesson is:

> build a stricter observation-first model than a generic ML stack, but a broader and more explicit determinant-and-bias model than the incumbent estimation process.

In other words, we should try to win in the middle:

- more mechanistic and observation-disciplined than generic predictor systems
- more subnational, determinant-aware, and bias-aware than the incumbent stack

## 8.1 What we must copy

We should treat these as mandatory:

- explicit observation ladder
- explicit surveillance-system grounding
- explicit country assumptions
- explicit sensitivity analyses for operational definitions
- explicit benchmark comparison to national cascade references
- explicit vetting-ready artifact outputs

## 8.2 What we should exceed

We should try to exceed the incumbent on:

- province-level heterogeneity
- KP x age x sex structure
- formal uncertainty around weak sources
- factorized upstream determinants
- explicit mesoscopic block factors
- metapopulation access and information structure
- stress-testing under missingness and source dropout

## 8.3 What we should not claim yet

We should not claim:

- universal superiority to AEM/Spectrum
- national reporting replacement
- direct superiority on all official metrics
- scenario reliability before our calibration is strong

The right near-term claim is narrower:

> we are building a Philippines-specific, subnational, determinant-aware HIV cascade model that can be benchmarked against the incumbent workflow and may outperform it on selected operational tasks.

## 9. Concrete Benchmark Tasks Where We Can Compete

The decks suggest the fairest benchmark targets.

## 9.1 Cascade reconstruction under sparse subnational data

Compete on:

- province or subnational diagnosed-stock fit
- province or subnational ART-stock fit
- documented suppression lower-bound consistency
- hierarchy reconciliation

## 9.2 Sensitivity transparency

Compete on:

- exposing 30-day versus 90-day retention sensitivity
- exposing sensitivity to viral-load testing gaps
- exposing sensitivity to stale IHBSS behavior inputs

The incumbent shows these sensitivities.
We should show them more systematically.

## 9.3 Determinant augmentation

Compete on whether mesoscopic factors improve fit without breaking official anchors.

Examples:

- access-friction factors
- service-fragility factors
- mobility-diffusion factors
- information-propagation factors

If those factors materially improve subnational fit while preserving national credibility, that is a real win over the incumbent process.

## 10. Immediate Implications for the Rescue Architecture

These decks strongly support several existing rescue decisions.

## 10.1 The observation-first rebuild is correct

The decks are dominated by:

- surveillance inputs
- coverage definitions
- operational cascade surfaces

So our observation-first rescue core is the right direction.

## 10.2 The Phase 1.5 mesoscopic factor engine is justified

The incumbent process has limited machinery for broad upstream determinants.

That is exactly where our:

- factor engine
- source-reliability layer
- mobility/percolation/information features

can add value.

## 10.3 Benchmarking against official references is non-negotiable

The decks are country-facing and stakeholder-facing.
That means our model must keep producing:

- official reference comparisons
- assumption audits
- sensitivity ladders

or it will not be credible as a competitor.

## 10.4 Phase 4 should remain blocked

Nothing in these decks suggests that we should rush into RL or large-scale optimization.

The incumbent competition is still winning mainly through:

- surveillance discipline
- estimation discipline
- assumption discipline

That means our next battle is still Phase 3 scientific credibility, not Phase 4 policy automation.

## 11. Bottom Line

The incumbent Philippines HIV stack is:

- stronger than a simple epidemiologic model
- broader than a single surveillance registry
- more institutionalized than a stand-alone modeling notebook

But it is also:

- survey-latent
- assumption-sensitive
- relatively coarse subnationally
- limited in formal bias modeling
- limited in broad determinant integration

So our path is not to dismiss it.
Our path is to build a challenger that:

- respects the same observation discipline,
- exposes assumptions more clearly,
- models uncertainty and bias more formally,
- uses a narrower mechanistic HIV core,
- and attaches a broader but controlled mesoscopic determinant layer.

That is the credible way to compete with AEM/Spectrum in the Philippines context.
