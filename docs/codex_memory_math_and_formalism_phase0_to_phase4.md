# Codex Memory Formalism: Mathematics and Model Structure, Phase 0 to Phase 4

This document is a **memory-derived mathematical specification** of the pipeline we designed in our Codex conversations.

It is not a line-by-line repository audit.

That means:

- some objects below are **high-confidence** because we discussed them repeatedly and tied them to artifacts or tests,
- some are **formalizations of design intent** where the conversation was architectural rather than implementation-complete.

The purpose of this document is to state the **mathematical objects, transformations, constraints, and gates** that define the pipeline.

---

## 1. Global Mathematical View

The pipeline can be represented as a sequence of maps:

\[
\mathcal{R}
\xrightarrow{\Phi_0}
\mathcal{E}
\xrightarrow{\Phi_1}
\mathcal{X}
\xrightarrow{\Phi_2}
\mathcal{C}
\xrightarrow{\Phi_3}
\mathcal{F}
\xrightarrow{\Phi_4}
\mathcal{P}
\]

where:

- \(\mathcal{R}\) is the raw world of documents, datasets, portal pages, tables, and structured repositories,
- \(\mathcal{E}\) is the extracted evidence layer,
- \(\mathcal{X}\) is the aligned tensor-ready evidence layer,
- \(\mathcal{C}\) is the curated candidate / inferential packaging layer,
- \(\mathcal{F}\) is the fitted mechanistic model layer,
- \(\mathcal{P}\) is the downstream production / policy layer.

The scientific philosophy is:

1. **widen first**,
2. **align and score**,
3. **curate and gate**,
4. **fit mechanistically**,
5. **only then simulate or promote**.

There is also a practical architectural split:

- the **scientific core** keeps canonical storage, exact retrieval, provenance, and gates,
- the **discovery sidecar** carries broader search and exploratory retrieval tools.

There is also a portability object that became clearer later in the conversation:

\[
\mathfrak{I}_{country,disease}
=
\left(
\text{global\_config},
\text{disease\_plugin},
\text{country\_pack},
\text{source\_adapter\_pack},
\text{benchmark\_pack},
\text{policy\_pack}
\right)
\]

This matters because country portability alone is not enough when state topology and intervention semantics differ across diseases.

There is also a multi-disease coupling pattern:

\[
\mathfrak{S}
=
\left(
\mathcal{M}^{(z_1)}, \dots, \mathcal{M}^{(z_m)},
E^{(z_a \rightarrow z_b)}
\right)
\]

where each disease keeps its own model and only low-dimensional exogenous summaries are exchanged across disease boundaries.

There is also an agreed selective acceleration roadmap:

- use PyTorch tensor math more in Phase 1, Phase 2, and Phase 3 transition computations,
- use GPyTorch only for alignment subproblems where GP interpolation is the actual bottleneck,
- defer NumPyro / JAX migration until the probabilistic core is intentionally migrated and JAX GPU works on the target machine,
- defer Ray until Phase 4 genuinely requires distributed rollout infrastructure.

That hybrid split became important later in the conversation.

---

## 2. Shared Notation

We use the following common notation.

### 2.1 Index sets

- \(s \in \mathcal{S}\): sources
- \(d \in \mathcal{D}\): documents
- \(o \in \mathcal{O}\): extracted observations
- \(c \in \mathcal{C}\): candidate subparameters
- \(b \in \mathcal{B}\): domain blocks
- \(p \in \mathcal{P}\): provinces
- \(r \in \mathcal{R}\): regions
- \(t \in \mathcal{T}\): time index
- \(k \in \mathcal{K}\): key-population group
- \(a \in \mathcal{A}\): age band
- \(x \in \mathcal{X}_{sex}\): sex axis
- \(g \in \mathcal{G}\): coarse CD4 bin
- \(\sigma \in \Sigma\): care-cascade state
- \(\delta \in \Delta\): duration bucket

### 2.2 Core sets

- \(\Sigma = \{U, D, A, V, L\}\)
  - \(U\): undiagnosed
  - \(D\): diagnosed, not on ART
  - \(A\): on ART, unsuppressed
  - \(V\): on ART, suppressed
  - \(L\): disengaged / lapsed / diagnosed-but-not-effectively-on-ART treatment-linked stock

- \(\mathcal{K}\) contains groups such as:
  - remaining population
  - MSM
  - TGW
  - FSW
  - clients of FSW
  - PWID
  - non-KP partners

- \(\mathcal{A}\) contains age bands such as:
  - 15-24
  - 25-34
  - 35-49
  - 50+

- \(\mathcal{G}\) contains CD4 bins such as:
  - \(<200\)
  - \(200-349\)
  - \(350-499\)
  - \(500+\)

### 2.3 Evidence tiers

Every record is tagged by a source tier:

- Tier 1: official anchor
- Tier 2: scientific literature
- Tier 3: structured repository / covariate source
- Tier 4: targeted scrape-only source

And by evidence role:

- anchor-eligible
- direct-truth-eligible
- prior-only
- chart-derived
- research-only

---

## 3. Phase 0 Formalism: Evidence Intake and Discovery

Before the formal objects, one important architectural note:

- OpenResearcher, heavier Docling usage, BGE-M3, and Chroma belong to the **discovery / retrieval enhancement layer**
- DuckDB, Parquet, FAISS, provenance gates, and registry promotion rules remain the **scientific production core**

So mathematically, Phase 0 is best viewed as two coupled sub-operators:

\[
\Phi_0 = \Phi_{0,\text{core}} \oplus \Phi_{0,\text{explore}}
\]

where:

- \(\Phi_{0,\text{core}}\) is authoritative and scientific-gate-aware,
- \(\Phi_{0,\text{explore}}\) is broad, search-heavy, and retrieval-oriented.

The memory-consistent cold-start rule is:

\[
\text{broad categories}
\rightarrow
\text{wide harvest}
\rightarrow
\text{soft extraction}
\rightarrow
\text{cheap filtering and scoring}
\rightarrow
\text{ablation / benchmark / scientific gating}
\]

So the formal logic of Phase 0 is not:

\[
\text{guess exact variables first}
\]

but:

\[
\text{generate a wide candidate field first, then narrow it}
\]

## 3.1 Source model

Each source record is a typed object:

\[
s = (\text{id}, \text{tier}, \text{org}, \text{url}, \text{doc\_type}, \text{geo}, \text{time}, \text{license}, \text{eligibility}, \text{metadata})
\]

with required scientific locator condition:

\[
\mathbf{1}_{\text{verifiable}}(s) =
\mathbf{1}\left[
\text{url}(s)\neq\varnothing
\;\vee\;
\text{doi}(s)\neq\varnothing
\;\vee\;
\text{pmid}(s)\neq\varnothing
\;\vee\;
\text{openalex\_id}(s)\neq\varnothing
\right]
\]

Phase 0 scientific admissibility requires:

\[
\mathbf{1}_{\text{verifiable}}(s)=1
\]

for any record entering the scientific registry path.

## 3.2 Document model

Each harvested document \(d\) is linked to one source \(s(d)\) and carries:

\[
d = (\text{source\_id}, \text{local\_path}, \text{file\_type}, \text{media\_type}, \text{checksum}, \text{parse\_eligible}, \text{parser\_hint}, \text{embedded\_payload})
\]

There is also a parser provenance sequence:

\[
\Pi_d = (\pi_1, \pi_2, \dots, \pi_m)
\]

that records:

- how the document was obtained,
- whether it was metadata-only,
- whether a remote snapshot was materialized,
- whether file type had to be sniffed or downgraded,
- which parser path was actually used.

## 3.3 Query-bank formalism

For plugin \(z\), corpus mode \(m\), and adapter \(h\), Phase 0 builds a query bank:

\[
Q(z,m,h)=Q_{default}(z)\cup Q_{hiv\_direct}(z,m)\cup Q_{upstream}(z,m)\cup Q_{geo}(z,m)\cup Q_{adapter}(h,m)
\]

Each query \(q \in Q\) carries metadata:

\[
q \mapsto (\text{domain\_bucket}(q), \text{query\_lane}(q), \text{geo\_focus}(q))
\]

where:

- `query_lane` is either `hiv_direct` or `upstream_determinant`,
- `geo_focus` is often `philippines` or `mixed`.

## 3.4 Harvest budget allocation

For a total target \(N\), each external source \(h\) receives a budget:

\[
N_h = \operatorname{round}(w_h N)
\]

subject to:

\[
\sum_h N_h = N
\]

where \(w_h\) are source weights over:

- PubMed
- Crossref
- OpenAlex
- Semantic Scholar
- arXiv
- bioRxiv
- Kaggle

The purpose is not statistical optimality.
It is controlled corpus diversification.

## 3.4A Cold-start category and query structure

At cold start, the initial universe should be generated from broad domain categories rather than a tiny hand-written variable list.

Let:

\[
\mathcal{B}_{cold}
=
\{\text{epi}, \text{clinical}, \text{economics}, \text{logistics}, \text{mobility}, \text{stigma}, \text{demography}, \text{policy}, \text{environment}, \dots\}
\]

Then the initial query universe is:

\[
Q_0 = \bigcup_{b \in \mathcal{B}_{cold}} Q_b
\]

where each \(Q_b\) is a domain-specific query family.

## 3.5 Observation extraction model

Each extracted observation \(o\) is represented as:

\[
o = (\text{parameter\_text}, \text{canonical\_name}, v, u, g, t, k, a, x, \ell, \rho)
\]

where:

- \(v\): numeric value
- \(u\): unit
- \(g\): geography
- \(t\): time
- \(k\): key-population field if known
- \(a\): age band if known
- \(x\): sex if known
- \(\ell\): extraction method label
- \(\rho\): provenance and eligibility metadata

The extraction stack can be viewed as:

\[
o = \Psi(\text{document text}, \text{table cells}, \text{rules}, \text{LLM normalizer})
\]

with strict schema validation:

\[
\mathbf{1}_{\text{schema}}(o)=1
\]

required for retention.

## 3.6 Canonicalization

Canonicalization is a two-stage map:

\[
\kappa : \text{raw phrase} \rightarrow \text{candidate canonical variable}
\]

implemented conceptually as:

\[
\kappa = \kappa_{\text{rules}} \oplus \kappa_{\text{LLM}}
\]

where:

- rules/dictionaries act first,
- the local LLM only resolves ambiguity,
- embeddings are not allowed to define the canonical variable on their own.

## 3.7 Dual lane relevance scoring

For each literature record \(i\), we define two main relevance scores:

\[
Q^{(H)}_i \in [0,1]
\qquad
Q^{(U)}_i \in [0,1]
\]

where:

- \(Q^{(H)}_i\): HIV-direct score
- \(Q^{(U)}_i\): upstream determinant score

The accepted lanes are:

\[
\mathcal{L}_{H} = \{i: Q^{(H)}_i \ge \tau_H\}
\]

\[
\mathcal{L}_{U} = \{i: Q^{(U)}_i \ge \tau_U\}
\]

and the union accepted set is:

\[
\mathcal{L}_{\cup} = \mathcal{L}_{H}\cup\mathcal{L}_{U}
\]

with review and reject bands defined by lower score intervals.

## 3.8 Linkage scoring

For upstream determinants, we defined pathway relevance targets:

\[
\Lambda = \{
\text{prevention\_access},
\text{testing\_uptake},
\text{linkage\_to\_care},
\text{retention\_adherence},
\text{suppression\_outcomes},
\text{mobility\_network\_mixing},
\text{health\_system\_reach},
\text{biological\_progression}
\}
\]

Each record \(i\) receives a linkage vector:

\[
\lambda_i = (\lambda_{i,\ell})_{\ell\in\Lambda}
\]

with:

\[
\lambda_{i,\ell}\in[0,1]
\]

and top linkage targets:

\[
\operatorname{TopLink}(i)=\{\ell : \lambda_{i,\ell}\text{ is among the largest components}\}
\]

This converts “broad determinant evidence” into structured causal hypotheses.

## 3.9 Soft ontology

Instead of forcing every record into a rigid ontology, we preserve a soft tag set:

\[
\Omega_i = \{\omega_1,\dots,\omega_m\}
\]

and soft subparameter hints:

\[
H_i = \{h_1,\dots,h_n\}
\]

These are carried forward for later clustering, linkage, and promotion.

## 3.10 Retrieval formalism

For each retrievable item \(i\), define an embedding:

\[
e_i \in \mathbb{R}^d
\]

with cosine similarity or inner product:

\[
\operatorname{sim}(i,j)=\frac{\langle e_i,e_j\rangle}{\|e_i\|\|e_j\|}
\]

Production retrieval stayed:

- exact / controlled over curated candidate space,
- with FAISS as the main ANN/exact vector engine,
- DuckDB and Parquet as the source of truth.

Chroma was retained only for:

- chunk retrieval,
- lane-specific exploration,
- metadata + document-filter search over curated subsets.

OpenResearcher can be represented as an optional discovery operator:

\[
\mathcal{Q}_{OR} : \text{seed query} \mapsto \text{expanded candidate document set}
\]

but it does not replace the scientific registry path:

\[
\mathcal{Q}_{OR}(q) \subseteq \mathcal{D}_{\text{proposed}}
\xrightarrow{\text{same source / parse / validation contracts}}
\mathcal{E}
\]

BGE-M3 similarly belongs to the optional embedding-upgrade family:

\[
e_i = f_{emb}(x_i)
\]

where \(f_{emb}\) may be a lighter local encoder by default and BGE-M3 when compute permits.

This also implies a silo-aware discovery design.

Let the corpus be partitioned into domain silos:

\[
\mathcal{D} = \bigsqcup_{m \in \mathcal{M}_{silo}} \mathcal{D}_m
\]

where \(\mathcal{M}_{silo}\) may include:

- economics
- logistics
- mobility
- stigma
- biology
- policy
- official anchors

Then retrieval and query expansion operate over:

\[
Q_m \times \mathcal{D}_m
\]

before later aggregation into shared candidate space.

The intended operational rule is:

- broad discovery uses a general embedder over siloed corpora
- narrower biomedical models remain downstream specialists for biomedical refinement tasks

## 3.11 Phase 0 sanity checks

We added explicit sanity functions for:

### Fallback reliance

\[
F = F_{\text{backend}} + F_{\text{heuristic}} + F_{\text{benchmark}} + F_{\text{graph}} + F_{\text{projection}}
\]

with thresholded statuses:

- pass
- warn
- fail

### Choice overload

This is conceptually a function of:

- candidate-to-curated ratio,
- singleton-support share,
- domain concentration,
- weak winner margins,
- weak graph edge separation.

Symbolically:

\[
\Xi = \Xi_{\text{registry}} + \Xi_{\text{benchmark}} + \Xi_{\text{graph}}
\]

and high \(\Xi\) indicates underdetermined selection rather than healthy breadth.

## 3.12 Candidate-only vs promotion-eligible distinction

Let:

\[
\mathcal{C}_{cand}
\]

denote broad candidate-only evidence and:

\[
\mathcal{C}_{prom}
\subseteq
\mathcal{C}_{cand}
\]

denote promotion-eligible evidence.

Then the intended rule is:

\[
\mathcal{C}_{prom}
=
\left\{
c \in \mathcal{C}_{cand}
:
\text{verifiable provenance}
\land
\text{usable alignment}
\land
\text{acceptable quality}
\land
\text{survives ablation / benchmark / scientific gates}
\right\}
\]

This is the formal version of:

- harvest widely
- extract softly
- preserve provenance
- narrow later

---

## 4. Phase 1 Formalism: Alignment and Tensorization

## 4.1 Purpose

Phase 1 converts extracted evidence rows into aligned model-facing tensors.

The basic map is:

\[
\Phi_1 : \mathcal{O} \rightarrow \mathcal{X}
\]

where \(\mathcal{X}\) contains aligned numeric surfaces.

## 4.2 Normalization operator

Each extracted observation \(o\) is normalized by:

\[
\mathcal{N}(o)=
(\tilde v, \tilde u, \tilde g, \tilde t, \tilde k, \tilde a, \tilde x, w, m)
\]

where:

- \(\tilde v\): normalized numeric value
- \(\tilde u\): normalized unit
- \(\tilde g\): normalized geography
- \(\tilde t\): normalized time resolution
- \(w\): confidence or source weight
- \(m\): provenance mask

## 4.3 Alignment tensor

A useful generic aligned tensor is:

\[
X_{p,t,f}
\]

or, when subgroup fields are present,

\[
X_{p,t,f,k,a,x}
\]

where:

- \(p\): province
- \(t\): time
- \(f\): canonical feature / subparameter
- \(k,a,x\): subgroup axes when available

Not every feature populates every subgroup axis.
Missingness must be tracked explicitly.

## 4.4 Alignment masks

Let:

\[
M_{p,t,f,k,a,x}\in\{0,1\}
\]

be an availability mask.

We also retain evidence weight:

\[
W_{p,t,f,k,a,x}\in[0,1]
\]

and provenance tier:

\[
T_{p,t,f,k,a,x}\in\{1,2,3,4\}
\]

These matter because not every aligned entry is scientifically equal.

## 4.5 Reconciliation

When multiple rows map to the same aligned cell, we conceptually use a weighted reconciliation operator:

\[
\hat X_{j}
=
\frac{\sum_{o \mapsto j} w_o \tilde v_o}{\sum_{o \mapsto j} w_o}
\]

for cell \(j\), subject to provenance and eligibility constraints.

This is not necessarily the exact implementation, but it is the right formal view of the phase.

## 4.6 Acceleration note

For Phase 1, the agreed acceleration direction is selective:

- move more normalization, weighting, and tensor assembly math into PyTorch tensors,
- keep canonical storage, manifests, and provenance structures outside the hot numerical loop,
- use GPyTorch only when Gaussian-process interpolation is the actual bottleneck for a specific alignment problem.

---

## 5. Phase 2 Formalism: Candidate Packaging and Promotion Staging

## 5.1 Three-layer registry

We repeatedly defined:

- candidate universe \(\mathcal{U}\)
- curated registry \(\mathcal{R}_c\)
- active inferential set \(\mathcal{A}\)

with nesting:

\[
\mathcal{A} \subseteq \mathcal{R}_c \subseteq \mathcal{U}
\]

## 5.2 Candidate object

Each candidate \(c\) is represented by:

\[
c=(\text{code}, \text{domain block}, \text{soft tags}, \text{literature refs}, \text{evidence strength}, \text{linkage profile}, \text{admissibility})
\]

## 5.3 Domain blocks

We discussed blocks:

\[
\mathcal{B}=
\{
\text{population},
\text{behavior},
\text{biology/clinical},
\text{logistics/health systems},
\text{mobility/network},
\text{economics/access},
\text{policy/implementation},
\text{environment/disruption},
\text{interaction-level}
\}
\]

Each candidate belongs to at least one block:

\[
b(c)\subseteq \mathcal{B}
\]

## 5.4 Deduplication and clustering

Candidates are clustered by:

- canonical concept,
- synonym family,
- mechanism family,
- units,
- evidence overlap.

Symbolically:

\[
c_i \sim c_j
\iff
\operatorname{canon}(c_i)\approx \operatorname{canon}(c_j)
\land
\operatorname{unit}(c_i)\approx \operatorname{unit}(c_j)
\land
\operatorname{evidenceOverlap}(c_i,c_j)\text{ is high}
\]

The curated registry is built over cluster representatives and reconciled concept groups.

## 5.5 Candidate scoring

A stylized selection score for candidate \(c\) is:

\[
S(c)=
\alpha Q_{\text{domain}}(c)
\beta Q_{\text{linkage}}(c)
\gamma Q_{\text{evidence}}(c)
\delta Q_{\text{semantic}}(c)
\eta Q_{\text{provenance}}(c)
- \zeta P_{\text{risk}}(c)
\]

where:

- \(Q_{\text{domain}}\): domain quality
- \(Q_{\text{linkage}}\): pathway linkage relevance
- \(Q_{\text{evidence}}\): evidence strength
- \(Q_{\text{semantic}}\): semantic retrieval / relevance support
- \(Q_{\text{provenance}}\): provenance strength
- \(P_{\text{risk}}\): leakage, ambiguity, weak provenance, or overload penalty

Again, the exact coefficients are not the point here.
The formal structure is.

## 5.6 Promotion rule

A candidate block \(B\subset \mathcal{R}_c\) is promotable only if:

\[
\Delta \text{MAE}_{\text{national}} \le \tau_1
\]

and/or improves at least one tracked target:

\[
\Delta \text{MAE}_{\text{regional}} < 0
\quad\text{or}\quad
\Delta \text{calibration} > 0
\quad\text{or}\quad
\Delta \text{boundary-hit rate} < 0
\quad\text{or}\quad
\Delta \text{mechanistic consistency} > 0
\]

without worsening protected metrics beyond tolerance.

That is the ablation/promote logic we repeatedly described.

## 5.7 Acceleration note

For Phase 2, the preferred direction is:

- more PyTorch tensor math for scoring, ranking, and blockwise selection,
- explicit registry and provenance logic retained as first-class objects,
- no forced commitment to NOTEARS or one specific continuous DAG solver unless we intentionally choose that path later.

---

## 6. Phase 3 Formalism: Mechanistic HIV Model

## 6.1 Core latent tensor

The intended enriched latent tensor is:

\[
X_{p,k,a,x,\sigma,\delta,t}
\]

where:

- \(p\): province
- \(k\): key-population group
- \(a\): age band
- \(x\): sex
- \(\sigma \in \{U,D,A,V,L\}\): cascade state
- \(\delta\): duration bucket
- \(t\): time

This is the main latent care tensor.

## 6.2 Coarse CD4 overlay

We chose **not** to make CD4 a full primary axis initially.

Instead, we represent a severity overlay:

\[
C_{p,k,a,x,g,t}
\]

for \(g\in\mathcal{G}\), interpreted as a simplex over CD4 bins within a cell.

Simplex constraint:

\[
\sum_{g\in\mathcal{G}} C_{p,k,a,x,g,t} = 1
\quad\text{and}\quad
C_{p,k,a,x,g,t}\ge 0
\]

for each valid \((p,k,a,x,t)\) cell where the overlay is defined.

## 6.3 Aggregate public cascade quantities

Let total PLHIV stock be:

\[
N_t = \sum_{p,k,a,x,\delta,\sigma} X_{p,k,a,x,\sigma,\delta,t}
\]

Then diagnosed stock:

\[
D^{\star}_t = \sum_{p,k,a,x,\delta} \left(X_{p,k,a,x,D,\delta,t}+X_{p,k,a,x,A,\delta,t}+X_{p,k,a,x,V,\delta,t}+X_{p,k,a,x,L,\delta,t}\right)
\]

On-ART stock:

\[
A^{\star}_t = \sum_{p,k,a,x,\delta} \left(X_{p,k,a,x,A,\delta,t}+X_{p,k,a,x,V,\delta,t}\right)
\]

Suppressed stock:

\[
V^{\star}_t = \sum_{p,k,a,x,\delta} X_{p,k,a,x,V,\delta,t}
\]

Then:

\[
\text{first95}_t = \frac{D^{\star}_t}{N_t}
\]

\[
\text{second95}_t = \frac{A^{\star}_t}{D^{\star}_t}
\]

\[
\text{trueThird95}_t = \frac{V^{\star}_t}{A^{\star}_t}
\]

This expresses the latent cascade.

## 6.4 Reservoir decomposition

We explicitly promoted the stock-flow decomposition:

- undiagnosed
- diagnosed-not-on-ART
- on-ART unsuppressed
- on-ART suppressed

Formally:

\[
R^{(U)}_t = \sum X_{U}
\]

\[
R^{(D)}_t = \sum (X_D + X_L)
\]

\[
R^{(A)}_t = \sum X_A
\]

\[
R^{(V)}_t = \sum X_V
\]

with:

\[
N_t = R^{(U)}_t + R^{(D)}_t + R^{(A)}_t + R^{(V)}_t
\]

This was one of the key mechanistic honesty checks.

## 6.5 Incidence decomposition

New infections enter the undiagnosed state:

\[
I_{p,k,a,x,t}\rightarrow U
\]

with conservation:

\[
\sum_{p,k,a,x} I_{p,k,a,x,t} = I_t
\]

where \(I_t\) is the total implied incidence / stock-growth-consistent inflow.

The key idea is that incidence is not just national.
It is decomposed by:

- geography,
- KP group,
- age,
- sex.

## 6.6 Care transitions

Within each cell, the major transitions are:

\[
U \rightarrow D
\]
\[
D \rightarrow A
\]
\[
A \rightarrow V
\]
\[
A \rightarrow L
\]
\[
L \rightarrow A
\]

These can be represented by transition probabilities:

\[
\pi^{UD}_{p,k,a,x,\delta,t},\;
\pi^{DA}_{p,k,a,x,\delta,t},\;
\pi^{AV}_{p,k,a,x,\delta,t},\;
\pi^{AL}_{p,k,a,x,\delta,t},\;
\pi^{LA}_{p,k,a,x,\delta,t}
\]

## 6.7 Low-rank transition parameterization

We explicitly preferred structured low-rank parameterization over random projection.

A generic logit form is:

\[
\operatorname{logit}\big(\pi^{m}_{p,k,a,x,\delta,t}\big)
=
\alpha^{m}
+ u^{m}_{p}
+ v^{m}_{k}
+ w^{m}_{a}
+ z^{m}_{x}
+ h^{m}_{\delta}
+ \langle \beta^{m}, Z_{p,t}\rangle
+ \Gamma^{m}_{k,a}
+ \Xi^{m}_{k,x}
+ \Lambda^{m}_{g}
\]

where:

- \(m\) indexes transition type,
- \(u_p\): province effect
- \(v_k\): KP effect
- \(w_a\): age effect
- \(z_x\): sex effect
- \(h_\delta\): duration effect
- \(Z_{p,t}\): aligned subparameter covariates
- \(\Gamma_{k,a}\): sparse KP-age interaction
- \(\Xi_{k,x}\): sparse KP-sex interaction
- \(\Lambda_g\): coarse CD4 modifier

This matches the modeling principle we discussed:

- additive structure,
- hierarchical pooling,
- sparse interactions,
- low-rank compression,
- no dense full Cartesian explosion.

## 6.8 State evolution

A schematic one-step update is:

\[
X_{t+1} = \mathcal{T}(X_t, I_t, \Pi_t, A_t, K_t)
\]

where:

- \(I_t\): incidence inflow
- \(\Pi_t\): transition parameter set
- \(A_t\): age progression operator
- \(K_t\): KP turnover operator

At the state level:

\[
X_{U,t+1}=X_{U,t}+I_t-F^{UD}_t-\text{other exits}
\]

\[
X_{D,t+1}=X_{D,t}+F^{UD}_t-F^{DA}_t+\text{returns from relevant states}
\]

\[
X_{A,t+1}=X_{A,t}+F^{DA}_t+F^{LA}_t-F^{AV}_t-F^{AL}_t
\]

\[
X_{V,t+1}=X_{V,t}+F^{AV}_t-\text{losses if modeled}
\]

\[
X_{L,t+1}=X_{L,t}+F^{AL}_t-F^{LA}_t
\]

where:

\[
F^{UD}_t = \pi^{UD}_t \cdot X_{U,t}
\quad\text{etc.}
\]

The exact treatment of mortality and exits was not fully formalized in the conversation, so this is a schematic memory-derived form.

## 6.9 Age progression

We explicitly discussed annual age progression:

\[
15\text{-}24 \rightarrow 25\text{-}34
\]
\[
25\text{-}34 \rightarrow 35\text{-}49
\]
\[
35\text{-}49 \rightarrow 50+
\]

This can be represented as an operator:

\[
\mathcal{A}: X_{p,k,a,x,\sigma,\delta,t}\mapsto X_{p,k,a',x,\sigma,\delta,t+1}
\]

with conservation:

\[
\sum_a X_{p,k,a,x,\sigma,\delta,t}
\approx
\sum_a X_{p,k,a,x,\sigma,\delta,t+1}
\]

up to inflow, exits, and mortality.

## 6.10 KP turnover

We also discussed sparse turnover between KP groups and partner categories:

\[
\mathcal{K}: X_{p,k,a,x,\sigma,\delta,t}\mapsto X_{p,k',a,x,\sigma,\delta,t+1}
\]

with a sparse turnover matrix:

\[
T^{KP}_{k\to k'}
\]

and conservation:

\[
\sum_k X_{p,k,a,x,\sigma,\delta,t}
\approx
\sum_k X_{p,k,a,x,\sigma,\delta,t+1}
\]

again up to inflow and exits.

## 6.11 Observation model

The key conceptual observation distinction was:

- latent cascade state,
- observed testing coverage,
- observed documented suppression,
- lower-bound observed suppression claims.

Let:

\[
y^{(1)}_t,\; y^{(2)}_t,\; y^{(3,\text{obs})}_t,\; y^{(\text{test})}_t
\]

be observed first-95, second-95, documented-third-95, and testing-coverage-style observations.

The model-implied latent quantities are:

\[
\theta^{(1)}_t = \text{first95}_t
\]
\[
\theta^{(2)}_t = \text{second95}_t
\]
\[
\theta^{(3,\text{true})}_t = \text{trueThird95}_t
\]

The critical rule we enforced conceptually is:

\[
\theta^{(3,\text{true})}_t
\not\gg
\text{documented suppression process}
\]

unless the posterior has real support for it.

That is the formal heart of the “latent suppression cannot outrun the testing/documentation process” constraint.

## 6.12 Hierarchy and shrinkage

We insisted that province effects remain inside the probabilistic model.

Formally, province-level parameters should be shrinkage estimates:

\[
u_p \sim \mathcal{N}(u_{r(p)}, \sigma^2_{prov})
\]

\[
u_r \sim \mathcal{N}(u_{nat}, \sigma^2_{reg})
\]

not post-hoc clipped surfaces.

This is central to:

- subnational stability,
- boundary-hit reduction,
- sparse-province robustness.

## 6.13 Acceptance inequalities

We discussed explicit scientific thresholds.

Examples:

\[
\text{mean hidden suppression reservoir error} \le 5 \text{ pp}
\]

\[
\text{mean documented lower-bound miss} \le 5 \text{ pp}
\]

\[
\text{each holdout-year second95 absolute error} \le 3 \text{ pp}
\]

plus regional and province boundary-hit thresholds.

These are not generic soft goals.
They are formal pass/fail inequalities.

## 6.14 Acceleration note

For Phase 3, the agreed roadmap is:

- use PyTorch tensor math more in transition and state-update computations,
- consider NumPyro / JAX only after an explicit decision to migrate the probabilistic core,
- require working JAX GPU support before treating that migration as the operational default.

## 6.15 Possible Phase 3 inference families

Let \(\Theta\) denote the full Phase 3 parameter collection and \(D\) the Phase 3 data bundle.

The scientific target is still the posterior object:

\[
p(\Theta \mid D)
\]

but we did **not** fix one mandatory inference family in conversation memory.

The admissible families include:

### MCMC

\[
\{\Theta^{(m)}\}_{m=1}^{M} \sim p(\Theta \mid D)
\]

or an asymptotically correct Markov-chain approximation to it.

This is the highest-fidelity reference family, but also the most operationally expensive.

### SVI

Choose an approximating family \(q_{\phi}(\Theta)\) and optimize:

\[
\phi^\star
=
\arg\min_{\phi}
D_{KL}\!\left(q_{\phi}(\Theta)\,\|\,p(\Theta \mid D)\right)
\]

equivalently by maximizing an ELBO-like objective.

This is allowed only if we intentionally adopt a variational backend. It is not the committed default.

### Laplace / variational hybrids

Approximate the posterior locally around a fitted mode \(\hat{\Theta}\):

\[
q(\Theta) \approx \mathcal{N}\!\left(\hat{\Theta}, H(\hat{\Theta})^{-1}\right)
\]

or combine local Gaussian structure with variational corrections.

This is the compromise family when full MCMC is too slow and pure variational inference is too coarse.

## 6.16 Where KL divergence belongs in Phase 3

KL divergence belongs in Phase 3 only conditionally:

- it belongs if we adopt SVI or another explicit variational approximation
- it does not become a universal Phase 3 loss just because KL is mathematically available

So the correct conditional statement is:

\[
\text{Use KL-driven objectives in Phase 3} \iff \text{variational inference is explicitly chosen}
\]

KL control does **not** imply perfect recovery of the true scientific system.

## 6.17 Commitment status for Phase 3

Committed:

- mechanistic state structure
- hierarchy and shrinkage
- scientific validation and benchmark gates
- increased PyTorch use in transition and state-update computations

Not yet committed:

- MCMC as the mandatory inference family
- SVI as the mandatory inference family
- Laplace / variational hybrids as the mandatory inference family
- NumPyro / JAX as the operational default backend

## 6.18 HIV-first rescue principle

Let:

\[
\mathcal{M}_{core}
\]

denote the narrow HIV scientific core and:

\[
\mathcal{D}_{prom}
\]

the promoted determinant set.

The rescue rule is:

\[
|\mathcal{D}_{prom}| \text{ must stay small enough that } \mathcal{M}_{core}
\]

remains observation-anchored, mechanistically interpretable, and benchmark-stable.

Operationally, that means:

- strengthen the observation ladder first
- admit determinants only through block tournaments
- enforce hard promotion budgets
- require stress-test survival before retention

---

## 7. Phase 4 Formalism: Simulation and Production Candidate Layer

## 7.1 Controlled rollout map

Phase 4 takes a fitted system and produces policy or scenario objects:

\[
\mathcal{P}: (X_t,\Pi_t,\mathcal{A}) \mapsto \text{policy frontier},\text{selected plan},\text{simulation outputs}
\]

## 7.2 Policy action space

Let \(u_t \in \mathcal{U}\) be an intervention or policy control vector.

Examples conceptually include actions that modify:

- testing intensity,
- linkage support,
- ART retention support,
- diagnostics reach,
- subnational targeting,
- KP-focused service allocation.

## 7.3 Counterfactual dynamics

A counterfactual rollout obeys:

\[
X_{t+1}^{(u)} = f(X_t^{(u)}, u_t, \Pi_t)
\]

with an objective vector:

\[
J(u)=
\left(
\Delta \text{diagnosis},
\Delta \text{treatment},
\Delta \text{suppression},
-\text{cost},
-\text{risk},
\dots
\right)
\]

## 7.4 Pareto frontier

The policy frontier is the nondominated set:

\[
\mathcal{F}_{Pareto}
=
\{u \in \mathcal{U} : \nexists u' \text{ such that } J(u') \succ J(u)\}
\]

This is the formal decision layer we discussed.

## 7.5 MPC framing

We also explicitly referenced MPC-style planning.

That means:

\[
u^{\star}_{1:H}
=
\arg\max_{u_{1:H}}
\sum_{h=1}^{H} \mathcal{L}(X_{t+h},u_{t+h})
\]

subject to:

\[
X_{t+h+1}=f(X_{t+h},u_{t+h},\Pi_{t+h})
\]

with a receding-horizon commitment rule such as:

- optimize over a horizon,
- commit only the first one or two steps,
- then replan.

## 7.6 Possible Phase 4 control families

We did **not** lock one mandatory Phase 4 controller.

The admissible family set currently includes:

### Heuristic frontier

Construct a controlled candidate set:

\[
\mathcal{U}_{cand} \subset \mathcal{U}
\]

evaluate each candidate under the fitted model, and return a nondominated or ranked subset:

\[
\mathcal{F}_{heuristic} \subseteq \mathcal{U}_{cand}
\]

This is the lowest-risk operational choice and the closest to the current conversation state.

### MPC

Use the receding-horizon formulation already written above:

\[
u^{\star}_{1:H}
=
\arg\max_{u_{1:H}}
\sum_{h=1}^{H} \mathcal{L}(X_{t+h},u_{t+h})
\]

with repeated replanning.

### PPO-like RL

Define a policy \(\pi_{\varphi}(u_t \mid X_t)\) and optimize expected return under rollout dynamics:

\[
\varphi^\star
=
\arg\max_{\varphi}
\mathbb{E}_{\pi_\varphi}\!\left[\sum_{h=0}^{H-1}\gamma^h R_{t+h}\right]
\]

This remains strictly optional and much more infrastructure-heavy.

## 7.7 Where KL divergence does and does not belong in Phase 4

KL divergence belongs in Phase 4 only if we adopt a PPO-like or trust-region-style controller, for example:

\[
D_{KL}\!\left(\pi_{\varphi_{old}} \,\|\, \pi_{\varphi_{new}}\right) \le \delta
\]

or as a penalty term inside the control objective.

It does **not** belong as:

- proof that the policy is scientifically valid
- a substitute for causal or mechanistic validation
- a required baseline ingredient of Phase 4 before a policy-gradient controller is even chosen

## 7.8 Commitment status for Phase 4

Committed:

- controlled scenario surfaces
- frontier-style downstream comparison surfaces
- post-Phase-3 gating before operational use

Not yet committed:

- heuristic frontier as the permanent default
- MPC as the permanent default
- PPO-like RL as the permanent default
- KL-penalized policy updates as the permanent default

## 7.9 Acceleration note

For Phase 4, a distributed rollout layer such as Ray is only justified if:

\[
\text{distributed simulation is the real bottleneck and Phase 4 is mature enough to need it}
\]

It is not part of the required baseline formalism before that point.

## 7.10 Disease-plugin portability note

Let \(z\) denote a disease plugin.

Then the model family is better written as:

\[
\mathcal{M}^{(z)}
\]

where \(z\) changes:

- state topology
- observation operator
- subgroup topology
- intervention semantics
- benchmark target map

So portability across diseases is not merely:

\[
\text{same code} + \text{different YAML}
\]

but rather:

\[
\text{shared framework} + \text{different disease plugin} + \text{different packs}
\]

This is what makes HIV, dengue, and TB comparable at the framework level but not reducible to one universal disease-specific tensor form.

## 7.11 Syndemic coupling note

For diseases \(z_a\) and \(z_b\), coupling should generally enter through exported summaries:

\[
E^{(z_a \rightarrow z_b)}_t \in \mathbb{R}^{q_{ab}}
\]

rather than through a forced joint latent tensor.

The practical diagnostic ladder is:

- cheap:
  - shared-driver overlap
  - lagged predictive gain
  - ablation benefit
- medium:
  - comparable-driver-set overlap
  - cross-model sensitivity comparison
- expensive:
  - CCM
  - transfer entropy

This ordering is important on constrained hardware because the cheap and medium diagnostics are much more realistic as baseline coupling tools.

The claim-strength ladder on top of that is:

- weak evidence:
  - exploratory overlap or loose synchronization only
- medium evidence:
  - predictive gain or ablation benefit from exchanged summaries
- strong evidence:
  - repeated, stable, multi-view coupling evidence across holdouts or settings

Only medium-to-strong evidence should support scientific retention, and only strong evidence should support narrow operational coupling claims.

---

## 8. Hard Scientific Contracts

These are mathematical constraints on what the pipeline is allowed to claim.

## 8.1 No fabricated references

For every scientific record \(i\):

\[
\mathbf{1}_{\text{verifiable}}(i)=1
\]

or the record is excluded from scientific use.

## 8.2 No fallback numeric placeholders in scientific claims

Let:

\[
\mathbf{1}_{\text{fallback\_numeric}}(z)
\]

indicate that an artifact \(z\) depends on synthetic or placeholder numeric backfill.

Then scientific claim eligibility requires:

\[
\mathbf{1}_{\text{fallback\_numeric}}(z)=0
\]

## 8.3 Required upstream artifacts

If a downstream scientific artifact claims to be based on upstream stage \(k\), then:

\[
\mathbf{1}_{\text{artifact exists}}(k)=1
\]

Otherwise the claim is invalid.

## 8.4 Research-only vs scientific-claim-eligible split

Every output \(z\) belongs to one of:

- research-only
- prior-driven
- synthetic
- scientific-claim-eligible

Only the last category may enter benchmark claims or production truth.

---

## 9. Overall Formal Summary

The conversation-defined model is best understood as a layered mathematical object:

### Phase 0

\[
\text{raw world}
\rightarrow
\text{typed evidence records}
\rightarrow
\text{dual-lane scored literature + extracted numeric candidates}
\]

### Phase 1

\[
\text{typed evidence}
\rightarrow
\text{aligned normalized tensors}
\]

### Phase 2

\[
\text{aligned tensors + scored evidence}
\rightarrow
\text{candidate universe}
\rightarrow
\text{curated registry}
\rightarrow
\text{active inferential set}
\]

### Phase 3

\[
\text{active inferential set}
\rightarrow
\text{mechanistic HIV state-space model}
\rightarrow
\text{fit / validation / benchmark artifacts}
\]

### Phase 4

\[
\text{fitted model}
\rightarrow
\text{counterfactual policies / frontier / MPC plan}
\]

The central scientific idea is:

\[
\text{wide discovery}
\;+\;
\text{strict gating}
\;+\;
\text{mechanistic structure}
\;+\;
\text{explicit provenance}
\]

not:

\[
\text{unconstrained end-to-end black-box prediction}
\]
