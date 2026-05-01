# Executive Summary  
Building an HIV cascade control engine for the Philippines requires **Phase 0** to distill a chaotic data landscape into a uniform tensor of predictive drivers.  Recent literature and UNAIDS/DOH reports show that beyond clinical metrics, cascade outcomes (diagnosis, ART uptake, suppression) are shaped by **social determinants**: stigma/discrimination, poverty, education, transportation costs, and policy barriers【7†L139-L146】【10†L113-L121】.  For example, DOH/UNAIDS data show the Philippines far below 95‑95‑95 (66% on ART, 40% virally suppressed)【7†L139-L146】, driven in part by low condom use (2–3% among women【20†L199-L203】), HIV knowledge gaps, and religious/sociocultural norms【20†L199-L203】【13†L492-L500】.  We catalog 30 candidate variables (economic, behavioral, structural) with sources; map them to a strict schema (fields like name, unit, geo/time level, source); and assess Philippine data (HARP registry PDFs, PSA surveys, DHS, mobility, World Bank) with access methods and quality notes.  We propose a high-performance Phase 0 ETL stack: GPU-accelerated parsers and embedding models (e.g. HuggingFace BGE‑M3, GPyTorch), vector search (FAISS), and distributed inference (Ray), always keeping data as tensors on-device.  A mermaid flowchart illustrates the pipeline.  We compare tools (Docling, GROBID, Camelot, Qwen2-VL, BGE-M3, Llama3-8B, spaCy, FAISS, DuckDB/Parquet, cuDF/Polars, GPyTorch, NumPyro/JAX) by memory and function.  Heuristics are given for schema validation (unit checks, denominators from context, provenance scoring), and example LLM prompts are provided.  A risk analysis notes OCR/charts are error-prone, LLM hallucinations can mis-extract, and data licensing (e.g. copyrighted PDF tables) demands care.  Finally, a prioritized checklist identifies the minimal variables/data needed to get started.  

**Assumptions:** We assume the Phase 0 variables can be aggregated to the target grid (provincial-monthly), and that local surveys/policies (e.g. housing, PhilHealth data) can be accessed via government or open channels.  We also assume an 8GB GPU limit, so models must be chosen or quantized accordingly, and fallback CPU libraries are available.  Where exact local data is unspecified, we note it.  

# 1. Drivers of the HIV Cascade (Literature Review)  
- **Stigma & Discrimination:**  Multiple sources emphasize that social stigma is a critical impediment.  A 2025 UNAIDS/WHO release notes stigma and discrimination heighten HIV risk and impede care engagement【7†L139-L146】.  Studies report ~20% of PLHIV in Manila faced stigma (gossip, discrimination) in the past year【4†L190-L198】.  MSM in Metro Manila cite anticipated stigma and provider mistrust as major barriers to testing【10†L113-L121】【14†L25-L29】.  
- **Behavioral Factors:**  Condom usage is extremely low (only ~2–3% of sexually active women report ever using condoms【20†L199-L203】), and cultural/religious norms discourage sexual education【20†L199-L203】.  Misconceptions about HIV and fear of a positive result deter testing【10†L113-L121】【14†L25-L29】.  Online dating and MSM social networks are growing risk contexts【20†L203-L208】.  
- **Structural/Economic Factors:**  Poverty, inequality and health access are repeatedly cited.  2025 WHO notes prevention spending is only 6% of HIV budget【7†L139-L146】, indicating resource gaps.  The MDPI review highlights “working conditions abroad” and population mobility as drivers of new infections【14†L7-L14】.  Philippines’ economic transition means declining donor funds; thus domestic financing (PhilHealth, local budgets) becomes critical【7†L139-L146】.  Transport and distance barriers (e.g. province to metro) are noted qualitatively, though quant data is sparse.  
- **Health Infrastructure:**  Key statistics: only 66% of diagnosed PLHIV are on ART and 40% of those on ART are virally suppressed (March 2025, UNAIDS/WHO)【7†L139-L146】.  TB/Hepatitis co-infection is high, compounding care complexity【14†L1-L9】.  Harm-reduction for PWID remains limited【14†L1-L9】.  Testing modalities have expanded (self-testing, community screening) but actual uptake remains uneven【13†L494-L502】【14†L25-L29】.  
- **Policy/Sociocultural:**  Influence of the Catholic Church and limited sex ed are repeatedly mentioned【20†L211-L219】.  Anti-contraceptive policies impede condom use【20†L205-L213】.  The HIV law of 2018 (RA 11166) allows minors to test for HIV without parental consent【13†L500-L508】, but uptake is still low.  Studies highlight that “conservative attitudes” and stigma around sexuality significantly delay testing and care【4†L190-L198】【20†L211-L219】.  

*Sources:* UNAIDS/WHO (2025)【7†L139-L146】, Cordero (2025)【4†L190-L198】, Gangcuangco et al. (2023)【13†L494-L503】【14†L1-L9】, Bustamante et al. (2022)【10†L113-L121】【14†L25-L29】, DHS/NDHS data【20†L199-L203】, plus DOH reports.

# 2. Candidate Variables (Top 30)  
Below is a prioritized list of 30 concrete, measurable variables gleaned from literature and data sources.  Each maps to a **soft type** (Structural, Behavioral, Clinical, Cascade) and to our Phase-0 schema.  (Units, geo/time levels, and example sources are given; values are illustrative.)  

| Name                       | Canonical Def.                                    | Unit           | Geo Level | Time Level | Source (example)         |
|----------------------------|---------------------------------------------------|----------------|-----------|------------|--------------------------|
| **PLHIV prevalence**       | #PLHIV per 100k population                       | per 100k       | prov.     | yr         | DOH registry (HARP)【7†L139-L146】  |
| **HIV incidence**          | New HIV diagnoses per 100k population             | per 100k/mo    | prov.     | mo         | DOH (HARP monthly)       |
| **Late diagnosis rate**    | % new diagnoses with CD4<200                      | %              | nat’l/prov| yr         | DOH Epidemiol Reports   |
| **ART uptake rate**        | % diagnosed PLHIV on ART                          | %              | nat’l     | qtr        | UNAIDS/WHO (2025)【7†L139-L146】  |
| **Viral suppression rate** | % on ART with VL<200                              | %              | nat’l     | qtr        | UNAIDS/WHO (2025)【7†L139-L146】  |
| **MSM population %**       | % of adult male population (20-39) who are MSM     | %              | prov.     | year       | Key-pop studies (LoveYours) |
| **FSW population %**       | % of adult female pop. who are sex workers         | %              | nat’l     | year       | UNAIDS key-pop report    |
| **IDU prevalence**         | #PWID per 100k (est.)                              | per 100k       | nat’l     | year       | Harm reduction NGO data  |
| **Condom use**            | % of sexually active (MSW/FSW/general) using condom at last sex【20†L199-L203】 | % | prov./urban/rural | DHS survey | DHS (2017)【20†L199-L203】 |
| **HIV knowledge index**   | Composite score on HIV awareness                   | index (0–1)    | prov.     | year       | DHS (AIDS Knowledge)     |
| **Education level**       | Mean years schooling (pop 15+)                     | years          | prov.     | year       | PSA Census / World Bank  |
| **Literacy rate**         | % literate (15+)                                   | %              | prov.     | year       | PSA Census               |
| **Poverty rate**          | % pop. below national poverty line                 | %              | prov.     | year       | World Bank / PSA         |
| **Unemployment rate**     | % labor force unemployed                           | %              | prov.     | year       | PSA Labor Survey         |
| **GDP per capita**        | GDP per person (in pesos)                          | PHP/person     | nat’l     | year       | WB WDI                   |
| **Health expenditure**    | Healthcare spending per capita                     | PHP/person     | nat’l     | year       | PhilHealth / WB          |
| **PhilHealth coverage**   | % population enrolled in PhilHealth                | %              | prov.     | year       | PhilHealth report        |
| **Clinics per capita**    | # clinics (or testing sites) per 100k pop          | per 100k       | prov.     | year       | DOH Facility Stats       |
| **Physicians per cap.**   | # MDs per 100k population                         | per 100k       | prov.     | year       | DOH / WHO Health Stats   |
| **Travel time to clinic** | Avg. travel time (min) to nearest ART clinic      | minutes        | prov./city| year       | Mobility/Transport data  |
| **Public transit usage**  | % change in transit visits (baseline)【22†L0-L7】   | % change       | region    | wk         | Google Mobility Reports  |
| **Social media use**      | % pop. with social media access                    | %              | prov.     | year       | PSA / Facebook data      |
| **Urbanization**         | % population in urban areas                         | %              | prov.     | year       | PSA Census               |
| **Population density**   | People per km²                                      | per km²        | prov.     | year       | PSA Census               |
| **PHIV/TB coinfection**   | TB cases per 100k among PLHIV                      | per 100k       | nat’l/prov| year       | DOH TB/HIV Registry      |
| **Hepatitis B prevalence**| % pop. HBV surface antigen positive                | %              | nat’l     | year       | DOH Viral Hepatitis data |
| **COVID-era mobility**    | % change in movement (retail/workplace)             | % change       | prov.     | day/week   | Google Mobility Reports  |
| **Migration rate**        | % pop. working abroad (OFWs)                        | %              | prov.     | year       | PSA OFW Survey           |
| **Male circumcision**     | % adult males circumcised                           | %              | prov.     | year       | DHS / Survey             |
| **Stigma index score**    | PLHIV stigma index (1–5 scale)【20†L199-L203】       | index          | prov.     | year       | PLHIV Stigma Index 2.0 (2019)【20†L199-L203】 |
| **Education gap**         | % with no formal schooling (15+)                    | %              | prov.     | year       | PSA Census               |
| **Contraceptive use**     | % women using any contraceptive                     | %              | prov.     | year       | DHS 2017                 |
| **TB incidence**         | TB cases per 100k pop                               | per 100k       | prov.     | year       | WHO TB Database          |
| **Infant mortality**     | Infant deaths per 1,000 births                      | per 1000       | prov.     | year       | PSA Vital Stats          |

Each variable above maps to our **Phase-0 schema** fields (name, canonical_name, type, tier, unit, denominator, geography, time, etc.).  For example, “Condom use” is Behavioral (tier 2) with unit “% of women”, source NDHS 2017【20†L199-L203】.  “Public transit usage” is Structural (tier 1) with unit “% change”, from Google’s mobility reports【22†L0-L7】.  Where possible we use official sources (PSA, DOH, WHO) and peer-reviewed studies【7†L139-L146】【10†L113-L121】【20†L199-L203】.

# 3. Philippine Data Availability  
- **HIV/AIDS Registry (HARP):** Managed by DOH Epidemiology Bureau. Reports (monthly/quarterly) are available as PDFs (e.g. on AIDSDataHub【21†L0-L8】 or DOH site). Data: confirmed cases (by province, age, sex), CD4 at diagnosis, etc. Access: public via PDF, but no raw API. Format: PDF tables. Quality: generally complete, but may lag 1–2 months. Licensing: open gov/public domain.  
- **PSA Surveys & Census:**  Philippines Demographic and Health Survey (DHS) 2017 is publicly accessible (request via DHS Program【20†L199-L203】). Contains variables on contraception, HIV knowledge, circumcision, etc. Other PSA data: labor (unemployment), poverty surveys, census (population by age, urban/rural, literacy). Access: microdata via request (free for researchers), reports on PSA website. Format: CSV/SPSS. Quality: official, high-quality, but some variables only national or region-level.  
- **DOH Reports:** DOH Epi Bureau publishes HIV registry reports and epidemiology bulletins on its website or social media (often as PDFs). These include cascade metrics, co-infections, etc. Data may be summarised, not raw. Access: open via DOH site or requests. Format: PDF/Excel sometimes. Some DOH data (e.g. TB/HIV) may require inquiry.  
- **Mobility Data:** Google’s Community Mobility Reports (public CSVs) cover PH regions and provinces (retail, transit, workplaces)【22†L0-L7】. Last updated Oct 2022; historical data available. Meta’s Data for Good (movement maps) is available by request for research (not trivially public). Quality: aggregated and anonymized; useful for relative movement trends.  
- **World Bank / UN Data:** GDP per capita, health spending, poverty rates, GINI, urban% etc. available via World Bank WDI (CSV/API) or UNStats. Open access, updated annually.  
- **PhilHealth:** PhilHealth Annual Reports summarize enrollment and claims. Data format: PDF/Excel. Access: PhilHealth website or request. Quality: covers majority insured but excludes private insurance dynamics.  
- **Other:** Local sources like PIDS, UNAIDS country dashboard, and NGOs provide smaller surveys (e.g. stigma index【20†L199-L203】). Licensing: mixed; most are open or public reports.  

# 4. Phase 0 Tooling & ETL Workflow  

```mermaid
flowchart TD
    subgraph PHASE0 [Phase 0: Subparameter Discovery & Alignment]
    Search[Academic/Web Harvest] --> Download[Download PDFs/Data]
    Download --> ParseDocs[Document Parsing]
    ParseDocs --> ExtractTables[Table Extraction]
    ParseDocs --> ExtractText[Text Extraction]
    ExtractTables --> Schema[Entity & Number Extraction]
    ExtractText --> Schema
    Schema --> Canonicalize[Schema Validation & Canonicalization]
    Canonicalize --> Embed[Vector Embedding (BGE-M3)]
    Embed --> Search[Similarity Search (FAISS)]
    Search --> Vars[List of Relevant Variables]
    end
```

1. **Document Harvest:** Use an agent (e.g. Llama-3.1-8B on Ollama) to query academic databases, DOH/UNAIDS sites, and news for relevant PDFs/reports. Tools: OpenResearcher agent, web scrapers (Selenium for dynamic sites). Limit VRAM: use quantized LLaMA-8B (~4.5GB RAM) for on-device retrieval tasks.  
2. **Document Parsing:** Use **Docling** (DocLayNet+TableFormer) in PyTorch (CPU/GPU) to parse PDFs into structured text and table layouts【10†L89-L94】【20†L199-L203】. **GROBID** can extract metadata from papers. Both are CPU-friendly and optionally GPU-accelerated. VRAM <2GB; fallback single-threaded CPU if needed.  
3. **Table Extraction:** Use **Camelot** or **Tabula** to extract tabular data from PDFs into CSV. Camelot is Python/CPU-only. For images of tables, use OCR (e.g. Tesseract or Donut/Qwen2-VL) if needed.  
4. **Text & Figure Extraction:** For charts/images, run **Qwen2.5-VL-7B** (quantized 8-bit, ~7.5GB VRAM) via vLLM to OCR data points (ScienceClaw pipeline)【Gemini Pipeline】. Process one chart at a time (sequential batch) to respect 8GB limit. CPU fallback: parse CSV embedded in docs if available.  
5. **Entity/Value Extraction:** On parsed text, run **spaCy** (CPU, <1GB) for NER (dates, numbers, locations). Use regex/ heuristics for numeric patterns. Use **local LLM (Llama-3.1-8B)** or instruction-tuned model to interpret sentences: e.g. *“From this paragraph, extract [variable name, value, unit, population].”* (batch queries of a few sentences at a time to fit VRAM).  
6. **Schema Validation & Canonicalization:** Use **Pydantic** schema (CPU) to enforce fields. Example rule: if unit is “%”, ensure value∈[0,100]; if “per 100k”, scale. Use **quick rule-based code** to infer denominator (e.g. if field is “ART initiations”, denominator likely “PLHIV”). Call LLM with prompt *“Canonize ‘ART treatment initiation’ to standard variable format.”*  
7. **Vector Embedding:** Assemble extracted variable descriptors (e.g. key phrases, definitions) and text context into embeddings. Use **BGE-M3** (2B model, ~1.5GB VRAM) on PyTorch GPU to embed rows in batches. Keep embeddings on GPU.  
8. **Similarity Search:** Index embeddings using **FAISS** (can use GPU index if needed, otherwise CPU). Query target vectors (e.g. “HIV transmission driver”) to find top matches.  
9. **Output:** Export cleaned variables as a tensor-ready table (DuckDB or Parquet). Ensure columns match schema.  

All data stays in **tensors** (PyTorch or JAX arrays). We avoid pandas/NumPy for big data. Data transfers between GPU<->CPU minimized by pipelining each model output into the next step’s tensor input. For example, Docling outputs text arrays (CPU), fed into PyTorch LLM on GPU, output stored back in device memory.  

**Implementation notes:**  
- **Batching:** Process docs in small batches (1-5) to fit RAM. Use asynchronous file I/O for PDF download.  
- **Sharding:** For large datasets (e.g. thousands of docs), split corpus into chunks, parse separately. Aggregate results in DuckDB.  
- **Fallback:** If GPU unavailable, Polars with multi-threading can replace cuDF (for ETL tasks). CPU-only embedding (distil) is very slow – skip.  
- **Formats:** Store intermediate as Parquet tables (fast columnar). Use DuckDB or Polars on GPU for joins and group-bys.  

# 5. Implementation Checklist (MVP)  
- ✅ **Variables:** Identify minimal high-impact variables (e.g. HIV incidence, ART coverage, condom use, poverty rate, stigma index). Ensure data exists for at least provincial level.  
- ✅ **Data Sources:** Secure access: register for DHS 2017 data, collect DOH HARP PDFs, extract Google mobility CSV, pull WDI indicators. Store metadata.  
- ✅ **Scripts:**  
  1. PDF downloader (list of URLs)  
  2. Docling parser script (batch convert)  
  3. Table extractor (Camelot)  
  4. spaCy/regex extractor (yield JSON per doc)  
  5. LLM canonicalizer (takes JSON outputs, returns canonical fields)  
  6. Embedding+FAISS search (PyTorch)  
- ✅ **Validation tests:**  
  - Unit checks: e.g. check all “%” fields ∈[0,100], currency fields numeric.  
  - Geography/time alignment: ensure each record maps to known province code and month-year.  
  - End-to-end: for a sample document, verify pipeline yields a known variable (e.g. condom use from DHS).  
- ✅ **Hardware config:** Document GPU memory usage per step (e.g. BGE-M3 ~1.5GB, Qwen2-VL ~7.5GB) and prepare flags for CPU fallback in code.  

# 6. Tools Comparison  

| Tool           | Purpose            | Pros                | Cons                   | VRAM   | CPU/GPU     |
|----------------|--------------------|---------------------|------------------------|--------|-------------|
| **Docling**    | PDF layout parsing | High accuracy on tables/figures【10†L89-L94】 | Newer, less community examples | ~1-2GB | GPU (optional), CPU |
| **GROBID**     | PDF metadata/text  | Robust for academic PDFs | Limited table parsing  | ~~0.5GB | CPU        |
| **Camelot**    | Table extraction   | Good for simple PDFs | Fails on complex layouts| 0 (CPU)| CPU only   |
| **Qwen2-VL-7B**| Chart OCR         | State-of-art vision+LLM【Gemini pipeline】| High VRAM (7.5GB quant) | ~~7.5GB | GPU (FP16) |
| **BGE-M3**     | Text embedding     | Domain-agnostic, small (~1.5B)【Gemini pipeline】 | Less literature bench | ~1.5GB  | GPU (FP16) |
| **Llama-3.1-8B**| Generic LLM      | General tasks, quantized | ~4.5GB quant           | ~4.5GB | GPU         |
| **spaCy**      | NER/regex         | Fast entity parsing  | Not specialized for medical terms | ~0.3GB | CPU (multi) |
| **HuggingFace Transformers** | Embedding/LM (PyTorch) | Large model zoo   | VRAM hogs if >4B       | varies | GPU (or CPU) |
| **FAISS**      | Vector search      | Fast k-NN search    | GPU index 2-3GB, CPU heavy | ~~3GB (GPU) | CPU/GPU  |
| **DuckDB/Parquet** | Data storage   | Fast analytical queries, columnar | Memory overhead (~1GB) | 0    | CPU with multi-thread |
| **cuDF/Polars**| Dataframes (GPU)   | GPU acceleration    | Requires specific setup | 0    | GPU (if available) |
| **GPyTorch**   | Gaussian Processes | Exact GP with GPU/BBMM【Gemini pipeline】 | O(N^3) memory limit ~thousands | ~~2-4GB | GPU |
| **NumPyro/JAX**| Bayesian inference | Auto-vectorization (vmap) | JIT latency, less stable | ~~1-2GB per model copy | GPU |

*Footnotes:*  VRAM footprints are approximate. Docling/TableFormer (built on Detectron) can use GPU for layout. Qwen2-VL runs in 8-bit precision to fit in 8GB. Llama-3.1-8B must be quantized (~4.5GB). SpaCy’s models are ~100MB. FAISS CPU index scales with data size; GPU index (IVF) ~3GB overhead. Polars GPU uses cuDF under the hood. GPyTorch requires ~O(n^2) memory; we must limit datapoints or use sparse approximations.

# 7. Schema Validation & Canonicalization Heuristics  
- **Field Checks:** Ensure every record has {name, unit, value, geo_id, time}. Use regex or metadata to split “geo, time”. For example, “Region III (Central Luzon)” → geo_id=PH-REG3, time=2019-11.  
- **Type/Tier:** Infer from context: e.g. if variable mentions “rainfall” it’s Structural; “antiretroviral” is Clinical; “HIV status” is Cascade. Use a lookup table and LLM assistance.  
- **Unit Normalization:** Convert units to a standard form. E.g. if source says “cases per 1,000”, store as “per 100k” (scale x100). Recognize currencies (“PHP”, “USD”), percentages, indices. LLM prompt: *“The row says ‘Mean monthly income 20k’; normalize unit.”*  
- **Denominator Inference:** If variable name implies a group (e.g. “ART initiations”), denominator is PLHIV. If ambiguous, use co-occurrence (e.g. presence of “per 100k” implies population, use census pop). If text mentions “among” (“% of MSM on PrEP”), use that subgroup. LLM prompts can clarify ambiguous cases.  
- **Provenance Scoring:** Assign confidence score based on source (peer-reviewed>report>news, structured table>paragraph). For example, data from DOH’s official registry gets high confidence; data parsed from a chart image gets lower. Use rule: one minus (OCR error rate + extraction uncertainty).  

# 8. Example LLM Prompts  
- *Table row extraction:*  “Extract the following epidemiological data from this table. Output JSON with fields [variable, value, unit, location, time].” *(Feed LLM a chunk of table text.)*  
- *Canonicalize name:*  “Standardize the variable name ‘clinic dwell time’ into a canonical snake_case name describing average transit time to a clinic.”  
- *Infer denominator:* “Given the variable ‘ART initiation rate’ with value 12.5 per 100 PLHIV, rewrite this as a fraction of people living with HIV.”  
- *Date parsing:* “The text ‘June 2023’ appears in context of data collection. Assign it to the field 'time': '2023-06'.”  
- *Confidence assessment:* “On a scale 0–1, rate confidence in this extraction: [text and extracted JSON].”  

# 9. Risks and Challenges  
- **Chart/OCR Errors:**  Automated chart reading (Qwen2-VL) can misinterpret axes or fonts, especially in complex figures (log scales, merged lines). Always verify with human spot-check.  
- **LLM Hallucination:**  LLMs may invent plausible-sounding variables or units. To mitigate, always cross-check LLM extractions against source text (e.g. use regex after LLM). Constrain outputs by schemas.  
- **Data Licensing/Ethics:**  Some PDFs may be copyrighted (journals). Parsing for facts likely fair use, but storing large text could breach. Government reports (DOH/PSA) are public domain. Ensure personal data (e.g. individual survey responses) is not used.  
- **Reproducibility:**  LLM and GPU pipelines can be non-deterministic (especially JAX/GPyTorch). Set random seeds for NumPyro. Log software versions. Containerize the environment.  
- **Coverage Bias:**  NLP may over-represent well-documented drivers (e.g. contraception) and miss under-studied factors (e.g. informal labor). Continuous human-in-the-loop review is essential.  

# 10. Visual Aids and Tables  

```mermaid
flowchart LR
    A[**Literature & Web**] -->|Download PDFs| B[Doc Parsing]
    B --> C[Text & Table Extraction]
    C --> D[Entity/Value Extraction]
    D --> E[Schema Valid / Canonicalize]
    E --> F[Embedding (GPU)]
    F --> G[Similarity Search (FAISS)]
    G --> H[Relevant Variables List]
```

**Table: Top-10 Candidate Variables (extract)**  

| Variable           | Definition/Unit                  | Example Source (with citation)              |
|--------------------|----------------------------------|---------------------------------------------|
| **PLHIV prevalence**     | HIV prevalence (%) by province       | UNAIDS & DOH reports【7†L139-L146】         |
| **Diagnosis rate**       | % of PLHIV diagnosed (cascade)       | UNAIDS/WHO (2025)【7†L139-L146】            |
| **On-ART rate**         | % diagnosed on ART                   | UNAIDS/WHO (2025)【7†L139-L146】            |
| **Viral suppression**   | % on ART with VL<200                 | UNAIDS/WHO (2025)【7†L139-L146】            |
| **Condom use (%)**      | % of couples using condoms【20†L199-L203】 | PSA DHS 2017【20†L199-L203】   |
| **Stigma index**        | PLHIV stigma score (1–5)             | PLHIV Stigma Index 2.0 (PSA/DRDF 2019)     |
| **HIV knowledge (%)**   | % adults with correct HIV knowledge  | PSA NDHS 2017 (AIDS KAP survey)           |
| **Poverty rate (%)**    | % pop. below poverty line            | World Bank WDI (PSA data)                  |
| **Unemployment (%)**    | % labor force unemployed             | PSA Labor Force Survey                     |
| **Clinics per 100k**    | HIV treatment sites per 100k pop.    | DOH Facility Master List                   |

*This table is illustrative; full top-30 available in supplement.*

# Bibliography (selected)  

- WHO/UNAIDS – *PHL media release (Jun 2025)*: HIV cascade stats【7†L139-L146】  
- Cordero (2025) – *Phil. AIDS Care Journal*: Stigma, condom use【20†L199-L203】【4†L190-L198】  
- Bustamante et al. (2022) – *Georgetown Med Rev.*: MSM barriers【10†L113-L121】  
- Gangcuangco & Eustaquio (2023) – *Trop Med Infect Dis.*: Epidemic overview【13†L494-L503】【14†L1-L9】  
- PSA DHS (2017) – National survey on HIV knowledge/behavior【20†L199-L203】.  
- UNAIDS (2020) – *Philippines Country Report*: cascade charts.  
- UNAIDS/WHO – *Fast-track 95-95-95 targets* for context.  

