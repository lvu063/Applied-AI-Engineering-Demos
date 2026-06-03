# Cohere Portfolio — Applied AI Engineering Demos

> Four focused demonstrations built to show practical skills for Cohere roles:
> Prompt Specialist, RevOps Analyst, FDE Agentic Platform, and Software Engineer Collect.
> All demos are runnable today with a free Cohere trial key.

**Live demo:** https://cohere-ai-playground.lovable.app

---

## Repository Structure

```
cohere-portfolio/
│
├── prompt-eval/
│   └── prompt_eval.py        # Prompt versioning + multi-dim evaluation framework
│
├── revops/
│   └── revops_pipeline.py    # GTM analytics: ARR waterfall, churn risk, customer 360
│
├── rag-agent/
│   └── rag_agent.py          # RAG pipeline: Cohere Embed + Rerank + Command A
│
└── docs/
    └── methodology.md         # Analytical decisions and design rationale
```

---

## Quick Start

```bash
pip install cohere pandas numpy scipy
export COHERE_API_KEY=your_trial_key_here
```

---

## Demo 1 — Prompt Evaluation Framework
*Targets: Forward Deployed Engineer, Prompt Specialist*

A systematic framework for iterating on and evaluating LLM prompts — the core daily
work of a Prompt Specialist. Tests four prompt variants (baseline, structured,
chain-of-thought, few-shot) against five test cases across four quality dimensions.

```bash
# Full evaluation across all variants
python prompt-eval/prompt_eval.py

# Single live demo query
python prompt-eval/prompt_eval.py --demo "What is the gold PnL?"

# Export results to CSV + JSON
python prompt-eval/prompt_eval.py --export

# Run without API key (mock mode)
python prompt-eval/prompt_eval.py --mock
```

**What it demonstrates:**
- Prompt versioning with documented rationale for each variant
- Multi-dimensional scoring: keyword coverage, hallucination resistance, conciseness, format
- Reproducible evaluation methodology (not "vibe checking")
- Enterprise use case grounding (financial analyst assistant)
- Production patterns: retry logic, error handling, mock fallback

**Sample output:**
```
PROMPT EVALUATION LEADERBOARD
======================================================================
Variant                       Overall     KW   Hall  Concise   Latency
----------------------------------------------------------------------
🥇 v4_few_shot                   0.891  0.880  0.940    0.920     340ms
🥈 v2_structured                 0.874  0.850  0.940    0.907     290ms
🥉 v3_chain_of_thought           0.856  0.830  0.940    0.880     420ms
   v1_baseline                   0.801  0.780  0.900    0.880     210ms
```

---

## Demo 2 — RevOps Analytics Pipeline
*Targets: RevOps Analyst (Analytics)*

Full revenue operations analytics stack on synthetic GTM data — 80 accounts,
200 opportunities, 240 usage records. Mirrors what a RevOps Analyst does daily:
pipeline coverage, churn risk scoring, customer 360, data hygiene checks.

```bash
# Full report
python revops/revops_pipeline.py

# Raw SQL queries via SQLite
python revops/revops_pipeline.py --sql

# Export all datasets to CSV
python revops/revops_pipeline.py --export
```

**What it demonstrates:**
- ARR waterfall by segment (new, expansion, churn, net)
- Pipeline coverage: win rate and avg deal size by lead source
- Customer 360: CRM + API usage + support tickets joined
- Churn risk model: composite score from health, usage trend, open tickets
- Data hygiene: 4 quality checks that would corrupt CRM reporting
- Raw SQL (SQLite): window functions, CTEs, CASE statements
- Mirrors BigQuery/Snowflake patterns used in Looker/Tableau

**SQL highlight:**
```sql
-- Expansion candidates: high usage growth, healthy, no critical tickets
WITH usage_trend AS (
    SELECT account_id,
           MAX(api_calls) - MIN(api_calls) AS usage_growth
    FROM usage GROUP BY account_id
)
SELECT a.account_name, a.segment, a.arr_usd, ut.usage_growth
FROM accounts a
JOIN usage_trend ut ON a.account_id = ut.account_id
WHERE a.health_score > 70 AND ut.usage_growth > 5000
ORDER BY ut.usage_growth DESC LIMIT 10
```

---

## Demo 3 — Metals RAG Agent
*Targets: Forward Deployed Engineer, Agentic Platform*

A production-style RAG pipeline that extends the Metals Risk Dashboard project
with a natural language interface. Uses Cohere's full search stack:
Embed v3 → cosine retrieval → Rerank v4 → Command A generation.

```bash
# Interactive chat
python rag-agent/rag_agent.py

# Single query
python rag-agent/rag_agent.py --query "What is the gold PnL?"

# Retrieval quality evaluation (Precision@1, Precision@3)
python rag-agent/rag_agent.py --eval

# Mock mode (no API key needed)
python rag-agent/rag_agent.py --mock
```

**What it demonstrates:**
- Full RAG pipeline: embed → retrieve → rerank → generate
- In-memory vector store with cosine similarity (numpy)
- Cohere Embed v3 for dense retrieval
- Cohere Rerank v4 for relevance re-ranking
- Grounded generation: model only answers from retrieved context
- Retrieval evaluation: Precision@1 and Precision@3
- Production patterns: retry logic, graceful degradation, mock fallback

**Architecture:**
```
User query
    │
    ▼
Cohere Embed v3 (query embedding)
    │
    ▼
Cosine similarity search → top-5 candidates
    │
    ▼
Cohere Rerank v4 → top-3 re-ranked
    │
    ▼
Command A (grounded generation with context)
    │
    ▼
Answer + source citations
```

---

## Demo 4 — Software Engineer Collect
*Targets: Software Engineer, Collect*

See: [Espace — Bilingual Community Platform](https://espace-cbc-rc.lovable.app)

React 18 · Node.js · Express.js · TypeScript · Jest · SVG

A full-stack React application with a 10-endpoint Express.js API, 32 tests
(integration + unit), bilingual state management via custom hook, and SVG
data visualisations built without external chart libraries. Demonstrates
the frontend/backend versatility and documentation quality the Collect team looks for.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Cohere Command A (`command-a-05-2025`) |
| Embeddings | Cohere Embed v3 (`embed-english-v3.0`) |
| Reranking | Cohere Rerank v4 (`rerank-v4.0-fast`) |
| Data analytics | Python · pandas · numpy · scipy · SQLite |
| Testing | pytest · mock fallback for all API calls |
| Frontend demo | React 18 · Lovable |

---

## Running Without an API Key

Every demo has `--mock` mode that runs with deterministic fake responses.
Real Cohere API calls are made only when `COHERE_API_KEY` is set.
Trial keys are free at [dashboard.cohere.com](https://dashboard.cohere.com/api-keys).

---

*Built as a portfolio project targeting engineering roles at Cohere.
Background in international economics and data engineering — strong on
domain translation, analytical rigour, and building things that actually get used.*
