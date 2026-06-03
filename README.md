# Cohere Portfolio — Applied AI Engineering

> Four production-inspired demos targeting engineering roles at Cohere.
> Every demo runs today with a free trial key — or without one, in mock mode.

**Live demo:** [cohere-ai-playground.lovable.app](https://cohere-ai-playground.lovable.app)

---

## What's in this repo

```
cohere-portfolio/
├── Dockerfile                          # Single containerised image for all demos
├── docker-compose.yml                  # Run each demo by name
├── requirements.txt
│
├── prompt-eval/
│   ├── prompt_eval.py                  # Prompt versioning + multi-dim eval framework
│   └── tool_agent.py                   # Function calling + ReAct agent loop
│
├── revops/
│   └── revops_pipeline.py              # GTM analytics: ARR, churn risk, customer 360
│
├── rag-agent/
│   └── rag_agent.py                    # RAG pipeline: Embed v3 + Rerank v4 + Command A
│
└── docs/
    └── methodology.md                  # Design decisions, GTM stack narrative, Salesforce
                                        # data model, production considerations
```

---

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Cohere trial key (free at dashboard.cohere.com/api-keys)
export COHERE_API_KEY=your_key_here

# Or run in mock mode — no key needed
python prompt-eval/prompt_eval.py --mock
```

**Docker (no Python setup required):**
```bash
docker build -t cohere-portfolio .
docker run -e COHERE_API_KEY=your_key cohere-portfolio
```

---

## Demo 1 — Prompt Evaluation Framework
*Forward Deployed Engineer, Prompt Specialist*

A systematic evaluation harness for iterating on LLM prompts — the actual
daily work of a Prompt Specialist. Vibe-checking is not a methodology.
Four prompt variants, five test cases, four scoring dimensions.

```bash
python prompt-eval/prompt_eval.py                          # full eval
python prompt-eval/prompt_eval.py --demo "What is gold PnL?"  # single query
python prompt-eval/prompt_eval.py --export                 # save CSV + JSON
python prompt-eval/prompt_eval.py --mock                   # no key needed
```

**Four prompt variants tested:**

| Variant | Strategy | Key hypothesis |
|---|---|---|
| `v1_baseline` | Minimal instruction | Control — what does the model do with no scaffolding? |
| `v2_structured` | Role + constraints | Does explicit role definition raise precision? |
| `v3_chain_of_thought` | Reason before answer | Does CoT reduce confident-sounding wrong answers? |
| `v4_few_shot` | Two exemplars | Do examples anchor format better than instructions? |

**Four scoring dimensions:**

| Dimension | Weight | Why it matters |
|---|---|---|
| Hallucination resistance | 30% | A confident wrong answer is worse than no answer |
| Keyword coverage | 35% | Proxies factual completeness |
| Conciseness | 20% | Verbose models hedge — verbosity is a reliability signal |
| Format quality | 15% | Structured outputs are easier to parse downstream |

**Sample leaderboard output:**
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

## Demo 2 — Tool Use Agent
*Forward Deployed Engineer, Prompt Specialist + Agentic Platform*

Extends the prompt eval work into agentic territory. A Prompt Specialist
doesn't just evaluate static prompts — they build agents that call tools
to complete multi-step tasks. This is a ReAct-style agent using Cohere
Command A's function calling capability.

```bash
python prompt-eval/tool_agent.py                                            # default query
python prompt-eval/tool_agent.py --query "What if gold drops 10%?"         # stress test
python prompt-eval/tool_agent.py --eval                                     # tool selection eval
python prompt-eval/tool_agent.py --mock                                     # no key needed
```

**Five tools the agent can call:**

| Tool | What it does |
|---|---|
| `get_position(symbol)` | Net quantity, spot price, cost basis, unrealised PnL |
| `get_pnl_summary(entity)` | PnL by desk or metal |
| `check_risk_limits(entity)` | Concentration % vs breach threshold |
| `get_volatility(symbol)` | 30-day annualised vol, 52W range |
| `calculate_stress_pnl(symbol, shock_pct)` | What-if price shock analysis |

**Agent loop (ReAct pattern):**
```
Query → Command A reasons → calls tool → observes result → reasons again → final answer
```

**Tool selection accuracy (mock mode):** 100% · **Parameter accuracy:** 80%

---

## Demo 3 — RevOps Analytics Pipeline
*RevOps Analyst (Analytics)*

Full revenue operations analytics stack on 80 synthetic accounts, 200
opportunities, 240 usage records, and 310 support tickets. Mirrors what
a RevOps Analyst does daily in BigQuery/Snowflake, surfaced in Looker.

```bash
python revops/revops_pipeline.py             # full report
python revops/revops_pipeline.py --sql       # raw SQL via SQLite
python revops/revops_pipeline.py --export    # save 6 CSV files
```

**What it produces:**

| Analysis | What it answers |
|---|---|
| ARR waterfall | Active vs churned ARR by segment — net ARR by cohort |
| Pipeline coverage | Win rate and avg deal size by stage × lead source |
| Customer 360 | CRM + API usage + support tickets in one joined view |
| Churn risk model | Composite score: health (40%) + usage trend (30%) + tickets (30%) |
| Data hygiene report | 4 quality checks that would corrupt CRM reporting |
| Raw SQL | Window functions, CTEs, CASE — SQLite-executable, BigQuery-portable |

**The Salesforce data model behind the synthetic data:**

| Our table | SFDC object | Key fields |
|---|---|---|
| `accounts` | Account | Segment → Type, ARR → AnnualRevenue |
| `opportunities` | Opportunity | Stage → StageName, Source → LeadSource |
| `usage` | Custom object | api_calls → Units__c |
| `tickets` | Case | priority → Priority, ttrs_hours → FirstResponseTime |

Full SOQL equivalents and GTM stack narrative in `docs/methodology.md`.

---

## Demo 4 — RAG Agent
*Forward Deployed Engineer, Agentic Platform*

Production-style retrieval-augmented generation using Cohere's full search
stack. Extends the Metals Risk Dashboard project with a natural language
query interface. The model only answers from retrieved context — grounded
answers are auditable answers.

```bash
python rag-agent/rag_agent.py                              # interactive chat
python rag-agent/rag_agent.py --query "Gold PnL?"          # single query
python rag-agent/rag_agent.py --eval                       # Precision@1, Precision@3
python rag-agent/rag_agent.py --mock                       # no key needed
```

**Pipeline:**
```
Query
  → Cohere Embed v3 (search_query input type)
  → Cosine similarity → top-5 candidates
  → Cohere Rerank v4 (rerank-v4.0-fast) → top-3
  → Command A (grounded generation, fabrication forbidden)
  → Answer + source citations [KB-001, KB-007]
```

**Why Rerank, not just cosine similarity?**
Cosine similarity retrieves by surface-level similarity. Rerank scores
candidates against actual query intent — a cross-encoder, not a dot product.
Three high-quality re-ranked documents outperform ten loosely-relevant ones.

---

## Demo 5 — Full-Stack React
*Software Engineer, Collect*

See: [espace-cbc-rc.lovable.app](https://espace-cbc-rc.lovable.app)
and: [github.com/[YOUR_GITHUB]/cbcrc-espace-prototype](https://github.com)

React 18 · Node.js · Express.js · TypeScript · Jest · SVG

A bilingual (FR/EN) community platform prototype for CBC Radio-Canada.
10-endpoint REST API, 32 tests (integration + unit), custom `useLanguage`
hook, SVG data visualisations built without chart libraries.

Why no external UI library? Building from scratch demonstrates understanding
of React's component model — not just the ability to configure pre-built tools.
The Collect team builds mission-critical internal tools. That distinction matters.

---

## Docker

```bash
# Build once
docker build -t cohere-portfolio .

# Run each demo
docker run -e COHERE_API_KEY=your_key cohere-portfolio python prompt-eval/prompt_eval.py --export
docker run -e COHERE_API_KEY=your_key cohere-portfolio python revops/revops_pipeline.py --sql
docker run -e COHERE_API_KEY=your_key cohere-portfolio python rag-agent/rag_agent.py --eval
docker run -e COHERE_API_KEY=your_key cohere-portfolio python prompt-eval/tool_agent.py --eval

# Or use docker compose
docker compose run prompt-eval
docker compose run revops
docker compose run rag-agent
docker compose run tool-agent
```

---

## Running without an API key

Every demo has `--mock` mode with deterministic responses. No key, no setup,
no friction. The mock mode also documents expected system behaviour — it is
executable specification.

Trial keys are free at [dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys).

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM + tool use | Cohere Command A (`command-a-05-2025`) |
| Embeddings | Cohere Embed v3 (`embed-english-v3.0`) |
| Reranking | Cohere Rerank v4 (`rerank-v4.0-fast`) |
| Vector store | In-memory numpy (cosine similarity) |
| Data analytics | Python · pandas · numpy · scipy · SQLite |
| Deployment | Docker · docker-compose |
| Testing | pytest · 100% mock fallback coverage |
| Frontend | React 18 · Node.js · TypeScript · Jest |

---

*Background in international economics and data engineering.
Strong on domain translation, analytical rigour, and building things that actually get used.*
