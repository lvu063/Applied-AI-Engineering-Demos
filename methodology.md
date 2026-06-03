# Methodology & Design Decisions
## Cohere Portfolio — Applied AI Engineering

> This document explains the analytical decisions, system design choices,
> and GTM stack narrative behind each demo. It exists because documentation
> is how engineers communicate intent — not just what the code does, but why.

---

## 1. Prompt Evaluation Framework

### The problem it solves

Most prompt iteration happens by feel — a practitioner runs a prompt,
reads the output, and decides "that's better." This is slow, non-reproducible,
and impossible to defend to a client. A Prompt Specialist's job is to replace
vibe-checking with a systematic evaluation loop.

### Design decisions

**Why four variants, not more?**
Four variants cover the four most meaningful prompt engineering strategies:
baseline (control), structured (role + constraints), chain-of-thought
(reasoning before answer), and few-shot (exemplar anchoring). Adding more
variants without a clear hypothesis is noise, not signal.

**Why these four scoring dimensions?**
- *Keyword coverage* — proxies for factual completeness
- *Hallucination resistance* — the highest-stakes metric for enterprise deployments
- *Conciseness* — verbosity is a reliability signal; verbose models often hedge
- *Format quality* — structured outputs are easier to parse downstream

Hallucination resistance is weighted at 0.30 (highest single weight) because
in financial and enterprise contexts, a confident wrong answer is worse than
no answer. This mirrors how Cohere's enterprise customers evaluate models.

**Why test hallucination resistance explicitly?**
TC-003 deliberately asks for information not in the context ("last quarter's
copper division revenue"). A production prompt must handle this gracefully.
Failing this test disqualifies a variant regardless of other scores.

### What this maps to in the Prompt Specialist role

The JD asks for "rigorous evaluations to ensure alignment with customer objectives"
and "metric-driven collaboration." This framework is a concrete implementation
of that — a harness you could hand to a client and say "here is how we will
measure whether your agent is improving."

---

## 2. RevOps Analytics Pipeline

### The GTM tech stack narrative

A modern B2B SaaS GTM stack has five layers:

```
Data sources          CRM & engagement        Warehouse           BI layer         Action
─────────────         ────────────────         ─────────           ────────         ──────
Product usage    →    Salesforce (CRM)    →    Snowflake/     →    Looker/     →    Alerts
Support tickets       HubSpot (marketing)      BigQuery            Tableau          Forecasts
Contract data         Outreach (sales)          (dbt models)        Metabase         Playbooks
Billing/CPQ           ZoomInfo (enrich)
```

The `revops_pipeline.py` simulates the **warehouse layer** — the step where
raw CRM + usage + support data is joined, cleaned, and modelled into
decision-ready tables. In production this would be a dbt project on
Snowflake or BigQuery, with models scheduled via Airflow.

### Quote-to-cash flow

The pipeline models the full quote-to-cash motion:

```
Lead → Opportunity (SFDC) → Proposal → Negotiation → Closed Won
                                                           │
                                                    Contract signed
                                                           │
                                                    Billing activated
                                                           │
                                                    ARR recognised
                                                           │
                                            Renewal / Expansion / Churn
```

The `generate_opportunities()` function models stages 1–5.
The `arr_waterfall()` function models the ARR recognition and churn step.
A production implementation would add a `billing` table joining to
Stripe or Zuora for actual revenue recognition.

### Salesforce data model

The synthetic data mirrors Salesforce's standard object model:

| Our table | Salesforce object | Key fields mapped |
|---|---|---|
| `accounts` | Account | Segment → Type, ARR → AnnualRevenue, CSM → OwnerId |
| `opportunities` | Opportunity | Stage → StageName, ARR → Amount, Source → LeadSource |
| `usage` | Custom object / Events | api_calls → Units__c, active_users → MAU__c |
| `tickets` | Case | priority → Priority, resolved → Status, ttrs_hours → FirstResponseTime |

In a real RevOps implementation, these would be Salesforce SOQL queries:

```sql
-- SOQL equivalent of our pipeline coverage query
SELECT StageName, LeadSource,
       COUNT(Id) opp_count,
       SUM(Amount) total_arr,
       AVG(Probability) avg_probability
FROM Opportunity
WHERE IsClosed = false
GROUP BY StageName, LeadSource
ORDER BY total_arr DESC
```

### Why churn risk scoring matters

The JD asks for "360 degree view of the customer" and "drive revenue growth."
Churn risk scoring is where these meet — it's the analytical output that
triggers a CSM action before a renewal conversation goes wrong.

The composite score weights:
- Health score (40%) — the lagging indicator already in the CRM
- Usage trend (30%) — the leading indicator most CRMs miss
- Open critical tickets (30%) — the operational signal that predicts churn fastest

Usage trend is the most predictive variable in practice. A customer whose
API call volume drops 20% month-over-month is at risk regardless of what
they tell their CSM. This is why usage data in the warehouse is more valuable
than survey scores in the CRM.

---

## 3. RAG Agent Architecture

### Why this pipeline, not a simpler one

The simplest RAG implementation is: embed query → cosine similarity → stuff context → generate.
This works for toy demos. It fails in production for two reasons:

1. **Cosine similarity retrieves by surface-level similarity, not relevance.**
   A query about "gold price movement" might retrieve a document about "gold
   futures settlement" when the user wants "gold spot PnL." Reranking fixes this
   by scoring retrieved candidates against the actual query intent.

2. **Context stuffing degrades generation quality.**
   Giving the model 10 loosely-relevant documents produces hedged, verbose answers.
   Three high-quality, re-ranked documents produces precise answers with fewer
   hallucinations. This is why Cohere's own documentation recommends Rerank as
   a standard pipeline component.

### The Cohere search stack in this pipeline

```
Query: "What is the gold PnL?"
   │
   ▼
Embed v3 (embed-english-v3.0)
   Produces a 1024-dim dense vector optimised for search_query input type
   Input type matters: search_query ≠ search_document embeddings
   │
   ▼
Cosine similarity → top-5 candidates
   In-memory numpy implementation (production: Pinecone, Weaviate, pgvector)
   │
   ▼
Rerank v4 (rerank-v4.0-fast)
   Cross-encoder model: scores each candidate against the full query
   Returns relevance scores, not just similarity scores
   top_n=3 → discards bottom 2 candidates
   │
   ▼
Command A (command-a-05-2025)
   Grounded generation: model instructed to answer ONLY from retrieved context
   System prompt explicitly forbids fabrication
   │
   ▼
Answer + source citations [KB-001, KB-007]
```

### Production considerations documented

Things this demo does that a prototype often skips:
- `input_type` parameter on embed calls (search_query vs search_document)
- Retry logic with exponential backoff on API calls
- Graceful degradation to mock mode if API unavailable
- Retrieval evaluation (Precision@1, Precision@3) to measure pipeline quality
- Source citation — grounded answers are auditable answers

---

## 4. Full-Stack React (SWE Collect)

### Architecture decisions in Espace

See: [github.com/[YOUR_GITHUB]/cbcrc-espace-prototype](https://github.com)

Key decisions worth explaining to an interviewer:

**Why no external UI library?**
Building from scratch demonstrates understanding of React's rendering model
and state management — not just the ability to configure pre-built components.
The Collect team builds internal tools; they need engineers who can go off the
beaten path when pre-built tools don't fit the use case.

**Why a custom `useLanguage` hook instead of i18next?**
The hook encapsulates bilingual state in a way that makes the migration to
i18next trivial — one import swap, no component changes. This is the right
abstraction boundary. Over-engineering to i18next at prototype stage would
have added complexity without benefit.

**Why Express.js + Jest for a prototype?**
The 10-endpoint REST API and 32 tests (integration + unit) exist to show that
"prototype" doesn't mean "untested." The Collect team builds mission-critical
internal tools. Test coverage is a signal that you understand what mission-critical means.

**What Next.js would add**
The honest gap: Espace uses Create React App, not Next.js. In a production
version, Next.js would add server-side rendering for initial load performance,
API routes replacing the separate Express server, and App Router for better
code organisation. The component logic is identical — the migration is a
structural refactor, not a rewrite.

---

## 5. Cross-cutting concerns

### Why one repo, not four

A single `cohere-portfolio` repo tells a coherent story: one engineer who
can move across the stack — from prompt engineering to data pipelines to
agentic systems to frontend. Four separate repos tell four disconnected stories.

### Mock mode as a design principle

Every demo runs without an API key. This is intentional — it means a hiring
manager can clone the repo and run it in 60 seconds without friction. Friction
kills demos. Mock mode also documents the expected behaviour of the real system,
which is a form of executable specification.

### Error handling philosophy

Every API call in this codebase has explicit error handling with graceful
degradation. This is not defensive programming for its own sake — it reflects
how production AI systems actually need to behave. An enterprise client's
workflow cannot stop because an LLM API returned a 429.

---

*This document was written as part of a portfolio project targeting engineering
roles at Cohere. The code is production-inspired but uses synthetic data throughout.
Nothing in this repo contains real financial data or real customer information.*
