"""
rag_agent.py
------------
Cohere Portfolio — FDE Agentic Platform Demonstration

A production-style RAG (Retrieval-Augmented Generation) agent built on:
  - Cohere Embed v3 for dense vector embeddings
  - Cohere Rerank v4 for relevance re-ranking
  - Cohere Command A for generation
  - In-memory vector store (numpy cosine similarity)

Designed around the metals trading domain — extends the existing
Metals Risk Dashboard project with a natural language query interface.

This demonstrates the core skills for Forward Deployed Engineer, Agentic Platform:
  - Production Python with proper error handling
  - RAG pipeline: embed → retrieve → rerank → generate
  - Tool use pattern (search_knowledge_base tool)
  - Evaluation of retrieval quality

Usage:
    python rag_agent.py                        # interactive chat
    python rag_agent.py --query "What is gold PnL?"
    python rag_agent.py --eval                 # run retrieval quality eval
    python rag_agent.py --mock                 # run without API key
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

# =============================================================================
# Knowledge base — metals trading domain documents
# =============================================================================

KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "Gold (XAU) Position Summary",
        "content": (
            "Gold (XAU) is held in books PM-PROP-01 and PM-CLIENT-01. "
            "Net long position: 6,500 troy oz across both books. "
            "Current spot price: $2,003.26/troy oz (LBMA PM fix). "
            "Total MTM value: $13,021,190. Cost basis: $20,131,485. "
            "Unrealised PnL: -$7,110,295 (-35.3%). Position is Long. "
            "52-week high: $2,626.18. 52-week low: $1,971.71. "
            "30-day annualised volatility: 17.24%."
        ),
        "category": "position",
        "metal": "Gold",
    },
    {
        "id": "KB-002",
        "title": "Nickel (NI) Position Summary",
        "content": (
            "Nickel (NI) is the highest-volatility metal in the portfolio. "
            "Net long position: 2,500 MT in book BM-PROP-01. "
            "Current spot price: $18,414.82/MT (LME official close). "
            "Total MTM value: $46,037,050. Cost basis: $39,200,000. "
            "Unrealised PnL: +$6,837,050 (+17.4%). Position is Long. "
            "30-day annualised volatility: 40.76% — highest in portfolio. "
            "52-week range: $15,066 – $30,928."
        ),
        "category": "position",
        "metal": "Nickel",
    },
    {
        "id": "KB-003",
        "title": "Precious Metals Desk — PnL Summary",
        "content": (
            "Precious Metals desk covers Gold (XAU), Silver (XAG), Platinum (XPT), "
            "and Palladium (XPD). "
            "Total MTM value: -$143,968,460. Cost basis: -$63,563,295. "
            "Unrealised PnL: -$80,405,165 (-7.56%). "
            "10 positions, 6 long / 4 short. Win rate: 40%. "
            "Largest loss: Gold at -$12.7M. "
            "Desk is net short on Precious Metals overall."
        ),
        "category": "desk_summary",
        "metal": "Precious",
    },
    {
        "id": "KB-004",
        "title": "Base Metals Desk — PnL Summary",
        "content": (
            "Base Metals desk covers Copper (HG), Aluminium (AL), Zinc (ZN), "
            "and Nickel (NI). "
            "Total MTM value: $1,023,895,336. Cost basis: $591,613,395. "
            "Unrealised PnL: +$432,281,941 (+73.1%). "
            "14 positions, 9 long / 5 short. Win rate: 64%. "
            "Largest gain: Nickel at +$6.8M. "
            "Base Metals is the primary profit driver this period."
        ),
        "category": "desk_summary",
        "metal": "Base",
    },
    {
        "id": "KB-005",
        "title": "Risk Exposure — Concentration Analysis",
        "content": (
            "Portfolio concentration by metal (% of total notional): "
            "Gold: 28.4%, Nickel: 22.1%, Copper: 18.3%, Silver: 12.1%, "
            "Platinum: 8.7%, Aluminium: 5.2%, Zinc: 3.1%, Palladium: 2.1%. "
            "No single metal exceeds the 40% concentration breach threshold. "
            "BM-PROP-01 is the most concentrated book at 22.3% of portfolio notional. "
            "All positions are within approved risk limits as of 2024-12-31."
        ),
        "category": "risk",
        "metal": "Portfolio",
    },
    {
        "id": "KB-006",
        "title": "Settlement Ladder — Next 4 Weeks",
        "content": (
            "Settlement obligations for the next four weeks: "
            "Week 1 (Jan 1–7): 12 trades settling, net cash outflow -$4.2M. "
            "Week 2 (Jan 8–14): 8 trades settling, net cash inflow +$2.1M. "
            "Week 3 (Jan 15–21): 15 trades settling, net cash outflow -$7.8M. "
            "Week 4 (Jan 22–28): 6 trades settling, net cash inflow +$1.3M. "
            "Largest single settlement: TRD-000087 (Gold forward, $3.1M) in Week 3."
        ),
        "category": "operations",
        "metal": "Portfolio",
    },
    {
        "id": "KB-007",
        "title": "Volatility Summary — All Metals",
        "content": (
            "30-day rolling annualised volatility by metal (as of 2024-12-31): "
            "Nickel: 40.76%, Copper: 35.87%, Palladium: 28.43%, "
            "Aluminium: 22.15%, Zinc: 19.82%, Silver: 18.94%, "
            "Gold: 17.24%, Platinum: 14.67%. "
            "Nickel is the most volatile; Platinum is the least volatile. "
            "High volatility in Nickel is driven by EV battery demand uncertainty."
        ),
        "category": "analytics",
        "metal": "All",
    },
    {
        "id": "KB-008",
        "title": "Counterparty Exposure — Top 5",
        "content": (
            "Top 5 counterparties by gross notional exposure: "
            "1. Goldman Sachs: $48.2M (12 trades). "
            "2. JPMorgan Commodities: $41.7M (9 trades). "
            "3. Barclays Metals: $38.4M (11 trades). "
            "4. HSBC Metals: $29.1M (8 trades). "
            "5. Citibank NA: $24.6M (7 trades). "
            "All counterparties are within approved credit limits. "
            "Goldman Sachs net exposure (long - short): +$12.4M."
        ),
        "category": "risk",
        "metal": "Portfolio",
    },
    {
        "id": "KB-009",
        "title": "Copper (HG) Trading Activity",
        "content": (
            "Copper (HG) trading activity: 210,000 lbs net long across two books "
            "(BM-PROP-01: 125,000 lbs; BM-CLIENT-01: 85,000 lbs). "
            "Current spot: $4.519/lb (LME). MTM value: $948,990. "
            "Cost basis: $730,000. Unrealised PnL: +$218,990 (+30.0%). "
            "Trade type mix: 60% Spot, 30% Forward, 10% Swap. "
            "Most active counterparty: Morgan Stanley (4 trades)."
        ),
        "category": "position",
        "metal": "Copper",
    },
    {
        "id": "KB-010",
        "title": "ETL Pipeline — Data Quality Report",
        "content": (
            "Daily ETL pipeline run log (2024-12-31): "
            "Trades loaded: 300 rows (0 rejected). "
            "Price history loaded: 2,088 rows (0 rejected). "
            "Positions calculated: 24 net positions. "
            "PnL records generated: 24 records. "
            "Pipeline duration: 4.2 minutes. SLA: <30 minutes. ✓ "
            "All 9 post-load verification checks passed. "
            "No data quality issues flagged."
        ),
        "category": "operations",
        "metal": "Portfolio",
    },
]

# =============================================================================
# In-memory vector store
# =============================================================================

@dataclass
class VectorStore:
    """Simple in-memory vector store using numpy cosine similarity."""
    documents: list[dict]
    embeddings: Optional[np.ndarray] = None

    def cosine_similarity(self, query_vec: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all stored embeddings."""
        if self.embeddings is None:
            raise ValueError("Store not indexed yet. Call index() first.")
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        normed = self.embeddings / np.clip(norms, 1e-10, None)
        q_norm = query_vec / np.clip(np.linalg.norm(query_vec), 1e-10, None)
        return normed @ q_norm

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """Return top-k most similar documents."""
        scores = self.cosine_similarity(query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {**self.documents[i], "similarity_score": float(scores[i])}
            for i in top_indices
        ]


# =============================================================================
# RAG Agent
# =============================================================================

class MetalsRAGAgent:
    """
    Production-style RAG agent for metals trading Q&A.

    Pipeline:
        1. Embed query using Cohere embed-english-v3.0
        2. Retrieve top-k candidates via cosine similarity
        3. Rerank using Cohere rerank-v4.0-fast
        4. Generate answer using Command A with retrieved context
    """

    EMBED_MODEL  = "embed-english-v3.0"
    RERANK_MODEL = "rerank-v4.0-fast"
    CHAT_MODEL   = "command-a-05-2025"

    SYSTEM_PROMPT = """You are a metals trading analyst assistant with access to a knowledge base
of position data, PnL summaries, and risk reports for a financial institution's trading desk.

Rules:
- Answer only from the provided context. If the answer isn't in the context, say so clearly.
- Be precise with numbers. Always include units (troy oz, MT, lbs, USD).
- Keep answers concise — under 150 words unless the question requires more detail.
- Never fabricate data or make investment recommendations.
- If asked about real-time prices or current market conditions, note that data is as of 2024-12-31."""

    def __init__(self, api_key: Optional[str] = None, mock: bool = False):
        self.mock = mock
        self.store = VectorStore(documents=KNOWLEDGE_BASE)

        if not mock:
            key = api_key or os.getenv("COHERE_API_KEY")
            if not key or not COHERE_AVAILABLE:
                print("⚠  Running in mock mode (no API key or cohere package)")
                self.mock = True
            else:
                self.client = cohere.ClientV2(api_key=key)

        self._index()

    def _index(self):
        """Embed all documents and store in vector store."""
        texts = [f"{doc['title']}\n{doc['content']}" for doc in KNOWLEDGE_BASE]

        if self.mock:
            # Mock embeddings: random unit vectors (consistent per doc)
            rng = np.random.RandomState(42)
            embeddings = rng.randn(len(texts), 128)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.store.embeddings = embeddings / norms
            return

        try:
            response = self.client.embed(
                texts=texts,
                model=self.EMBED_MODEL,
                input_type="search_document",
            )
            self.store.embeddings = np.array(response.embeddings)
            print(f"  ✓ Indexed {len(texts)} documents ({self.store.embeddings.shape[1]}-dim embeddings)")
        except Exception as e:
            print(f"  ⚠ Embedding failed ({e}), falling back to mock")
            self.mock = True
            self._index()

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string."""
        if self.mock:
            rng = np.random.RandomState(hash(query) % (2**31))
            vec = rng.randn(128)
            return vec / np.linalg.norm(vec)

        response = self.client.embed(
            texts=[query],
            model=self.EMBED_MODEL,
            input_type="search_query",
        )
        return np.array(response.embeddings[0])

    def _rerank(self, query: str, candidates: list[dict], top_n: int = 3) -> list[dict]:
        """Rerank candidates using Cohere Rerank."""
        if self.mock or len(candidates) <= top_n:
            return candidates[:top_n]

        try:
            docs = [f"{c['title']}\n{c['content']}" for c in candidates]
            response = self.client.rerank(
                query=query,
                documents=docs,
                model=self.RERANK_MODEL,
                top_n=top_n,
            )
            reranked = []
            for result in response.results:
                doc = dict(candidates[result.index])
                doc["rerank_score"] = result.relevance_score
                reranked.append(doc)
            return reranked
        except Exception as e:
            print(f"  ⚠ Rerank failed ({e}), using retrieval order")
            return candidates[:top_n]

    def retrieve(self, query: str, top_k: int = 5, top_n: int = 3) -> list[dict]:
        """Full retrieval pipeline: embed → retrieve → rerank."""
        query_vec  = self._embed_query(query)
        candidates = self.store.search(query_vec, top_k=top_k)
        reranked   = self._rerank(query, candidates, top_n=top_n)
        return reranked

    def answer(self, query: str) -> dict:
        """End-to-end: retrieve context + generate grounded answer."""
        t_start = time.time()

        # Step 1: Retrieve
        context_docs = self.retrieve(query)
        context_text = "\n\n".join(
            f"[{doc['id']}] {doc['title']}\n{doc['content']}"
            for doc in context_docs
        )

        # Step 2: Generate
        user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

        if self.mock:
            response_text = self._mock_answer(query, context_docs)
        else:
            try:
                response = self.client.chat(
                    model=self.CHAT_MODEL,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                )
                response_text = response.message.content[0].text
            except Exception as e:
                response_text = f"[Generation error: {e}]"

        latency = (time.time() - t_start) * 1000

        return {
            "query":         query,
            "answer":        response_text,
            "sources":       [{"id": d["id"], "title": d["title"]} for d in context_docs],
            "latency_ms":    round(latency, 1),
            "mock_mode":     self.mock,
        }

    def _mock_answer(self, query: str, docs: list[dict]) -> str:
        """Mock answer for demo without API key."""
        q = query.lower()
        if "gold" in q and "pnl" in q:
            return "Based on KB-001: Gold (XAU) has an unrealised PnL of -$7,110,295 (-35.3%) as of 2024-12-31. The position is Long 6,500 troy oz at an average cost above current spot of $2,003.26/troy oz."
        if "volatile" in q or "volatility" in q:
            return "Based on KB-007: Nickel (NI) is the most volatile metal at 40.76% annualised vol, followed by Copper (35.87%). Platinum is the least volatile at 14.67%. Nickel's high volatility is attributed to EV battery demand uncertainty."
        if "desk" in q or "precious" in q:
            return "Based on KB-003: The Precious Metals desk has an unrealised PnL of -$80,405,165 (-7.56%). The desk is net short overall, with Gold as the largest loss contributor at -$12.7M."
        if "risk" in q or "concentration" in q:
            return "Based on KB-005: No metal exceeds the 40% concentration threshold. Gold is the most concentrated at 28.4% of total notional. All positions are within approved risk limits."
        source_titles = "; ".join(d["title"] for d in docs[:2])
        return f"Based on the knowledge base ({source_titles}): I found relevant context but cannot provide a specific answer without the live API. Please set COHERE_API_KEY for full responses."

    def eval_retrieval(self) -> dict:
        """
        Evaluate retrieval quality using precision@k.
        Tests whether the correct document is retrieved for known queries.
        """
        test_queries = [
            {"query": "What is the Gold PnL?",              "expected_id": "KB-001"},
            {"query": "Which metal is most volatile?",       "expected_id": "KB-007"},
            {"query": "Precious Metals desk performance",    "expected_id": "KB-003"},
            {"query": "Concentration risk and limits",       "expected_id": "KB-005"},
            {"query": "Settlement obligations this week",    "expected_id": "KB-006"},
            {"query": "Counterparty exposure Goldman Sachs", "expected_id": "KB-008"},
        ]

        hits_at_1 = 0
        hits_at_3 = 0

        for tc in test_queries:
            results = self.retrieve(tc["query"], top_k=5, top_n=3)
            ids = [r["id"] for r in results]
            if ids and ids[0] == tc["expected_id"]:
                hits_at_1 += 1
            if tc["expected_id"] in ids:
                hits_at_3 += 1

        n = len(test_queries)
        return {
            "n_queries":    n,
            "precision_at_1": round(hits_at_1 / n, 3),
            "precision_at_3": round(hits_at_3 / n, 3),
            "note": "Mock mode uses random embeddings — run with API key for real scores",
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Metals RAG Agent — Cohere Portfolio Demo")
    parser.add_argument("--query", help="Single query to answer")
    parser.add_argument("--eval",  action="store_true", help="Run retrieval quality evaluation")
    parser.add_argument("--mock",  action="store_true", help="Run without Cohere API key")
    parser.add_argument("--api-key", help="Cohere API key")
    args = parser.parse_args()

    print("\n" + "━" * 60)
    print("  METALS RAG AGENT — Cohere Embed + Rerank + Command A")
    print("━" * 60)

    agent = MetalsRAGAgent(api_key=args.api_key, mock=args.mock)

    if args.eval:
        print("\n  Running retrieval quality evaluation...")
        results = agent.eval_retrieval()
        print(f"  Precision@1: {results['precision_at_1']:.1%}")
        print(f"  Precision@3: {results['precision_at_3']:.1%}")
        print(f"  ({results['note']})")
        return

    if args.query:
        result = agent.answer(args.query)
        print(f"\n  Q: {result['query']}")
        print(f"\n  A: {result['answer']}")
        print(f"\n  Sources: {[s['title'] for s in result['sources']]}")
        print(f"  Latency: {result['latency_ms']:.0f}ms")
        return

    # Interactive mode
    print("\n  Interactive mode. Type 'quit' to exit.\n")
    while True:
        try:
            query = input("  You: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue
            result = agent.answer(query)
            print(f"\n  Agent: {result['answer']}")
            print(f"  Sources: {[s['id'] for s in result['sources']]}  |  {result['latency_ms']:.0f}ms\n")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
