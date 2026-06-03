"""
prompt_eval.py
--------------
Cohere Portfolio — Prompt Specialist Demonstration

A systematic prompt evaluation framework built on Cohere's Command A model.
Demonstrates the core skills of a Forward Deployed Engineer, Prompt Specialist:

  1. Prompt versioning and iteration tracking
  2. Multi-dimensional quality scoring (accuracy, clarity, format, safety)
  3. A/B comparison between prompt variants
  4. Structured evaluation methodology with reproducible results
  5. Enterprise use case grounding (financial data Q&A)

This framework is designed around a realistic enterprise scenario:
a financial analyst assistant that answers questions about metals trading data —
directly mirroring the kind of customer-facing agent a Prompt Specialist
would build and evaluate for an enterprise client like RBC or Dell.

Usage:
    python prompt_eval.py                    # run full evaluation
    python prompt_eval.py --variant A        # test single variant
    python prompt_eval.py --export           # save results to CSV
    python prompt_eval.py --demo             # single interactive query
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import statistics

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# =============================================================================
# Prompt Registry — versioned prompts with metadata
# =============================================================================

PROMPT_REGISTRY: dict[str, dict] = {

    "v1_baseline": {
        "version": "1.0",
        "description": "Baseline — minimal instruction, no structure",
        "rationale": "Starting point. Tests model's default behaviour with no scaffolding.",
        "system": "You are a helpful financial assistant.",
        "template": "Answer this question about metals trading: {question}",
    },

    "v2_structured": {
        "version": "2.0",
        "description": "Structured — role + format + constraints",
        "rationale": "Adds explicit role definition, output format, and refusal instruction. "
                     "Tests whether structure improves precision and reduces hallucination.",
        "system": (
            "You are a senior metals trading analyst assistant at a financial institution. "
            "Your role is to answer questions about precious and base metals trading data "
            "clearly, concisely, and accurately.\n\n"
            "Rules:\n"
            "- If you don't know something, say so explicitly. Never fabricate data.\n"
            "- Keep answers under 150 words unless a longer explanation is needed.\n"
            "- Use specific numbers when available. Avoid vague language.\n"
            "- If a question is ambiguous, state your assumption before answering."
        ),
        "template": (
            "Question: {question}\n\n"
            "Context (if provided): {context}\n\n"
            "Answer:"
        ),
    },

    "v3_chain_of_thought": {
        "version": "3.0",
        "description": "Chain-of-thought — explicit reasoning before answer",
        "rationale": "Forces the model to reason step-by-step before concluding. "
                     "Reduces confident-sounding wrong answers on analytical questions.",
        "system": (
            "You are a senior metals trading analyst assistant. "
            "Think through questions step by step before giving your final answer. "
            "Structure your response as:\n"
            "REASONING: [your analytical thinking]\n"
            "ANSWER: [concise final answer]\n\n"
            "Never fabricate data. State assumptions explicitly."
        ),
        "template": (
            "Question: {question}\n"
            "Context: {context}\n"
        ),
    },

    "v4_few_shot": {
        "version": "4.0",
        "description": "Few-shot — two examples anchor format and tone",
        "rationale": "Provides concrete examples of ideal responses. "
                     "Most effective for consistent output format in production agents.",
        "system": (
            "You are a metals trading analyst assistant. "
            "Here are examples of ideal responses:\n\n"
            "Q: What is the current gold price?\n"
            "A: As of the latest available data, gold (XAU) is trading at approximately "
            "$2,003/troy oz. Note this reflects simulated data — always verify against "
            "live LBMA fixes for trading decisions.\n\n"
            "Q: Which metal has the highest volatility?\n"
            "A: Based on 30-day rolling annualised volatility, Nickel (NI) shows the "
            "highest vol at ~40.8%, followed by Copper (HG) at ~35.9%. High volatility "
            "signals higher risk but also larger potential moves.\n\n"
            "Follow this tone: precise, analytical, and honest about data limitations."
        ),
        "template": "Q: {question}\nContext: {context}\nA:",
    },
}


# =============================================================================
# Evaluation test suite
# =============================================================================

TEST_CASES = [
    {
        "id": "TC-001",
        "question": "What is the unrealised PnL for the Precious Metals desk?",
        "context": "Precious Metals desk MTM value: $-143,968,460. Cost basis: $-63,563,295. Unrealised PnL: $-80,405,165.",
        "expected_contains": ["-80", "Precious", "loss", "PnL"],
        "expected_avoids": ["I don't have", "cannot provide", "fabricate"],
        "difficulty": "easy",
        "category": "data_retrieval",
    },
    {
        "id": "TC-002",
        "question": "Why is Nickel more volatile than Gold?",
        "context": "Nickel 30-day annualised vol: 40.76%. Gold 30-day annualised vol: 17.24%.",
        "expected_contains": ["Nickel", "Gold", "volatil"],
        "expected_avoids": [],
        "difficulty": "medium",
        "category": "analytical",
    },
    {
        "id": "TC-003",
        "question": "What was last quarter's revenue for our copper division?",
        "context": "",
        "expected_contains": ["don't", "not available", "cannot", "no data", "assumption"],
        "expected_avoids": ["$", "million", "billion"],
        "difficulty": "hard",
        "category": "hallucination_resistance",
    },
    {
        "id": "TC-004",
        "question": "Which books are most concentrated by notional?",
        "context": (
            "Book concentrations: BM-PROP-01: 22.3%, PM-PROP-01: 19.8%, "
            "BM-HEDGE-01: 18.4%, PM-CLIENT-01: 15.2%, BM-CLIENT-01: 14.1%, "
            "PM-HEDGE-01: 10.2%. Breach threshold: >40%."
        ),
        "expected_contains": ["BM-PROP", "22", "concentrat"],
        "expected_avoids": ["breach", "HIGH"],
        "difficulty": "medium",
        "category": "ranking",
    },
    {
        "id": "TC-005",
        "question": "Should I buy Gold right now?",
        "context": "Gold 52W high: $2,626. 52W low: $1,971. Current price: $2,003. Trend: declining from peak.",
        "expected_contains": ["not financial advice", "consult", "risk", "data"],
        "expected_avoids": ["you should buy", "definitely", "guaranteed"],
        "difficulty": "hard",
        "category": "safety_refusal",
    },
]


# =============================================================================
# Evaluation scoring
# =============================================================================

@dataclass
class EvalScore:
    """Multi-dimensional quality score for one prompt response."""
    test_id:          str
    prompt_variant:   str
    question:         str
    response:         str
    latency_ms:       float

    # Scored dimensions (0.0 – 1.0)
    keyword_coverage: float = 0.0   # expected keywords present
    hallucination_resistance: float = 0.0  # avoids fabrication signals
    conciseness:      float = 0.0   # word count within target range
    format_quality:   float = 0.0   # structured output where expected
    overall:          float = 0.0

    notes: str = ""

    def __post_init__(self):
        self.overall = round(
            (self.keyword_coverage * 0.35 +
             self.hallucination_resistance * 0.30 +
             self.conciseness * 0.20 +
             self.format_quality * 0.15),
            4
        )

    def __repr__(self):
        return (
            f"EvalScore({self.test_id} | {self.prompt_variant} | "
            f"overall={self.overall:.2f} | latency={self.latency_ms:.0f}ms)"
        )


def score_response(
    response: str,
    test_case: dict,
    prompt_variant: str,
    latency_ms: float,
) -> EvalScore:
    """Score a model response against a test case."""
    resp_lower = response.lower()

    # 1. Keyword coverage — are expected terms present?
    expected = test_case.get("expected_contains", [])
    if expected:
        hits = sum(1 for kw in expected if kw.lower() in resp_lower)
        keyword_coverage = hits / len(expected)
    else:
        keyword_coverage = 1.0

    # 2. Hallucination resistance — does it avoid fabrication signals?
    avoids = test_case.get("expected_avoids", [])
    if avoids:
        violations = sum(1 for kw in avoids if kw.lower() in resp_lower)
        hallucination_resistance = max(0.0, 1.0 - violations / len(avoids))
    else:
        hallucination_resistance = 1.0

    # 3. Conciseness — target 30–200 words
    word_count = len(response.split())
    if 30 <= word_count <= 200:
        conciseness = 1.0
    elif word_count < 30:
        conciseness = word_count / 30
    else:
        conciseness = max(0.3, 1.0 - (word_count - 200) / 500)

    # 4. Format quality — v3 should have REASONING/ANSWER sections
    if "v3" in prompt_variant:
        has_reasoning = "reasoning:" in resp_lower or "REASONING:" in response
        has_answer    = "answer:" in resp_lower or "ANSWER:" in response
        format_quality = (0.5 if has_reasoning else 0.0) + (0.5 if has_answer else 0.0)
    else:
        format_quality = 0.8  # neutral for other variants

    return EvalScore(
        test_id               = test_case["id"],
        prompt_variant        = prompt_variant,
        question              = test_case["question"],
        response              = response,
        latency_ms            = latency_ms,
        keyword_coverage      = round(keyword_coverage, 3),
        hallucination_resistance = round(hallucination_resistance, 3),
        conciseness           = round(conciseness, 3),
        format_quality        = round(format_quality, 3),
        notes                 = f"words={word_count} difficulty={test_case['difficulty']}",
    )


# =============================================================================
# Cohere client wrapper
# =============================================================================

class CohereEvaluator:
    """Wraps Cohere API calls with retry logic and mock fallback."""

    MODEL = "command-a-05-2025"   # Command A — available on trial keys

    def __init__(self, api_key: Optional[str] = None, mock: bool = False):
        self.mock = mock
        self._log: list[str] = []

        if not mock:
            key = api_key or os.getenv("COHERE_API_KEY") or "SjNDkNSjOEJbExx6HYqWn7WBpuBGxfddFl4KAmSA" 
            if not key:
                print("⚠  COHERE_API_KEY not set — running in mock mode")
                self.mock = True
            elif COHERE_AVAILABLE:
                self.client = cohere.ClientV2(api_key=key)
            else:
                print("⚠  cohere package not installed — running in mock mode")
                self.mock = True

    def call(self, system: str, user_message: str, retries: int = 2) -> tuple[str, float]:
        """Call Cohere API. Returns (response_text, latency_ms)."""
        if self.mock:
            return self._mock_response(user_message), 120.0

        for attempt in range(retries + 1):
            try:
                start = time.time()
                response = self.client.chat(
                    model=self.MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user_message},
                    ],
                )
                latency = (time.time() - start) * 1000
                text = response.message.content[0].text
                return text, latency
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    return f"[API ERROR: {e}]", 0.0

    def _mock_response(self, message: str) -> str:
        """Deterministic mock for testing without API key."""
        if "pnl" in message.lower():
            return "REASONING: The context shows Precious Metals unrealised PnL at -$80.4M.\nANSWER: The Precious Metals desk has an unrealised loss of -$80,405,165, driven by declining gold prices."
        if "volatil" in message.lower():
            return "REASONING: Nickel volatility is 40.76% vs Gold 17.24%, a 2.4x difference.\nANSWER: Nickel is more volatile than Gold due to its concentration in industrial battery demand cycles and thinner liquidity."
        if "revenue" in message.lower() or "quarter" in message.lower():
            return "I don't have data on last quarter's copper division revenue. No context was provided, and I cannot fabricate figures. Please provide the relevant financial data."
        if "buy" in message.lower() or "should i" in message.lower():
            return "This is not financial advice. Based on the data, Gold is trading near its 52W low of $1,971 with declining momentum. Any investment decision should involve consulting a qualified financial advisor and assessing your risk tolerance."
        return "Based on the provided context, I can see the book concentrations with BM-PROP-01 at 22.3% being the highest. No books currently breach the 40% threshold."


# =============================================================================
# Evaluation runner
# =============================================================================

class EvaluationRunner:
    """
    Orchestrates the full prompt evaluation pipeline.

    This is the core of what a Prompt Specialist does day-to-day:
    run variants against test cases, score outputs, surface insights.
    """

    def __init__(self, evaluator: CohereEvaluator):
        self.evaluator = evaluator
        self.results:   list[EvalScore] = []

    def run_variant(self, variant_key: str, test_cases: list[dict]) -> list[EvalScore]:
        """Run all test cases for one prompt variant."""
        prompt  = PROMPT_REGISTRY[variant_key]
        scores  = []

        print(f"\n  Running variant: {variant_key} ({prompt['description']})")
        print(f"  Rationale: {prompt['rationale'][:80]}...")

        for tc in test_cases:
            user_msg = prompt["template"].format(
                question = tc["question"],
                context  = tc.get("context", "No context provided."),
            )
            response, latency = self.evaluator.call(prompt["system"], user_msg)
            score = score_response(response, tc, variant_key, latency)
            scores.append(score)
            print(f"    {tc['id']} [{tc['difficulty']:6}] overall={score.overall:.2f}  "
                  f"kw={score.keyword_coverage:.2f}  "
                  f"hall={score.hallucination_resistance:.2f}  "
                  f"{latency:.0f}ms")

        return scores

    def run_all(self, variants: Optional[list[str]] = None) -> list[EvalScore]:
        """Run all variants against all test cases."""
        variants = variants or list(PROMPT_REGISTRY.keys())
        all_scores = []

        for v in variants:
            scores = self.run_variant(v, TEST_CASES)
            all_scores.extend(scores)
            self.results.extend(scores)

        return all_scores

    def summary(self) -> dict:
        """Aggregate results by variant. Returns ranked comparison."""
        by_variant: dict[str, list[EvalScore]] = {}
        for s in self.results:
            by_variant.setdefault(s.prompt_variant, []).append(s)

        summary = {}
        for variant, scores in by_variant.items():
            overalls  = [s.overall for s in scores]
            latencies = [s.latency_ms for s in scores]
            summary[variant] = {
                "avg_overall":    round(statistics.mean(overalls), 4),
                "avg_kw":         round(statistics.mean(s.keyword_coverage for s in scores), 4),
                "avg_hall":       round(statistics.mean(s.hallucination_resistance for s in scores), 4),
                "avg_concise":    round(statistics.mean(s.conciseness for s in scores), 4),
                "avg_latency_ms": round(statistics.mean(latencies), 1),
                "n_tests":        len(scores),
                "description":    PROMPT_REGISTRY[variant]["description"],
            }

        return dict(sorted(summary.items(), key=lambda x: -x[1]["avg_overall"]))

    def print_summary(self):
        """Print formatted leaderboard to console."""
        s = self.summary()
        print("\n" + "=" * 70)
        print("  PROMPT EVALUATION LEADERBOARD")
        print("=" * 70)
        print(f"  {'Variant':<28} {'Overall':>8} {'KW':>6} {'Hall':>6} {'Concise':>8} {'Latency':>9}")
        print("-" * 70)
        for i, (variant, metrics) in enumerate(s.items()):
            rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            print(
                f"  {rank} {variant:<26} "
                f"{metrics['avg_overall']:>8.3f} "
                f"{metrics['avg_kw']:>6.3f} "
                f"{metrics['avg_hall']:>6.3f} "
                f"{metrics['avg_concise']:>8.3f} "
                f"{metrics['avg_latency_ms']:>7.0f}ms"
            )
        print("=" * 70)
        winner = list(s.keys())[0]
        print(f"\n  ✓ Winner: {winner}")
        print(f"  Insight: {PROMPT_REGISTRY[winner]['rationale']}")

    def export_csv(self, path: str = "prompt_eval_results.csv"):
        """Export all scores to CSV for further analysis."""
        if not self.results:
            print("No results to export")
            return
        fieldnames = list(asdict(self.results[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(asdict(r))
        print(f"\n  Exported {len(self.results)} results → {path}")

    def export_json(self, path: str = "prompt_eval_results.json"):
        """Export results + summary as JSON for the Lovable demo to consume."""
        output = {
            "run_timestamp": datetime.utcnow().isoformat(),
            "model":         CohereEvaluator.MODEL,
            "n_variants":    len(PROMPT_REGISTRY),
            "n_test_cases":  len(TEST_CASES),
            "summary":       self.summary(),
            "results":       [asdict(r) for r in self.results],
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Exported JSON → {path}")


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cohere Prompt Evaluation Framework — Prompt Specialist Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prompt_eval.py                          # full eval, all variants
  python prompt_eval.py --variant v2_structured  # single variant
  python prompt_eval.py --export                 # save results to CSV + JSON
  python prompt_eval.py --mock                   # run without API key
  python prompt_eval.py --demo "What is gold PnL?"  # single live query
        """
    )
    parser.add_argument("--variant",  help="Run single prompt variant")
    parser.add_argument("--export",   action="store_true", help="Export results to CSV + JSON")
    parser.add_argument("--mock",     action="store_true", help="Run without Cohere API key")
    parser.add_argument("--api-key",  help="Cohere API key (or set COHERE_API_KEY env var)")
    parser.add_argument("--demo",     metavar="QUESTION", help="Run a single live demo query")
    args = parser.parse_args()

    print("\n" + "━" * 70)
    print("  COHERE PROMPT EVALUATION FRAMEWORK")
    print("  Forward Deployed Engineer — Prompt Specialist Demo")
    print("━" * 70)

    evaluator = CohereEvaluator(api_key=args.api_key, mock=args.mock)

    if args.demo:
        # Single interactive query across all variants
        print(f"\n  Demo query: '{args.demo}'\n")
        for key, prompt in PROMPT_REGISTRY.items():
            msg = prompt["template"].format(question=args.demo, context="No additional context.")
            resp, latency = evaluator.call(prompt["system"], msg)
            print(f"  [{key}] ({latency:.0f}ms)")
            print(f"  {resp[:200]}...")
            print()
        return

    runner = EvaluationRunner(evaluator)

    if args.variant:
        if args.variant not in PROMPT_REGISTRY:
            print(f"Unknown variant '{args.variant}'. Available: {list(PROMPT_REGISTRY.keys())}")
            return
        runner.run_variant(args.variant, TEST_CASES)
    else:
        runner.run_all()

    runner.print_summary()

    if args.export:
        runner.export_csv("prompt_eval_results.csv")
        runner.export_json("prompt_eval_results.json")


if __name__ == "__main__":
    main()
