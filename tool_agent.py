"""
tool_agent.py
-------------
Cohere Portfolio — Tool Use Agent Demonstration

Extends the prompt evaluation work into agentic territory.
A Prompt Specialist at Cohere doesn't just evaluate static prompts —
they build and evaluate agents that USE TOOLS to complete multi-step tasks.

This agent uses Cohere Command A's tool use capability to answer
metals trading questions by calling structured tools rather than
relying purely on parametric knowledge.

Tools implemented:
  get_position(symbol)          → returns position data for a metal
  get_pnl_summary(desk)         → returns PnL for a desk or metal
  check_risk_limits(book_id)    → returns concentration and breach status
  get_volatility(symbol)        → returns vol metrics for a metal
  calculate_stress_pnl(symbol, shock_pct) → applies price shock to position

This demonstrates:
  - Cohere tool use (function calling) with the v2 client
  - ReAct-style agent loop: reason → act → observe → respond
  - Tool evaluation: did the agent call the right tool with right params?
  - Enterprise-grade patterns: input validation, structured outputs, audit trail

Usage:
    python prompt-eval/tool_agent.py
    python prompt-eval/tool_agent.py --query "What happens to gold PnL if price drops 10%?"
    python prompt-eval/tool_agent.py --eval
    python prompt-eval/tool_agent.py --mock
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


# =============================================================================
# Tool definitions — the schema Cohere uses to understand available tools
# =============================================================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": (
                "Retrieve current position data for a specific metal. "
                "Returns net quantity, unit, average cost, and long/short status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Metal symbol: XAU, XAG, XPT, XPD, HG, AL, ZN, or NI",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pnl_summary",
            "description": (
                "Get unrealised PnL summary. Can query by desk name "
                "('Precious Metals', 'Base Metals') or metal symbol."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Desk name or metal symbol to query PnL for",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_risk_limits",
            "description": (
                "Check whether a book or metal is within approved risk limits. "
                "Returns concentration percentage and breach status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Book ID (e.g. BM-PROP-01) or metal symbol to check",
                    }
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility",
            "description": "Get 30-day annualised volatility and 52-week range for a metal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Metal symbol",
                    }
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_stress_pnl",
            "description": (
                "Calculate the stressed PnL impact if a metal's price moves by a given percentage. "
                "Use this for what-if analysis and scenario testing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Metal symbol to stress test",
                    },
                    "shock_pct": {
                        "type": "number",
                        "description": "Price shock as a percentage (e.g. -10 for a 10% price drop)",
                    },
                },
                "required": ["symbol", "shock_pct"],
            },
        },
    },
]


# =============================================================================
# Tool implementations — the actual functions the agent can call
# =============================================================================

POSITIONS_DB = {
    "XAU": {"symbol": "XAU", "metal": "Gold",      "net_qty": 6500,    "unit": "troy oz", "avg_cost": 3097.15, "spot": 2003.26, "long_short": "Long"},
    "XAG": {"symbol": "XAG", "metal": "Silver",    "net_qty": 31000,   "unit": "troy oz", "avg_cost": 29.94,   "spot": 24.32,   "long_short": "Long"},
    "XPT": {"symbol": "XPT", "metal": "Platinum",  "net_qty": -8500,   "unit": "troy oz", "avg_cost": 738.82,  "spot": 947.52,  "long_short": "Short"},
    "XPD": {"symbol": "XPD", "metal": "Palladium", "net_qty": -4200,   "unit": "troy oz", "avg_cost": 1166.67, "spot": 1014.92, "long_short": "Short"},
    "HG":  {"symbol": "HG",  "metal": "Copper",    "net_qty": 210000,  "unit": "lbs",     "avg_cost": 3.48,    "spot": 4.52,    "long_short": "Long"},
    "AL":  {"symbol": "AL",  "metal": "Aluminium", "net_qty": -1800,   "unit": "MT",      "avg_cost": 2833.33, "spot": 2458.37, "long_short": "Short"},
    "ZN":  {"symbol": "ZN",  "metal": "Zinc",      "net_qty": 3200,    "unit": "MT",      "avg_cost": 2450.00, "spot": 2812.19, "long_short": "Long"},
    "NI":  {"symbol": "NI",  "metal": "Nickel",    "net_qty": 2500,    "unit": "MT",      "avg_cost": 15680.00,"spot": 18414.82,"long_short": "Long"},
}

PNL_DB = {
    "Precious Metals": {"desk": "Precious Metals", "mtm": -143968460, "cost": -63563295, "pnl": -80405165, "pnl_pct": -7.56},
    "Base Metals":     {"desk": "Base Metals",     "mtm": 1023895336, "cost": 591613395, "pnl": 432281941,  "pnl_pct": 73.1},
    "XAU": {"metal": "Gold",      "pnl": -7110295,  "pnl_pct": -35.3},
    "XAG": {"metal": "Silver",    "pnl": -174339,   "pnl_pct": -18.8},
    "XPT": {"metal": "Platinum",  "pnl": 1773750,   "pnl_pct": 28.2},
    "NI":  {"metal": "Nickel",    "pnl": 6837050,   "pnl_pct": 17.4},
    "HG":  {"metal": "Copper",    "pnl": 218990,    "pnl_pct": 30.0},
}

RISK_DB = {
    "XAU": {"concentration_pct": 28.4, "breach": False, "limit": 40.0},
    "NI":  {"concentration_pct": 22.1, "breach": False, "limit": 40.0},
    "HG":  {"concentration_pct": 18.3, "breach": False, "limit": 40.0},
    "BM-PROP-01":  {"concentration_pct": 22.3, "breach": False, "limit": 40.0},
    "PM-PROP-01":  {"concentration_pct": 19.8, "breach": False, "limit": 40.0},
}

VOL_DB = {
    "NI":  {"symbol": "NI",  "metal": "Nickel",    "vol_30d_ann_pct": 40.76, "high_52w": 30928, "low_52w": 15066},
    "HG":  {"symbol": "HG",  "metal": "Copper",    "vol_30d_ann_pct": 35.87, "high_52w": 6.91,  "low_52w": 3.64},
    "XPD": {"symbol": "XPD", "metal": "Palladium", "vol_30d_ann_pct": 28.43, "high_52w": 1580,  "low_52w": 820},
    "XAU": {"symbol": "XAU", "metal": "Gold",      "vol_30d_ann_pct": 17.24, "high_52w": 2626,  "low_52w": 1971},
    "XAG": {"symbol": "XAG", "metal": "Silver",    "vol_30d_ann_pct": 18.94, "high_52w": 32.5,  "low_52w": 21.4},
}


def get_position(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    pos = POSITIONS_DB.get(symbol)
    if not pos:
        return {"error": f"No position found for symbol '{symbol}'"}
    mtm    = round(pos["net_qty"] * pos["spot"], 2)
    cost   = round(pos["net_qty"] * pos["avg_cost"], 2)
    pnl    = round(mtm - cost, 2)
    pnl_pct = round(pnl / abs(cost) * 100, 2) if cost != 0 else 0
    return {**pos, "mtm_usd": mtm, "cost_basis_usd": cost,
            "unrealised_pnl_usd": pnl, "pnl_pct": pnl_pct}


def get_pnl_summary(entity: str) -> dict:
    entity = entity.strip()
    result = PNL_DB.get(entity) or PNL_DB.get(entity.upper())
    if not result:
        return {"error": f"No PnL data for '{entity}'. Try: 'Precious Metals', 'Base Metals', or a symbol like 'XAU'"}
    return result


def check_risk_limits(entity: str) -> dict:
    entity = entity.strip().upper()
    result = RISK_DB.get(entity)
    if not result:
        return {"status": "no_data", "message": f"No risk data for '{entity}'"}
    return {
        **result,
        "status": "BREACH" if result["breach"] else "OK",
        "message": f"Concentration {result['concentration_pct']}% vs limit {result['limit']}% — {'⚠ BREACH' if result['breach'] else '✓ Within limits'}"
    }


def get_volatility(symbol: str) -> dict:
    symbol = symbol.upper().strip()
    result = VOL_DB.get(symbol)
    if not result:
        return {"error": f"No volatility data for '{symbol}'"}
    return result


def calculate_stress_pnl(symbol: str, shock_pct: float) -> dict:
    symbol = symbol.upper().strip()
    pos    = POSITIONS_DB.get(symbol)
    if not pos:
        return {"error": f"No position for '{symbol}'"}
    shocked_price  = pos["spot"] * (1 + shock_pct / 100)
    stressed_mtm   = round(pos["net_qty"] * shocked_price, 2)
    base_mtm       = round(pos["net_qty"] * pos["spot"], 2)
    cost           = round(pos["net_qty"] * pos["avg_cost"], 2)
    stressed_pnl   = round(stressed_mtm - cost, 2)
    pnl_delta      = round(stressed_mtm - base_mtm, 2)
    return {
        "symbol":         symbol,
        "metal":          pos["metal"],
        "shock_pct":      shock_pct,
        "base_spot":      pos["spot"],
        "shocked_spot":   round(shocked_price, 4),
        "base_pnl_usd":   round(base_mtm - cost, 2),
        "stressed_pnl_usd": stressed_pnl,
        "pnl_delta_usd":  pnl_delta,
        "direction":      "gain" if pnl_delta > 0 else "loss",
    }


TOOL_REGISTRY = {
    "get_position":        get_position,
    "get_pnl_summary":     get_pnl_summary,
    "check_risk_limits":   check_risk_limits,
    "get_volatility":      get_volatility,
    "calculate_stress_pnl": calculate_stress_pnl,
}


# =============================================================================
# Agent runner
# =============================================================================

@dataclass
class AgentTrace:
    """Full audit trail of one agent run."""
    query:        str
    tools_called: list[dict]
    final_answer: str
    latency_ms:   float
    n_steps:      int
    mock_mode:    bool


class ToolAgent:
    """
    ReAct-style tool use agent using Cohere Command A.

    The agent loop:
        1. Send query + tool definitions to Command A
        2. If model returns tool_use: execute the tool, return result
        3. Repeat until model returns text (final answer)
        4. Return answer + full audit trail
    """

    MODEL         = "command-a-05-2025"
    MAX_STEPS     = 5
    SYSTEM_PROMPT = (
        "You are a metals trading analyst assistant with access to live trading tools. "
        "Use the available tools to answer questions accurately. "
        "Always call the most relevant tool before answering — do not guess from memory. "
        "When presenting numbers: always include units (troy oz, MT, USD). "
        "If a tool returns an error, say so clearly rather than fabricating data."
    )

    def __init__(self, api_key: Optional[str] = None, mock: bool = False):
        self.mock = mock
        if not mock:
            key = api_key or os.getenv("COHERE_API_KEY")
            if not key or not COHERE_AVAILABLE:
                print("⚠  Running in mock mode")
                self.mock = True
            else:
                self.client = cohere.ClientV2(api_key=key)

    def run(self, query: str) -> AgentTrace:
        t_start      = time.time()
        messages     = [{"role": "user", "content": query}]
        tools_called = []
        n_steps      = 0
        final_answer = ""

        if self.mock:
            return self._mock_run(query, t_start)

        for step in range(self.MAX_STEPS):
            n_steps += 1
            try:
                response = self.client.chat(
                    model=self.MODEL,
                    system=self.SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOL_DEFINITIONS,
                )
            except Exception as e:
                final_answer = f"[API error: {e}]"
                break

            msg = response.message

            # Check for tool calls
            tool_use_blocks = [b for b in msg.content if b.type == "tool_use"] if msg.content else []

            if not tool_use_blocks:
                # Model returned text — we're done
                text_blocks = [b for b in msg.content if b.type == "text"] if msg.content else []
                final_answer = text_blocks[0].text if text_blocks else "[No response]"
                break

            # Execute each tool call
            tool_results = []
            for tool_call in tool_use_blocks:
                name   = tool_call.name
                params = tool_call.parameters if hasattr(tool_call, "parameters") else {}

                tool_fn = TOOL_REGISTRY.get(name)
                if tool_fn:
                    result = tool_fn(**params)
                else:
                    result = {"error": f"Unknown tool: {name}"}

                tools_called.append({
                    "step":   step + 1,
                    "tool":   name,
                    "params": params,
                    "result": result,
                })

                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })

            # Add assistant message and tool results to history
            messages.append({"role": "assistant", "content": msg.content})
            messages.extend(tool_results)

        latency = (time.time() - t_start) * 1000
        return AgentTrace(
            query=query,
            tools_called=tools_called,
            final_answer=final_answer,
            latency_ms=round(latency, 1),
            n_steps=n_steps,
            mock_mode=False,
        )

    def _mock_run(self, query: str, t_start: float) -> AgentTrace:
        """Deterministic mock showing the agent would call the right tools."""
        q = query.lower()
        tools_called = []

        if "gold" in q and ("pnl" in q or "position" in q):
            tools_called.append({"step": 1, "tool": "get_position", "params": {"symbol": "XAU"}, "result": get_position("XAU")})
            final_answer = (
                f"Based on get_position(XAU): Gold is net Long 6,500 troy oz at an average cost of $3,097.15/oz. "
                f"Current spot is $2,003.26/oz, giving a MTM value of $13,021,190 against a cost basis of $20,131,485. "
                f"Unrealised PnL: -$7,110,295 (-35.3%)."
            )
        elif "stress" in q or "drop" in q or "shock" in q or "%" in q:
            symbol = "XAU" if "gold" in q else "NI" if "nickel" in q else "XAU"
            shock  = -10.0
            tools_called.append({"step": 1, "tool": "calculate_stress_pnl", "params": {"symbol": symbol, "shock_pct": shock}, "result": calculate_stress_pnl(symbol, shock)})
            result = calculate_stress_pnl(symbol, shock)
            final_answer = (
                f"Stress test result for {result['metal']} with a {shock}% price shock: "
                f"spot moves from ${result['base_spot']:,.2f} to ${result['shocked_spot']:,.2f}. "
                f"PnL delta: ${result['pnl_delta_usd']:,.0f} ({result['direction']}). "
                f"Stressed unrealised PnL: ${result['stressed_pnl_usd']:,.0f}."
            )
        elif "volatile" in q or "volatility" in q:
            tools_called.append({"step": 1, "tool": "get_volatility", "params": {"symbol": "NI"}, "result": get_volatility("NI")})
            tools_called.append({"step": 2, "tool": "get_volatility", "params": {"symbol": "XAU"}, "result": get_volatility("XAU")})
            final_answer = "Based on get_volatility: Nickel (NI) is the most volatile at 40.76% annualised vol, more than twice Gold's 17.24%. Nickel's 52W range ($15,066–$30,928) reflects significant price swings driven by EV battery demand uncertainty."
        elif "risk" in q or "limit" in q or "breach" in q:
            tools_called.append({"step": 1, "tool": "check_risk_limits", "params": {"entity": "XAU"}, "result": check_risk_limits("XAU")})
            final_answer = "Based on check_risk_limits: Gold (XAU) concentration is 28.4% of portfolio notional — within the 40% breach threshold. No current limit breaches across the portfolio."
        else:
            tools_called.append({"step": 1, "tool": "get_pnl_summary", "params": {"entity": "Base Metals"}, "result": get_pnl_summary("Base Metals")})
            final_answer = "Based on get_pnl_summary: Base Metals desk has an unrealised PnL of +$432,281,941 (+73.1%), making it the primary profit driver. Precious Metals desk is at -$80,405,165 (-7.56%)."

        return AgentTrace(
            query=query,
            tools_called=tools_called,
            final_answer=final_answer,
            latency_ms=round((time.time() - t_start) * 1000, 1),
            n_steps=len(tools_called),
            mock_mode=True,
        )


# =============================================================================
# Tool call evaluation — did the agent use the right tools?
# =============================================================================

EVAL_CASES = [
    {"query": "What is our current gold position?",                "expected_tool": "get_position",        "expected_param_contains": "XAU"},
    {"query": "What happens to nickel PnL if price drops 15%?",   "expected_tool": "calculate_stress_pnl", "expected_param_contains": "NI"},
    {"query": "Which metal is most volatile?",                     "expected_tool": "get_volatility",       "expected_param_contains": None},
    {"query": "Is the gold book within risk limits?",              "expected_tool": "check_risk_limits",    "expected_param_contains": "XAU"},
    {"query": "Show me Precious Metals desk PnL",                  "expected_tool": "get_pnl_summary",      "expected_param_contains": "Precious"},
]


def run_tool_eval(agent: ToolAgent) -> dict:
    """Evaluate whether the agent calls the correct tool for each query."""
    correct_tool    = 0
    correct_params  = 0
    n               = len(EVAL_CASES)

    for tc in EVAL_CASES:
        trace = agent.run(tc["query"])
        if not trace.tools_called:
            continue
        first_tool = trace.tools_called[0]["tool"]
        if first_tool == tc["expected_tool"]:
            correct_tool += 1
        if tc["expected_param_contains"]:
            params_str = json.dumps(trace.tools_called[0].get("params", {}))
            if tc["expected_param_contains"] in params_str:
                correct_params += 1
        else:
            correct_params += 1

    return {
        "tool_selection_accuracy":  round(correct_tool / n, 3),
        "param_accuracy":           round(correct_params / n, 3),
        "n_cases":                  n,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tool Use Agent — Cohere Portfolio Demo")
    parser.add_argument("--query",   help="Single query to run")
    parser.add_argument("--eval",    action="store_true", help="Run tool call evaluation")
    parser.add_argument("--mock",    action="store_true", help="Run without API key")
    parser.add_argument("--api-key", help="Cohere API key")
    args = parser.parse_args()

    print("\n" + "━" * 60)
    print("  TOOL USE AGENT — Cohere Command A + Function Calling")
    print("━" * 60)

    agent = ToolAgent(api_key=args.api_key, mock=args.mock)

    if args.eval:
        print("\n  Running tool selection evaluation...")
        results = run_tool_eval(agent)
        print(f"  Tool selection accuracy: {results['tool_selection_accuracy']:.1%}")
        print(f"  Parameter accuracy:      {results['param_accuracy']:.1%}")
        return

    query = args.query or "What happens to our gold position if price drops 10%?"
    print(f"\n  Query: {query}\n")
    trace = agent.run(query)

    if trace.tools_called:
        print("  Tool calls:")
        for tc in trace.tools_called:
            print(f"    Step {tc['step']}: {tc['tool']}({json.dumps(tc['params'])})")
            print(f"    → {json.dumps(tc['result'])[:120]}...")

    print(f"\n  Answer: {trace.final_answer}")
    print(f"\n  Steps: {trace.n_steps}  |  Latency: {trace.latency_ms:.0f}ms  |  Mock: {trace.mock_mode}")


if __name__ == "__main__":
    main()
