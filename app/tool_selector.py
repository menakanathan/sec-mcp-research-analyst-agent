"""
app/tool_selector.py
────────────────────
LLM-driven tool selector — Gap 2 fix.

Instead of an if/elif keyword chain hardcoded in the agent, this module:
  1. Receives the LIVE tool list discovered from the MCP server.
  2. Builds a compact prompt describing each tool's name, description,
     and parameter names.
  3. Asks Ollama to return a JSON object naming the tool to call and its
     arguments.
  4. Validates the selection against the known tool list.
  5. Falls back gracefully if Ollama is unavailable or returns bad JSON.

This makes the agent genuinely agentic: the LLM decides what to do at
runtime based on schemas it received over the wire — not code written
by the developer.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from app.local_llm import OllamaClient, LocalLLMError

logger = logging.getLogger(__name__)

# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class ToolCall:
    tool_name:  str
    arguments:  dict = field(default_factory=dict)
    reasoning:  str  = ""     # why the LLM chose this tool (for evidence log)
    llm_driven: bool = True   # False when keyword fallback was used


# ── Prompt templates ──────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the routing brain of an SEC financial research agent.

You will receive:
  - A list of available MCP tools (name, description, parameter names)
  - A user research question

Your job:
  Select exactly ONE tool and provide its arguments.

Rules:
  - Use only tool names from the list — no invented tools.
  - Use only parameter names shown in the schema — no invented params.
  - Ticker symbols must be real stock exchange symbols (AAPL, MSFT, NVDA,
    TSLA, AMZN, GOOGL, META, JPM, INTC, AMD, etc.). Never use ordinary
    English words as tickers — RISK, ITEM, FACT, FORM, DATA, LIST, SHOW,
    FIND, INFO, HELP are not tickers. If no ticker is mentioned in the
    question, default to AAPL.
  - For compare_companies provide BOTH ticker_a AND ticker_b.
  - For the "item" parameter of extract_filing_section use the BARE
    section number only: "1", "1A", "7", or "8" — never "Item 1A" or
    "ITEM 1A".
  - Respond ONLY with a JSON object — no markdown fences, no preamble.

Required response format (JSON only):
{
  "tool": "<exact tool name from list>",
  "arguments": { "<param>": "<value>", ... },
  "reasoning": "<one sentence explaining your choice>"
}
"""


def _build_user_prompt(question: str, tools: list[dict]) -> str:
    """
    Compact tool list: name + description + parameter names only.
    Keeping it short reduces token usage and improves Ollama reliability.
    """
    lines = ["Available MCP tools:"]
    for t in tools:
        props  = (t.get("inputSchema") or {}).get("properties") or {}
        params = ", ".join(props.keys()) if props else "none"
        lines.append(
            f"  - {t['name']}: {t['description']}  [params: {params}]"
        )
    lines.append("")
    lines.append(f"Research question: {question}")
    lines.append("")
    lines.append("Select the best tool and respond with JSON only.")
    return "\n".join(lines)


# ── JSON extraction ───────────────────────────────────────────────────────

def _extract_json(raw: str) -> dict:
    """
    Parse the LLM response as JSON.

    Handles three cases:
      1. Clean JSON string  → json.loads directly
      2. JSON buried inside prose → find first {...} with regex
      3. Nothing parseable → raises ValueError
    """
    stripped = raw.strip()

    # Case 1: the whole string is JSON
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Case 2: JSON embedded inside text (despite system prompt instructions)
    match = re.search(r"\{.*?\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not extract valid JSON from LLM output. "
        f"First 300 chars: {raw[:300]!r}"
    )


# ── Public entry point ────────────────────────────────────────────────────

def select_tool(question: str,
                tools: list[dict],
                llm: OllamaClient) -> ToolCall:
    """
    Ask Ollama which MCP tool to call for this research question.

    Parameters
    ----------
    question : str
        Natural language research question from the user.
    tools : list[dict]
        Tools discovered live from the MCP server via list_tools().
        Each dict has keys: name, description, inputSchema.
    llm : OllamaClient
        Local Ollama instance.

    Returns
    -------
    ToolCall
        Selected tool name, arguments dict, and reasoning.
        llm_driven=True  → Ollama made the selection.
        llm_driven=False → keyword fallback was used.
    """
    if not tools:
        logger.warning("No tools available — returning default")
        return _default_call("AAPL", reason="no tools available")

    user_prompt = _build_user_prompt(question, tools)
    known_tools = {t["name"] for t in tools}

    try:
        raw    = llm.chat(_SYSTEM_PROMPT, user_prompt)
        parsed = _extract_json(raw)

        tool_name = parsed.get("tool", "").strip()
        arguments = parsed.get("arguments") or {}
        reasoning = parsed.get("reasoning", "")

        # Validate: LLM must have chosen a real tool
        if tool_name not in known_tools:
            raise ValueError(
                f"LLM selected '{tool_name}' which is not in the tool list. "
                f"Known: {sorted(known_tools)}"
            )

        # Validate: arguments must be a dict
        if not isinstance(arguments, dict):
            raise ValueError(
                f"LLM returned arguments of type {type(arguments).__name__}, "
                f"expected dict"
            )

        logger.info(
            f"[tool_selector] LLM selected '{tool_name}' "
            f"args={arguments} — {reasoning}"
        )
        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            reasoning=reasoning,
            llm_driven=True,
        )

    except (LocalLLMError, ValueError, KeyError, TypeError) as exc:
        logger.warning(
            f"[tool_selector] LLM selection failed ({exc}); "
            f"falling back to keyword routing"
        )
        return _keyword_fallback(question, known_tools, str(exc))


# ── Fallback (keyword) ────────────────────────────────────────────────────

def _keyword_fallback(question: str,
                      known_tools: set[str],
                      error_reason: str) -> ToolCall:
    """
    Deterministic fallback used only when Ollama fails.
    Clearly marked llm_driven=False in the returned ToolCall so the
    caller can signal to the user that agentic routing was not used.
    """
    from app.tool_registry import extract_tickers

    lower   = question.lower()
    tickers = extract_tickers(question)
    ticker  = tickers[0] if tickers else "AAPL"

    reason_prefix = f"Keyword fallback (LLM error: {error_reason})"

    if "compare" in lower and len(tickers) >= 2 \
            and "compare_companies" in known_tools:
        return ToolCall(
            tool_name="compare_companies",
            arguments={"ticker_a": tickers[0], "ticker_b": tickers[1]},
            reasoning=f"{reason_prefix}: detected 'compare' + two tickers",
            llm_driven=False,
        )

    if ("risk" in lower or "1a" in lower or "item 1a" in lower) \
            and "extract_filing_section" in known_tools:
        return ToolCall(
            tool_name="extract_filing_section",
            arguments={"ticker": ticker, "item": "1A"},
            reasoning=f"{reason_prefix}: detected risk/1A keyword",
            llm_driven=False,
        )

    if any(w in lower for w in ["financial", "revenue", "income",
                                  "earnings", "facts", "snapshot"]) \
            and "get_financial_snapshot" in known_tools:
        return ToolCall(
            tool_name="get_financial_snapshot",
            arguments={"ticker": ticker},
            reasoning=f"{reason_prefix}: detected financial keyword",
            llm_driven=False,
        )

    if any(w in lower for w in ["filing", "10-k", "10-q", "list"]) \
            and "list_recent_filings" in known_tools:
        return ToolCall(
            tool_name="list_recent_filings",
            arguments={"ticker": ticker},
            reasoning=f"{reason_prefix}: detected filing keyword",
            llm_driven=False,
        )

    return _default_call(ticker, f"{reason_prefix}: no keyword match")


def _default_call(ticker: str, reason: str) -> ToolCall:
    return ToolCall(
        tool_name="create_analyst_brief",
        arguments={"ticker": ticker},
        reasoning=reason,
        llm_driven=False,
    )