"""
app/mcp_agent.py
────────────────
MCPAgent — the fully agentic, MCP-compliant agent.

What makes this different from the original SecLLMResearchAgent:

  Gap 1 (transport) — FIXED
    Tools are discovered and invoked via mcp.ClientSession over stdio,
    not by importing Python modules and calling them directly.

  Gap 2 (routing) — FIXED
    The LLM reads the live tool schemas and selects which tool to invoke.
    No if/elif keyword chain in the hot path; keyword routing is an
    explicit fallback used only when Ollama is unreachable.

Pipeline for each query:
  1. list_tools()    — MCP protocol → server returns tool schemas
  2. select_tool()   — Ollama reads schemas, picks tool + args
  3. call_tool()     — MCP protocol → server executes tool, returns data
  4. synthesize()    — Ollama writes analyst response from tool output
"""

import json
import logging
from typing import Any

from app.config import LLM_PROVIDER
from app.local_llm import OllamaClient, LocalLLMError
from app.mcp_client import SyncSecMCPClient, MCPClientError
from app.models import AskResponse
from app.tool_selector import select_tool, ToolCall

logger = logging.getLogger(__name__)


# ── Synthesis prompt ──────────────────────────────────────────────────────

_SYNTHESIS_SYSTEM = """\
You are a careful SEC filings research analyst assistant.
Use ONLY the SEC tool evidence provided — do not invent facts.
Do not give investment advice.

Write a concise analyst-style response with these sections:
## Executive Answer
## Evidence Used
## Key Observations
## Risks and Limitations
## Next Research Steps
"""


# ── Agent ─────────────────────────────────────────────────────────────────

class MCPAgent:
    """
    Agentic SEC research analyst — all tool interactions go through MCP.
    """

    def __init__(self) -> None:
        self.mcp = SyncSecMCPClient()
        self.llm = OllamaClient()

    # ── Public entry point ────────────────────────────────────────────────

    def answer(self, question: str,
               use_llm: bool = True) -> AskResponse:
        """
        Full agentic pipeline.

        Parameters
        ----------
        question : str
            Natural language research question.
        use_llm : bool
            True  → LLM selects tools and synthesises response (agentic).
            False → keyword fallback selects tool; deterministic summary.
        """
        tools:      list[dict] = []
        evidence:   list[dict] = []
        tools_used: list[str]  = []

        # ── Step 1: Discover tools from MCP server ────────────────────────
        try:
            tools = self.mcp.list_tools()
            logger.info(
                f"[mcp_agent] MCP tool discovery OK — "
                f"{len(tools)} tools: "
                + ", ".join(t["name"] for t in tools)
            )
        except MCPClientError as exc:
            logger.error(f"[mcp_agent] MCP discovery failed: {exc}")
            return AskResponse(
                question=question,
                mode="error_mcp_discovery",
                tools_used=[],
                answer=_mcp_error_message(exc),
                evidence=[],
            )

        # ── Step 2: LLM selects which tool to call ────────────────────────
        llm_available = use_llm and self.llm.is_available()

        if llm_available:
            tool_call = select_tool(question, tools, self.llm)
        else:
            # No LLM — use the keyword fallback directly
            from app.tool_selector import _keyword_fallback
            known = {t["name"] for t in tools}
            tool_call = _keyword_fallback(
                question, known,
                "Ollama unavailable" if use_llm else "use_llm=False"
            )

        logger.info(
            f"[mcp_agent] Tool selected: {tool_call.tool_name} "
            f"args={tool_call.arguments} "
            f"llm_driven={tool_call.llm_driven} "
            f"| {tool_call.reasoning}"
        )

        # ── Step 3: Execute tool via MCP protocol ─────────────────────────
        try:
            output = self.mcp.call_tool(
                tool_call.tool_name,
                tool_call.arguments,
            )
            tools_used.append(tool_call.tool_name)
            evidence.append({
                "tool":       tool_call.tool_name,
                "input":      tool_call.arguments,
                "output":     output,
                "reasoning":  tool_call.reasoning,
                "llm_driven": tool_call.llm_driven,
            })
        except MCPClientError as exc:
            logger.error(
                f"[mcp_agent] call_tool({tool_call.tool_name}) failed: {exc}"
            )
            return AskResponse(
                question=question,
                mode="error_mcp_tool_call",
                tools_used=tools_used,
                answer=(
                    f"MCP tool `{tool_call.tool_name}` returned an error:\n\n"
                    f"```\n{exc}\n```\n\n"
                    "Check the MCP server logs for details."
                ),
                evidence=evidence,
            )

        # ── Step 4: LLM synthesises the result ────────────────────────────
        routing_label = "llm" if tool_call.llm_driven else "keyword_fallback"

        if llm_available:
            try:
                answer = self._llm_synthesize(question, evidence)
                mode   = f"mcp_agentic_ollama_{routing_label}"
            except LocalLLMError as exc:
                logger.warning(
                    f"[mcp_agent] LLM synthesis failed ({exc}); "
                    f"using deterministic summary"
                )
                answer = self._deterministic_summary(question, evidence)
                mode   = f"mcp_agentic_fallback_{routing_label}"
        else:
            answer = self._deterministic_summary(question, evidence)
            mode   = f"mcp_agentic_deterministic_{routing_label}"

        return AskResponse(
            question=question,
            mode=mode,
            tools_used=tools_used,
            answer=answer,
            evidence=evidence,
        )

    # ── LLM synthesis ─────────────────────────────────────────────────────

    def _llm_synthesize(self, question: str,
                         evidence: list[dict]) -> str:
        """Ask Ollama to write an analyst brief from the tool evidence."""
        compact = [
            {
                "tool":           e["tool"],
                "input":          e["input"],
                "reasoning":      e.get("reasoning", ""),
                "llm_driven":     e.get("llm_driven", False),
                "output_excerpt": json.dumps(
                    e["output"], indent=2, default=str
                )[:8000],
            }
            for e in evidence
        ]
        user_prompt = (
            f"Research question:\n{question}\n\n"
            f"MCP tool evidence (tool was invoked over MCP protocol):\n"
            + json.dumps(compact, indent=2)
        )
        return self.llm.chat(_SYNTHESIS_SYSTEM, user_prompt)

    # ── Deterministic summary (no LLM) ────────────────────────────────────

    def _deterministic_summary(self, question: str,
                                evidence: list[dict]) -> str:
        """
        Structured text summary produced without any LLM call.
        Used when Ollama is unavailable or use_llm=False.
        """
        lines = [
            "# Analyst Response  *(deterministic mode — Ollama not used)*",
            "",
            f"**Question:** {question}",
            "",
            "## Tools Invoked via MCP Protocol",
        ]

        for e in evidence:
            driven = "LLM-selected" if e.get("llm_driven") else "keyword-fallback"
            lines.append(
                f"- `{e['tool']}` ({driven})  args: `{e['input']}`"
                f"  — {e.get('reasoning', '')}"
            )

        lines += ["", "## Evidence Summary"]

        for e in evidence:
            tool   = e["tool"]
            output = e["output"]

            if tool == "create_analyst_brief" and isinstance(output, dict):
                co = output.get("company", {})
                fi = output.get("filing_used", {})
                lines.append(
                    f"**Company:** {co.get('title')} ({co.get('ticker')})"
                )
                lines.append(
                    f"**Filing:** {fi.get('form')} filed {fi.get('filing_date')}"
                )
                risk = output.get("risk_excerpt", "")
                if risk:
                    lines += ["", "**Risk excerpt:**", risk[:800] + "…"]
                mdna = output.get("mdna_excerpt", "")
                if mdna:
                    lines += ["", "**MD&A excerpt:**", mdna[:800] + "…"]
                for m in output.get("financial_snapshot", [])[:8]:
                    lines.append(
                        f"- {m.get('name')}: {m.get('value')} "
                        f"{m.get('unit')} FY{m.get('fiscal_year')}"
                    )

            elif tool == "extract_filing_section" and isinstance(output, dict):
                lines.append(
                    f"**Extracted:** Item {output.get('item')} "
                    f"— {output.get('title')}  "
                    f"({output.get('char_count')} chars)"
                )
                lines.append(output.get("text", "")[:1000] + "…")

            elif tool == "get_financial_snapshot" and isinstance(output, list):
                for m in output[:8]:
                    lines.append(
                        f"- {m.get('name')}: {m.get('value')} "
                        f"{m.get('unit')} FY{m.get('fiscal_year')}"
                    )

            elif tool == "list_recent_filings" and isinstance(output, list):
                for f in output:
                    lines.append(
                        f"- {f.get('form')} filed {f.get('filing_date')} "
                        f"| {f.get('accession_number')}"
                    )

            elif tool == "compare_companies" and isinstance(output, dict):
                for label in ("company_a", "company_b"):
                    co = output.get(label, {}).get("company", {})
                    lines.append(
                        f"**{label.replace('_', ' ').title()}:** "
                        f"{co.get('title')} ({co.get('ticker')})"
                    )
                    for m in output.get(label, {}).get(
                        "financial_snapshot", []
                    )[:4]:
                        lines.append(
                            f"  - {m.get('name')}: {m.get('value')} "
                            f"{m.get('unit')} FY{m.get('fiscal_year')}"
                        )

            elif tool == "resolve_company" and isinstance(output, dict):
                lines.append(
                    f"**Resolved:** {output.get('title')} "
                    f"| CIK: {output.get('cik')}"
                )

            else:
                lines.append(f"```\n{json.dumps(output, indent=2, default=str)[:600]}\n```")

        lines += [
            "",
            "## Limitations",
            "- Automated extraction may require validation against the source filing.",
            "- This is an academic research prototype — not financial advice.",
        ]
        return "\n".join(lines)


# ── Helpers ───────────────────────────────────────────────────────────────

def _mcp_error_message(exc: Exception) -> str:
    return (
        "## MCP Server Unreachable\n\n"
        f"Error: `{exc}`\n\n"
        "**To fix:** start the MCP server in a terminal:\n"
        "```bash\n"
        "python -m app.mcp_server\n"
        "```\n\n"
        "Then reload this page / re-run your query."
    )
