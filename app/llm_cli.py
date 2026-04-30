"""
app/llm_cli.py
──────────────
CLI interface for the agentic SEC research analyst.

Updated to use MCPAgent — all tool calls go through MCP protocol.
The output panel shows clearly which routing path was used (LLM vs fallback).
"""

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.mcp_agent import MCPAgent

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SEC MCP Research Analyst — agentic mode (Ollama + MCP)"
    )
    parser.add_argument(
        "question",
        help='Research question in quotes, e.g. "Show AAPL financial facts"',
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip Ollama — use keyword fallback for tool selection and deterministic summary",
    )
    args = parser.parse_args()

    agent = MCPAgent()

    console.print(
        Panel(
            "[bold]SEC MCP Research Analyst[/bold]\n"
            "[dim]Discovering tools from MCP server → LLM selects tool "
            "→ invoking via MCP → synthesising response[/dim]",
            border_style="blue",
        )
    )

    with console.status("Running agentic pipeline…", spinner="dots"):
        response = agent.answer(args.question, use_llm=not args.no_llm)

    # ── Status panel ──────────────────────────────────────────────────────
    mode       = response.mode
    tools_used = response.tools_used

    status_color = "green" if "mcp_agentic_ollama_llm" in mode else "yellow"
    routing_note = (
        "LLM selected tool from MCP schema"
        if "llm" in mode
        else "Keyword fallback (Ollama unavailable or --no-llm)"
    )

    console.print(
        Panel(
            f"[bold]Mode:[/bold]          {mode}\n"
            f"[bold]Routing:[/bold]       {routing_note}\n"
            f"[bold]Tools via MCP:[/bold] {', '.join(tools_used) or 'none'}",
            title="Pipeline Status",
            border_style=status_color,
        )
    )

    # ── Evidence table ────────────────────────────────────────────────────
    if response.evidence:
        tbl = Table(title="MCP Tool Evidence", show_lines=True)
        tbl.add_column("Tool", style="cyan", no_wrap=True)
        tbl.add_column("Input args", style="dim")
        tbl.add_column("LLM-driven", justify="center")
        tbl.add_column("Reasoning", style="dim")

        for e in response.evidence:
            tbl.add_row(
                e.get("tool", ""),
                str(e.get("input", {})),
                "✓" if e.get("llm_driven") else "✗",
                e.get("reasoning", ""),
            )
        console.print(tbl)

    # ── Answer ────────────────────────────────────────────────────────────
    console.rule("[bold]Analyst Response[/bold]")
    console.print(response.answer)


if __name__ == "__main__":
    main()
