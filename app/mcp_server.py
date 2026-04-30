"""
app/mcp_server.py
─────────────────
FastMCP server — exposes SEC research tools over the MCP protocol.

Run standalone (for MCP client connections):
    python -m app.mcp_server

The single change from the original: mcp.run(transport="stdio") is now
explicit.  This is required when the server is spawned as a subprocess
by SecMCPClient / Claude Desktop — the stdio transport must be declared
so the process reads MCP messages from stdin and writes to stdout rather
than trying to bind a network port.
"""

from mcp.server.fastmcp import FastMCP

from app.tool_registry import SecToolRegistry
from app.llm_agent import SecLLMResearchAgent

mcp = FastMCP("sec-mcp-local-llm-research-analyst")

tools     = SecToolRegistry()
llm_agent = SecLLMResearchAgent()


@mcp.tool()
def ask_sec_research_agent(question: str, use_llm: bool = True) -> dict:
    """Ask the local-LLM-connected SEC research analyst a natural-language question."""
    return llm_agent.answer(question, use_llm=use_llm).model_dump()


@mcp.tool()
def local_llm_health() -> dict:
    """Check whether local Ollama LLM is reachable."""
    return {"provider": "ollama", "available": llm_agent.local_llm.is_available()}


@mcp.tool()
def resolve_company(ticker: str) -> dict:
    """Resolve stock ticker to SEC company identity and CIK."""
    return tools.resolve_company(ticker)


@mcp.tool()
def list_recent_filings(ticker: str, form: str = "10-K", limit: int = 5) -> list[dict]:
    """List recent SEC filings for a ticker."""
    return tools.list_recent_filings(ticker, form=form, limit=limit)


@mcp.tool()
def get_financial_snapshot(ticker: str) -> list[dict]:
    """Get selected latest financial facts from SEC XBRL company facts."""
    return tools.get_financial_snapshot(ticker)


@mcp.tool()
def extract_filing_section(ticker: str, item: str = "1A", form: str = "10-K") -> dict:
    """Extract a section from an SEC filing. The item argument must be a bare number: "1" (Business), "1A" (Risk Factors), "7" (MD&A), or "8" (Financial Statements). Do not include the word Item — use "1A" not "Item 1A"."""
    return tools.extract_filing_section(ticker, item=item, form=form)


@mcp.tool()
def create_analyst_brief(ticker: str, form: str = "10-K") -> dict:
    """Create structured analyst brief from latest filing and company facts."""
    return tools.create_analyst_brief(ticker, form=form)


@mcp.tool()
def compare_companies(ticker_a: str, ticker_b: str, form: str = "10-K") -> dict:
    """Compare two companies using SEC filings and structured facts."""
    return tools.compare_companies(ticker_a, ticker_b, form=form)


if __name__ == "__main__":
    # transport="stdio" is required for subprocess clients (SecMCPClient,
    # Claude Desktop).  Without it FastMCP tries to start an HTTP server.
    mcp.run(transport="stdio")