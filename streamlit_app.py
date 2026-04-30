"""
streamlit_app.py
────────────────
Streamlit UI for the SEC MCP Research Analyst Agent.

Updated to use MCPAgent (Gap 1 + Gap 2 fixed):
  - All tool calls go through the MCP protocol (mcp.ClientSession / stdio)
  - The LLM selects which tool to invoke from the live tool schema list
  - Sidebar shows MCP server health + live tool discovery
  - Mode badge makes the routing path (LLM-agentic vs fallback) visible
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from app.mcp_agent import MCPAgent          # ← replaces SecLLMResearchAgent
from app.tool_registry import extract_tickers
from app.config import OLLAMA_MODEL, OLLAMA_BASE_URL

st.set_page_config(
    page_title="SEC MCP Research Analyst",
    page_icon="",
    layout="wide",
)


# ── Cached agent ──────────────────────────────────────────────────────────

@st.cache_resource
def get_agent() -> MCPAgent:
    return MCPAgent()


# ── Rendering helpers ─────────────────────────────────────────────────────

def render_metric_table(metrics: list) -> None:
    if not metrics:
        st.info("No financial metrics returned.")
        return
    df   = pd.DataFrame(metrics)
    cols = [c for c in ["name", "value", "unit", "fiscal_year",
                          "fiscal_period", "filed"] if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def render_mode_badge(mode: str) -> None:
    """Colour-coded badge showing which execution path was used."""
    if "mcp_agentic_ollama_llm" in mode:
        st.success(
            "  Full agentic mode — LLM selected tool via MCP protocol"
        )
    elif "mcp_agentic_ollama_keyword" in mode:
        st.warning(
            "  MCP transport active — tool was keyword-selected "
            "(Ollama tool-selector fell back)"
        )
    elif "mcp_agentic_fallback" in mode or "mcp_agentic_deterministic" in mode:
        st.warning(
            "  MCP transport active — LLM synthesis unavailable, "
            "deterministic summary shown"
        )
    elif "error_mcp" in mode:
        st.error(
            "  MCP server unreachable — check terminal and restart the server"
        )
    else:
        st.info(f"Mode: {mode}")


def render_evidence(evidence: list) -> None:
    if not evidence:
        st.info("No evidence available.")
        return

    for idx, item in enumerate(evidence, start=1):
        tool       = item.get("tool", "unknown_tool")
        tool_input = item.get("input", {})
        output     = item.get("output", {})
        reasoning  = item.get("reasoning", "")
        llm_driven = item.get("llm_driven", False)

        header = f"Evidence {idx}: `{tool}`"
        if llm_driven:
            header += "  *(LLM-selected)*"
        else:
            header += "  *(keyword fallback)*"

        with st.expander(header, expanded=False):
            if reasoning:
                st.caption(f"**Selection reasoning:** {reasoning}")

            st.markdown("**Tool input (sent over MCP protocol)**")
            st.json(tool_input)

            if tool == "get_financial_snapshot" and isinstance(output, list):
                st.markdown("**Financial snapshot**")
                render_metric_table(output)

            elif tool == "extract_filing_section" and isinstance(output, dict):
                st.markdown(
                    f"**Extracted:** {output.get('item')} — "
                    f"{output.get('title')}"
                )
                st.caption(f"Character count: {output.get('char_count')}")
                st.text_area(
                    "Section excerpt",
                    output.get("text", "")[:5000],
                    height=260,
                    key=f"section_{idx}",
                )

            elif tool == "create_analyst_brief" and isinstance(output, dict):
                company = output.get("company", {})
                filing  = output.get("filing_used", {})
                st.markdown(
                    f"**Company:** {company.get('title')} ({company.get('ticker')})"
                )
                st.markdown(
                    f"**Filing:** {filing.get('form')} filed "
                    f"{filing.get('filing_date')}"
                )
                st.markdown("**Financial snapshot**")
                render_metric_table(output.get("financial_snapshot", []))
                st.markdown("**Risk excerpt (Item 1A)**")
                st.text_area(
                    "Risk",
                    output.get("risk_excerpt", "")[:5000],
                    height=220,
                    key=f"risk_{idx}",
                )
                st.markdown("**MD&A excerpt (Item 7)**")
                st.text_area(
                    "MD&A",
                    output.get("mdna_excerpt", "")[:5000],
                    height=220,
                    key=f"mdna_{idx}",
                )

            elif tool == "compare_companies" and isinstance(output, dict):
                col1, col2 = st.columns(2)
                for col, label in [(col1, "company_a"), (col2, "company_b")]:
                    with col:
                        data    = output.get(label, {})
                        company = data.get("company", {})
                        filing  = data.get("filing_used", {})
                        st.markdown(
                            f"### {company.get('title')} "
                            f"({company.get('ticker')})"
                        )
                        st.caption(
                            f"{filing.get('form')} filed "
                            f"{filing.get('filing_date')}"
                        )
                        render_metric_table(data.get("financial_snapshot", []))
                        st.text_area(
                            "Risk excerpt",
                            data.get("risk_excerpt", "")[:3000],
                            height=180,
                            key=f"{label}_risk_{idx}",
                        )

            elif tool == "list_recent_filings" and isinstance(output, list):
                for f in output:
                    st.markdown(
                        f"- **{f.get('form')}** filed {f.get('filing_date')} "
                        f"| [{f.get('accession_number')}]({f.get('filing_url', '#')})"
                    )

            else:
                st.json(output)


def suggested_questions() -> list[str]:
    return [
        "Create an analyst brief for Apple using latest 10-K",
        "Summarize Tesla Item 1A risk factors from latest 10-K",
        "Show NVIDIA latest financial facts",
        "List recent Microsoft 10-K filings",
        "Compare MSFT and NVDA using latest 10-K risks and financial facts",
        "What are the main risks disclosed by Amazon in its latest 10-K?",
    ]


# ── App layout ────────────────────────────────────────────────────────────

agent = get_agent()

st.title(" SEC MCP Research Analyst Agent")
st.caption(
    "Local LLM (Ollama) · MCP protocol · SEC EDGAR · "
    "evidence based analyst responses"
)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    st.write("**LLM Provider:** Ollama (local)")
    st.write(f"**Model:** `{OLLAMA_MODEL}`")
    st.write(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")

    if st.button("Check Ollama health"):
        if agent.llm.is_available():
            st.success("Ollama is reachable ✓")
        else:
            st.error("Ollama not reachable — run: `ollama serve`")

    st.divider()

    # MCP server health — live tool discovery over protocol
    st.header("MCP Server")
    if st.button("Discover MCP tools"):
        with st.spinner("Connecting to MCP server…"):
            try:
                from app.mcp_client import SyncSecMCPClient
                tools_list = SyncSecMCPClient().list_tools()
                st.success(
                    f"MCP server reachable — "
                    f"{len(tools_list)} tools discovered ✓"
                )
                with st.expander("Tool schemas (from server)"):
                    for t in tools_list:
                        props = (t.get("inputSchema") or {}).get(
                            "properties", {}
                        )
                        params = ", ".join(props.keys()) if props else "none"
                        st.markdown(
                            f"**`{t['name']}`** — {t['description']}  \n"
                            f"params: `{params}`"
                        )
            except Exception as exc:
                st.error(f"MCP server unreachable: {exc}")
                st.info(
                    "Start the server in a separate terminal:\n"
                    "```\npython -m app.mcp_server\n```"
                )

    st.divider()
    st.header("Run Settings")
    use_llm      = st.toggle("Use Ollama LLM", value=True)
    show_evidence = st.toggle("Show evidence panel", value=True)
    show_raw_json = st.toggle("Show raw JSON", value=False)

    st.divider()
    st.header("Demo Questions")
    selected = st.selectbox("Choose a sample", suggested_questions())
    if st.button("Use selected question"):
        st.session_state["question"] = selected

# ── Main area ─────────────────────────────────────────────────────────────

if "question" not in st.session_state:
    st.session_state["question"] = suggested_questions()[0]

question = st.text_area(
    "Research question",
    value=st.session_state["question"],
    height=90,
    placeholder=(
        "Example: Compare MSFT and NVDA using latest 10-K "
        "risks and financial facts"
    ),
)

col_a, col_b, _ = st.columns([1, 1, 3])
with col_a:
    run = st.button("Run Analyst", type="primary")
with col_b:
    if st.button("Clear Results"):
        st.session_state.pop("last_response", None)

# Ticker detection hint
detected = extract_tickers(question)
if not detected:
    st.warning("No ticker detected — agent will default to AAPL in fallback mode.")

# ── Query execution ───────────────────────────────────────────────────────

if run:
    with st.spinner(
        "Discovering MCP tools → LLM selects tool → "
        "invoking via MCP → synthesising response…"
    ):
        response = agent.answer(question, use_llm=use_llm)
        st.session_state["last_response"] = response.model_dump()
        st.session_state["last_run_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

# ── Response display ──────────────────────────────────────────────────────

if "last_response" in st.session_state:
    resp = st.session_state["last_response"]

    st.divider()
    
    #st.subheader("Analyst Response")

    #m1, m2, m3 = st.columns(3)
    #with m1:
    #    st.metric("Mode", resp.get("mode", "unknown"))
    #with m2:
    #    st.metric("Tools via MCP", len(resp.get("tools_used", [])))
    #with m3:
    #    st.metric("Run time", st.session_state.get("last_run_time", ""))
    # Colour-coded mode badge
    #render_mode_badge(resp.get("mode", ""))

    tools_used = resp.get("tools_used", [])
    if tools_used:
        st.markdown(
            "**Tools invoked over MCP:** "
            + ", ".join(f"`{t}`" for t in tools_used)
        )

    st.markdown(resp.get("answer", ""))

    if show_evidence:
        st.divider()
        st.subheader("Evidence Pack  *(tool inputs + outputs via MCP)*")
        render_evidence(resp.get("evidence", []))

    if show_raw_json:
        st.divider()
        st.subheader("Raw response JSON")
        st.json(resp)

st.divider()
st.caption(
    "Prototype only. Outputs are automated research drafts "
    "and not financial advice. Validate findings against original SEC filings."
)
