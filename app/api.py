from pydantic import BaseModel
from fastapi import FastAPI, Query

from app.models import AskRequest
from app.sec_client import SecClient
from app.section_extractor import extract_section
from app.tool_registry import SecToolRegistry
from app.llm_agent import SecLLMResearchAgent

app = FastAPI(title="SEC MCP Local LLM Research Analyst API", version="3.0.0")

sec = SecClient()
tools = SecToolRegistry()
llm_agent = SecLLMResearchAgent()


class CompareRequest(BaseModel):
    ticker_a: str
    ticker_b: str
    form: str = "10-K"


@app.get("/health")
def health():
    return {"status": "ok", "service": "sec-mcp-local-llm-research-analyst"}


@app.get("/llm-health")
def llm_health():
    available = llm_agent.local_llm.is_available()
    return {
        "provider": "ollama",
        "available": available,
        "message": "Ollama is reachable" if available else "Ollama is not reachable. Start it with: ollama serve"
    }


@app.post("/ask")
def ask(req: AskRequest):
    return llm_agent.answer(req.question, use_llm=req.use_llm)


@app.get("/company/{ticker}")
def company(ticker: str):
    return sec.resolve_ticker(ticker)


@app.get("/filings/{ticker}")
def filings(ticker: str, form: str | None = Query(default=None), limit: int = 10):
    return sec.recent_filings(ticker, form=form, limit=limit)


@app.get("/facts/{ticker}")
def facts(ticker: str):
    return sec.financial_snapshot(ticker)


@app.get("/section/{ticker}")
def section(ticker: str, item: str = "1A", form: str = "10-K"):
    text = sec.download_filing_text(ticker, form=form)
    return extract_section(text, item=item)


@app.get("/brief/{ticker}")
def brief(ticker: str, form: str = "10-K"):
    return tools.create_analyst_brief(ticker, form=form)


@app.post("/compare")
def compare(req: CompareRequest):
    return tools.compare_companies(req.ticker_a, req.ticker_b, form=req.form)
