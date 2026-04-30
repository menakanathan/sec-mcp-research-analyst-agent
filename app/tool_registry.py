import re
from app.sec_client import SecClient
from app.section_extractor import extract_section


def extract_tickers(question: str) -> list[str]:
    known = {
        "APPLE": "AAPL", "MICROSOFT": "MSFT", "NVIDIA": "NVDA", "TESLA": "TSLA",
        "JPMORGAN": "JPM", "JPMORGAN CHASE": "JPM", "AMAZON": "AMZN",
        "ALPHABET": "GOOGL", "GOOGLE": "GOOGL", "META": "META", "INTEL": "INTC",
    }
    upper = question.upper()
    tickers = []
    for name, ticker in known.items():
        if name in upper and ticker not in tickers:
            tickers.append(ticker)
    candidates = re.findall(r"\b[A-Z]{1,5}\b", upper)
    # Extended stop-word list — common English words and domain terms that
    # are NOT stock tickers and must be excluded from ticker candidates.
    stop = {
        "SEC", "MCP", "LLM", "RAG", "API", "USA", "US", "FY", "CEO", "CFO",
        "AI", "ML", "NLP", "ETF", "IPO", "PE", "EPS", "ROE", "ROA", "DCF",
        "RISK", "ITEM", "FACT", "FORM", "DATA", "LIST", "SHOW", "FIND",
        "INFO", "HELP", "USE", "GET", "SET", "RUN", "THE", "AND", "FOR",
        "FROM", "WITH", "THIS", "THAT", "WHAT", "WHEN", "HOW", "WHY",
        "LATEST", "RECENT", "LAST", "NEXT", "BEST", "TOP", "KEY",
        "ANNUAL", "FISCAL", "QUARTER", "YEAR", "MONTH", "DATE",
        "REVENUE", "INCOME", "ASSETS", "EQUITY", "CASH", "DEBT",
    }
    for c in candidates:
        if c not in stop and c not in tickers:
            tickers.append(c)
    return tickers[:4]


class SecToolRegistry:
    def __init__(self):
        self.sec = SecClient()

    def resolve_company(self, ticker: str) -> dict:
        return self.sec.resolve_ticker(ticker).model_dump()

    def list_recent_filings(self, ticker: str, form: str = "10-K", limit: int = 5) -> list[dict]:
        return [f.model_dump() for f in self.sec.recent_filings(ticker, form=form, limit=limit)]

    def get_financial_snapshot(self, ticker: str) -> list[dict]:
        return [m.model_dump() for m in self.sec.financial_snapshot(ticker)]

    def extract_filing_section(self, ticker: str, item: str = "1A", form: str = "10-K") -> dict:
        text = self.sec.download_filing_text(ticker, form=form)
        return extract_section(text, item=item).model_dump()

    def create_analyst_brief(self, ticker: str, form: str = "10-K") -> dict:
        company = self.sec.resolve_ticker(ticker)
        filing = self.sec.latest_filing(ticker, form=form)
        text = self.sec.download_filing_text(ticker, accession_number=filing.accession_number, form=form)
        risk = extract_section(text, item="1A", max_chars=10000)
        mdna = extract_section(text, item="7", max_chars=10000)
        facts = self.get_financial_snapshot(ticker)
        return {
            "company": company.model_dump(),
            "filing_used": filing.model_dump(),
            "risk_excerpt": risk.text[:2500],
            "mdna_excerpt": mdna.text[:2500],
            "financial_snapshot": facts,
            "limitations": ["Automated filing extraction may require validation.", "This is not financial advice."],
        }

    def compare_companies(self, ticker_a: str, ticker_b: str, form: str = "10-K") -> dict:
        return {"company_a": self.create_analyst_brief(ticker_a, form=form),
                "company_b": self.create_analyst_brief(ticker_b, form=form)}