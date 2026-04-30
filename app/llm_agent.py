import json
from app.config import LLM_PROVIDER
from app.local_llm import OllamaClient
from app.models import AskResponse
from app.tool_registry import SecToolRegistry, extract_tickers


class SecLLMResearchAgent:
    def __init__(self):
        self.tools = SecToolRegistry()
        self.local_llm = OllamaClient()

    def answer(self, question: str, use_llm: bool = True) -> AskResponse:
        lower = question.lower()
        tickers = extract_tickers(question)
        evidence = []
        tools_used = []
        try:
            if "compare" in lower and len(tickers) >= 2:
                result = self.tools.compare_companies(tickers[0], tickers[1])
                evidence.append({"tool": "compare_companies", "input": {"ticker_a": tickers[0], "ticker_b": tickers[1]}, "output": result})
                tools_used.append("compare_companies")
            elif "risk" in lower or "item 1a" in lower:
                ticker = tickers[0] if tickers else "AAPL"
                result = self.tools.extract_filing_section(ticker, item="1A")
                facts = self.tools.get_financial_snapshot(ticker)
                evidence.append({"tool": "extract_filing_section", "input": {"ticker": ticker, "item": "1A"}, "output": result})
                evidence.append({"tool": "get_financial_snapshot", "input": {"ticker": ticker}, "output": facts})
                tools_used.extend(["extract_filing_section", "get_financial_snapshot"])
            elif "financial" in lower or "revenue" in lower or "income" in lower or "facts" in lower:
                ticker = tickers[0] if tickers else "AAPL"
                result = self.tools.get_financial_snapshot(ticker)
                evidence.append({"tool": "get_financial_snapshot", "input": {"ticker": ticker}, "output": result})
                tools_used.append("get_financial_snapshot")
            elif "filing" in lower or "10-k" in lower or "10-q" in lower:
                ticker = tickers[0] if tickers else "AAPL"
                result = self.tools.list_recent_filings(ticker)
                evidence.append({"tool": "list_recent_filings", "input": {"ticker": ticker}, "output": result})
                tools_used.append("list_recent_filings")
            else:
                ticker = tickers[0] if tickers else "AAPL"
                result = self.tools.create_analyst_brief(ticker)
                evidence.append({"tool": "create_analyst_brief", "input": {"ticker": ticker}, "output": result})
                tools_used.append("create_analyst_brief")
        except Exception as exc:
            return AskResponse(question=question, mode="error", tools_used=tools_used, answer=f"Error while calling SEC tools: {exc}", evidence=evidence)

        if use_llm and LLM_PROVIDER.lower() == "ollama":
            try:
                answer = self._local_llm_synthesize(question, evidence)
                return AskResponse(question=question, mode="local_llm_ollama", tools_used=tools_used, answer=answer, evidence=evidence)
            except Exception as exc:
                fallback = "Local LLM was requested, but Ollama failed or was unavailable. Reason: " + str(exc) + "\\n\\n" + self._fallback_synthesize(question, evidence)
                return AskResponse(question=question, mode="fallback_due_to_local_llm_error", tools_used=tools_used, answer=fallback, evidence=evidence)

        return AskResponse(question=question, mode="deterministic_fallback", tools_used=tools_used, answer=self._fallback_synthesize(question, evidence), evidence=evidence)

    def _local_llm_synthesize(self, question: str, evidence: list[dict]) -> str:
        compact_evidence = []
        for e in evidence:
            compact_evidence.append({
                "tool": e["tool"],
                "input": e["input"],
                "output_excerpt": json.dumps(e["output"], indent=2)[:10000],
            })

        system_prompt = (
            "You are a careful SEC filings research analyst assistant. "
            "Use only the provided SEC tool evidence. Do not provide investment advice. "
            "Write in concise analyst style with sections: Executive Answer, Evidence Used, Key Observations, Risks and Limitations, Next Research Steps."
        )
        user_prompt = "Research question:\\n" + question + "\\n\\nSEC tool evidence:\\n" + json.dumps(compact_evidence, indent=2)
        return self.local_llm.chat(system_prompt, user_prompt)

    def _fallback_synthesize(self, question: str, evidence: list[dict]) -> str:
        lines = ["# Analyst Response", "", f"Question: {question}", "", "## Tools Used"]
        for e in evidence:
            lines.append(f"- {e['tool']} with input {e['input']}")
        lines.append("")
        lines.append("## Evidence Summary")
        for e in evidence:
            tool = e["tool"]
            output = e["output"]
            if tool == "create_analyst_brief":
                company = output.get("company", {})
                filing = output.get("filing_used", {})
                lines.append(f"- Company: {company.get('title')} ({company.get('ticker')})")
                lines.append(f"- Filing: {filing.get('form')} filed {filing.get('filing_date')}")
                lines.append(f"- Risk excerpt: {output.get('risk_excerpt', '')[:800]}...")
                lines.append(f"- MD&A excerpt: {output.get('mdna_excerpt', '')[:800]}...")
            elif tool == "extract_filing_section":
                lines.append(f"- Extracted {output.get('item')} - {output.get('title')}, {output.get('char_count')} characters.")
                lines.append(f"- Excerpt: {output.get('text', '')[:1000]}...")
            elif tool == "get_financial_snapshot":
                for m in output[:8]:
                    lines.append(f"- {m.get('name')}: {m.get('value')} {m.get('unit')} FY{m.get('fiscal_year')}")
            elif tool == "list_recent_filings":
                for f in output:
                    lines.append(f"- {f.get('form')} filed {f.get('filing_date')} accession {f.get('accession_number')}")
            elif tool == "compare_companies":
                a = output.get("company_a", {}).get("company", {})
                b = output.get("company_b", {}).get("company", {})
                lines.append(f"- Compared {a.get('ticker')} and {b.get('ticker')} using latest filing evidence.")
        lines.append("")
        lines.append("## Limitations")
        lines.append("- Automated extraction can be imperfect due to filing format variation.")
        lines.append("- This is an academic research assistant, not financial advice.")
        return "\\n".join(lines)
