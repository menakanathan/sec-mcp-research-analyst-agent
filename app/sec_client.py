import re
import time
from functools import lru_cache
from typing import Any, Optional

import requests
from bs4 import BeautifulSoup

from app.config import (
    SEC_USER_AGENT,
    SEC_REQUEST_TIMEOUT,
    SEC_TICKER_URL,
    SEC_SUBMISSIONS_URL,
    SEC_COMPANY_FACTS_URL,
    SEC_ARCHIVES_BASE,
)
from app.models import CompanyIdentity, FilingSummary, FinancialMetric


class SecClient:
    def __init__(self, user_agent: str = SEC_USER_AGENT, pause_seconds: float = 0.15):
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
        self.data_headers = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}
        self.pause_seconds = pause_seconds

    @staticmethod
    def _pad_cik(cik: str | int) -> str:
        return str(cik).zfill(10)

    def _get_json(self, url: str, data_host: bool = False) -> Any:
        time.sleep(self.pause_seconds)
        headers = self.data_headers if data_host else self.headers
        resp = requests.get(url, headers=headers, timeout=SEC_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        time.sleep(self.pause_seconds)
        resp = requests.get(url, headers=self.headers, timeout=SEC_REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.text

    @lru_cache(maxsize=1)
    def ticker_map(self) -> dict[str, CompanyIdentity]:
        data = self._get_json(SEC_TICKER_URL)
        out = {}
        for _, row in data.items():
            ticker = row["ticker"].upper()
            out[ticker] = CompanyIdentity(
                ticker=ticker,
                cik=self._pad_cik(row["cik_str"]),
                title=row["title"],
            )
        return out

    # Common English words that Ollama sometimes sends as ticker arguments.
    _NOT_TICKERS = {
        "RISK", "ITEM", "FACT", "FORM", "DATA", "LIST", "SHOW", "FIND",
        "INFO", "HELP", "USE", "GET", "SET", "RUN", "THE", "AND", "FOR",
        "FROM", "WITH", "THIS", "THAT", "WHAT", "WHEN", "HOW", "WHY",
        "LATEST", "RECENT", "LAST", "NEXT", "BEST", "TOP", "KEY",
        "ANNUAL", "FISCAL", "REVENUE", "INCOME", "ASSETS",
    }

    def resolve_ticker(self, ticker: str) -> CompanyIdentity:
        ticker = ticker.upper().strip()
        if ticker in self._NOT_TICKERS:
            raise ValueError(
                f"\"{ticker}\" is not a stock ticker symbol. "
                f"Please provide a valid exchange-listed ticker such as "
                f"AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, or JPM."
            )
        company = self.ticker_map().get(ticker)
        if not company:
            raise ValueError(
                f"Ticker '{ticker}' not found in the SEC ticker map. "
                f"Ensure it is a valid US-listed company symbol."
            )
        return company

    def submissions(self, ticker: str) -> dict:
        company = self.resolve_ticker(ticker)
        return self._get_json(SEC_SUBMISSIONS_URL.format(cik=company.cik), data_host=True)

    def recent_filings(self, ticker: str, form: Optional[str] = None, limit: int = 10) -> list[FilingSummary]:
        company = self.resolve_ticker(ticker)
        recent = self.submissions(ticker).get("filings", {}).get("recent", {})
        filings = []

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, f in enumerate(forms):
            if form and f.upper() != form.upper():
                continue

            accession = accessions[i]
            accession_no_dashes = accession.replace("-", "")
            primary_doc = primary_docs[i]
            cik_no_leading = str(int(company.cik))
            filing_url = f"{SEC_ARCHIVES_BASE}/{cik_no_leading}/{accession_no_dashes}/{primary_doc}"

            filings.append(
                FilingSummary(
                    accession_number=accession,
                    form=f,
                    filing_date=filing_dates[i],
                    report_date=report_dates[i] if i < len(report_dates) else None,
                    primary_document=primary_doc,
                    filing_url=filing_url,
                )
            )

            if len(filings) >= limit:
                break

        return filings

    def latest_filing(self, ticker: str, form: str = "10-K") -> FilingSummary:
        filings = self.recent_filings(ticker, form=form, limit=1)
        if not filings:
            raise ValueError(f"No {form} filings found for {ticker}")
        return filings[0]

    def download_filing_text(self, ticker: str, accession_number: Optional[str] = None, form: str = "10-K") -> str:
        if accession_number:
            # Build the URL directly from the accession number — avoids a
            # redundant list scan that fails when the filing sits beyond the
            # first 100 results returned by recent_filings().
            filing = self._filing_from_accession(ticker, accession_number)
        else:
            filing = self.latest_filing(ticker, form=form)

        raw = self._get_text(filing.filing_url)
        return self.clean_html(raw)

    def _filing_from_accession(self, ticker: str,
                                accession_number: str) -> "FilingSummary":
        """
        Resolve a known accession number to a FilingSummary without searching.

        Strategy:
          1. Try the submissions JSON first (already cached from earlier calls).
          2. If not found there, derive the filing URL directly from the
             accession number and fetch the primary document name from the index.
        """
        company = self.resolve_ticker(ticker)
        recent  = self.submissions(ticker).get("filings", {}).get("recent", {})

        accessions   = recent.get("accessionNumber", [])
        forms        = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        primary_docs = recent.get("primaryDocument", [])

        # ── Check submissions cache first ─────────────────────────────────
        for i, acc in enumerate(accessions):
            if acc == accession_number:
                acc_nodash  = acc.replace("-", "")
                cik_no_lead = str(int(company.cik))
                primary_doc = primary_docs[i] if i < len(primary_docs) else ""
                filing_url  = (
                    f"{SEC_ARCHIVES_BASE}/{cik_no_lead}"
                    f"/{acc_nodash}/{primary_doc}"
                )
                return FilingSummary(
                    accession_number=acc,
                    form=forms[i] if i < len(forms) else "10-K",
                    filing_date=filing_dates[i] if i < len(filing_dates) else "",
                    report_date=report_dates[i] if i < len(report_dates) else None,
                    primary_document=primary_doc,
                    filing_url=filing_url,
                )

        # ── Fallback: derive URL directly from accession number ───────────
        acc_nodash  = accession_number.replace("-", "")
        cik_no_lead = str(int(company.cik))
        index_url   = (
            f"{SEC_ARCHIVES_BASE}/{cik_no_lead}"
            f"/{acc_nodash}/{accession_number}-index.htm"
        )
        try:
            index_html  = self._get_text(index_url)
            primary_doc = self._extract_primary_doc(index_html)
        except Exception:
            primary_doc = f"{accession_number}.htm"

        filing_url = (
            f"{SEC_ARCHIVES_BASE}/{cik_no_lead}"
            f"/{acc_nodash}/{primary_doc}"
        )
        return FilingSummary(
            accession_number=accession_number,
            form="10-K",
            filing_date="",
            report_date=None,
            primary_document=primary_doc,
            filing_url=filing_url,
        )

    @staticmethod
    def _extract_primary_doc(index_html: str) -> str:
        """Pull the primary document filename from the filing index page."""
        from bs4 import BeautifulSoup
        soup  = BeautifulSoup(index_html, "lxml")
        links = soup.find_all("a", href=True)
        for link in links:
            href = link["href"]
            name = href.split("/")[-1].lower()
            if name.endswith((".htm", ".html")) and "index" not in name:
                return href.split("/")[-1]
        return links[0]["href"].split("/")[-1] if links else ""

    @staticmethod
    def clean_html(raw: str) -> str:
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "ix:header"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
        return re.sub(r"\s+", " ", text).strip()

    def company_facts(self, ticker: str) -> dict:
        company = self.resolve_ticker(ticker)
        return self._get_json(SEC_COMPANY_FACTS_URL.format(cik=company.cik), data_host=True)

    def financial_snapshot(self, ticker: str) -> list[FinancialMetric]:
        facts = self.company_facts(ticker)
        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        tags = {
            "RevenueFromContractWithCustomerExcludingAssessedTax": "Revenue",
            "Revenues": "Revenue",
            "NetIncomeLoss": "Net Income",
            "OperatingIncomeLoss": "Operating Income",
            "Assets": "Assets",
            "Liabilities": "Liabilities",
            "StockholdersEquity": "Stockholders Equity",
            "CashAndCashEquivalentsAtCarryingValue": "Cash and Cash Equivalents",
        }

        out = []
        seen = set()

        for tag, label in tags.items():
            if label in seen or tag not in us_gaap:
                continue

            units = us_gaap[tag].get("units", {})
            unit = "USD" if "USD" in units else next(iter(units.keys()), None)
            if not unit:
                continue

            entries = units[unit]
            annual = [e for e in entries if e.get("form") in ("10-K", "10-K/A") and e.get("fy")]
            annual = annual or entries
            annual = sorted(annual, key=lambda e: (e.get("filed", ""), e.get("fy", 0)), reverse=True)
            latest = annual[0]

            out.append(FinancialMetric(
                name=label,
                value=latest.get("val"),
                unit=unit,
                fiscal_year=latest.get("fy"),
                fiscal_period=latest.get("fp"),
                filed=latest.get("filed"),
            ))
            seen.add(label)

        return out