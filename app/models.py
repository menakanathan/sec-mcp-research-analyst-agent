from typing import Any, Optional
from pydantic import BaseModel, Field


class CompanyIdentity(BaseModel):
    ticker: str
    cik: str
    title: str


class FilingSummary(BaseModel):
    accession_number: str
    form: str
    filing_date: str
    report_date: Optional[str] = None
    primary_document: str
    filing_url: str


class FinancialMetric(BaseModel):
    name: str
    value: Any
    unit: Optional[str] = None
    fiscal_year: Optional[int] = None
    fiscal_period: Optional[str] = None
    filed: Optional[str] = None


class ExtractedSection(BaseModel):
    item: str
    title: str
    text: str
    char_count: int


class AskRequest(BaseModel):
    question: str = Field(..., examples=["Create an analyst brief for Apple using latest 10-K"])
    use_llm: bool = True


class AskResponse(BaseModel):
    question: str
    mode: str
    tools_used: list[str]
    answer: str
    evidence: list[dict]
