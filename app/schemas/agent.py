from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.retrieval import Citation


class IntentType(str, Enum):
    lookup = "lookup"
    comparison = "comparison"
    risk_analysis = "risk_analysis"
    investment_thesis = "investment_thesis"
    valuation_reasoning = "valuation_reasoning"


class CalculationResult(BaseModel):
    name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)


class ClaimEvidence(BaseModel):
    claim: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    support_score: float = 0.0
    status: str = "unknown"


class AnalystDraft(BaseModel):
    answer_summary: str
    supporting_reasoning: list[str] = Field(default_factory=list)
    numeric_calculations: list[CalculationResult] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    major_claims: list[str] = Field(default_factory=list)
    cited_chunk_ids: list[str] = Field(default_factory=list)
    claim_evidence: list[ClaimEvidence] = Field(default_factory=list)


class FactCheckVerdict(str, Enum):
    approve = "approve"
    revise = "revise"
    retrieve_more = "retrieve_more"


class ClaimCheck(BaseModel):
    claim: str
    verdict: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    notes: str | None = None


class FactCheckReport(BaseModel):
    verdict: FactCheckVerdict
    checks: list[ClaimCheck] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    weak_claims: list[str] = Field(default_factory=list)
    citation_consistent: bool = True
    feedback: str | None = None


class FinalAnswer(BaseModel):
    concise_answer: str
    reasoning: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence_level: float = 0.0
    unsupported_or_uncertain: list[str] = Field(default_factory=list)
