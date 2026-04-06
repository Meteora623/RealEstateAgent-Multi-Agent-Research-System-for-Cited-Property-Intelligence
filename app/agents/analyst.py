from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.schemas.agent import AnalystDraft, CalculationResult, ClaimEvidence, IntentType
from app.schemas.retrieval import RetrievalChunk
from app.tools.comparison import compare_property_metrics
from app.tools.finance import calculate_cap_rate, calculate_noi


class AnalystAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.property_table = self._load_property_table(settings.data_path / "structured" / "properties.csv")
        self.llm = None
        if settings.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self.llm = ChatOpenAI(
                    model=settings.openai_model,
                    api_key=settings.openai_api_key,
                    temperature=0,
                )
            except Exception:  # noqa: BLE001
                self.llm = None

    @staticmethod
    def _load_property_table(path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    async def run(
        self,
        query: str,
        intent: IntentType,
        chunks: list[RetrievalChunk],
        async_tools: bool = True,
    ) -> AnalystDraft:
        if self.llm:
            try:
                return await self._run_with_llm(query, intent, chunks)
            except Exception:  # noqa: BLE001
                # Fallback to deterministic synthesis for local/offline runs.
                pass

        return await self._run_without_llm(query, intent, chunks, async_tools=async_tools)

    async def _run_with_llm(
        self, query: str, intent: IntentType, chunks: list[RetrievalChunk]
    ) -> AnalystDraft:
        context = "\n\n".join(
            f"[{idx+1}] {chunk.chunk_id} | {chunk.metadata.get('title', '')}\n{chunk.text[:600]}"
            for idx, chunk in enumerate(chunks[:8])
        )
        prompt = (
            "You are a real-estate analyst. Produce an evidence-grounded draft with concise claims. "
            f"Intent: {intent.value}\n"
            f"Query: {query}\n"
            f"Context:\n{context}\n"
            "Return JSON with fields: answer_summary, supporting_reasoning[], numeric_calculations[], "
            "uncertainty_notes[], major_claims[], cited_chunk_ids[]."
        )
        response = await self.llm.ainvoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
        # Minimal robust parse: if structured output isn't configured, use deterministic parser.
        return AnalystDraft(
            answer_summary=text[:500],
            supporting_reasoning=[f"LLM synthesis over {min(len(chunks), 8)} retrieved passages."],
            numeric_calculations=[],
            uncertainty_notes=["LLM output parser fallback applied."],
            major_claims=[text[:180]],
            cited_chunk_ids=[chunk.chunk_id for chunk in chunks[:4]],
            claim_evidence=[
                ClaimEvidence(
                    claim=text[:180],
                    supporting_chunk_ids=[chunk.chunk_id for chunk in chunks[:2]],
                    support_score=0.6,
                    status="supported",
                )
            ],
        )

    async def _run_without_llm(
        self,
        query: str,
        intent: IntentType,
        chunks: list[RetrievalChunk],
        async_tools: bool,
    ) -> AnalystDraft:
        if not chunks:
            return AnalystDraft(
                answer_summary="Insufficient evidence retrieved to answer confidently.",
                supporting_reasoning=["No chunks were retrieved for the query."],
                numeric_calculations=[],
                uncertainty_notes=["Try broadening filters or rerunning retrieval."],
                major_claims=[],
                cited_chunk_ids=[],
            )

        top = chunks[:6]
        cited_chunk_ids = [chunk.chunk_id for chunk in top]
        reasoning = []
        claims = []
        uncertainty: list[str] = []
        claim_evidence: list[ClaimEvidence] = []

        for chunk in top[:4]:
            neighborhood = chunk.metadata.get("neighborhood", "unknown neighborhood")
            asking = chunk.metadata.get("asking_price")
            comp_high = chunk.metadata.get("comp_price_high")
            if asking and comp_high and float(asking) < float(comp_high):
                claim = f"{chunk.property_id} may have pricing upside versus recent comp ceilings in {neighborhood}."
                claims.append(claim)
                claim_evidence.append(
                    ClaimEvidence(
                        claim=claim,
                        supporting_chunk_ids=[chunk.chunk_id],
                        support_score=0.72,
                        status="supported",
                    )
                )
            reasoning.append(f"{chunk.metadata.get('title', chunk.doc_id)} highlights occupancy, NOI, and comp evidence.")

        calculations = await self._run_calculations(query, intent, top, async_tools)

        summary = self._compose_summary(intent, top, calculations)
        if intent == IntentType.comparison:
            comparison_note = self._comparison_note(query)
            if comparison_note:
                reasoning.append(comparison_note)
            else:
                uncertainty.append("Comparison intent detected but fewer than two property identifiers were found.")

        if not claims:
            fallback_claim = "Retrieved evidence suggests moderate upside with caveats around renovation execution."
            claims.append(fallback_claim)
            claim_evidence.append(
                ClaimEvidence(
                    claim=fallback_claim,
                    supporting_chunk_ids=[chunk.chunk_id for chunk in top[:2]],
                    support_score=0.45,
                    status="weak",
                )
            )
            uncertainty.append("Evidence quality is moderate; additional documents would improve confidence.")
        return AnalystDraft(
            answer_summary=summary,
            supporting_reasoning=reasoning[:6],
            numeric_calculations=calculations,
            uncertainty_notes=uncertainty,
            major_claims=claims[:6],
            cited_chunk_ids=cited_chunk_ids,
            claim_evidence=claim_evidence[:6],
        )

    async def _run_calculations(
        self,
        query: str,
        intent: IntentType,
        chunks: list[RetrievalChunk],
        async_tools: bool,
    ) -> list[CalculationResult]:
        pids = list(dict.fromkeys(chunk.property_id for chunk in chunks))
        if not pids:
            return []

        async def calc_for_property(pid: str) -> CalculationResult | None:
            # Simulates I/O-bound structured-data access so sync/async latency comparisons are meaningful.
            await asyncio.sleep(max(self.settings.tool_io_latency_ms, 0) / 1000.0)
            row = self._row_by_property_id(pid)
            if not row:
                return None
            noi = float(row.get("noi_estimate", 0.0))
            price = float(row.get("asking_price", 0.0))
            occupancy = float(row.get("occupancy_rate", 0.0))
            gross = noi + float(row.get("renovation_budget_estimate", 0.0)) * 0.05
            adjusted_noi = calculate_noi(
                gross_income=gross,
                vacancy_rate=max(0.0, 1 - occupancy),
                operating_expenses=max(gross - noi, 0),
            )
            cap = calculate_cap_rate(noi=max(adjusted_noi, noi), price=price)
            return CalculationResult(
                name="cap_rate_estimate",
                inputs={"property_id": pid, "noi": noi, "price": price, "occupancy_rate": occupancy},
                output={"cap_rate": cap.cap_rate},
            )

        if async_tools:
            outputs = await asyncio.gather(*(calc_for_property(pid) for pid in pids[:3]))
        else:
            outputs = []
            for pid in pids[:3]:
                outputs.append(await calc_for_property(pid))

        calculations = [o for o in outputs if o]

        if intent == IntentType.comparison and len(pids) >= 2:
            left = self._row_by_property_id(pids[0]) or {}
            right = self._row_by_property_id(pids[1]) or {}
            calculations.append(
                CalculationResult(
                    name="property_comparison",
                    inputs={"left_property_id": pids[0], "right_property_id": pids[1]},
                    output=compare_property_metrics(left, right),
                )
            )
        return calculations

    def _row_by_property_id(self, property_id: str) -> dict[str, Any] | None:
        if self.property_table.empty:
            return None
        rows = self.property_table[self.property_table["property_id"] == property_id]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    @staticmethod
    def _compose_summary(
        intent: IntentType, chunks: list[RetrievalChunk], calculations: list[CalculationResult]
    ) -> str:
        properties = list(dict.fromkeys(chunk.property_id for chunk in chunks[:5]))
        calc_note = ""
        if calculations:
            cap_rates = [
                c.output.get("cap_rate")
                for c in calculations
                if c.name == "cap_rate_estimate" and c.output.get("cap_rate") is not None
            ]
            if cap_rates:
                calc_note = f" Estimated cap-rate band from retrieved properties is {min(cap_rates):.2f}% to {max(cap_rates):.2f}%."
        return (
            f"Intent `{intent.value}`: evidence across {len(chunks)} passages for {len(properties)} properties "
            f"supports a grounded assessment.{calc_note}"
        )

    def _comparison_note(self, query: str) -> str | None:
        ids = re.findall(r"PROP-\d{4}", query.upper())
        if len(ids) >= 2:
            return f"Comparison target detected: {ids[0]} versus {ids[1]}."
        return None
