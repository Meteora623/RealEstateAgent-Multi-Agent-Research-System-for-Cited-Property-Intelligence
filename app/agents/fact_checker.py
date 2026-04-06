from __future__ import annotations

import re

from app.schemas.agent import AnalystDraft, ClaimCheck, FactCheckReport, FactCheckVerdict
from app.schemas.retrieval import RetrievalChunk


def _tokens(text: str) -> set[str]:
    raw = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "has",
        "have",
        "are",
        "was",
        "were",
        "in",
        "on",
        "of",
        "to",
        "a",
        "an",
        "is",
        "it",
    }
    return {tok for tok in raw if tok not in stop and len(tok) > 2}


class FactCheckerAgent:
    def run(
        self,
        draft: AnalystDraft,
        chunks: list[RetrievalChunk],
        retrieval_round: int,
        max_retrieval_rounds: int,
    ) -> FactCheckReport:
        chunk_map = {chunk.chunk_id: chunk for chunk in chunks}
        chunk_tokens = {cid: _tokens(chunk.text) for cid, chunk in chunk_map.items()}

        checks: list[ClaimCheck] = []
        unsupported: list[str] = []
        weak: list[str] = []
        contradiction_tokens = {"no", "not", "never", "without", "none"}

        for claim in draft.major_claims:
            claim_tokens = _tokens(claim)
            best_overlap = 0.0
            best_ids: list[str] = []
            contradiction_detected = False
            for cid, tokens in chunk_tokens.items():
                denom = max(len(claim_tokens), 1)
                overlap = len(claim_tokens.intersection(tokens)) / denom
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_ids = [cid]

                if claim_tokens.intersection(contradiction_tokens) and not tokens.intersection(
                    contradiction_tokens
                ):
                    if overlap >= 0.2:
                        contradiction_detected = True

            declared = next((item for item in draft.claim_evidence if item.claim == claim), None)
            declared_ids = declared.supporting_chunk_ids if declared else []
            missing_declared_ids = [cid for cid in declared_ids if cid not in chunk_map]
            if missing_declared_ids:
                contradiction_detected = True

            if contradiction_detected:
                verdict = "unsupported"
                unsupported.append(claim)
                notes = "Contradiction or missing supporting citation detected."
            elif best_overlap >= 0.42:
                verdict = "supported"
                notes = f"Claim support score {best_overlap:.2f}."
            elif best_overlap >= 0.22:
                verdict = "weak"
                weak.append(claim)
                notes = f"Weak support {best_overlap:.2f}; consider additional evidence."
            else:
                verdict = "unsupported"
                unsupported.append(claim)
                notes = f"No convincing support in retrieved chunks (max overlap {best_overlap:.2f})."
            checks.append(
                ClaimCheck(
                    claim=claim,
                    verdict=verdict,
                    supporting_chunk_ids=best_ids,
                    notes=notes,
                )
            )

        missing_citations = [
            cid for cid in draft.cited_chunk_ids if cid not in chunk_map
        ]
        citation_consistent = len(missing_citations) == 0
        if missing_citations:
            unsupported.extend(
                [f"Citation references missing chunk {cid}." for cid in missing_citations]
            )

        if unsupported and retrieval_round < max_retrieval_rounds:
            verdict = FactCheckVerdict.retrieve_more
            feedback = "Unsupported claims detected; trigger another retrieval round."
        elif unsupported:
            verdict = FactCheckVerdict.revise
            feedback = "Unsupported claims remain after max retrieval rounds."
        elif weak:
            verdict = FactCheckVerdict.revise
            feedback = "Claims are mostly supported but some are weak."
        else:
            verdict = FactCheckVerdict.approve
            feedback = "All major claims have adequate evidence coverage."

        return FactCheckReport(
            verdict=verdict,
            checks=checks,
            unsupported_claims=unsupported,
            weak_claims=weak,
            citation_consistent=citation_consistent,
            feedback=feedback,
        )
