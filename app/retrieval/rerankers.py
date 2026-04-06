from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.core.config import Settings
from app.schemas.retrieval import RetrievalChunk


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


class Reranker(ABC):
    provider_name: str = "unknown"

    @abstractmethod
    def rerank(self, query: str, chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
        raise NotImplementedError


class OverlapReranker(Reranker):
    provider_name = "overlap"

    def rerank(self, query: str, chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
        q = _tokens(query)
        rescored: list[RetrievalChunk] = []
        for chunk in chunks:
            c = _tokens(chunk.text)
            denom = max(len(q), 1)
            overlap = len(q.intersection(c)) / denom
            chunk.rerank_score = overlap
            chunk.fused_score = 0.7 * chunk.fused_score + 0.3 * overlap
            rescored.append(chunk)
        return sorted(rescored, key=lambda c: c.fused_score, reverse=True)


class CohereReranker(Reranker):
    provider_name = "cohere"

    def __init__(self, api_key: str):
        import cohere

        self.client = cohere.ClientV2(api_key=api_key)

    def rerank(self, query: str, chunks: list[RetrievalChunk]) -> list[RetrievalChunk]:
        if not chunks:
            return []
        documents = [chunk.text for chunk in chunks]
        response = self.client.rerank(
            model="rerank-v3.5",
            query=query,
            documents=documents,
            top_n=len(documents),
        )
        for result in response.results:
            chunk = chunks[result.index]
            chunk.rerank_score = float(result.relevance_score)
            chunk.fused_score = 0.6 * chunk.fused_score + 0.4 * chunk.rerank_score
        return sorted(chunks, key=lambda c: c.fused_score, reverse=True)


def build_reranker(settings: Settings) -> Reranker:
    if settings.use_cohere_reranker and settings.cohere_api_key:
        return CohereReranker(settings.cohere_api_key)
    return OverlapReranker()
