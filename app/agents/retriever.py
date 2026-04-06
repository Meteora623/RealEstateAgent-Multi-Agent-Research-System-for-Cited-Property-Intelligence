from __future__ import annotations

from app.retrieval.service import HybridRetriever
from app.schemas.retrieval import MetadataFilter, RetrievalChunk, RetrieverDebugInfo


class RetrieverAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def run(
        self,
        query: str,
        filters: MetadataFilter | None,
        top_k: int,
        rerank: bool,
        retrieval_mode: str,
        candidate_pool_size: int | None = None,
        fusion_alpha: float | None = None,
        rerank_top_n: int | None = None,
    ) -> tuple[list[RetrievalChunk], RetrieverDebugInfo]:
        return self.retriever.retrieve(
            query=query,
            metadata_filter=filters,
            top_k=top_k,
            rerank=rerank,
            retrieval_mode=retrieval_mode,
            candidate_pool_size=candidate_pool_size,
            fusion_alpha=fusion_alpha,
            rerank_top_n=rerank_top_n,
        )
