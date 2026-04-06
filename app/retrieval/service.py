from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.config import Settings
from app.retrieval.bm25_store import BM25Index
from app.retrieval.dense_store import DenseIndex, build_dense_index
from app.retrieval.embeddings import EmbeddingProvider, build_embedding_provider
from app.retrieval.fusion import weighted_fusion
from app.retrieval.rerankers import Reranker, build_reranker
from app.schemas.retrieval import Citation, MetadataFilter, RetrievalChunk, RetrieverDebugInfo


@dataclass
class RetrieverComponents:
    embedding_provider: EmbeddingProvider
    dense_index: DenseIndex
    bm25_index: BM25Index
    reranker: Reranker


def build_retriever_components(settings: Settings) -> RetrieverComponents:
    return RetrieverComponents(
        embedding_provider=build_embedding_provider(settings),
        dense_index=build_dense_index(settings),
        bm25_index=BM25Index(settings.indices_path),
        reranker=build_reranker(settings),
    )


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        components: RetrieverComponents | None = None,
    ):
        self.settings = settings
        self.components = components or build_retriever_components(settings)

    def retrieve(
        self,
        query: str,
        metadata_filter: MetadataFilter | None = None,
        top_k: int | None = None,
        rerank: bool = True,
        retrieval_mode: str | None = None,
        candidate_pool_size: int | None = None,
        fusion_alpha: float | None = None,
        rerank_top_n: int | None = None,
    ) -> tuple[list[RetrievalChunk], RetrieverDebugInfo]:
        retrieval_mode = retrieval_mode or self.settings.default_retrieval_mode
        top_k = top_k or self.settings.default_top_k
        candidate_pool_size = candidate_pool_size or max(
            top_k * self.settings.default_candidate_pool_multiplier, top_k
        )
        fusion_alpha = self.settings.fusion_alpha if fusion_alpha is None else fusion_alpha
        rerank_top_n = rerank_top_n or self.settings.rerank_top_n
        filt = (metadata_filter or MetadataFilter()).as_match_dict()

        timings: dict[str, float] = {}
        dense_results: list[dict] = []
        sparse_results: list[dict] = []

        if retrieval_mode in {"dense", "hybrid"}:
            t0 = time.perf_counter()
            qvec = self.components.embedding_provider.embed_query(query)
            dense_results = self.components.dense_index.query(
                qvec, top_k=candidate_pool_size, metadata_filter=filt
            )
            dense_results = self._apply_advanced_filters(dense_results, metadata_filter)
            timings["dense_ms"] = (time.perf_counter() - t0) * 1000.0

        if retrieval_mode in {"sparse", "hybrid"}:
            t1 = time.perf_counter()
            sparse_results = self.components.bm25_index.query(
                query, top_k=candidate_pool_size, metadata_filter=filt
            )
            sparse_results = self._apply_advanced_filters(sparse_results, metadata_filter)
            timings["sparse_ms"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        if retrieval_mode == "dense":
            fused = weighted_fusion(dense_results, [], alpha=1.0, top_k=candidate_pool_size)
        elif retrieval_mode == "sparse":
            fused = weighted_fusion([], sparse_results, alpha=0.0, top_k=candidate_pool_size)
        else:
            fused = weighted_fusion(
                dense_results,
                sparse_results,
                alpha=fusion_alpha,
                top_k=candidate_pool_size,
            )
        timings["fusion_ms"] = (time.perf_counter() - t2) * 1000.0

        fused_before_rerank = [chunk.model_copy(deep=True) for chunk in fused]
        rerank_applied = bool(rerank and fused)
        if rerank_applied:
            t3 = time.perf_counter()
            reranked = self.components.reranker.rerank(query, fused[:rerank_top_n])
            fused = reranked[:top_k]
            timings["rerank_ms"] = (time.perf_counter() - t3) * 1000.0
        else:
            fused = fused[:top_k]

        debug = RetrieverDebugInfo(
            retrieval_mode=retrieval_mode,
            dense_hits=len(dense_results),
            sparse_hits=len(sparse_results),
            fused_hits=len(fused),
            rerank_applied=rerank_applied,
            reranker_provider=self.components.reranker.provider_name if rerank_applied else "none",
            filter_applied=filt,
            timings_ms=timings,
            dense_rankings=self._rows_to_ranking(dense_results, "score"),
            sparse_rankings=self._rows_to_ranking(sparse_results, "score"),
            fused_rankings=self._chunks_to_ranking(fused_before_rerank),
            reranked_rankings=self._chunks_to_ranking(fused) if rerank_applied else [],
        )
        return fused, debug

    @staticmethod
    def _apply_advanced_filters(
        results: list[dict],
        metadata_filter: MetadataFilter | None,
    ) -> list[dict]:
        if metadata_filter is None:
            return results

        filtered: list[dict] = []
        for row in results:
            meta = row.get("metadata", {})
            price = float(meta.get("asking_price", 0.0) or 0.0)
            year = int(meta.get("year_built", 0) or 0)
            if metadata_filter.min_price is not None and price < metadata_filter.min_price:
                continue
            if metadata_filter.max_price is not None and price > metadata_filter.max_price:
                continue
            if metadata_filter.min_year is not None and year < metadata_filter.min_year:
                continue
            if metadata_filter.max_year is not None and year > metadata_filter.max_year:
                continue
            filtered.append(row)
        return filtered

    @staticmethod
    def _rows_to_ranking(rows: list[dict], score_key: str) -> list[dict]:
        ranking: list[dict] = []
        for idx, row in enumerate(rows, start=1):
            ranking.append(
                {
                    "rank": idx,
                    "chunk_id": row.get("chunk_id"),
                    "doc_id": row.get("doc_id"),
                    "property_id": row.get("property_id"),
                    "score": float(row.get(score_key, 0.0)),
                }
            )
        return ranking

    @staticmethod
    def _chunks_to_ranking(chunks: list[RetrievalChunk]) -> list[dict]:
        ranking: list[dict] = []
        for idx, chunk in enumerate(chunks, start=1):
            ranking.append(
                {
                    "rank": idx,
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "property_id": chunk.property_id,
                    "fused_score": round(chunk.fused_score, 6),
                    "dense_score": round(chunk.dense_score, 6),
                    "sparse_score": round(chunk.sparse_score, 6),
                    "rerank_score": round(chunk.rerank_score, 6)
                    if chunk.rerank_score is not None
                    else None,
                }
            )
        return ranking

    @staticmethod
    def map_citations(chunks: list[RetrievalChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for idx, chunk in enumerate(chunks, start=1):
            citations.append(
                Citation(
                    citation_id=f"S{idx}",
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    property_id=chunk.property_id,
                    title=chunk.metadata.get("title"),
                    source=chunk.metadata.get("source"),
                    snippet=chunk.text[:260],
                )
            )
        return citations
