from __future__ import annotations

from typing import Any

from app.schemas.retrieval import RetrievalChunk


def _normalize(items: list[dict[str, Any]], score_key: str) -> dict[str, float]:
    if not items:
        return {}
    scores = [float(item.get(score_key, 0.0)) for item in items]
    min_s = min(scores)
    max_s = max(scores)
    scale = max(max_s - min_s, 1e-9)
    normalized: dict[str, float] = {}
    for item in items:
        cid = item["chunk_id"]
        score = float(item.get(score_key, 0.0))
        normalized[cid] = (score - min_s) / scale
    return normalized


def weighted_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    alpha: float,
    top_k: int,
) -> list[RetrievalChunk]:
    dense_norm = _normalize(dense_results, "score")
    sparse_norm = _normalize(sparse_results, "score")

    merged: dict[str, RetrievalChunk] = {}

    for item in dense_results:
        cid = item["chunk_id"]
        dense_score = dense_norm.get(cid, 0.0)
        merged[cid] = RetrievalChunk(
            chunk_id=cid,
            doc_id=item["doc_id"],
            property_id=item["property_id"],
            text=item["text"],
            metadata=item.get("metadata", {}),
            dense_score=dense_score,
            sparse_score=0.0,
            fused_score=alpha * dense_score,
        )

    for item in sparse_results:
        cid = item["chunk_id"]
        sparse_score = sparse_norm.get(cid, 0.0)
        if cid not in merged:
            merged[cid] = RetrievalChunk(
                chunk_id=cid,
                doc_id=item["doc_id"],
                property_id=item["property_id"],
                text=item["text"],
                metadata=item.get("metadata", {}),
                dense_score=0.0,
                sparse_score=sparse_score,
                fused_score=(1 - alpha) * sparse_score,
            )
        else:
            entry = merged[cid]
            entry.sparse_score = sparse_score
            entry.fused_score = alpha * entry.dense_score + (1 - alpha) * sparse_score

    ranked = sorted(merged.values(), key=lambda c: c.fused_score, reverse=True)
    return ranked[:top_k]

