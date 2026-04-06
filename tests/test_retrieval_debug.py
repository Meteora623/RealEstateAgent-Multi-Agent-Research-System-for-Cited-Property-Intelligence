from app.core.config import Settings
from app.retrieval.service import HybridRetriever, RetrieverComponents
from app.schemas.retrieval import RetrievalChunk


class _FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text):
        return [0.1, 0.2]


class _FakeDense:
    def upsert(self, chunks, embeddings):
        return None

    def query(self, query_embedding, top_k, metadata_filter=None):
        return [
            {"chunk_id": "c1", "doc_id": "d1", "property_id": "p1", "text": "alpha", "metadata": {}, "score": 0.9},
            {"chunk_id": "c2", "doc_id": "d2", "property_id": "p2", "text": "beta", "metadata": {}, "score": 0.8},
        ]

    def backend_name(self):
        return "fake"


class _FakeSparse:
    def build(self, chunks):
        return None

    def query(self, query, top_k, metadata_filter=None):
        return [
            {"chunk_id": "c2", "doc_id": "d2", "property_id": "p2", "text": "beta", "metadata": {}, "score": 10.0},
            {"chunk_id": "c3", "doc_id": "d3", "property_id": "p3", "text": "gamma", "metadata": {}, "score": 8.0},
        ]


class _FakeReranker:
    provider_name = "fake_reranker"

    def rerank(self, query, chunks):
        for idx, chunk in enumerate(chunks):
            chunk.rerank_score = 1.0 - idx * 0.1
            chunk.fused_score += 0.05
        return chunks


def test_retrieval_debug_contains_rank_transitions():
    settings = Settings()
    components = RetrieverComponents(
        embedding_provider=_FakeEmbedder(),
        dense_index=_FakeDense(),
        bm25_index=_FakeSparse(),
        reranker=_FakeReranker(),
    )
    retriever = HybridRetriever(settings=settings, components=components)
    chunks, debug = retriever.retrieve(
        query="test",
        top_k=2,
        rerank=True,
        retrieval_mode="hybrid",
        candidate_pool_size=3,
    )
    assert len(chunks) == 2
    assert debug.reranker_provider == "fake_reranker"
    assert len(debug.dense_rankings) >= 1
    assert len(debug.sparse_rankings) >= 1
    assert len(debug.fused_rankings) >= 1
    assert len(debug.reranked_rankings) >= 1
