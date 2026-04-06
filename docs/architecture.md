# Architecture Notes

## Agent Roles

- Supervisor: intent classification, plan generation, metadata filter inference.
- Retriever: dense + sparse retrieval, fusion, reranking, debug telemetry.
- Analyst: evidence synthesis and numeric tool usage.
- Fact-Checker: claim support checks and citation consistency validation.

## Control Flow

1. Supervisor sets `intent`, `plan`, inferred filters.
2. Retriever pulls top-k context using configured retrieval strategy.
3. Analyst drafts answer, claims, calculations, uncertainty.
4. Fact-checker validates support and can request another retrieval round.
5. Finalizer composes concise user answer with citations and confidence.

## Retrieval Stack

- Dense: Pinecone (optional) or local cosine index fallback.
- Sparse: BM25 local index.
- Fusion: weighted dense+sparse score blending with per-config alpha.
- Rerank: Cohere (optional) or token-overlap fallback (provider-tagged in metrics).
- Chunking strategies: fixed, semantic sentence-bounded, section-aware semantic.
- Debug traces: dense/sparse/fused/reranked rank transitions returned by retrieval debug endpoint.

## State and Schemas

- Graph state tracked with typed fields in `app/schemas/state.py`.
- API I/O enforced by Pydantic models in `app/schemas/api.py`.
- Analyst/fact-check/final outputs enforced by `app/schemas/agent.py`.
