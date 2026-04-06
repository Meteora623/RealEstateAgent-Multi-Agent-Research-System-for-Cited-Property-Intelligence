# Validated Claims

## Claim Verification Matrix
| Status | Claim | Evidence Source |
|---|---|---|
| Verified | Indexed 18,720 property-related passages/chunks. | scale_metrics.json |
| Verified | Improved retrieval hit_rate@k by +0.880 from dense baseline. | comparison_table.csv |
| Unsupported | Best config lacks numeric official RAGAs for required metrics. | official_ragas_gate.json |
| Verified | Reduced average latency by 35.24% using async tool execution. | latency_metrics.json |

## Verified Metrics Snapshot
- indexed_documents: 4680
- indexed_chunks: 18720
- best_configuration: hybrid_rerank
- official_ragas_gate: False
- sync_avg_ms: 354.01
- async_avg_ms: 229.26
- async_latency_reduction_pct: 35.24

## Conservative project-summary wording
- LangGraph multi-agent research workflow indexed 18,720+ property passages with citation-backed responses.
- Hybrid retrieval (dense + BM25 + reranking + chunking tuning) improved hit_rate@k from 0.0 to 0.88.
- Per-agent latency tracing shows 35.24% average response-time reduction with async execution.
