# Evaluation Summary

## Scale Metrics
- total_indexed_documents: 4680
- total_indexed_chunks: 18720
- total_evaluation_queries: 50
- total_metadata_fields_used_for_filtering: 14

## Config Comparison
| Config | Source | Faithfulness | Answer Relevancy | Context Recall | Hit@k | MRR@k | Unsupported Claim Rate |
|---|---|---:|---:|---:|---:|---:|---:|
| dense_only_baseline | fallback_quality | 0.1899 | 0.1963 | 0.2135 | 0.0 | 0.0 | 0.96 |
| hybrid_baseline | fallback_quality | 0.3718 | 0.1963 | 0.4481 | 0.82 | 0.5557 | 0.325 |
| hybrid_rerank | fallback_quality | 0.4406 | 0.1963 | 0.522 | 0.88 | 0.6285 | 0.235 |
| hybrid_rerank_semantic | fallback_quality | 0.3006 | 0.1963 | 0.4158 | 0.86 | 0.6339 | 0.36 |

Best configuration: **hybrid_rerank**
Official RAGAs gate: **False**
Gate reason: run_ragas=false.

## Baseline to Best Deltas
- faithfulness: 0.1899 -> 0.4406 (delta +0.2507)
- answer_relevancy: 0.1963 -> 0.1963 (delta +0.0000)
- context_recall: 0.2135 -> 0.5220 (delta +0.3085)
- hit_rate_at_k: 0.0 -> 0.88
- mrr_at_k: 0.0 -> 0.6285
- unsupported_claim_rate: 0.96 -> 0.235

## Latency Experiments
- sync_vs_async_tools avg reduction (%): 35.24
- rerank_on_vs_off avg delta (%): -1.5