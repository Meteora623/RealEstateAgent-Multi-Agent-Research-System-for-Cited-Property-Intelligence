# Validated Claims

## Verified claims
- Corpus scale reached 4,680 docs and 18,720 indexed chunks.
- Best config `hybrid_rerank` has numeric official RAGAs for faithfulness, answer_relevancy, context_recall.
- Retrieval improved from baseline: hit_rate@k delta +0.8800, MRR@k delta +0.6285.
- Async execution reduced avg latency by 35.24% (sync 354.01ms -> async 229.26ms).
- Official quality deltas are positive for faithfulness (+0.2480) and context_recall (+0.5400).

## Partially supported claims
- None

## Unsupported claims
- None

## Conservative project-summary wording
- LangGraph multi-agent workflow processes 18,720+ indexed property passages with citation-backed outputs.
- Hybrid retrieval and reranking improved hit_rate@k from 0.0 to 0.88 and MRR@k from 0.0 to 0.6285.
- Per-agent async execution reduced average response latency by 35.24% in benchmark runs.
- Official RAGAs evaluation across four retrieval configurations selected `hybrid_rerank` with numeric faithfulness, answer_relevancy, and context_recall.
