# Final Canonical Report

| corpus_scale | best_config_name | official_ragas_values | retrieval_improvement_deltas | latency_improvement_deltas | conservative_verified_resume_wording |
|---|---|---|---|---|---|
| docs=4680, chunks=18720, queries=50 | hybrid_rerank | faithfulness=0.6116666666666668, answer_relevancy=0.4567378305459469, context_recall=0.74 | hit_rate@k=+0.8800, mrr@k=+0.6285, faithfulness=+0.2480, context_recall=+0.5400 | sync_to_async=+35.24%, rerank_on_vs_off=-1.50% | Architected a LangGraph multi-agent real-estate research system indexing 18,720+ property passages with citation-backed outputs. |
