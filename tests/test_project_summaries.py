from app.evaluation.runner import EvaluationRunner
from app.schemas.evaluation import ConfigEvaluationResult


def test_project_summary_candidates_only_from_verified_claims():
    baseline = ConfigEvaluationResult(
        config_name="dense_only_baseline",
        retrieval_mode="dense",
        rerank=False,
        semantic_chunking=False,
        ragas_metrics={"faithfulness": 0.3, "answer_relevancy": 0.3, "context_recall": 0.3, "context_precision": 0.3},
        retrieval_metrics={"hit_rate_at_k": 0.2, "mrr_at_k": 0.2, "citation_coverage": 0.2, "unsupported_claim_rate": 0.2},
        latency_metrics={"avg_ms": 100.0, "p50_ms": 100.0, "p95_ms": 120.0},
    )
    best = ConfigEvaluationResult(
        config_name="hybrid_rerank_semantic",
        retrieval_mode="hybrid",
        rerank=True,
        semantic_chunking=True,
        ragas_metrics={"faithfulness": 0.6, "answer_relevancy": 0.6, "context_recall": 0.6, "context_precision": 0.6},
        retrieval_metrics={"hit_rate_at_k": 0.7, "mrr_at_k": 0.7, "citation_coverage": 0.7, "unsupported_claim_rate": 0.1},
        latency_metrics={"avg_ms": 60.0, "p50_ms": 60.0, "p95_ms": 80.0},
    )
    claims = [
        ("Verified", "Indexed 16,000 property-related passages/chunks.", "scale_metrics.json"),
        ("Partially supported", "Other claim", "x"),
        ("Verified", "Improved retrieval hit_rate@k by +0.500 from dense baseline.", "comparison_table.csv"),
    ]
    bullets = EvaluationRunner._project_summary_candidates(
        claim_rows=claims,
        scale_metrics={"total_indexed_chunks": 16000},
        baseline=baseline,
        best=best,
        async_reduction=35.0,
    )
    assert 1 <= len(bullets) <= 5
    assert any("16,000+" in bullet or "16000" in bullet for bullet in bullets)
