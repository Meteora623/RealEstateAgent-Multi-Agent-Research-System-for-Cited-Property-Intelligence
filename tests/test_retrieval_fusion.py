from app.retrieval.fusion import weighted_fusion


def test_weighted_fusion_combines_dense_and_sparse_scores():
    dense = [
        {"chunk_id": "c1", "doc_id": "d1", "property_id": "p1", "text": "a", "metadata": {}, "score": 0.9},
        {"chunk_id": "c2", "doc_id": "d2", "property_id": "p2", "text": "b", "metadata": {}, "score": 0.5},
    ]
    sparse = [
        {"chunk_id": "c2", "doc_id": "d2", "property_id": "p2", "text": "b", "metadata": {}, "score": 10.0},
        {"chunk_id": "c3", "doc_id": "d3", "property_id": "p3", "text": "c", "metadata": {}, "score": 8.0},
    ]
    fused = weighted_fusion(dense, sparse, alpha=0.5, top_k=3)
    ids = [item.chunk_id for item in fused]
    assert "c2" in ids
    assert len(fused) == 3
    assert fused[0].fused_score >= fused[-1].fused_score

