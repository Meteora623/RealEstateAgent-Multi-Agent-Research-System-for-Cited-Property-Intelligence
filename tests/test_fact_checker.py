from app.agents.fact_checker import FactCheckerAgent
from app.schemas.agent import AnalystDraft
from app.schemas.retrieval import RetrievalChunk


def test_fact_checker_flags_unsupported_claims():
    draft = AnalystDraft(
        answer_summary="Summary",
        supporting_reasoning=[],
        numeric_calculations=[],
        uncertainty_notes=[],
        major_claims=["This building has a heliport and private marina."],
        cited_chunk_ids=["DOC-1::chunk-000"],
    )
    chunks = [
        RetrievalChunk(
            chunk_id="DOC-1::chunk-000",
            doc_id="DOC-1",
            property_id="PROP-1",
            text="Building includes updated lobby and boiler improvements.",
            metadata={},
        )
    ]
    report = FactCheckerAgent().run(draft, chunks, retrieval_round=1, max_retrieval_rounds=2)
    assert report.unsupported_claims or report.weak_claims
    assert report.verdict.value != "approve"
