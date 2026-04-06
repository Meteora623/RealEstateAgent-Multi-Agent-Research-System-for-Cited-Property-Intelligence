from app.retrieval.service import HybridRetriever
from app.schemas.retrieval import RetrievalChunk


def test_citation_mapping_references_chunk_and_doc_ids():
    chunks = [
        RetrievalChunk(
            chunk_id="DOC-0001::chunk-000",
            doc_id="DOC-0001",
            property_id="PROP-0001",
            text="Evidence passage one.",
            metadata={"title": "Doc 1", "source": "synthetic"},
        ),
        RetrievalChunk(
            chunk_id="DOC-0002::chunk-001",
            doc_id="DOC-0002",
            property_id="PROP-0002",
            text="Evidence passage two.",
            metadata={"title": "Doc 2", "source": "synthetic"},
        ),
    ]
    citations = HybridRetriever.map_citations(chunks)
    assert citations[0].doc_id == "DOC-0001"
    assert citations[1].chunk_id == "DOC-0002::chunk-001"
    assert citations[0].citation_id == "S1"

