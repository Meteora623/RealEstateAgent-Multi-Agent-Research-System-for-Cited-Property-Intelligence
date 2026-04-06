from app.retrieval.chunking import ChunkingConfig, chunk_document
from app.schemas.data import PropertyDocument


def test_fixed_chunking_overlap_and_count():
    text = "A" * 1200
    doc = PropertyDocument(
        doc_id="DOC-1",
        property_id="PROP-1",
        doc_type="listing_description",
        title="Test Doc",
        text=text,
        metadata={},
    )
    chunks = chunk_document(
        doc,
        ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            semantic_chunking=False,
            strategy="fixed",
        ),
    )
    assert len(chunks) == 3
    assert chunks[0].chunk_id.endswith("chunk-000")
    assert len(chunks[1].text) > 0


def test_semantic_chunking_preserves_sentence_boundaries():
    text = "One short sentence. Two sentence with more details. Third sentence closes."
    doc = PropertyDocument(
        doc_id="DOC-2",
        property_id="PROP-2",
        doc_type="market_commentary",
        title="Semantic Test",
        text=text,
        metadata={},
    )
    chunks = chunk_document(
        doc,
        ChunkingConfig(
            chunk_size=45,
            chunk_overlap=0,
            semantic_chunking=True,
            strategy="semantic",
        ),
    )
    assert len(chunks) >= 2
    assert chunks[0].text.endswith(".")


def test_section_semantic_chunking_splits_by_headers():
    text = "## Overview\nA short sentence. Another sentence.\n\n## Risk\nRisk one. Risk two. Risk three."
    doc = PropertyDocument(
        doc_id="DOC-3",
        property_id="PROP-3",
        doc_type="market_commentary",
        title="Section Test",
        text=text,
        metadata={},
    )
    chunks = chunk_document(
        doc,
        ChunkingConfig(
            chunk_size=40,
            chunk_overlap=0,
            semantic_chunking=True,
            strategy="section_semantic",
        ),
    )
    assert len(chunks) >= 2
    assert any("Overview" in chunk.text for chunk in chunks)
