from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.data import DocumentChunk, PropertyDocument


@dataclass
class ChunkingConfig:
    chunk_size: int = 700
    chunk_overlap: int = 120
    semantic_chunking: bool = False
    strategy: str = "fixed"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _fixed_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def _semantic_chunk_text(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in _sentences(text):
        if current and current_len + len(sentence) > chunk_size:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _split_sections(text: str) -> list[str]:
    if not text.strip():
        return []
    sections = re.split(
        r"\n(?=(?:##\s+[A-Za-z]|[A-Z][A-Za-z ]{3,30}:|\[[A-Z_ ]{3,30}\]))",
        text.strip(),
    )
    return [section.strip() for section in sections if section.strip()]


def _section_semantic_chunk_text(text: str, chunk_size: int) -> list[str]:
    sections = _split_sections(text)
    if not sections:
        return []

    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        chunks.extend(_semantic_chunk_text(section, chunk_size))
    return chunks


def chunk_document(document: PropertyDocument, config: ChunkingConfig) -> list[DocumentChunk]:
    strategy = config.strategy
    if config.semantic_chunking and strategy == "fixed":
        strategy = "semantic"

    if strategy == "section_semantic":
        text_chunks = _section_semantic_chunk_text(document.text, config.chunk_size)
    elif strategy == "semantic":
        text_chunks = _semantic_chunk_text(document.text, config.chunk_size)
    else:
        text_chunks = _fixed_chunk_text(document.text, config.chunk_size, config.chunk_overlap)

    results: list[DocumentChunk] = []
    for idx, text in enumerate(text_chunks):
        chunk_id = f"{document.doc_id}::chunk-{idx:03d}"
        metadata = {**document.metadata, "doc_type": document.doc_type, "title": document.title}
        results.append(
            DocumentChunk(
                chunk_id=chunk_id,
                doc_id=document.doc_id,
                property_id=document.property_id,
                text=text,
                metadata=metadata,
            )
        )
    return results
