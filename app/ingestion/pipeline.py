from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.ingestion.generator import generate_synthetic_dataset
from app.ingestion.loader import load_jsonl_documents
from app.retrieval.chunking import ChunkingConfig, chunk_document
from app.retrieval.service import build_retriever_components
from app.schemas.data import DocumentChunk, IngestionStats

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.components = build_retriever_components(settings)

    def ensure_sample_data(
        self,
        dataset_profile: str | None = None,
        num_properties: int | None = None,
        docs_per_property: int | None = None,
        eval_queries: int | None = None,
    ) -> dict[str, int]:
        profile = dataset_profile or self.settings.dataset_profile
        docs_path = self.settings.data_path / "raw" / "property_documents.jsonl"
        eval_path = self.settings.data_path / "eval" / "eval_queries.jsonl"

        if profile == "benchmark":
            target_docs = max(3000, self.settings.benchmark_num_properties * self.settings.benchmark_docs_per_property)
            p_count = num_properties or self.settings.benchmark_num_properties
            d_count = docs_per_property or self.settings.benchmark_docs_per_property
            q_count = eval_queries or self.settings.benchmark_eval_queries
            seed = self.settings.benchmark_seed
        else:
            target_docs = max(300, (num_properties or 120) * (docs_per_property or 4))
            p_count = num_properties or 120
            d_count = docs_per_property or 4
            q_count = eval_queries or 60
            seed = 11

        existing_docs = 0
        existing_queries = 0
        if docs_path.exists():
            with docs_path.open("r", encoding="utf-8") as f:
                existing_docs = sum(1 for _ in f)
        if eval_path.exists():
            with eval_path.open("r", encoding="utf-8") as f:
                existing_queries = sum(1 for _ in f)

        if existing_docs >= target_docs and existing_queries >= min(60, q_count):
            return {"properties": -1, "documents": existing_docs, "eval_queries": existing_queries}

        return generate_synthetic_dataset(
            data_dir=self.settings.data_path,
            num_properties=p_count,
            docs_per_property=d_count,
            eval_queries=q_count,
            seed=seed,
        )

    def run(
        self,
        semantic_chunking: bool | None = None,
        chunking_strategy: str | None = None,
        dataset_profile: str | None = None,
        force_reindex: bool = False,
    ) -> IngestionStats:
        self.ensure_sample_data(dataset_profile=dataset_profile)
        documents = load_jsonl_documents(self.settings.data_path / "raw" / "property_documents.jsonl")
        if not documents:
            raise RuntimeError("No documents found for ingestion.")

        strategy = chunking_strategy or self.settings.chunking_strategy
        if semantic_chunking is True and strategy == "fixed":
            strategy = "semantic"
        if semantic_chunking is False and strategy != "fixed":
            strategy = "fixed"

        cfg = ChunkingConfig(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            semantic_chunking=strategy != "fixed",
            strategy=strategy,
        )

        chunks: list[DocumentChunk] = []
        for doc in documents:
            chunks.extend(chunk_document(doc, cfg))

        texts = [chunk.text for chunk in chunks]
        embeddings = self.components.embedding_provider.embed_documents(texts)
        self.components.dense_index.upsert(chunks, embeddings)
        self.components.bm25_index.build(chunks)

        chunk_catalog_path = self.settings.indices_path / "chunk_catalog.jsonl"
        with chunk_catalog_path.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(chunk.model_dump_json() + "\n")

        metadata_fields_used = sorted(
            {
                key
                for chunk in chunks
                for key in chunk.metadata.keys()
                if key
                in {
                    "city",
                    "neighborhood",
                    "property_type",
                    "building_class",
                    "year_built",
                    "asking_price",
                    "noi_estimate",
                    "occupancy_rate",
                    "property_tax_annual",
                    "ltv_ratio",
                    "interest_rate",
                    "dscr",
                    "market_cap_low",
                    "market_cap_high",
                }
            }
        )

        chunk_lengths = [len(chunk.text) for chunk in chunks]
        chunks_by_doc_type = self._chunks_by_doc_type(chunks)
        metadata_coverage_pct = self._metadata_coverage(chunks, metadata_fields_used)
        citation_integrity = self._citation_integrity(chunks, documents)

        stats = IngestionStats(
            total_documents=len(documents),
            total_chunks=len(chunks),
            dense_index_backend=self.components.dense_index.backend_name(),
            bm25_documents=len(chunks),
            semantic_chunking=strategy != "fixed",
            chunking_strategy=strategy,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            metadata_fields_used=metadata_fields_used,
            average_chunk_length=round(float(np.mean(chunk_lengths)) if chunk_lengths else 0.0, 2),
            median_chunk_length=round(float(median(chunk_lengths)) if chunk_lengths else 0.0, 2),
            p95_chunk_length=round(float(np.percentile(chunk_lengths, 95)) if chunk_lengths else 0.0, 2),
            chunks_by_doc_type=chunks_by_doc_type,
            metadata_coverage_pct=metadata_coverage_pct,
            citation_map_integrity=citation_integrity,
        )

        (self.settings.indices_path / "ingestion_stats.json").write_text(
            stats.model_dump_json(indent=2), encoding="utf-8"
        )
        self._write_corpus_reports(stats, dataset_profile or self.settings.dataset_profile)
        logger.info(
            "Ingestion complete",
            extra={"event": "ingestion_complete", "docs": len(documents), "chunks": len(chunks)},
        )
        return stats

    @staticmethod
    def _chunks_by_doc_type(chunks: list[DocumentChunk]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in chunks:
            doc_type = str(chunk.metadata.get("doc_type", "unknown"))
            counts[doc_type] = counts.get(doc_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda item: item[0]))

    @staticmethod
    def _metadata_coverage(chunks: list[DocumentChunk], fields: list[str]) -> dict[str, float]:
        coverage: dict[str, float] = {}
        total = max(len(chunks), 1)
        for field in fields:
            present = sum(1 for chunk in chunks if chunk.metadata.get(field) not in (None, "", []))
            coverage[field] = round((present / total) * 100.0, 2)
        return coverage

    @staticmethod
    def _citation_integrity(chunks: list[DocumentChunk], documents) -> dict[str, int | bool]:
        doc_ids = {doc.doc_id for doc in documents}
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        valid_doc_refs = sum(1 for chunk in chunks if chunk.doc_id in doc_ids)
        return {
            "unique_chunk_ids": len(set(chunk_ids)),
            "duplicate_chunk_ids": len(chunk_ids) - len(set(chunk_ids)),
            "valid_doc_links": valid_doc_refs,
            "all_doc_links_valid": valid_doc_refs == len(chunks),
        }

    def _write_corpus_reports(self, stats: IngestionStats, dataset_profile: str) -> None:
        self.settings.reports_path.mkdir(parents=True, exist_ok=True)
        json_path = self.settings.reports_path / "corpus_stats.json"
        md_path = self.settings.reports_path / "corpus_stats.md"
        payload: dict[str, Any] = stats.model_dump()
        payload["dataset_profile"] = dataset_profile
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        lines = [
            "# Corpus Stats",
            "",
            f"- dataset_profile: {dataset_profile}",
            f"- total_raw_documents: {stats.total_documents}",
            f"- total_indexed_chunks: {stats.total_chunks}",
            f"- average_chunk_length: {stats.average_chunk_length}",
            f"- median_chunk_length: {stats.median_chunk_length}",
            f"- p95_chunk_length: {stats.p95_chunk_length}",
            "",
            "## Chunks by Doc Type",
        ]
        for doc_type, count in stats.chunks_by_doc_type.items():
            lines.append(f"- {doc_type}: {count}")
        lines.append("")
        lines.append("## Metadata Coverage (%)")
        for field, pct in stats.metadata_coverage_pct.items():
            lines.append(f"- {field}: {pct}")
        lines.append("")
        lines.append("## Citation Map Integrity")
        for key, value in stats.citation_map_integrity.items():
            lines.append(f"- {key}: {value}")

        md_path.write_text("\n".join(lines), encoding="utf-8")


def load_chunk_catalog(indices_path: Path) -> list[DocumentChunk]:
    path = indices_path / "chunk_catalog.jsonl"
    if not path.exists():
        return []
    chunks: list[DocumentChunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(DocumentChunk.model_validate(json.loads(line)))
    return chunks

