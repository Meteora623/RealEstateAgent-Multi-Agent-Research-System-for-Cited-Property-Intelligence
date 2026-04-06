from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import Settings
from app.core.logging import get_logger
from app.schemas.data import DocumentChunk

logger = get_logger(__name__)


class DenseIndex(ABC):
    @abstractmethod
    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(
        self, query_embedding: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def backend_name(self) -> str:
        raise NotImplementedError


class LocalDenseIndex(DenseIndex):
    def __init__(self, indices_dir: Path):
        self.indices_dir = indices_dir
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.indices_dir / "local_dense_vectors.npy"
        self.meta_path = self.indices_dir / "local_dense_metadata.json"
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self.metadata: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.vectors_path.exists() and self.meta_path.exists():
            self.vectors = np.load(self.vectors_path)
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        np.save(self.vectors_path, self.vectors)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self.vectors = np.array(embeddings, dtype=np.float32)
        self.metadata = []
        for chunk in chunks:
            self.metadata.append(
                {
                    "id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "property_id": chunk.property_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        self._save()

    def query(
        self, query_embedding: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.vectors.size == 0:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        dots = self.vectors @ q
        norms = np.linalg.norm(self.vectors, axis=1) * q_norm
        sims = np.divide(dots, norms, where=norms != 0)

        candidates = np.argsort(-sims)
        results: list[dict[str, Any]] = []
        for idx in candidates:
            meta = self.metadata[int(idx)]
            if metadata_filter and not _match_filter(meta.get("metadata", {}), metadata_filter):
                continue
            results.append(
                {
                    "chunk_id": meta["id"],
                    "score": float(sims[idx]),
                    "doc_id": meta["doc_id"],
                    "property_id": meta["property_id"],
                    "text": meta["text"],
                    "metadata": meta["metadata"],
                }
            )
            if len(results) >= top_k:
                break
        return results

    def backend_name(self) -> str:
        return "local"


class PineconeDenseIndex(DenseIndex):
    def __init__(self, settings: Settings):
        if not settings.pinecone_api_key:
            raise ValueError("Pinecone API key missing.")
        try:
            from pinecone import Pinecone, ServerlessSpec
        except Exception as exc:  # noqa: BLE001
            raise ImportError("pinecone package is not installed.") from exc

        self._pinecone_serverless_spec = ServerlessSpec
        self.settings = settings
        self.client = Pinecone(api_key=settings.pinecone_api_key)
        self.index_name = settings.pinecone_index_name
        self._ensure_index()
        self.index = self.client.Index(self.index_name)

    def _ensure_index(self) -> None:
        existing = {idx["name"] for idx in self.client.list_indexes()}
        if self.index_name in existing:
            return
        self.client.create_index(
            name=self.index_name,
            dimension=256,
            metric="cosine",
            spec=self._pinecone_serverless_spec(
                cloud=self.settings.pinecone_cloud, region=self.settings.pinecone_region
            ),
        )
        logger.info("Created Pinecone index %s", self.index_name)

    def upsert(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        vectors = []
        for chunk, emb in zip(chunks, embeddings):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": emb,
                    "metadata": {
                        **chunk.metadata,
                        "doc_id": chunk.doc_id,
                        "property_id": chunk.property_id,
                        "text": chunk.text,
                    },
                }
            )
        if vectors:
            self.index.upsert(vectors=vectors)

    def query(
        self, query_embedding: list[float], top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=metadata_filter or {},
        )
        results: list[dict[str, Any]] = []
        for match in response.get("matches", []):
            meta = match.get("metadata", {})
            results.append(
                {
                    "chunk_id": match["id"],
                    "score": float(match["score"]),
                    "doc_id": meta.get("doc_id", ""),
                    "property_id": meta.get("property_id", ""),
                    "text": meta.get("text", ""),
                    "metadata": meta,
                }
            )
        return results

    def backend_name(self) -> str:
        return "pinecone"


def _match_filter(meta: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
    for key, value in metadata_filter.items():
        if key not in meta:
            return False
        if str(meta.get(key)).lower() != str(value).lower():
            return False
    return True


def build_dense_index(settings: Settings) -> DenseIndex:
    if settings.use_pinecone and settings.pinecone_api_key:
        try:
            return PineconeDenseIndex(settings)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Falling back to local dense index. Pinecone unavailable: %s", exc)
    return LocalDenseIndex(settings.indices_path)
