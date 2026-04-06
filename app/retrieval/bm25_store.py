from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.schemas.data import DocumentChunk


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _match_filter(meta: dict[str, Any], metadata_filter: dict[str, Any]) -> bool:
    for key, value in metadata_filter.items():
        if str(meta.get(key, "")).lower() != str(value).lower():
            return False
    return True


class BM25Index:
    def __init__(self, indices_dir: Path):
        self.indices_dir = indices_dir
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.indices_dir / "bm25.pkl"
        self.meta_path = self.indices_dir / "bm25_metadata.json"
        self.bm25: BM25Okapi | None = None
        self.metadata: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            with self.index_path.open("rb") as f:
                self.bm25 = pickle.load(f)
            self.metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def build(self, chunks: list[DocumentChunk]) -> None:
        corpus = [_tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.metadata = [
            {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "property_id": chunk.property_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        self._save()

    def _save(self) -> None:
        with self.index_path.open("wb") as f:
            pickle.dump(self.bm25, f)
        self.meta_path.write_text(json.dumps(self.metadata, indent=2), encoding="utf-8")

    def query(
        self, query: str, top_k: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results: list[dict[str, Any]] = []
        for idx in order:
            meta = self.metadata[idx]
            if metadata_filter and not _match_filter(meta.get("metadata", {}), metadata_filter):
                continue
            results.append(
                {
                    "chunk_id": meta["chunk_id"],
                    "doc_id": meta["doc_id"],
                    "property_id": meta["property_id"],
                    "text": meta["text"],
                    "metadata": meta["metadata"],
                    "score": float(scores[idx]),
                }
            )
            if len(results) >= top_k:
                break
        return results

