from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from app.core.config import Settings


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, settings: Settings):
        from langchain_openai import OpenAIEmbeddings

        self._embedder = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)


class HashEmbeddingProvider(EmbeddingProvider):
    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dimensions, dtype=np.float32)
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return vec.tolist()

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


def batch_iter(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    if settings.openai_api_key:
        try:
            return OpenAIEmbeddingProvider(settings)
        except Exception:  # noqa: BLE001
            return HashEmbeddingProvider()
    return HashEmbeddingProvider()
