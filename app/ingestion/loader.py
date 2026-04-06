from __future__ import annotations

import json
from pathlib import Path

from app.schemas.data import PropertyDocument


def load_jsonl_documents(path: Path) -> list[PropertyDocument]:
    docs: list[PropertyDocument] = []
    if not path.exists():
        return docs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(PropertyDocument.model_validate(json.loads(line)))
    return docs

