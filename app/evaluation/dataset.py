from __future__ import annotations

import json
from pathlib import Path

from app.schemas.evaluation import EvaluationQuery


def load_eval_queries(path: Path) -> list[EvaluationQuery]:
    if not path.exists():
        return []
    rows: list[EvaluationQuery] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(EvaluationQuery.model_validate(json.loads(line)))
    return rows

