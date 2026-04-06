from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from app.core.config import Settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def configure_langsmith(settings: Settings) -> None:
    if not settings.langsmith_enabled:
        return
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint


@dataclass
class RequestTrace:
    request_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
    events: list[dict[str, str | float]] = field(default_factory=list)

    def add_event(self, stage: str, event: str, duration_ms: float | None = None) -> None:
        payload: dict[str, str | float] = {"stage": stage, "event": event}
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        self.events.append(payload)

    def stop_stage(self, stage: str, started_at: float) -> None:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        self.stage_durations_ms[stage] = duration_ms
        logger.info(
            "Stage completed",
            extra={"request_id": self.request_id, "stage": stage, "duration_ms": duration_ms},
        )

    @property
    def total_duration_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0


@contextmanager
def track_stage(trace: RequestTrace, stage: str):
    started_at = time.perf_counter()
    try:
        yield
    finally:
        trace.stop_stage(stage, started_at)

