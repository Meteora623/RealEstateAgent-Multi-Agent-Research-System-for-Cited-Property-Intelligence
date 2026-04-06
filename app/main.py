from __future__ import annotations

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.observability import configure_langsmith

settings = get_settings()
configure_logging(settings.log_level)
configure_langsmith(settings)

app = FastAPI(
    title="realestate-agent",
    version="0.1.0",
    description="LangGraph multi-agent real estate research backend.",
)
app.include_router(router)

