from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.evaluation.runner import run_evaluation_sync
from app.graph.workflow import RealEstateResearchSystem
from app.ingestion.pipeline import IngestionPipeline


@lru_cache(maxsize=1)
def get_system() -> RealEstateResearchSystem:
    settings = get_settings()
    return RealEstateResearchSystem(settings)


@lru_cache(maxsize=1)
def get_pipeline() -> IngestionPipeline:
    settings = get_settings()
    return IngestionPipeline(settings)


def get_settings_dep() -> Settings:
    return get_settings()


def run_evaluation(
    settings: Settings,
    run_ragas: bool,
    ragas_mode: str | None,
    dataset_profile: str | None,
    max_queries: int | None,
    output_tag: str | None,
):
    return run_evaluation_sync(
        settings=settings,
        run_ragas=run_ragas,
        ragas_mode=ragas_mode,
        dataset_profile=dataset_profile,
        max_queries=max_queries,
        output_tag=output_tag,
    )
