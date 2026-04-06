from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = Field(default="realestate-agent", alias="APP_NAME")
    env: str = Field(default="dev", alias="ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_project: str = Field(default="realestate-agent", alias="LANGSMITH_PROJECT")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="realestate-agent-index", alias="PINECONE_INDEX_NAME")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")
    use_pinecone: bool = Field(default=False, alias="USE_PINECONE")

    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    use_cohere_reranker: bool = Field(default=False, alias="USE_COHERE_RERANKER")

    default_retrieval_mode: str = Field(default="hybrid", alias="DEFAULT_RETRIEVAL_MODE")
    default_top_k: int = Field(default=8, alias="DEFAULT_TOP_K")
    default_candidate_pool_multiplier: int = Field(default=4, alias="DEFAULT_CANDIDATE_POOL_MULTIPLIER")
    fusion_alpha: float = Field(default=0.65, alias="FUSION_ALPHA")
    rerank_top_n: int = Field(default=20, alias="RERANK_TOP_N")
    chunk_size: int = Field(default=700, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, alias="CHUNK_OVERLAP")
    semantic_chunking: bool = Field(default=False, alias="SEMANTIC_CHUNKING")
    chunking_strategy: str = Field(default="fixed", alias="CHUNKING_STRATEGY")

    data_dir: str = Field(default="data", alias="DATA_DIR")
    indices_dir: str = Field(default="data/indices", alias="INDICES_DIR")
    eval_output_dir: str = Field(default="artifacts/eval", alias="EVAL_OUTPUT_DIR")
    reports_dir: str = Field(default="reports", alias="REPORTS_DIR")

    dataset_profile: str = Field(default="benchmark", alias="DATASET_PROFILE")
    benchmark_num_properties: int = Field(default=780, alias="BENCHMARK_NUM_PROPERTIES")
    benchmark_docs_per_property: int = Field(default=6, alias="BENCHMARK_DOCS_PER_PROPERTY")
    benchmark_eval_queries: int = Field(default=80, alias="BENCHMARK_EVAL_QUERIES")
    benchmark_seed: int = Field(default=13, alias="BENCHMARK_SEED")

    ragas_mode: str = Field(default="auto", alias="RAGAS_MODE")
    ragas_fallback_enabled: bool = Field(default=True, alias="RAGAS_FALLBACK_ENABLED")
    require_official_ragas_best_config: bool = Field(
        default=True, alias="REQUIRE_OFFICIAL_RAGAS_BEST_CONFIG"
    )

    max_retrieval_rounds: int = 2
    tool_io_latency_ms: int = Field(default=35, alias="TOOL_IO_LATENCY_MS")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def indices_path(self) -> Path:
        path = Path(self.indices_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def eval_output_path(self) -> Path:
        path = Path(self.eval_output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_path(self) -> Path:
        path = Path(self.reports_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def langsmith_enabled(self) -> bool:
        return bool(self.langsmith_tracing and self.langsmith_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
