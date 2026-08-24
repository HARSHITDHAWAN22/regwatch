"""
Central configuration. Loaded once, imported everywhere.
Using pydantic-settings so all config is env-driven -> works identically
locally, in Docker, and on Render/Railway (just set env vars in their dashboard).
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "sqlite:///./regwatch.db"

    # LLM - Google Gemini (genuinely free tier: no card required, 60 req/min
    # on the current free Flash model as of mid-2026 - see aistudio.google.com/apikey)
    # Using the "-latest" alias so this doesn't break again when Google
    # renames/deprecates the underlying model version, which happens often.
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"

    # Embeddings / retrieval
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    similarity_threshold: float = 0.55
    rerank_top_k: int = 5
    verification_min_score: int = 3

    # Auth
    jwt_secret: str = "dev_secret_change_me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    # Versioning - stamped onto every impact assessment so old results
    # remain comparable/traceable even after you improve the prompt or pipeline
    prompt_version: str = "v1"
    pipeline_version: str = "v1"

    # Paths
    faiss_index_dir: str = "./data/faiss_index"

    # Redis - backs job status tracking and the LLM response cache so both
    # survive across multiple worker processes, not just one in-memory dict.
    # Falls back to in-memory behavior automatically if unreachable (see
    # app/redis_client.py) - this is not a hard dependency to run the app.
    redis_url: str = "redis://localhost:6379/0"

    class Config:
        env_file = ".env"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
