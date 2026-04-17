"""
Core configuration module.
Loads environment variables from .env using pydantic-settings.
"""

import logging
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Gemini ──────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str

    # ── Gemini model names ───────────────────────────────────────────────────
    GEMINI_MODEL: str = "models/gemini-2.5-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"

    # ── RAG tuning ───────────────────────────────────────────────────────────
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 80
    TOP_K_RESULTS: int = 4

    # ── Conversation memory ──────────────────────────────────────────────────
    MEMORY_WINDOW: int = 3          # last N user-assistant pairs kept

    # ── FAISS index path ─────────────────────────────────────────────────────
    FAISS_INDEX_PATH: str = "data/faiss_index"

    # ── Source data ──────────────────────────────────────────────────────────
    DATA_FILE_PATH: str = "data/mitratech_data.txt"
    WEBSITE_URL: str = "https://www.mitratechgroup.com"

    # ── Logging ──────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton of Settings."""
    return Settings()


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
