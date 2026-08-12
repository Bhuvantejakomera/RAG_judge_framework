from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

DEFAULT_GEMINI_JUDGE_MODEL = "models/gemini-3.6-flash"


class ChunkingConfig(BaseModel):
    """Chunking-related settings for document preprocessing."""

    size: int = Field(..., ge=1)
    overlap: int = Field(..., ge=0)


class EmbeddingConfig(BaseModel):
    """Embedding model settings."""

    model: str
    dimension: int = Field(..., ge=1)


class LLMConfig(BaseModel):
    """Large language model configuration."""

    local_model: str
    judge_model: str


class JudgeConfig(BaseModel):
    """Primary and fallback judge configuration."""

    provider: str = "gemini"
    model: str = DEFAULT_GEMINI_JUDGE_MODEL
    fallback_provider: str = "ollama"
    fallback_model: str = "qwen3:8b"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)


class RetrievalConfig(BaseModel):
    """Settings used for retrieval and ranking."""

    top_k: int = Field(..., ge=1)
    display_k: int = Field(..., ge=1)
    similarity_threshold: float = Field(..., ge=0.0, le=1.0)


class MemoryConfig(BaseModel):
    """Settings for conversational sliding window memory."""

    sliding_window_turns: int = Field(..., ge=1)


class EvaluationConfig(BaseModel):
    """Settings for automatic evaluation mode selection."""

    session_threshold: int = Field(default=5, ge=0)
    dataset_sample_size: int = Field(default=50, ge=1)
    random_seed: int = 42
    sampling_strategy: str = "random"
    bert_score_model: str = "roberta-base"


class WikipediaConfig(BaseModel):
    """Wikipedia client settings."""

    user_agent: str
    language: str = "en"


class IndexingConfig(BaseModel):
    """Indexing and ingestion settings."""

    max_articles: int | None = 100
    max_workers: int = 4


class LoggingConfig(BaseModel):
    """Logging destination and format settings."""

    level: str = Field(default="INFO")
    directory: str = Field(default="./logs")


class AppSettings(BaseModel):
    """Aggregate application configuration."""

    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    memory: MemoryConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    wikipedia: WikipediaConfig
    logging: LoggingConfig


class ConfigManager:
    """Singleton wrapper for application settings loaded from YAML."""

    _instance: ConfigManager | None = None
    _settings: AppSettings

    def __new__(cls, config_path: str | Path | None = None) -> ConfigManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._settings = cls._instance._load_settings(config_path)
        return cls._instance

    @property
    def settings(self) -> AppSettings:
        """Returns the loaded application settings."""
        return self._settings

    def reload(self, config_path: str | Path | None = None) -> None:
        """Reloads the application settings from the YAML configuration file."""
        self._settings = self._load_settings(config_path)

    def _load_settings(self, config_path: str | Path | None = None) -> AppSettings:
        path = Path(config_path or Path.cwd() / "config" / "config.yaml")
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found at {path}")

        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw_config, dict):
            raise ValueError("Configuration content must be a mapping at the top level.")

        try:
            return AppSettings(**raw_config)
        except ValidationError as error:
            raise ValueError(f"Invalid configuration schema: {error}") from error
