from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve the project root independently of the terminal's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Shared configuration for the API, ingestion and retrieval modules."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        # Handles .env files created by Windows PowerShell with a UTF-8 BOM.
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )

    app_name: str = Field(default="PolicyPilot", min_length=1)
    app_version: str = "0.1.0"
    app_env: Literal["development", "test", "production"] = "development"

    gemini_api_key: SecretStr | None = None
    gemini_model: str = Field(default="gemini-2.5-flash", min_length=1)
    gemini_embedding_model: str = Field(default="gemini-embedding-001", min_length=1)
    embedding_dimensions: int = 768

    @field_validator("embedding_dimensions")
    @classmethod
    def check_embedding_dimensions(cls, value: int) -> int:
        if value not in (768, 1536, 3072):
            raise ValueError("EMBEDDING_DIMENSIONS must be 768, 1536 or 3072")
        return value

    agent_tool_transport: Literal["mcp", "direct"] = "mcp"
    qdrant_url: str | None = None
    qdrant_path: Path = PROJECT_ROOT / "data" / "qdrant"
    qdrant_collection_prefix: str = Field(default="policypilot", min_length=1)

    @property
    def qdrant_collection(self) -> str:
        """Collection name tied to the embedding model and dimension."""
        model = self.gemini_embedding_model.replace("/", "_").replace(".", "_")
        return f"{self.qdrant_collection_prefix}_{model}_{self.embedding_dimensions}"


@lru_cache
def get_settings() -> Settings:
    """Load settings once per process instead of reading .env on each request."""
    return Settings()
