"""Runtime configuration for Agentic Catalog Engine.

Configuration is loaded from environment variables first, with optional local
`.env` support for development. Production deployments should provide secrets
through the hosting platform environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Typed application settings sourced from environment variables."""

    openai_api_key: str | None
    llama_cloud_api_key: str | None
    pinecone_api_key: str | None
    openai_chat_model: str
    openai_embedding_model: str
    pinecone_index_name: str

    @property
    def has_openai_credentials(self) -> bool:
        """Return whether OpenAI-backed extraction and embeddings can run."""
        return bool(self.openai_api_key)

    @property
    def has_llama_parse_credentials(self) -> bool:
        """Return whether LlamaParse-backed complex PDF parsing can run."""
        return bool(self.llama_cloud_api_key)

    @property
    def has_pinecone_credentials(self) -> bool:
        """Return whether Pinecone-backed semantic indexing can run."""
        return bool(self.pinecone_api_key)


def load_config() -> AppConfig:
    """Load application configuration from the process environment."""
    return AppConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llama_cloud_api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        openai_chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o"),
        openai_embedding_model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "ace-catalog"),
    )
