"""
Config module.
"""

from enum import StrEnum, auto

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class SearchStrategy(StrEnum):
    SIMPLE = auto()
    MULTIQUERY = auto()


class LLMProvider(StrEnum):
    OLLAMA = auto()
    OPENAI = auto()


# TODO break down into multiple config files depending on which entrypoint is used
class Settings(BaseSettings):
    mealie_api_url: str = Field(
        "http://localhost:9000/api/recipes", description="Mealie API URL"
    )
    mealie_external_url: str = Field(
        "http://localhost:9000", description="Mealie External URL (for links)"
    )
    mealie_token: SecretStr | None = Field(None, description="Mealie API Token")

    vectordb_url: str = Field(
        "http://localhost:6333", description="Vector DB (qdrant) URL"
    )
    vectordb_collection_name: str = Field(
        "mealie_recipes", description="Qdrant Collection Name"
    )
    vectordb_k: int = Field(3, description="Number of results to return when searching")
    # embedding_model: str = "nomic-embed-text"
    embedding_model: str = Field("mealie-rag-embedding", description="Embedding Model")

    llm_provider: LLMProvider = Field(LLMProvider.OLLAMA, description="LLM Provider")
    llm_base_url: str = Field(
        "http://localhost:11434", description="LLM Provider Base URL"
    )
    llm_api_key: SecretStr | None = Field(None, description="LLM Provider API Key")

    llm_model: str = Field("mealie-rag-llm", description="LLM Model")
    llm_temperature: float = Field(0.2, description="LLM Temperature")
    llm_seed: int | None = Field(None, description="LLM Seed")

    ui_port: int = Field(7860, description="Port to serve the UI on")
    ui_username: str = Field("mealie", description="UI Username")
    ui_password: SecretStr = Field("rag", description="UI Password")

    search_strategy: SearchStrategy = Field(
        SearchStrategy.SIMPLE, description="Search Strategy"
    )

    # ingest specific settings
    delete_collection_if_exists: bool = Field(
        False, description="Delete the collection if it exists before ingesting"
    )

    log_level: str = Field("INFO", description="Log level for the application")
    dependency_log_level: str = Field(
        "WARNING", description="Log level for dependencies"
    )

    tracing_base_url: str = Field(
        "https://cloud.langfuse.com", description="Langfuse Base URL"
    )
    tracing_public_key: SecretStr = Field(
        "pk-lf-test", description="Langfuse Public Key"
    )
    tracing_secret_key: SecretStr = Field(
        "sk-lf-test", description="Langfuse Secret Key"
    )
    tracing_environment: str = Field("development", description="Langfuse Environment")
    tracing_enabled: bool = Field(False, description="Enable tracing")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
