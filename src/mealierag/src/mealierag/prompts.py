from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from langfuse import Langfuse

from .config import settings


class PromptType(StrEnum):
    CHAT_GENERATION = "chat-generation"
    MULTI_QUERY_BUILDER = "multi-query-builder-generation"


class PromptManager(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_prompt(self, prompt_type: PromptType, label: str | None = None) -> Any:
        """Get a prompt from the manager. Returns a prompt object that can be compiled."""
        pass


class LangfusePromptManager(PromptManager):
    def __init__(self, langfuse_client: Langfuse = None):
        super().__init__()
        self.client = (
            langfuse_client
            if langfuse_client
            else Langfuse(
                base_url=settings.tracing_base_url,
                public_key=settings.tracing_public_key.get_secret_value(),
                secret_key=settings.tracing_secret_key.get_secret_value(),
                environment=settings.tracing_environment,
                tracing_enabled=settings.tracing_enabled,
            )
        )

    def get_prompt(self, prompt_type: PromptType, label: str | None = None) -> Any:
        label = label if label else settings.tracing_environment
        return self.client.get_prompt(prompt_type.value, label=label)
