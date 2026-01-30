from enum import StrEnum

from langfuse import Langfuse, get_client


class PromptType(StrEnum):
    GENERATION_SYSTEM = "generation-system"
    GENERATION_USER = "generation-user"


class PromptManager:
    def __init__(self):
        pass


class LangfusePromptManager(PromptManager):
    def __init__(self, langfuse_client: Langfuse = None):
        super().__init__()
        self.client = langfuse_client if langfuse_client else get_client()

    def get_prompt(self, prompt_type: PromptType, **kwargs) -> str:
        return self.client.get_prompt(prompt_type).compile(**kwargs)
