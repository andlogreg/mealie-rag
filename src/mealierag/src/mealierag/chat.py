"""
Chat module.

Contains the system prompt and functions to populate messages for RAG.
"""

from qdrant_client.models import ScoredPoint

from .api import ChatMessages
from .config import settings
from .prompts import PromptManager, PromptType


def populate_context(hits: list[ScoredPoint]) -> str:
    """Populate context from search hits"""
    # Format Context
    context_text = ""
    for hit in hits:
        context_text += f"[RECIPE_START]\nRecipeName: {hit.payload['name']}\nRecipeID: {hit.payload['recipe_id']}\n{hit.payload['text']}[RECIPE_END]\n"
    return context_text


def populate_messages(
    query: str, context_results: list[ScoredPoint], prompt_manager: PromptManager
) -> ChatMessages:
    """Populate messages for RAG"""
    context_text = populate_context(context_results)

    prompt = prompt_manager.get_prompt(PromptType.CHAT_GENERATION)
    messages = prompt.compile(
        external_url=settings.mealie_external_url,
        context_text=context_text,
        query=query,
    )

    return ChatMessages(messages=messages, prompt=prompt)
