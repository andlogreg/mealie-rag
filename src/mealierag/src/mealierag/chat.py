"""
Chat module.

Contains the system prompt and functions to populate messages for RAG.
"""

from qdrant_client.models import ScoredPoint

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
) -> list[dict[str, str]]:
    """Populate messages for RAG"""
    context_text = populate_context(context_results)

    system_content = prompt_manager.get_prompt(
        PromptType.GENERATION_SYSTEM, external_url=settings.mealie_external_url
    )

    user_message = prompt_manager.get_prompt(
        PromptType.GENERATION_USER, context_text=context_text, query=query
    )

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    return messages
