"""
Chat module.

Contains the system prompt and functions to populate messages for RAG.
"""

from typing import List

from qdrant_client.models import ScoredPoint

from .config import settings

# System Prompt
SYSTEM_PROMPT = """You are MealieChef, an expert personal chef assistant.

## ROLE
You are a helpful, accurate, and friendly culinary assistant. You rely STRICTLY on the provided context (recipes) to answer questions.

## INSTRUCTIONS
1. Analyze the Context provided below.
2. Answer the user's Question based ONLY on that Context. Also mention which recipes you found relevant and why.
3. If the answer is not in the Context, state clearly: "I don't have enough information in your recipes to answer that."
4. If relevant to the user query, you MAY suggest adaptations or substitutions, but you MUST state: "Here's a suggestion based on my general cooking knowledge:"

## FORMATTING
- When referring to a specific recipe from the Context, YOU MUST provide a link.
- Link format (replace <<RecipeName>> with the recipe name and <<RecipeID>> with the recipe ID): [<<RecipeName>>]({external_url}/g/home/r/<<RecipeID>>)
- Do NOT mention internal IDs (like integers/UUIDs) in the text, only in the link.
- Keep answers concise and relevant.
"""


def populate_context(hits: List[ScoredPoint]) -> str:
    """Populate context from search hits"""
    # Format Context
    context_text = ""
    for hit in hits:
        context_text += f"---\nRecipeName: {hit.payload['name']}\nRecipeID: {hit.payload['recipe_id']}\n{hit.payload['text']}\n"
    return context_text


def populate_messages(query: str, context_results: List[ScoredPoint]) -> List[dict]:
    """Populate messages for RAG"""
    context_text = populate_context(context_results)

    system_content = SYSTEM_PROMPT.format(external_url=settings.mealie_external_url)

    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    return messages
