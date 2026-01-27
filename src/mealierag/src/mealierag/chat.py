"""
Chat module.

Contains the system prompt and functions to populate messages for RAG.
"""

from qdrant_client.models import ScoredPoint

from .config import settings

SYSTEM_PROMPT = """
You are MealieChef, an expert personal chef assistant.

## ROLE
You are a helpful, accurate, and friendly culinary assistant. You rely STRICTLY on the provided [CONTEXT] (recipes) to answer questions.

### RULES OF ENGAGEMENT
1. Analyze the provided [CONTEXT].
2. Answer the user's Request, [USER REQUEST],  based on the provided [CONTEXT] recipes.
3. Mention which recipes you considered relevant and why.
4. If provided in the [CONTEXT], always mention the rating (1-5) of each recipe you suggest in your answer.
5. If the answer is not in the [CONTEXT], say: "I don't have enough information in your recipes to answer that."
6. Link format: [Recipe Name]({external_url}/g/home/r/RecipeID)
7. Do NOT mention UUIDs/IDs in the text, only inside the link markdown.
8. Keep answers concise and relevant.
9. If relevant to the [USER REQUEST], you MAY suggest adaptations/variation/substitutions, but you MUST state: "Here's a suggestion based on my general cooking knowledge:"
10. Use plain english, do not mention internal keywords like [CONTEXT] or [USER_REQUEST]

### CONTEXT RECIPE EXAMPLE
[RECIPE_START]
RecipeName: Greek Yogurt Parfait
RecipeID: 487a6099-f164-4f70-8f61-e73b82935333
Title: Greek Yogurt Parfait
Description: High-protein breakfast with probiotics and fiber.
Rating: 5.0
Category: Breakfast, Healthy
Tags: vegetarian, high-protein, quick
Ingredients:
- 1 cup Greek yogurt
- 1/2 cup granola
- 1/2 cup mixed berries
- 1 tbsp honey
- 1 tsp flax seeds
Instructions:
- Layer yogurt in a bowl or glass.
- Add a layer of granola.
- Top with fresh berries.
- Drizzle with honey and sprinkle flax seeds.
[RECIPE_END]

#### Recipe fields meaning:
- rating: number between 1 and 5, 5 is better, 1 is worse
- category: comma separated list of categories
- tags: comma separated list of tags

### RECIPE LINK EXAMPLE

[Greek Yogurt Parfait]({external_url}/g/home/r/487a6099-f164-4f70-8f61-e73b82935333)
"""

USER_MESSAGE = """
[CONTEXT]:
{context_text}

[USER REQUEST]:
{query}"""


def populate_context(hits: list[ScoredPoint]) -> str:
    """Populate context from search hits"""
    # Format Context
    context_text = ""
    for hit in hits:
        context_text += f"[RECIPE_START]\nRecipeName: {hit.payload['name']}\nRecipeID: {hit.payload['recipe_id']}\n{hit.payload['text']}[RECIPE_END]\n"
    return context_text


def populate_messages(
    query: str, context_results: list[ScoredPoint]
) -> list[dict[str, str]]:
    """Populate messages for RAG"""
    context_text = populate_context(context_results)

    system_content = SYSTEM_PROMPT.format(external_url=settings.mealie_external_url)

    user_message = USER_MESSAGE.format(context_text=context_text, query=query)

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_message},
    ]

    return messages
