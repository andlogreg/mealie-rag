import logging

from pydantic import Field, create_model
from qdrant_client.models import PointStruct

from .api import ChatMessages
from .config import settings
from .llm_client import LLMClient
from .models import NormalizedRecipeIngredients, Recipe

logger = logging.getLogger(__name__)


def normalize_ingredients(
    recipe: Recipe,
    llm_client: LLMClient,
    system_prompt: str,
) -> Recipe:
    user_input = str([ing.display for ing in recipe.recipeIngredients])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    chat_messages = ChatMessages(messages=messages, prompt=None)
    temperature = 0.2
    response = llm_client.chat(
        chat_messages=chat_messages,
        model=settings.llm_model,
        temperature=temperature,
        seed=settings.llm_seed,
        response_model=NormalizedRecipeIngredients,
    )
    recipe.normalizedRecipeIngredients = response
    return recipe


def enrich_recipe_properties(
    recipe: Recipe,
    llm_client: LLMClient,
    system_prompt: str,
) -> Recipe:
    missing_fields = {}
    target_fields = [
        "recipeCategory",
        "tags",
        "tools",
        "method",
        "is_healthy",
        "total_time_minutes",
    ]

    for field_name in target_fields:
        current_value = getattr(recipe, field_name)
        # Check if value is "empty" (None or empty list)
        if current_value is None or (
            isinstance(current_value, list) and not current_value
        ):
            field_info = Recipe.model_fields[field_name]
            # Use the original model's type annotation and description
            missing_fields[field_name] = (
                field_info.annotation,
                Field(description=field_info.description),
            )

    if not missing_fields:
        return recipe

    EnrichmentModel = create_model("EnrichmentModel", **missing_fields)

    user_input = recipe.get_text_representation(
        ["name", "description", "recipeIngredients", "recipeInstructions"]
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]
    chat_messages = ChatMessages(messages=messages, prompt=None)
    temperature = 0.2

    try:
        response = llm_client.chat(
            chat_messages=chat_messages,
            model=settings.llm_model,
            temperature=temperature,
            seed=settings.llm_seed,
            response_model=EnrichmentModel,
        )
    except Exception as e:
        logger.exception(f"Error enriching recipe {recipe.name}: {e}")
        return recipe

    # Update recipe with response data
    response_data = response.model_dump()
    for field, value in response_data.items():
        if value is not None:
            setattr(recipe, field, value)

    return recipe


def create_point_from_recipe(recipe: Recipe, embedding: list[float]) -> PointStruct:
    """
    Create a Qdrant PointStruct from a Recipe object.
    Standardizes payload structure and ID generation.
    """
    payload = {
        "recipe_id": recipe.id,
        "name": recipe.name,
        "slug": recipe.slug,
        "total_time_minutes": recipe.total_time_minutes,
        "description": recipe.description,
        "category": recipe.recipeCategory,
        "tags": recipe.tags,
        "tools": recipe.tools,
        "method": recipe.method,
        "rating": recipe.rating,
        "is_healthy": recipe.is_healthy,
        "text": recipe.get_text_for_embedding(),
        "ingredients": [ing.display for ing in recipe.recipeIngredients],
        "instructions": [inst.text for inst in recipe.recipeInstructions],
        "normalized_ingredients": recipe.normalizedRecipeIngredients.flatten(),
        # Temporary: nested model_dump to ease recipe recreation at query/search time
        "model_dump": recipe.model_dump(),
    }

    if recipe.id is None:
        raise ValueError(f"Recipe '{recipe.name}' has no ID.")

    return PointStruct(
        id=recipe.id,
        vector=embedding,
        payload=payload,
    )
