from unittest.mock import MagicMock

from qdrant_client.http.models import ScoredPoint

from mealierag.chat import (
    populate_context,
    populate_messages,
)
from mealierag.config import settings
from mealierag.prompts import PromptType


def test_populate_context():
    """Test context population from hits."""
    hits = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "name": "Recipe 1",
                "recipe_id": "uuid-1",
                "text": "Description of Recipe 1",
                "model_dump": {
                    "name": "Recipe 1",
                    "slug": "recipe-1",
                    "id": "uuid-1",
                    "rating": None,
                    "recipeIngredient": [],
                    "recipeInstructions": [],
                },
            },
        ),
        ScoredPoint(
            id=2,
            version=1,
            score=0.8,
            payload={
                "name": "Recipe 2",
                "recipe_id": "uuid-2",
                "text": "Description of Recipe 2",
                "model_dump": {
                    "name": "Recipe 2",
                    "slug": "recipe-2",
                    "id": "uuid-2",
                    "rating": None,
                    "recipeIngredient": [],
                    "recipeInstructions": [],
                },
            },
        ),
    ]

    context = populate_context(hits)

    assert "[RECIPE_START]" in context
    assert "RecipeName: Recipe 1" in context
    assert "RecipeID: uuid-1" in context
    # Description is no longer in the context
    # assert "Description of Recipe 1" in context
    assert "RecipeName: Recipe 2" in context
    assert "Rating: None" in context
    assert "Ingredients:" in context
    assert "Instructions:" in context


def test_populate_messages():
    """Test message population."""
    hits = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={
                "name": "Recipe 1",
                "recipe_id": "uuid-1",
                "text": "Description",
                "model_dump": {
                    "name": "Recipe 1",
                    "slug": "recipe-1",
                    "id": "uuid-1",
                    "rating": None,
                    "recipeIngredient": [],
                    "recipeInstructions": [],
                },
            },
        )
    ]
    query = "What can I cook?"

    mock_prompt_manager = MagicMock()
    mock_prompt = MagicMock()
    mock_prompt.compile.return_value = [
        {"role": "system", "content": "System Prompt"},
        {"role": "user", "content": "User Message"},
    ]
    mock_prompt_manager.get_prompt.return_value = mock_prompt

    messages = populate_messages(query, hits, mock_prompt_manager)

    assert messages.messages_count == 2
    assert messages.messages[0]["role"] == "system"
    assert messages.messages[0]["content"] == "System Prompt"
    assert messages.messages[1]["role"] == "user"
    assert messages.messages[1]["content"] == "User Message"

    mock_prompt_manager.get_prompt.assert_any_call(PromptType.CHAT_GENERATION)

    # Check compile call on the prompt object
    expected_context = (
        "[RECIPE_START]\n"
        "RecipeName: Recipe 1\n"
        "RecipeID: uuid-1\n"
        "Rating: None\n"
        "Ingredients:\n"
        "Instructions:\n"
        "[RECIPE_END]\n"
    )
    mock_prompt.compile.assert_called_with(
        external_url=settings.mealie_external_url,
        context_text=expected_context,
        query="What can I cook?",
    )
