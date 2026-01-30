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
            },
        ),
    ]

    context = populate_context(hits)

    assert "[RECIPE_START]" in context
    assert "RecipeName: Recipe 1" in context
    assert "RecipeID: uuid-1" in context
    assert "Description of Recipe 1" in context
    assert "RecipeName: Recipe 2" in context


def test_populate_messages():
    """Test message population."""
    hits = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.9,
            payload={"name": "Recipe 1", "recipe_id": "uuid-1", "text": "Description"},
        )
    ]
    query = "What can I cook?"

    mock_prompt_manager = MagicMock()
    mock_prompt_manager.get_prompt.side_effect = [
        "System Prompt",
        "User Message",
    ]

    messages = populate_messages(query, hits, mock_prompt_manager)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System Prompt"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User Message"

    mock_prompt_manager.get_prompt.assert_any_call(
        PromptType.GENERATION_SYSTEM, external_url=settings.mealie_external_url
    )
