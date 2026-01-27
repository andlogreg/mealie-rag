from qdrant_client.http.models import ScoredPoint

from mealierag.chat import (
    SYSTEM_PROMPT,
    USER_MESSAGE,
    populate_context,
    populate_messages,
)
from mealierag.config import settings


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

    messages = populate_messages(query, hits)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT.format(
        external_url=settings.mealie_external_url
    )
    assert messages[1]["role"] == "user"

    context_text = populate_context(hits)
    expected_user_msg = USER_MESSAGE.format(context_text=context_text, query=query)

    assert messages[1]["content"] == expected_user_msg
