from unittest.mock import MagicMock

from qdrant_client.http.models import ScoredPoint

from mealierag.models import QueryExtraction
from mealierag.run_qa_ui import chat_fn, print_hits


def test_print_hits():
    """Test hits table formatting."""
    hits = [
        ScoredPoint(
            id=1,
            version=1,
            score=1.0,
            payload={
                "name": "Recipe 1",
                "rating": 5,
                "tags": ["t1"],
                "category": "c1",
                "recipe_id": "123",
            },
        )
    ]

    table = print_hits(hits)
    # The new format includes a markdown link so we check for that or a partial match that makes sense
    assert "[Recipe 1]" in table
    assert "(http" in table
    assert "123)" in table
    assert "| 5 | t1 | c1 |" in table


def test_chat_fn(mocker):
    """Test chat generator function."""
    mock_service_instance = MagicMock()
    mocker.patch("mealierag.run_qa_ui.service", mock_service_instance)

    mock_service_instance.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service_instance.retrieve_recipes.return_value = [
        ScoredPoint(
            id=1, version=1, score=1.0, payload={"name": "Recipe 1", "recipe_id": "123"}
        )
    ]
    mock_service_instance.populate_messages.return_value = []

    # Mock chat stream
    mock_service_instance.chat.return_value = iter(["Hello"])

    generator = chat_fn("test message", [])

    responses = list(generator)

    # Check progression of messages
    assert any("Consulting" in r for r, _ in responses)
    assert any("Finding" in r for r, _ in responses)
    assert any("Hello" in r for r, _ in responses)
    assert "Recipe 1" in responses[-1][1].value

    mock_service_instance.generate_queries.assert_called_once()
    mock_service_instance.retrieve_recipes.assert_called_once()
    mock_service_instance.chat.assert_called_once()


def test_chat_fn_no_results(mocker):
    """Test chat function with no results."""
    mock_service_instance = MagicMock()
    mocker.patch("mealierag.run_qa_ui.service", mock_service_instance)

    mock_service_instance.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service_instance.retrieve_recipes.return_value = []

    generator = chat_fn("test message", [])
    responses = list(generator)

    assert any("couldn't find" in r for r, _ in responses)
