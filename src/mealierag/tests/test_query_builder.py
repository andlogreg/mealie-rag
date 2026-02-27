from unittest.mock import MagicMock

from mealierag.models import QueryExtraction
from mealierag.query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder


def _make_builder(mock_client, enable_expand=True, enable_culinary_brainstorm=True):
    """Helper to construct a MultiQueryQueryBuilder with the given feature flags."""
    return MultiQueryQueryBuilder(
        llm_client=mock_client,
        model="test-model",
        temperature=0.7,
        seed=42,
        prompt_manager=MagicMock(),
        enable_expand=enable_expand,
        enable_culinary_brainstorm=enable_culinary_brainstorm,
    )


def test_default_query_builder():
    """Test DefaultQueryBuilder."""
    builder = DefaultQueryBuilder()
    result = builder("test query")
    assert result.expanded_queries == ["test query"]


def test_multi_query_builder_both_enabled(mock_ollama_client):
    """Expand=True, Brainstorm=True (default behaviour).

    Expects 1 LLM call for expansion + 1 per expanded query for brainstorm.
    """
    builder = _make_builder(mock_ollama_client)

    mock_ollama_client.chat.side_effect = [
        QueryExtraction(expanded_queries=["Query 1", "Query 2", "Query 3"]),
        "Refined Query 1",
        "Refined Query 2",
        "Refined Query 3",
    ]

    response = builder("original query")

    # 1 expansion + 3 brainstorm calls
    assert mock_ollama_client.chat.call_count == 4

    first_call_kwargs = mock_ollama_client.chat.call_args_list[0].kwargs
    assert first_call_kwargs["model"] == "test-model"
    assert first_call_kwargs["temperature"] == 0.7
    assert first_call_kwargs["seed"] == 42
    assert first_call_kwargs["response_model"] == QueryExtraction

    assert response.expanded_queries == [
        "Refined Query 1",
        "Refined Query 2",
        "Refined Query 3",
    ]


def test_multi_query_builder_expand_only(mock_ollama_client):
    """Expand=True, Brainstorm=False.

    Expects 1 LLM call for expansion and raw expanded queries returned unchanged.
    """
    builder = _make_builder(mock_ollama_client, enable_culinary_brainstorm=False)

    mock_ollama_client.chat.side_effect = [
        QueryExtraction(expanded_queries=["Query 1", "Query 2", "Query 3"]),
    ]

    response = builder("original query")

    # Only the expansion call
    assert mock_ollama_client.chat.call_count == 1
    assert response.expanded_queries == ["Query 1", "Query 2", "Query 3"]


def test_multi_query_builder_brainstorm_only(mock_ollama_client):
    """Expand=False, Brainstorm=True.

    The user input is passed as-is to brainstorm — expects 1 brainstorm LLM call.
    """
    builder = _make_builder(mock_ollama_client, enable_expand=False)

    mock_ollama_client.chat.side_effect = ["Brainstormed query"]

    response = builder("original query")

    # 0 expansion calls + 1 brainstorm call for the single passthrough query
    assert mock_ollama_client.chat.call_count == 1
    assert response.expanded_queries == ["Brainstormed query"]


def test_multi_query_builder_both_disabled(mock_ollama_client):
    """Expand=False, Brainstorm=False.

    No LLM calls at all; user input returned as the sole query.
    """
    builder = _make_builder(
        mock_ollama_client, enable_expand=False, enable_culinary_brainstorm=False
    )

    response = builder("original query")

    assert mock_ollama_client.chat.call_count == 0
    assert response.expanded_queries == ["original query"]
