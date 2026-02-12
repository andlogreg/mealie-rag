from unittest.mock import MagicMock

from mealierag.models import QueryExtraction
from mealierag.query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder


def test_default_query_builder():
    """Test DefaultQueryBuilder."""
    builder = DefaultQueryBuilder()
    result = builder("test query")
    assert result.expanded_queries == ["test query"]


def test_multi_query_query_builder(mock_ollama_client):
    """Test MultiQueryQueryBuilder."""
    builder = MultiQueryQueryBuilder(
        llm_client=mock_ollama_client,
        model="test-model",
        temperature=0.7,
        seed=42,
        prompt_manager=MagicMock(),
    )

    # Mock LLM response
    mock_ollama_client.chat.return_value = QueryExtraction(
        expanded_queries=["Query 1", "Query 2", "Query 3"]
    )

    response = builder("original query")
    queries = response.expanded_queries

    # Check if chat was called correctly
    mock_ollama_client.chat.assert_called_once()
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["seed"] == 42
    assert call_kwargs["response_model"] == QueryExtraction

    # Check parsing logic
    assert len(queries) == 3
    assert "Query 1" in queries
    assert "Query 2" in queries
    assert "Query 3" in queries
