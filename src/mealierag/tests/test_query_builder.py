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
    # We expect 1 call for initial query generation
    # And then 1 call for each generated query (3 in this case)
    mock_ollama_client.chat.side_effect = [
        QueryExtraction(expanded_queries=["Query 1", "Query 2", "Query 3"]),
        "Refined Query 1",
        "Refined Query 2",
        "Refined Query 3",
    ]

    response = builder("original query")
    queries = response.expanded_queries

    # Check if chat was called correctly (1 initial + 3 refinements)
    assert mock_ollama_client.chat.call_count == 4

    # Check the first call arguments
    first_call_kwargs = mock_ollama_client.chat.call_args_list[0].kwargs
    assert first_call_kwargs["model"] == "test-model"
    assert first_call_kwargs["temperature"] == 0.7
    assert first_call_kwargs["seed"] == 42
    assert first_call_kwargs["response_model"] == QueryExtraction

    # Check parsing logic
    assert len(queries) == 3
    assert "Refined Query 1" in queries
    assert "Refined Query 2" in queries
    assert "Refined Query 3" in queries
