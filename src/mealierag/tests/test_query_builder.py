from unittest.mock import MagicMock

from mealierag.query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder


def test_default_query_builder():
    """Test DefaultQueryBuilder."""
    builder = DefaultQueryBuilder()
    queries = builder("test query")
    assert queries == ["test query"]


def test_multi_query_query_builder(mock_ollama_client):
    """Test MultiQueryQueryBuilder."""
    builder = MultiQueryQueryBuilder(
        ollama_client=mock_ollama_client, model="test-model", temperature=0.7, seed=42
    )

    # Mock LLM response
    response_text = "1. Query 1\n- Query 2\nQuery 3"
    mock_ollama_client.chat.return_value = response_text

    queries = builder("original query")

    # Check if chat was called correctly
    mock_ollama_client.chat.assert_called_once()
    call_kwargs = mock_ollama_client.chat.call_args.kwargs
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["seed"] == 42

    # Check parsing logic
    assert len(queries) == 3
    assert "Query 1" in queries
    assert "Query 2" in queries
    assert "Query 3" in queries


def test_multi_query_parsing():
    """Test parsing logic specifically."""
    builder = MultiQueryQueryBuilder(MagicMock(), "model", 0.1, 1)

    raw_response = """
    1. First Query
    2. Second Query
    - Third Query
    * Fourth Query
    Fifth Query
    """

    queries = builder._parse_response(raw_response)

    expected = [
        "First Query",
        "Second Query",
        "Third Query",
        "Fourth Query",
        "Fifth Query",
    ]

    assert queries == expected
