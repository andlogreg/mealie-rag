from unittest.mock import MagicMock

import pytest
from qdrant_client import models

from mealierag.vectordb import (
    get_vector_db_client,
    retrieve_results_rrf,
    retrieve_results_simple,
)


def test_get_vector_db_client(mock_qdrant_client):
    """Test getting Qdrant client."""
    client = get_vector_db_client("http://test")

    assert client == mock_qdrant_client


def test_retrieve_results_simple(mock_qdrant_client):
    """Test simple retrieval."""
    mock_results = MagicMock()
    mock_results.points = ["result1", "result2"]
    mock_qdrant_client.query_points.return_value = mock_results

    results = retrieve_results_simple(
        [[0.1, 0.2]], mock_qdrant_client, "test_collection", k=2
    )

    mock_qdrant_client.query_points.assert_called_with(
        collection_name="test_collection", query=[0.1, 0.2], limit=2, query_filter=None
    )
    assert results == ["result1", "result2"]


def test_retrieve_results_simple_error(mock_qdrant_client):
    """Test simple retrieval error with multiple vectors."""
    with pytest.raises(
        ValueError, match="Simple retrieval supports exactly one query vector"
    ):
        retrieve_results_simple([[0.1], [0.2]], mock_qdrant_client, "test_collection")


def test_retrieve_results_rrf(mock_qdrant_client):
    """Test RRF retrieval."""
    mock_results = MagicMock()
    mock_results.points = ["result1", "result2"]
    mock_qdrant_client.query_points.return_value = mock_results

    query_vectors = [[0.1, 0.2], [0.3, 0.4]]

    results = retrieve_results_rrf(
        query_vectors, mock_qdrant_client, "test_collection", k=2
    )

    # Generally check if call args are correct
    call_args = mock_qdrant_client.query_points.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    assert call_args.kwargs["query"].fusion == models.Fusion.RRF
    assert call_args.kwargs["limit"] == 2
    assert len(call_args.kwargs["prefetch"]) == 2

    assert results == ["result1", "result2"]


def test_build_filters_none():
    """Test _build_filters with None."""
    from mealierag.vectordb import _build_filters

    assert _build_filters(None) is None


def test_build_filters_empty():
    """Test _build_filters with empty extraction."""
    from mealierag.models import QueryExtraction
    from mealierag.vectordb import _build_filters

    qe = QueryExtraction(expanded_queries=["q1"])
    assert _build_filters(qe) is None


def test_build_filters_negative_ingredients():
    """Test _build_filters with negative ingredients."""
    from mealierag.models import QueryExtraction
    from mealierag.vectordb import _build_filters

    qe = QueryExtraction(
        expanded_queries=["q1"], negative_ingredients=["onion", "garlic"]
    )
    filters = _build_filters(qe)

    assert filters is not None
    assert filters.must is None
    assert len(filters.must_not) == 2
    assert filters.must_not[0].key == "ingredients"
    assert filters.must_not[0].match.text == "onion"


def test_build_filters_ratings():
    """Test _build_filters with rating range."""
    from mealierag.models import QueryExtraction
    from mealierag.vectordb import _build_filters

    qe = QueryExtraction(expanded_queries=["q1"], min_rating=4, max_rating=5)
    filters = _build_filters(qe)

    assert filters is not None
    assert filters.must_not is None
    assert len(filters.must) == 1
    assert filters.must[0].key == "rating"
    assert filters.must[0].range.gte == 4
    assert filters.must[0].range.lt == 5


def test_build_filters_all():
    """Test _build_filters with all conditions."""
    from mealierag.models import QueryExtraction
    from mealierag.vectordb import _build_filters

    qe = QueryExtraction(
        expanded_queries=["q1"],
        negative_ingredients=["onion"],
        min_rating=3,
    )
    filters = _build_filters(qe)

    assert filters is not None
    assert len(filters.must_not) == 1
    assert len(filters.must) == 1
