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
