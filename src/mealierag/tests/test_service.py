from unittest.mock import MagicMock

import pytest

from mealierag.config import settings
from mealierag.service import MealieRAGService, SearchStrategy


@pytest.fixture
def mock_embedding_func(mocker):
    return mocker.patch(
        "mealierag.service.get_embedding", return_value=[[0.1, 0.2, 0.3]]
    )


@pytest.fixture
def mock_retrieve_results(mocker):
    return mocker.patch(
        "mealierag.service.retrieve_results_simple",
        return_value=[MagicMock(id="1", payload={"name": "Recipe 1"})],
    )


def test_service_initialization(mock_settings, mock_qdrant_client, mock_ollama_client):
    """Test service initialization with default settings"""
    service = MealieRAGService()
    assert service.ollama_client is not None
    assert service.vector_db_client is not None


def test_generate_queries(mock_settings, mock_qdrant_client, mock_ollama_client):
    """Test generating queries with default strategy"""
    service = MealieRAGService()
    queries = service.generate_queries("test query")
    assert queries == ["test query"]


def test_retrieve_recipes(
    mock_settings,
    mock_qdrant_client,
    mock_ollama_client,
    mock_embedding_func,
    mock_retrieve_results,
):
    """Test retrieving recipes"""
    service = MealieRAGService()
    recipes = service.retrieve_recipes(["test query"])

    assert len(recipes) == 1
    assert recipes[0].payload["name"] == "Recipe 1"
    mock_embedding_func.assert_called_once()
    mock_retrieve_results.assert_called_once()


def test_check_health(mock_settings, mock_qdrant_client, mock_ollama_client):
    """Test health check"""
    mock_qdrant_client.collection_exists.return_value = True
    service = MealieRAGService()
    assert service.check_health() is True

    mock_qdrant_client.collection_exists.return_value = False
    assert service.check_health() is False


def test_multiquery_strategy(
    mock_settings, mock_qdrant_client, mock_ollama_client, mocker
):
    """Test multiquery strategy initialization"""
    # Temporarily switch strategy
    previous_strategy = settings.search_strategy
    settings.search_strategy = SearchStrategy.MULTIQUERY

    try:
        mocker.patch(
            "mealierag.service.MultiQueryQueryBuilder",
            return_value=lambda x: ["q1", "q2"],
        )

        service = MealieRAGService()
        queries = service.generate_queries("test")
        assert queries == ["q1", "q2"]
        assert service._retrieve_results.__name__ == "retrieve_results_rrf"

    finally:
        settings.search_strategy = previous_strategy
