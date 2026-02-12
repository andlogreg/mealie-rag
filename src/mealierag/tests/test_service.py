from unittest.mock import MagicMock

import pytest

from mealierag.api import ChatMessages
from mealierag.models import QueryExtraction
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


@pytest.fixture
def mock_dependencies(mocker):
    return {
        "llm_client": MagicMock(),
        "vector_db_client": MagicMock(),
        "prompt_manager": MagicMock(),
        "query_builder": MagicMock(),
        "retrieve_results_fn": MagicMock(),
    }


def test_service_initialization(mock_dependencies):
    """Test service initialization with injected dependencies"""
    service = MealieRAGService(**mock_dependencies)
    assert service.llm_client == mock_dependencies["llm_client"]
    assert service.vector_db_client == mock_dependencies["vector_db_client"]
    assert service.prompt_manager == mock_dependencies["prompt_manager"]
    assert service.query_builder == mock_dependencies["query_builder"]
    assert service._retrieve_results == mock_dependencies["retrieve_results_fn"]


def test_generate_queries(mock_dependencies):
    """Test generating queries"""
    service = MealieRAGService(**mock_dependencies)

    expected_extraction = QueryExtraction(expanded_queries=["test query"])
    mock_dependencies["query_builder"].return_value = expected_extraction

    result = service.generate_queries("test query")

    assert result == expected_extraction
    mock_dependencies["query_builder"].assert_called_once_with("test query")


def test_retrieve_recipes(mock_dependencies, mock_embedding_func):
    """Test retrieving recipes"""
    service = MealieRAGService(**mock_dependencies)

    mock_dependencies["retrieve_results_fn"].return_value = [
        MagicMock(id="1", payload={"name": "Recipe 1"})
    ]

    extraction = QueryExtraction(expanded_queries=["test query"])
    recipes = service.retrieve_recipes(extraction)

    assert len(recipes) == 1
    assert recipes[0].payload["name"] == "Recipe 1"

    mock_embedding_func.assert_called_once()
    mock_dependencies["retrieve_results_fn"].assert_called_once()


def test_populate_messages(mock_dependencies):
    """Test populate_messages delegates to prompt_manager via util"""
    # service = MealieRAGService(**mock_dependencies)

    # TODO: Test populate_messages
    pass


def test_chat(mock_dependencies):
    """Test chat delegates to ollama_client"""
    service = MealieRAGService(**mock_dependencies)
    messages = ChatMessages(messages=[{"role": "user", "content": "hi"}])

    service.chat(messages)

    mock_dependencies["llm_client"].streaming_chat.assert_called_once()


def test_check_health(mock_dependencies):
    """Test health check"""
    service = MealieRAGService(**mock_dependencies)

    mock_dependencies["vector_db_client"].collection_exists.return_value = True
    assert service.check_health() is True

    mock_dependencies["vector_db_client"].collection_exists.return_value = False
    assert service.check_health() is False


def test_create_mealie_rag_service(mocker):
    """Test factory function creates service with correct dependencies."""
    from mealierag.config import LLMProvider

    mock_settings = MagicMock()
    mock_settings.search_strategy = SearchStrategy.SIMPLE
    mock_settings.llm_provider = LLMProvider.OLLAMA
    mock_settings.llm_base_url = "http://test-ollama"
    mock_settings.vectordb_url = "http://test-qdrant"

    mocker.patch("mealierag.service.OllamaClient")
    mocker.patch("mealierag.service.OpenAIClient")
    mocker.patch("mealierag.service.get_vector_db_client")
    mocker.patch("mealierag.service.LangfusePromptManager")
    mocker.patch("mealierag.service.DefaultQueryBuilder")

    from mealierag.service import create_mealie_rag_service

    # Test Ollama
    service = create_mealie_rag_service(mock_settings)

    assert isinstance(service, MealieRAGService)
    assert service.llm_client is not None
    assert service.vector_db_client is not None

    # Test OpenAI
    mock_settings.llm_provider = LLMProvider.OPENAI
    mock_settings.llm_api_key = MagicMock(get_secret_value=lambda: "test-key")

    service_openai = create_mealie_rag_service(mock_settings)
    assert isinstance(service_openai, MealieRAGService)
    assert service_openai.llm_client is not None
