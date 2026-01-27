from unittest.mock import MagicMock

import pytest

from mealierag.config import Settings


@pytest.fixture
def mock_env(monkeypatch):
    """
    Mock environment variables for testing.
    """
    monkeypatch.setenv("MEALIE_API_URL", "http://test-mealie/api/recipes")
    monkeypatch.setenv("MEALIE_TOKEN", "test-token")
    monkeypatch.setenv("VECTORDB_URL", "http://test-qdrant")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-ollama")


@pytest.fixture
def mock_settings(mock_env):
    """
    Return a Settings object with mocked environment variables.
    """
    return Settings()


@pytest.fixture
def mock_qdrant_client(mocker):
    """
    Mock the QdrantClient.
    """
    mock_client = MagicMock()
    mocker.patch("mealierag.vectordb.QdrantClient", return_value=mock_client)
    return mock_client


@pytest.fixture
def mock_ollama_client(mocker):
    """
    Mock the OllamaClient.
    """
    mock_client = MagicMock()
    mocker.patch("mealierag.llm_client.ollama.Client", return_value=mock_client)
    return mock_client
