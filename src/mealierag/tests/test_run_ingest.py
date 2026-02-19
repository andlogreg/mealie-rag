import pytest

from mealierag.config import LLMProvider
from mealierag.models import Recipe
from mealierag.run_ingest import main


def test_run_ingest_main(mocker, mock_settings, mock_qdrant_client):
    """Test ingest main function."""
    mocker.patch("mealierag.run_ingest.settings", mock_settings)
    mock_settings.llm_provider = LLMProvider.OLLAMA

    # Mock data
    mock_recipes = [Recipe(name="Test Recipe", slug="test", id="1")]
    mocker.patch("mealierag.run_ingest.fetch_full_recipes", return_value=mock_recipes)

    # Mock embeddings
    mock_get_embedding = mocker.patch(
        "mealierag.run_ingest.get_embedding", return_value=[[0.1, 0.2]]
    )

    # Mock Ollama client class
    mock_ollama_class = mocker.patch("mealierag.run_ingest.OllamaClient")
    mock_ollama_instance = mock_ollama_class.return_value

    # Mock vector DB client factory
    mocker.patch(
        "mealierag.run_ingest.get_vector_db_client", return_value=mock_qdrant_client
    )

    # Mock Langfuse prompt manager
    mocker.patch("mealierag.run_ingest.LangfusePromptManager")

    # Setup Qdrant behavior
    mock_qdrant_client.collection_exists.return_value = False

    main()

    # Verify client initialization
    mock_ollama_class.assert_called_once()

    # Verify logic
    mock_qdrant_client.create_collection.assert_called_once()
    mock_qdrant_client.upsert.assert_called_once()

    call_args = mock_qdrant_client.upsert.call_args
    assert call_args.kwargs["collection_name"] == mock_settings.vectordb_collection_name
    assert len(call_args.kwargs["points"]) == 1
    assert call_args.kwargs["points"][0].payload["name"] == "Test Recipe"
    assert "model_dump" in call_args.kwargs["points"][0].payload
    assert call_args.kwargs["points"][0].payload["model_dump"]["name"] == "Test Recipe"

    # Verify embedding called with correct client
    assert mock_get_embedding.call_args[0][1] == mock_ollama_instance


def test_run_ingest_recreate_collection(mocker, mock_settings, mock_qdrant_client):
    """Test recreating collection if it exists."""
    mocker.patch("mealierag.run_ingest.settings", mock_settings)
    mock_settings.delete_collection_if_exists = True
    mock_settings.llm_provider = LLMProvider.OLLAMA

    mocker.patch("mealierag.run_ingest.fetch_full_recipes", return_value=[])
    mocker.patch("mealierag.run_ingest.get_embedding", return_value=[[0.1]])
    mocker.patch("mealierag.run_ingest.OllamaClient")
    mocker.patch(
        "mealierag.run_ingest.get_vector_db_client", return_value=mock_qdrant_client
    )
    mocker.patch("mealierag.run_ingest.LangfusePromptManager")

    mock_qdrant_client.collection_exists.return_value = True

    main()

    mock_qdrant_client.delete_collection.assert_called_once_with(
        mock_settings.vectordb_collection_name
    )
    mock_qdrant_client.create_collection.assert_called()


def test_run_ingest_existing_collection_error(
    mocker, mock_settings, mock_qdrant_client
):
    """Test error if collection exists and delete is False."""
    mocker.patch("mealierag.run_ingest.settings", mock_settings)
    mock_settings.delete_collection_if_exists = False
    mock_settings.llm_provider = LLMProvider.OLLAMA

    mocker.patch("mealierag.run_ingest.fetch_full_recipes", return_value=[])
    mocker.patch("mealierag.run_ingest.get_embedding", return_value=[[0.1]])
    mocker.patch("mealierag.run_ingest.OllamaClient")
    mocker.patch(
        "mealierag.run_ingest.get_vector_db_client", return_value=mock_qdrant_client
    )
    mocker.patch("mealierag.run_ingest.LangfusePromptManager")

    mock_qdrant_client.collection_exists.return_value = True

    with pytest.raises(Exception, match="already exists"):
        main()
