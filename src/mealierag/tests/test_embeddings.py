import pytest

from mealierag.embeddings import get_embedding


def test_get_embedding(mock_settings, mock_ollama_client):
    """Test embedding generation."""
    texts = ["test1", "test2"]

    mock_ollama_client.embed.return_value = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}

    embeddings = get_embedding(texts, mock_ollama_client, mock_settings)

    assert len(embeddings) == 2
    assert embeddings[0] == [0.1, 0.2]
    mock_ollama_client.embed.assert_called_with(
        model=mock_settings.embedding_model, input=texts
    )


def test_get_embedding_error(mock_settings, mock_ollama_client):
    """Test embedding generation error."""
    mock_ollama_client.embed.side_effect = Exception("Ollama Error")

    with pytest.raises(Exception, match="Error generating embedding"):
        get_embedding(["test"], mock_ollama_client, mock_settings)
