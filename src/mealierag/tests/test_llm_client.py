from mealierag.api import ChatMessages
from mealierag.llm_client import OllamaClient


def test_ollama_client_init(mock_ollama_client):
    """Test initialization."""
    client = OllamaClient("http://test")
    assert client.url == "http://test"


def test_ollama_client_streaming_chat(mock_ollama_client):
    """Test streaming chat."""
    client = OllamaClient("http://test")

    messages = ChatMessages(messages=[{"role": "user", "content": "hello"}])

    mock_ollama_client.chat.return_value = iter(
        [{"message": {"content": "chunk1"}}, {"message": {"content": "chunk2"}}]
    )

    response = client.streaming_chat(messages, "model", 0.5, 42)

    result = list(response)

    mock_ollama_client.chat.assert_called_with(
        model="model",
        messages=messages.messages,
        stream=True,
        options={"temperature": 0.5, "seed": 42},
    )
    assert result == ["chunk1", "chunk2"]


def test_ollama_client_chat(mock_ollama_client):
    """Test non-streaming chat."""
    client = OllamaClient("http://test")

    messages = ChatMessages(messages=[{"role": "user", "content": "hello"}])

    mock_ollama_client.chat.return_value = {"message": {"content": "response"}}

    response = client.chat(messages, "model", 0.5, 42)

    mock_ollama_client.chat.assert_called_with(
        model="model",
        messages=messages.messages,
        stream=False,
        options={"temperature": 0.5, "seed": 42},
        format=None,
    )
    assert response == "response"


def test_ollama_client_embed(mock_ollama_client):
    """Test embed."""
    client = OllamaClient("http://test")

    mock_ollama_client.embed.return_value = {"embeddings": [[0.1]]}

    response = client.embed(model="model", input="text")

    mock_ollama_client.embed.assert_called_with(model="model", input="text")
    assert response == {"embeddings": [[0.1]]}
