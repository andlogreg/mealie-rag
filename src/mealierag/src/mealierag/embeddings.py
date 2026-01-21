"""
Embeddings module.

Contains functions to generate embeddings.
"""

import ollama

from .config import Settings


def get_embedding(
    text: str, ollama_client: ollama.Client, settings: Settings
) -> list[float]:
    """
    Generate embedding for a given text.

    Args:
        text: Text to generate embedding for
        ollama_client: Ollama client
        settings: Settings

    Returns:
        Embedding for the given text
    """
    try:
        response = ollama_client.embed(model=settings.embedding_model, input=[text])
        return response["embeddings"][0]
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")
