"""
Embeddings module.

Contains functions to generate embeddings.
"""

import logging

import ollama

from .config import Settings

logger = logging.getLogger(__name__)


def get_embedding(
    texts: list[str], ollama_client: ollama.Client, settings: Settings
) -> list[list[float]]:
    """
    Generate embedding for a list of texts.

    Args:
        texts: List of texts to generate embedding for
        ollama_client: Ollama client
        settings: Settings

    Returns:
        List of embeddings for the given texts
    """
    try:
        logger.debug("Generating embedding", extra={"texts": texts})
        response = ollama_client.embed(model=settings.embedding_model, input=texts)
        return response["embeddings"]
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")
