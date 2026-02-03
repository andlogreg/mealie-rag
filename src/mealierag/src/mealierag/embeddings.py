"""
Embeddings module.

Contains functions to generate embeddings.
"""

import logging

from .config import Settings
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


def get_embedding(
    texts: list[str], llm_client: LLMClient, settings: Settings
) -> list[list[float]]:
    """
    Generate embedding for a list of texts.

    Args:
        texts: List of texts to generate embedding for
        llm_client: LLM client
        settings: Settings

    Returns:
        List of embeddings for the given texts
    """
    try:
        logger.debug("Generating embedding", extra={"texts": texts})
        response = llm_client.embed(model=settings.embedding_model, input=texts)
        return response["embeddings"]
    except Exception as e:
        raise Exception(f"Error generating embedding: {e}")
