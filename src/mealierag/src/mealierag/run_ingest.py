"""
Ingest Mealie recipes into the vector database.
"""

import logging

from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import LLMProvider, settings
from .embeddings import get_embedding
from .llm_client import OllamaClient, OpenAIClient
from .mealie import fetch_full_recipes
from .vectordb import get_vector_db_client

logger = logging.getLogger(__name__)


def main():
    # 1. Initialize Clients
    logger.info(f"Connecting to Qdrant at {settings.vectordb_url}...")
    vector_db_client = get_vector_db_client(settings.vectordb_url)

    if settings.llm_provider == LLMProvider.OLLAMA:
        llm_client = OllamaClient(base_url=settings.llm_base_url)
    elif settings.llm_provider == LLMProvider.OPENAI:
        llm_client = OpenAIClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    # 2. Get Data
    # TODO: We fetch full recipes to memory. This is not scalable. We should process them in batches.
    recipes = fetch_full_recipes(
        settings.mealie_api_url, settings.mealie_token.get_secret_value()
    )

    # 3. Create Collection if not exists
    logger.info("Determining embedding dimension...")
    dummy_text = "test"
    dummy_embedding = get_embedding([dummy_text], llm_client, settings)[0]
    vector_size = len(dummy_embedding)
    logger.info(f"Embedding dimension: {vector_size}")

    if vector_db_client.collection_exists(settings.vectordb_collection_name):
        if settings.delete_collection_if_exists:
            logger.info(
                f"Collection '{settings.vectordb_collection_name}' already exists. Recreating..."
            )
            vector_db_client.delete_collection(settings.vectordb_collection_name)
        else:
            raise Exception(
                f"Collection '{settings.vectordb_collection_name}' already exists."
            )

    logger.info(f"Creating collection '{settings.vectordb_collection_name}'...")
    vector_db_client.create_collection(
        collection_name=settings.vectordb_collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # 4. Process and Upsert
    logger.info("Processing and indexing recipes...")
    points = []
    for idx, r in enumerate(recipes):
        # Generate embedding
        embedding = get_embedding([r.get_text_for_embedding()], llm_client, settings)[0]

        # Create Point
        # We use idx as ID (integer). In production, consider UUIDs to avoid collisions in future ingestions.
        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "recipe_id": r.id,
                "slug": r.slug,
                "name": r.name,
                "category": r.recipeCategory,
                "tags": r.tags,
                "rating": r.rating,
                "text": r.get_text_for_embedding(),
            },
        )
        points.append(point)

    # Upsert batch
    vector_db_client.upsert(
        collection_name=settings.vectordb_collection_name, points=points
    )
    logger.info(f"Successfully indexed {len(points)} recipes.")


if __name__ == "__main__":
    main()
