"""
Ingest Mealie recipes into the vector database.
"""

import logging

import ollama
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings
from .embeddings import get_embedding
from .mealie import fetch_full_recipes
from .vectordb import get_vector_db_client

# Client initialization
ollama_client = ollama.Client(host=settings.ollama_base_url)

logger = logging.getLogger(__name__)


def main():
    # 1. Initialize Qdrant Client
    logger.info(f"Connecting to Qdrant at {settings.qdrant_url}...")
    client = get_vector_db_client(settings.qdrant_url)

    # 2. Get Data
    # TODO: We fetch full recipes to memory. This is not scalable. We should process them in batches.
    recipes = fetch_full_recipes(settings.mealie_api_url, settings.mealie_token)

    # 3. Create Collection if not exists
    logger.info("Determining embedding dimension...")
    dummy_text = "test"
    dummy_embedding = get_embedding(dummy_text, ollama_client, settings)
    vector_size = len(dummy_embedding)
    logger.info(f"Embedding dimension: {vector_size}")

    if client.collection_exists(settings.collection_name):
        if settings.delete_collection_if_exists:
            logger.info(
                f"Collection '{settings.collection_name}' already exists. Recreating..."
            )
            client.delete_collection(settings.collection_name)
        else:
            raise Exception(f"Collection '{settings.collection_name}' already exists.")

    logger.info(f"Creating collection '{settings.collection_name}'...")
    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # 4. Process and Upsert
    logger.info("Processing and indexing recipes...")
    points = []
    for idx, r in enumerate(recipes):
        # Generate embedding
        embedding = get_embedding(r.get_text_for_embedding(), ollama_client, settings)

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
    client.upsert(collection_name=settings.collection_name, points=points)
    logger.info(f"Successfully indexed {len(points)} recipes.")


if __name__ == "__main__":
    main()
