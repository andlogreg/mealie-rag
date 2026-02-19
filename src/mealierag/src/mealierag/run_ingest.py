"""
Ingest Mealie recipes into the vector database.
"""

import logging

from qdrant_client.models import Distance, VectorParams

from .config import LLMProvider, settings
from .embeddings import get_embedding
from .ingest import (
    create_point_from_recipe,
    enrich_recipe_properties,
    normalize_ingredients,
)
from .llm_client import OllamaClient, OpenAIClient
from .mealie import fetch_full_recipes
from .prompts import LangfusePromptManager, PromptType
from .vectordb import get_vector_db_client

logger = logging.getLogger(__name__)


def main():
    # 1. Initialize Clients
    logger.info("Connecting to Qdrant...")
    vector_db_client = get_vector_db_client(
        url=settings.vectordb_url, path=settings.vectordb_path
    )

    if settings.llm_provider == LLMProvider.OLLAMA:
        llm_client = OllamaClient(base_url=settings.llm_base_url)
    elif settings.llm_provider == LLMProvider.OPENAI:
        llm_client = OpenAIClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    # 2. Load prompts from Langfuse
    prompt_manager = LangfusePromptManager()
    normalize_prompt = prompt_manager.get_prompt(
        PromptType.INGEST_NORMALIZE_INGREDIENTS
    ).compile()
    logger.info("Loaded 'ingest-normalize-ingredients' prompt from Langfuse.")
    enrich_prompt = prompt_manager.get_prompt(
        PromptType.INGEST_ENRICH_RECIPES
    ).compile()
    logger.info("Loaded 'ingest-enrich-recipes' prompt from Langfuse.")

    # 3. Get Data
    # TODO: We fetch full recipes to memory. This is not scalable. We should process them in batches.
    recipes = fetch_full_recipes(
        settings.mealie_api_url,
        settings.mealie_token.get_secret_value() if settings.mealie_token else None,
    )

    # 4. Create Collection if not exists
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

    # 5. Process and Upsert
    logger.info("Processing and indexing recipes...")
    points = []
    for idx, r in enumerate(recipes):
        logger.info(f"Processing recipe {idx + 1}/{len(recipes)}: {r.name}...")
        r = normalize_ingredients(r, llm_client, system_prompt=normalize_prompt)
        r = enrich_recipe_properties(r, llm_client, system_prompt=enrich_prompt)
        # Generate embedding
        embedding = get_embedding([r.get_text_for_embedding()], llm_client, settings)[0]

        # Create Point
        point = create_point_from_recipe(r, embedding)
        points.append(point)

    # Upsert batch
    vector_db_client.upsert(
        collection_name=settings.vectordb_collection_name, points=points
    )
    logger.info(f"Successfully indexed {len(points)} recipes.")


if __name__ == "__main__":
    main()
