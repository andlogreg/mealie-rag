"""
Ingest enriched recipes from a local JSON file into Qdrant.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from qdrant_client.models import Distance, VectorParams
from tqdm import tqdm

from mealierag.config import LLMProvider, settings
from mealierag.embeddings import get_embedding
from mealierag.ingest import create_point_from_recipe
from mealierag.llm_client import OllamaClient, OpenAIClient
from mealierag.models import Recipes
from mealierag.vectordb import get_vector_db_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = (
    Path("../enriched") / f"{datetime.now().strftime('%Y%m%d')}_recipes_enriched.json"
)


def load_recipes(path: Path) -> Recipes:
    """Load and validate a ``Recipes`` collection from *path*."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return Recipes.model_validate_json(path.read_text())


def build_llm_client(provider: LLMProvider):
    """Instantiate LLM client based on the configured *provider*."""
    if provider == LLMProvider.OLLAMA:
        return OllamaClient(base_url=settings.llm_base_url)
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(
            api_key=settings.llm_api_key.get_secret_value()
            if settings.llm_api_key
            else "",
            base_url=settings.llm_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def main(dataset_path: Path) -> None:
    """Index recipes from *dataset_path* into Qdrant."""
    # 1. Connect to Qdrant
    if settings.vectordb_path:
        logger.info("Connecting to local Qdrant at %s...", settings.vectordb_path)
    else:
        logger.info("Connecting to Qdrant at %s...", settings.vectordb_url)

    vector_db_client = get_vector_db_client(
        url=settings.vectordb_url,
        path=settings.vectordb_path,
    )
    llm_client = build_llm_client(settings.llm_provider)

    # 2. Load recipes
    logger.info("Loading recipes from %s...", dataset_path)
    recipes = load_recipes(dataset_path)
    logger.info("Loaded %d recipes.", len(recipes))

    # 3. Probe embedding dimension
    logger.info("Determining embedding dimension...")
    dummy_embedding = get_embedding(["test"], llm_client, settings)[0]
    vector_size = len(dummy_embedding)
    logger.info("Embedding dimension: %d", vector_size)

    # 4. Create or recreate the Qdrant collection
    collection_name = settings.vectordb_collection_name
    if vector_db_client.collection_exists(collection_name):
        if not settings.delete_collection_if_exists:
            raise RuntimeError(
                f"Collection '{collection_name}' already exists. "
                "Set DELETE_COLLECTION_IF_EXISTS=true to overwrite it."
            )
        logger.info(
            "Collection '%s' already exists — deleting and recreating...",
            collection_name,
        )
        vector_db_client.delete_collection(collection_name)

    logger.info("Creating collection '%s'...", collection_name)
    vector_db_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    # 5. Generate embeddings and upsert
    logger.info("Generating embeddings and indexing recipes...")
    points = []
    for r in tqdm(recipes, desc="Embedding recipes"):
        try:
            embedding = get_embedding(
                [r.get_text_for_embedding()], llm_client, settings
            )[0]
        except Exception as e:
            logger.error("Failed to embed recipe '%s': %s", r.name, e)
            continue
        points.append(create_point_from_recipe(r, embedding))

    if points:
        vector_db_client.upsert(collection_name=collection_name, points=points)
        logger.info("Successfully indexed %d recipes.", len(points))
    else:
        logger.warning("No recipes were indexed — check embedding errors above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest recipes from a JSON file into the vector database."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the enriched recipes JSON file. Defaults to {DEFAULT_DATASET_PATH}",
    )
    args = parser.parse_args()
    main(args.dataset_path)
