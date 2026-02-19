"""
Enrich a raw recipe dump with normalised ingredients and LLM-inferred properties.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from mealierag.config import LLMProvider, settings
from mealierag.ingest import enrich_recipe_properties, normalize_ingredients
from mealierag.llm_client import OllamaClient, OpenAIClient
from mealierag.models import Recipes
from mealierag.prompts import LangfusePromptManager, PromptType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_INPUT_PATH = (
    Path("../raw_data") / f"{datetime.now().strftime('%Y%m%d')}_recipes.json"
)
DEFAULT_OUTPUT_PATH = (
    Path("../enriched") / f"{datetime.now().strftime('%Y%m%d')}_recipes_enriched.json"
)


def build_llm_client(provider: LLMProvider):
    """Instantiate LLM client based on the configured *provider*."""
    if provider == LLMProvider.OLLAMA:
        return OllamaClient(base_url=settings.llm_base_url)
    if provider == LLMProvider.OPENAI:
        return OpenAIClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )
    raise ValueError(f"Unknown LLM provider: {provider}")


def main(input_path: Path, output_path: Path) -> None:
    """Load, enrich, and persist recipes."""
    logger.info("Loading recipes from %s...", input_path)
    recipes = Recipes.model_validate_json(input_path.read_text())
    logger.info("Loaded %d recipes.", len(recipes))

    llm_client = build_llm_client(settings.llm_provider)

    prompt_manager = LangfusePromptManager()
    normalize_prompt = prompt_manager.get_prompt(
        PromptType.INGEST_NORMALIZE_INGREDIENTS
    ).compile()
    logger.info("Loaded 'ingest-normalize-ingredients' prompt from Langfuse.")
    enrich_prompt = prompt_manager.get_prompt(
        PromptType.INGEST_ENRICH_RECIPES
    ).compile()
    logger.info("Loaded 'ingest-enrich-recipes' prompt from Langfuse.")

    for idx, r in enumerate(recipes):
        logger.info("Processing recipe %d / %d: %s", idx + 1, len(recipes), r.name)
        r = normalize_ingredients(r, llm_client, normalize_prompt)
        r = enrich_recipe_properties(r, llm_client, enrich_prompt)

    logger.info("Writing enriched recipes to %s...", output_path)
    output_path.write_text(recipes.model_dump_json(indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich recipes with normalized ingredients and properties."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the raw recipes JSON. Defaults to {DEFAULT_INPUT_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination for the enriched recipes JSON. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()
    main(args.input_path, args.output_path)
