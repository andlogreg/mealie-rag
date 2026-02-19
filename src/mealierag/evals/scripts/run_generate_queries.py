"""
Generate queries for the recipes.
"""

import logging

from pydantic import BaseModel

from mealierag.api import ChatMessages
from mealierag.config import LLMProvider, settings
from mealierag.llm_client import OllamaClient, OpenAIClient
from mealierag.models import Recipes
from mealierag.prompts import LangfusePromptManager, PromptType

logger = logging.getLogger(__name__)


class GeneratedQueries(BaseModel):
    queries: list[str]


def main(input_dataset: str, output_path: str):
    with open(input_dataset, "r") as f:
        data = f.read()
    recipes = Recipes.model_validate_json(data)

    if settings.llm_provider == LLMProvider.OLLAMA:
        llm_client = OllamaClient(base_url=settings.llm_base_url)
    elif settings.llm_provider == LLMProvider.OPENAI:
        llm_client = OpenAIClient(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    prompt_manager = LangfusePromptManager()
    system_prompt = prompt_manager.get_prompt(
        PromptType.DATA_QUERY_GENERATION
    ).compile()

    all_queries = []
    for idx, recipe in enumerate(recipes):
        logger.info(f"Processing recipe {idx + 1}/{len(recipes)}")
        user_input = recipe.get_text_for_context()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]
        chat_messages = ChatMessages(messages=messages, prompt=None)
        temperature = 0.9
        response = llm_client.chat(
            chat_messages=chat_messages,
            model=settings.llm_model,
            temperature=temperature,
            seed=settings.llm_seed,
            response_model=GeneratedQueries,
        )
        # recipe.normalizedRecipeIngredients = response
        all_queries.extend(response.queries)

    final_queries = GeneratedQueries(queries=all_queries)
    with open(output_path, "w") as f:
        f.write(final_queries.model_dump_json(indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate queries for the recipes.")
    parser.add_argument(
        "input_dataset", type=str, help="Path to the input JSON dataset."
    )
    parser.add_argument("output_path", type=str, help="Path to the output JSON file.")
    args = parser.parse_args()
    main(args.input_dataset, args.output_path)
