"""
Fetch all recipes from Mealie and print them.

Mostly for debugging purposes.
"""

import logging

from .config import settings
from .mealie import fetch_full_recipes

logger = logging.getLogger(__name__)


def main():
    recipes = fetch_full_recipes(settings.mealie_api_url, settings.mealie_token)
    logger.info(f"Successfully fetched {len(recipes)} recipes.")

    if recipes:
        logger.info("Here are the first 5 recipes:")
        for recipe in recipes[:5]:
            logger.info("##################### RECIPE #####################")
            logger.info(recipe.get_text_for_embedding())
            logger.info("\n")


if __name__ == "__main__":
    main()
