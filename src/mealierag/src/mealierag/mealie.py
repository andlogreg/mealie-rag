"""
Mealie module.

Contains functions to interact with Mealie, including fetching recipes and recipe details.
"""

import logging

import requests

from .models import Recipe, RecipeResponse

logger = logging.getLogger(__name__)


def fetch_recipes(
    mealie_api_url: str, mealie_token: str, per_page: int = 10
) -> list[Recipe]:
    """
    Fetch all recipes from Mealie.

    Args:
        mealie_api_url: Mealie API URL
        mealie_token: Mealie token

    Returns:
        List of recipes
    """
    logger.info(f"Fetching recipes from {mealie_api_url}...")
    all_recipes: list[Recipe] = []
    page = 1

    try:
        while True:
            logger.info(f"Fetching page {page}...")
            response = requests.get(
                mealie_api_url,
                headers={"Authorization": f"Bearer {mealie_token}"},
                params={"page": page, "perPage": per_page},
            )
            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict) and "items" in data:
                try:
                    paginated_response = RecipeResponse(**data)
                    all_recipes.extend(paginated_response.items)

                    if page >= paginated_response.total_pages:
                        break

                    page += 1
                except Exception as validation_err:
                    raise Exception(
                        f"Validation error: {validation_err}"
                    ) from validation_err
            else:
                raise Exception(
                    f"Unexpected response format: {type(data)}, Full response: {data}"
                )

        logger.info(f"Fetched {len(all_recipes)} recipes.")
        return all_recipes
    except Exception as e:
        raise Exception(f"Error fetching recipes: {e}") from e


def fetch_full_recipe(recipe: Recipe, mealie_api_url: str, mealie_token: str) -> Recipe:
    """
    Fetch full recipe details from Mealie.

    Args:
        recipe: Recipe to fetch full details for
        mealie_api_url: Mealie API URL
        mealie_token: Mealie token

    Returns:
        Full recipe details
    """
    logger.info(f"Fetching recipe {recipe.name}...")
    try:
        response = requests.get(
            f"{mealie_api_url}/{recipe.id}",
            headers={"Authorization": f"Bearer {mealie_token}"},
        )
        response.raise_for_status()
        return Recipe(**response.json())
    except Exception as e:
        raise Exception(f"Error fetching recipe {recipe.id}: {e}") from e


def fetch_full_recipes(mealie_api_url: str, mealie_token: str) -> list[Recipe]:
    """
    Fetch all recipes with full details from Mealie.

    Args:
        mealie_api_url: Mealie API URL
        mealie_token: Mealie token

    Returns:
        List of recipes
    """
    recipes = fetch_recipes(mealie_api_url, mealie_token)
    return [
        fetch_full_recipe(recipe, mealie_api_url, mealie_token) for recipe in recipes
    ]
