from unittest.mock import MagicMock

import pytest

from mealierag.mealie import fetch_full_recipe, fetch_full_recipes, fetch_recipes
from mealierag.models import Recipe


def test_fetch_recipes_pagination(mocker):
    """
    Test fetching recipes with pagination.
    """
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"

    # Mock responses
    response1 = MagicMock()
    response1.json.return_value = {
        "page": 1,
        "per_page": 2,
        "total": 3,
        "total_pages": 2,
        "items": [
            {"name": "Recipe 1", "slug": "recipe-1", "id": "1"},
            {"name": "Recipe 2", "slug": "recipe-2", "id": "2"},
        ],
    }
    response1.raise_for_status.return_value = None

    response2 = MagicMock()
    response2.json.return_value = {
        "page": 2,
        "per_page": 2,
        "total": 3,
        "total_pages": 2,
        "items": [
            {"name": "Recipe 3", "slug": "recipe-3", "id": "3"},
        ],
    }
    response2.raise_for_status.return_value = None

    mock_get = mocker.patch("requests.get", side_effect=[response1, response2])

    recipes = fetch_recipes(base_url, token, per_page=2)

    assert len(recipes) == 3
    assert recipes[0].name == "Recipe 1"
    assert recipes[1].name == "Recipe 2"
    assert recipes[2].name == "Recipe 3"
    assert mock_get.call_count == 2


def test_fetch_recipes_error(mocker):
    """
    Test error handling when fetching recipes.
    """
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"

    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API Error")
    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(Exception, match="Error fetching recipes"):
        fetch_recipes(base_url, token)


def test_fetch_full_recipe(mocker):
    """
    Test fetching a full recipe.
    """
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"
    recipe_id = "1"
    recipe = Recipe(name="Simple", slug="simple", id=recipe_id)

    full_recipe_data = {
        "name": "Full Recipe",
        "slug": "full-recipe",
        "id": recipe_id,
        "description": "Full details",
    }

    mock_response = MagicMock()
    mock_response.json.return_value = full_recipe_data
    mock_response.raise_for_status.return_value = None

    mocker.patch("requests.get", return_value=mock_response)

    full_recipe = fetch_full_recipe(recipe, base_url, token)

    assert full_recipe.name == "Full Recipe"
    assert full_recipe.description == "Full details"


def test_fetch_full_recipes(mocker):
    """
    Test fetching all full recipes.
    """
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"

    mock_recipes = [
        Recipe(name="Recipe 1", slug="recipe-1", id="1"),
        Recipe(name="Recipe 2", slug="recipe-2", id="2"),
    ]
    mocker.patch("mealierag.mealie.fetch_recipes", return_value=mock_recipes)

    # Mock fetch_full_recipe
    mock_full_recipe = Recipe(name="Full Recipe", slug="full", id="999")
    mocker.patch("mealierag.mealie.fetch_full_recipe", return_value=mock_full_recipe)

    full_recipes = fetch_full_recipes(base_url, token)

    assert len(full_recipes) == 2
    assert full_recipes.items == [mock_full_recipe, mock_full_recipe]


def test_fetch_recipes_validation_error(mocker):
    """
    Test validation error when fetching recipes.
    """
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "items": [{"name": "Invalid Recipe"}]  # Missing required fields like slug
    }
    mock_response.raise_for_status.return_value = None

    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(Exception, match="Validation error"):
        fetch_recipes(base_url, token)


def test_fetch_recipes_unexpected_format(mocker):
    """Test unexpected response format."""
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"

    mock_response = MagicMock()
    # Return list instead of dict
    mock_response.json.return_value = ["item1", "item2"]
    mock_response.raise_for_status.return_value = None

    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(Exception, match="Unexpected response format"):
        fetch_recipes(base_url, token)


def test_fetch_full_recipe_error(mocker):
    """Test error fetching full recipe."""
    base_url = "http://test-mealie/api/recipes"
    token = "test-token"
    recipe = Recipe(name="Simple", slug="simple", id="1")

    mocker.patch("requests.get", side_effect=Exception("Boom"))

    with pytest.raises(Exception, match="Error fetching recipe 1"):
        fetch_full_recipe(recipe, base_url, token)
