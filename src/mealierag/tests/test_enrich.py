from unittest.mock import MagicMock

from pydantic import BaseModel

from mealierag.ingest import enrich_recipe_properties
from mealierag.llm_client import LLMClient
from mealierag.models import Recipe


def test_enrich_recipe_properties_fills_missing_category():
    # Arrange
    recipe = Recipe(
        name="Test Recipe",
        slug="test-recipe",
        recipeIngredients=[],
        recipeInstructions=[],
        # recipeCategory is empty by default
    )
    assert not recipe.recipeCategory

    mock_llm_client = MagicMock(spec=LLMClient)

    class MockResponse(BaseModel):
        recipeCategory: list[str] = ["Dinner"]
        tags: list[str] = ["Test"]
        tools: list[str] = ["Knife"]

    mock_llm_client.chat.return_value = MockResponse()

    enriched_recipe = enrich_recipe_properties(
        recipe, mock_llm_client, system_prompt="test prompt"
    )

    # Assert
    assert enriched_recipe.recipeCategory == ["Dinner"]
    mock_llm_client.chat.assert_called_once()

    call_args = mock_llm_client.chat.call_args
    _, kwargs = call_args
    DynamicModel = kwargs["response_model"]
    schema = DynamicModel.model_json_schema()
    properties = schema["properties"]
    assert "recipeCategory" in properties
    assert "tags" in properties
    assert "tools" in properties
    assert "rating" not in properties
    assert "is_healthy" in properties
    assert "total_time_minutes" in properties
    assert "method" in properties


def test_enrich_recipe_properties_skips_existing():
    # Arrange
    recipe = Recipe(
        name="Test Recipe",
        slug="test-recipe",
        recipeCategory=["Lunch"],
        tags=["Easy"],
        tools=["Spoon"],
        method=["Boiled"],
        rating=5.0,
        is_healthy=True,
        total_time_minutes=30,
        recipeIngredients=[],
        recipeInstructions=[],
    )

    mock_llm_client = MagicMock(spec=LLMClient)

    enriched_recipe = enrich_recipe_properties(
        recipe, mock_llm_client, system_prompt="test prompt"
    )

    # Assert
    # Should perform no enrichment because all potentially missing fields are present
    mock_llm_client.chat.assert_not_called()
    assert enriched_recipe.recipeCategory == ["Lunch"]


def test_enrich_recipe_properties_partial():
    # Arrange
    recipe = Recipe(
        name="Test Recipe",
        slug="test-recipe",
        recipeCategory=["Lunch"],
        # tags missing
        tools=["Spoon"],
        # method missing (default empty list)
        rating=5.0,
        recipeIngredients=[],
        recipeInstructions=[],
    )

    mock_llm_client = MagicMock(spec=LLMClient)

    class MockResponse(BaseModel):
        tags: list[str] = ["NewTag"]
        method: list[str] = ["Fried"]

    mock_llm_client.chat.return_value = MockResponse()

    enriched_recipe = enrich_recipe_properties(
        recipe, mock_llm_client, system_prompt="test prompt"
    )

    # Assert
    assert enriched_recipe.tags == ["NewTag"]
    assert enriched_recipe.method == ["Fried"]
    assert enriched_recipe.recipeCategory == ["Lunch"]  # Unchanged

    call_args = mock_llm_client.chat.call_args
    _, kwargs = call_args
    DynamicModel = kwargs["response_model"]
    schema = DynamicModel.model_json_schema()
    properties = schema["properties"]
    assert "tags" in properties
    assert "method" in properties
    assert "recipeCategory" not in properties  # Should not be asked for
    assert "tools" not in properties
    assert "rating" not in properties
    # is_healthy was None in recipe (default), so it should be in properties
    assert "is_healthy" in properties
    assert "total_time_minutes" in properties


def test_enrich_recipe_properties_is_healthy():
    # Arrange
    recipe = Recipe(
        name="Healthy Salad",
        slug="healthy-salad",
        recipeIngredients=[],
        recipeInstructions=[],
    )
    assert recipe.is_healthy is None

    mock_llm_client = MagicMock(spec=LLMClient)

    class MockResponse(BaseModel):
        is_healthy: bool = True

    mock_llm_client.chat.return_value = MockResponse()

    enriched_recipe = enrich_recipe_properties(
        recipe, mock_llm_client, system_prompt="test prompt"
    )

    # Assert
    assert enriched_recipe.is_healthy is True

    call_args = mock_llm_client.chat.call_args
    _, kwargs = call_args
    DynamicModel = kwargs["response_model"]
    schema = DynamicModel.model_json_schema()
    properties = schema["properties"]
    assert "is_healthy" in properties
    assert "total_time_minutes" in properties
    assert "method" in properties


def test_enrich_recipe_properties_total_time():
    # Arrange
    recipe = Recipe(
        name="Quick Pasta",
        slug="quick-pasta",
        recipeIngredients=[],
        recipeInstructions=[],
    )
    assert recipe.total_time_minutes is None

    mock_llm_client = MagicMock(spec=LLMClient)

    class MockResponse(BaseModel):
        total_time_minutes: int = 15

    mock_llm_client.chat.return_value = MockResponse()

    enriched_recipe = enrich_recipe_properties(
        recipe, mock_llm_client, system_prompt="test prompt"
    )

    # Assert
    assert enriched_recipe.total_time_minutes == 15

    call_args = mock_llm_client.chat.call_args
    _, kwargs = call_args
    DynamicModel = kwargs["response_model"]
    schema = DynamicModel.model_json_schema()
    properties = schema["properties"]
    assert "total_time_minutes" in properties
    assert "method" in properties
