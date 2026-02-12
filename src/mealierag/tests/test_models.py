from mealierag.models import Recipe, RecipeIngredient, RecipeInstruction


def test_recipe_model_embedding_text():
    """
    Test that the recipe text for embedding is generated correctly.
    """
    recipe = Recipe(
        id="1",
        name="Test Recipe",
        slug="test-recipe",
        description="A delicious test.",
        rating=5.0,
        recipeCategory=["Dinner", "Test"],
        tags=["Easy", "Quick"],
        recipeIngredient=[
            RecipeIngredient(display="1 cup of tests"),
            RecipeIngredient(display="2 spoons of verification"),
        ],
        recipeInstructions=[
            RecipeInstruction(text="Mix tests."),
            RecipeInstruction(text="Verify result."),
        ],
    )

    text = recipe.get_text_for_embedding()

    expected_parts = [
        "Test Recipe",
        "A delicious test.",
        "Easy, Quick",
        "1 cup of tests",
        "2 spoons of verification",
        "Mix tests.",
        "Verify result.",
    ]

    for part in expected_parts:
        assert part in text
