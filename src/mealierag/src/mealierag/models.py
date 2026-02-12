"""
Models module.

Contains Pydantic models for Mealie API responses.
"""

from pydantic import BaseModel, ConfigDict, Field


class RecipeIngredient(BaseModel):
    display: str

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        return self.display

    def get_text_for_context(self):
        return self.display


class RecipeInstruction(BaseModel):
    text: str

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        return self.text

    def get_text_for_context(self):
        return self.text


class Recipe(BaseModel):
    id: str | None = None
    userId: str | None = None
    householdId: str | None = None
    groupId: str | None = None
    name: str
    slug: str
    image: str | None = None
    recipeServings: float | None = None
    recipeYieldQuantity: float | None = None
    recipeYield: str | None = None
    totalTime: str | None = None
    prepTime: str | None = None
    cookTime: str | None = None
    performTime: str | None = None
    description: str | None = None
    recipeCategory: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    rating: float | None = None
    orgURL: str | None = None
    dateAdded: str | None = None
    dateUpdated: str | None = None
    createdAt: str | None = None
    updatedAt: str | None = None
    lastMade: str | None = None
    recipeIngredient: list[RecipeIngredient] = Field(default_factory=list)
    recipeInstructions: list[RecipeInstruction] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        text_content = f"{self.name}. {self.description}\n{', '.join(self.tags)}\n"
        for ing in self.recipeIngredient:
            text_content += f"{ing.get_text_for_embedding()}\n"
        for step in self.recipeInstructions:
            text_content += f"{step.get_text_for_embedding()}\n"
        return text_content

    def get_text_for_context(self):
        text_content = (
            f"RecipeName: {self.name}\nRecipeID: {self.id}\nRating: {self.rating}\n"
        )
        text_content += "Ingredients:\n"
        for ing in self.recipeIngredient:
            text_content += f"- {ing.get_text_for_context()}\n"
        text_content += "Instructions:\n"
        for step in self.recipeInstructions:
            text_content += f"- {step.get_text_for_context()}\n"
        return text_content


class RecipeResponse(BaseModel):
    page: int
    per_page: int = Field(alias="per_page")
    total: int
    total_pages: int
    items: list[Recipe]
    next: str | None = None
    previous: str | None = None

    model_config = ConfigDict(extra="ignore")


class QueryExtraction(BaseModel):
    """
    Extracted search parameters for culinary retrieval.
    """

    expanded_queries: list[str] = Field(
        description="5 diverse search variations. Transform general terms (meat) to specific ones (chicken, beef, etc)."
    )
    negative_ingredients: list[str] | None = Field(
        None,
        description="Individual food items to exclude. Singular lowercase nouns only (e.g., 'shrimp', 'mushroom').",
    )
    other_negative_constraints: list[str] | None = Field(
        None,
        description="Non-ingredient exclusions like equipment (no oven), time (no long prep), or diet (no fried).",
    )
    min_rating: int | None = Field(
        None,
        description="Minimum recipe rating filter, inclusive. Only use if explicitly specified.",
    )
    max_rating: int | None = Field(
        None,
        description="Maximum recipe rating filter, exclusive. Only use if explicitly specified.",
    )
