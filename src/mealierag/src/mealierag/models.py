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


class RecipeInstruction(BaseModel):
    text: str

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
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
        text_content = f"Title: {self.name}\nDescription: {self.description}\nRating: {self.rating}\nCategory: {', '.join(self.recipeCategory)}\nTags: {', '.join(self.tags)}\nIngredients:\n"
        for ing in self.recipeIngredient:
            text_content += f"- {ing.get_text_for_embedding()}\n"
        text_content += "Instructions:\n"
        for step in self.recipeInstructions:
            text_content += f"- {step.get_text_for_embedding()}\n"
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
