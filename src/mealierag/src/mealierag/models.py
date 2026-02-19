"""
Models module.

Contains Pydantic models for Mealie API responses.
"""

from collections.abc import Iterator
from itertools import chain

from pydantic import BaseModel, ConfigDict, Field


class RecipeIngredient(BaseModel):
    display: str

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        return self.display

    def get_text_for_context(self):
        return self.display

    def get_text_representation(self):
        return self.display


class NormalizedRecipeIngredient(BaseModel):
    names: list[str]

    def get_payload_text(self):
        return " ".join(self.names)

    def get_text_representation(self):
        return ", ".join(self.names)


class NormalizedRecipeIngredients(BaseModel):
    ingredients: list[NormalizedRecipeIngredient] = Field(default_factory=list)

    def flatten(self):
        return list(chain.from_iterable([ing.names for ing in self.ingredients]))

    def get_text_representation(self):
        return ", ".join([ing.get_text_representation() for ing in self.ingredients])


class RecipeInstruction(BaseModel):
    text: str

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        return self.text

    def get_text_for_context(self):
        return self.text

    def get_text_representation(self):
        return self.text


class Recipe(BaseModel):
    id: str | None = None
    name: str
    slug: str
    image: str | None = None
    totalTime: str | None = Field(
        default=None,
        description="Total time to make the recipe, including prep and cook time",
    )
    total_time_minutes: int | None = Field(
        default=None,
        description="Total (Prep + Cook) time in minutes. Infers from totalTime or estimates based on instructions.",
    )
    description: str | None = Field(
        default=None, description="Description of the recipe"
    )
    recipeCategory: list[str] = Field(
        default_factory=list,
        description="Categories of the recipe, e.g. ['Dinner', 'Lunch', 'Breakfast']",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags of the recipe, e.g. ['Healthy', 'Quick', 'Easy']",
    )
    tools: list[str] = Field(
        default_factory=list,
        description="Tools needed to make the recipe, e.g. ['Oven', 'Stove', 'Microwave']",
    )
    method: list[str] = Field(
        default_factory=list,
        description="Cooking methods used, e.g. ['Fried', 'Baked', 'Grilled', 'Slow Cooked']",
    )
    rating: float | None = Field(
        default=None, description="Rating of the recipe, between 1 and 5"
    )
    is_healthy: bool | None = Field(
        default=None,
        description="True if the recipe is considered healthy based on ingredients and cooking method",
    )
    recipeIngredients: list[RecipeIngredient] = Field(
        default_factory=list, description="Ingredients of the recipe"
    )
    recipeInstructions: list[RecipeInstruction] = Field(
        default_factory=list, description="Instructions of the recipe"
    )
    normalizedRecipeIngredients: NormalizedRecipeIngredients = Field(
        default_factory=NormalizedRecipeIngredients,
        description="Normalized ingredients of the recipe",
    )

    model_config = ConfigDict(extra="ignore")

    def get_text_for_embedding(self):
        text_content = f"{self.name}. {self.description}\n{', '.join(self.tags)}\n"
        for ing in self.recipeIngredients:
            text_content += f"{ing.get_text_for_embedding()}\n"
        for step in self.recipeInstructions:
            text_content += f"{step.get_text_for_embedding()}\n"
        return text_content

    def get_text_for_context(self):
        text_content = (
            f"RecipeName: {self.name}\nRecipeID: {self.id}\nRating: {self.rating}\n"
        )
        text_content += "Ingredients:\n"
        for ing in self.recipeIngredients:
            text_content += f"- {ing.get_text_for_context()}\n"
        text_content += "Instructions:\n"
        for step in self.recipeInstructions:
            text_content += f"- {step.get_text_for_context()}\n"
        return text_content

    def get_text_representation(self, properties_to_include: list):
        text_content = ""
        for prop in properties_to_include:
            if hasattr(self, prop):
                value = getattr(self, prop)
                if value is None:
                    continue

                text_content += f"**{prop}:**"

                if (
                    isinstance(value, list)
                    and value
                    and hasattr(value[0], "get_text_representation")
                ):
                    text_content += "\n"
                    for val in value:
                        text_content += f"- {val.get_text_representation()}\n"
                elif hasattr(value, "get_text_representation") and not isinstance(
                    value, list
                ):
                    text_content += "\n"
                    text_content += f"- {value.get_text_representation()}\n"
                elif isinstance(value, list):
                    text_content += "\n"
                    text_content += ", ".join(str(v) for v in value) + "\n"
                else:
                    text_content += " "
                    text_content += f"{value}\n"
            else:
                raise ValueError(f"Property {prop} not found in Recipe model")
        return text_content.strip()


class Recipes(BaseModel):
    items: list[Recipe]

    def __len__(self):
        return len(self.items)

    def __iter__(self) -> Iterator[Recipe]:
        return iter(self.items)

    def __contains__(self, item):
        return item in self.items


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
