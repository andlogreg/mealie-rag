"""
Models module.

Contains Pydantic models for Mealie API responses.
"""

from typing import List, Optional

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
    id: Optional[str] = None
    userId: Optional[str] = None
    householdId: Optional[str] = None
    groupId: Optional[str] = None
    name: str
    slug: str
    image: Optional[str] = None
    recipeServings: Optional[float] = None
    recipeYieldQuantity: Optional[float] = None
    recipeYield: Optional[str] = None
    totalTime: Optional[str] = None
    prepTime: Optional[str] = None
    cookTime: Optional[str] = None
    performTime: Optional[str] = None
    description: Optional[str] = None
    recipeCategory: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    rating: Optional[float] = None
    orgURL: Optional[str] = None
    dateAdded: Optional[str] = None
    dateUpdated: Optional[str] = None
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None
    lastMade: Optional[str] = None
    recipeIngredient: List[RecipeIngredient] = Field(default_factory=list)
    recipeInstructions: List[RecipeInstruction] = Field(default_factory=list)

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
    items: List[Recipe]
    next: Optional[str] = None
    previous: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
