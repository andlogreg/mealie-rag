"""
Mock Mealie API

NOTE: Currently outdated
"""

import json
from fastapi import FastAPI, HTTPException

app = FastAPI()


def load_recipes():
    with open("recipes.json", "r") as f:
        return json.load(f)


recipes = load_recipes()


@app.get("/api/recipes")
def get_recipes(page: int = 1, per_page: int = 50):
    start = (page - 1) * per_page
    end = start + per_page
    paginated_items = recipes[start:end]

    total = len(recipes)
    total_pages = (total + per_page - 1) // per_page

    return {
        "page": page,
        "per_page": per_page,
        "total": total,
        "total_pages": total_pages,
        "items": paginated_items,
        "next": f"/api/recipes?page={page + 1}&per_page={per_page}"
        if page < total_pages
        else None,
        "previous": f"/api/recipes?page={page - 1}&per_page={per_page}"
        if page > 1
        else None,
    }


@app.get("/api/recipes/{slug}")
def get_recipe(slug: str):
    for recipe in recipes:
        if recipe["slug"] == slug:
            return recipe
    raise HTTPException(status_code=404, detail="Recipe not found")


@app.get("/")
def read_root():
    return {"message": "Welcome to Mock Mealie API"}
