"""
Fetch all recipes from Mealie and print them.

Mostly for debugging purposes.
"""

from .config import settings
from .mealie import fetch_full_recipes


def main():
    recipes = fetch_full_recipes(settings.mealie_api_url, settings.mealie_token)
    print(f"Successfully fetched {len(recipes)} recipes.")

    if recipes:
        print("Here are the first 5 recipes:")
        for recipe in recipes[:5]:
            print("##################### RECIPE #####################")
            print(recipe.get_text_for_embedding())
            print("\n")


if __name__ == "__main__":
    main()
