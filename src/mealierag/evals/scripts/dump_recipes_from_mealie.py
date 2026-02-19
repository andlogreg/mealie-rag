"""
Fetch all recipes from Mealie and dump them to a JSON file.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

from mealierag.config import settings
from mealierag.mealie import fetch_full_recipes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = (
    Path("../raw_data") / f"{datetime.now().strftime('%Y%m%d')}_recipes.json"
)


def main(output_path: Path) -> None:
    """Fetch all recipes from Mealie and write them to *output_path* as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching all recipes from %s...", settings.mealie_api_url)
    recipes = fetch_full_recipes(
        settings.mealie_api_url, settings.mealie_token.get_secret_value()
    )
    logger.info("Successfully fetched %d recipes.", len(recipes))

    logger.info("Writing recipes to %s...", output_path)
    output_path.write_text(recipes.model_dump_json(indent=2))
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Destination JSON file. Defaults to {DEFAULT_OUTPUT_PATH}",
    )
    args = parser.parse_args()
    main(args.output_path)
