"""
Convert YAML query dataset into a Langfuse dataset.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
from langfuse import Langfuse
from langfuse.api.resources.commons.errors.not_found_error import NotFoundError
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_YAML_DATASET_PATH = (
    Path("../raw_data") / f"{datetime.now().strftime('%Y%m%d')}_queries.yaml"
)


def load_yaml_dataset(path: Path) -> list[dict]:
    """Load and parse YAML query file."""
    if not path.exists():
        raise FileNotFoundError(f"YAML dataset not found at: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def main(
    yaml_dataset_path: Path,
    dataset_name: str,
    description: str | None,
    public_key: str | None,
    secret_key: str | None,
    host: str | None,
    append_if_exists: bool = False,
) -> None:
    logger.info("Loading YAML dataset from %s...", yaml_dataset_path)
    raw_data = load_yaml_dataset(yaml_dataset_path)
    logger.info("Loaded %d raw items.", len(raw_data))

    # Build Langfuse client kwargs
    client_kwargs: dict = {}
    if public_key:
        client_kwargs["public_key"] = public_key
    if secret_key:
        client_kwargs["secret_key"] = secret_key
    if host:
        client_kwargs["host"] = host

    langfuse = Langfuse(**client_kwargs)

    # Check for existing dataset if appending is not allowed
    if not append_if_exists:
        try:
            langfuse.get_dataset(dataset_name)
            logger.error(
                "Dataset '%s' already exists in Langfuse. Use --append-if-exists to append to it.",
                dataset_name,
            )
            return
        except NotFoundError:
            logger.info("Dataset '%s' does not exist. Creating it...", dataset_name)
            pass

    # Create the dataset (idempotent: returns existing dataset if name already exists)
    create_kwargs: dict = {"name": dataset_name}
    if description:
        create_kwargs["description"] = description
    langfuse.create_dataset(**create_kwargs)
    logger.info("Dataset '%s' ready in Langfuse.", dataset_name)

    skipped = 0
    for item in tqdm(raw_data, desc="Uploading items"):
        query = item.get("query")
        if not query:
            logger.warning("Skipping item with missing 'query' field: %s", item)
            skipped += 1
            continue

        langfuse.create_dataset_item(
            dataset_name=dataset_name,
            input={"question": query},
            expected_output={
                "expected_properties": item.get("expected_properties", {})
            },
            metadata={
                "id": item.get("id"),
                **item.get("metadata", {}),
            },
        )

    # Flush any buffered requests before exiting
    langfuse.flush()

    logger.info(
        "Dataset '%s' populated with %d items (%d skipped).",
        dataset_name,
        len(raw_data) - skipped,
        skipped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YAML query dataset into a Langfuse dataset."
    )
    parser.add_argument(
        "--yaml-dataset-path",
        type=Path,
        default=DEFAULT_YAML_DATASET_PATH,
        help=f"Path to the input YAML dataset. Defaults to {DEFAULT_YAML_DATASET_PATH}",
    )
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        default=None,
        help=(
            "Name for the Langfuse dataset. "
            "Defaults to the stem of --yaml-dataset-path."
        ),
    )
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional description for the Langfuse dataset.",
    )
    parser.add_argument(
        "--append-if-exists",
        action="store_true",
        help="Append to the dataset if it already exists. Defaults to False (exit if exists).",
    )
    # Credential overrides — if omitted, the SDK reads LANGFUSE_PUBLIC_KEY,
    # LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from the environment.
    parser.add_argument(
        "--public-key",
        type=str,
        default=None,
        help="Langfuse public key (overrides LANGFUSE_PUBLIC_KEY env var).",
    )
    parser.add_argument(
        "--secret-key",
        type=str,
        default=None,
        help="Langfuse secret key (overrides LANGFUSE_SECRET_KEY env var).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Langfuse host URL (overrides LANGFUSE_HOST env var).",
    )
    args = parser.parse_args()

    output_dataset_name = args.output_dataset_name or args.yaml_dataset_path.stem
    main(
        yaml_dataset_path=args.yaml_dataset_path,
        dataset_name=output_dataset_name,
        description=args.description,
        public_key=args.public_key,
        secret_key=args.secret_key,
        host=args.host,
        append_if_exists=args.append_if_exists,
    )
