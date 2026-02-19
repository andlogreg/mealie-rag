"""
Convert YAML query dataset into Ragas dataset.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
from ragas import Dataset
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
    yaml_dataset_path: Path, output_dataset_name: str, dataset_root_dir: str
) -> None:
    """Load YAML query file and write Ragas CSV dataset."""
    logger.info("Loading YAML dataset from %s...", yaml_dataset_path)
    raw_data = load_yaml_dataset(yaml_dataset_path)
    logger.info("Loaded %d raw items.", len(raw_data))

    # TODO: Consider setting data_model on the Dataset for schema validation.
    dataset = Dataset(
        name=output_dataset_name,
        backend="local/csv",
        root_dir=dataset_root_dir,
    )

    skipped = 0
    for item in tqdm(raw_data, desc="Building dataset"):
        query = item.get("query")
        if not query:
            logger.warning("Skipping item with missing 'query' field: %s", item)
            skipped += 1
            continue
        dataset.append(
            {
                "id": item.get("id"),
                "question": query,
                "expected_properties": item.get("expected_properties", {}),
                "metadata": item.get("metadata", {}),
            }
        )

    dataset.save()
    logger.info(
        "Dataset '%s' created with %d samples (%d skipped).",
        output_dataset_name,
        len(dataset),
        skipped,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YAML query dataset into Ragas dataset."
    )
    parser.add_argument(
        "--yaml-dataset-path",
        type=Path,
        default=DEFAULT_YAML_DATASET_PATH,
        help=f"Path to the input YAML dataset. Defaults to {DEFAULT_YAML_DATASET_PATH}",
    )
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        default="..",
        help="Root directory for the Ragas dataset output. Defaults to '..'",
    )
    parser.add_argument(
        "--output-dataset-name",
        type=str,
        default=None,
        help=(
            "Name for the output Ragas dataset. "
            "Defaults to the stem of --yaml-dataset-path."
        ),
    )
    args = parser.parse_args()

    output_dataset_name = args.output_dataset_name or args.yaml_dataset_path.stem
    main(args.yaml_dataset_path, output_dataset_name, args.dataset_root_dir)
