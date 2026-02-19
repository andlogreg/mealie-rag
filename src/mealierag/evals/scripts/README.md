# Evaluation Scripts

Offline evaluation of retrieval and generation quality.
Run all commands from this directory (`src/mealierag/evals/scripts/`).

> [!NOTE]
> Data files (`raw_data/`, `enriched/`, `datasets/`) are tracked with DVC, not git.

---

## 1. Dump recipes from Mealie

Fetches all recipes from a Mealie instance and saves them as a JSON snapshot.

```bash
MEALIE_API_URL="http://localhost:8000" \
MEALIE_TOKEN="your-token" \
uv run dump_recipes_from_mealie.py
# Output: ../raw_data/<YYYYMMDD>_recipes.json
```

## 2. Enrich recipes

Normalises ingredients and fills in missing properties (e.g. `is_healthy`) using
the configured LLM.

```bash
uv run enrich_recipes.py \
  --input-path INPUT_FILE \
  --output-path OUTPUT_FILE
```

## 3. Ingest into Qdrant

Embeds the enriched recipes and upserts them into a Qdrant collection.

```bash
VECTORDB_PATH="../qdrant_local_data" \
VECTORDB_COLLECTION_NAME="mealie_recipes_local" \
uv run ingest_from_file.py
```

## 4. Create Ragas dataset

Converts a YAML query file into a Ragas-compatible CSV dataset.

```bash
uv run create_ragas_dataset.py
```

## 5. Run evaluation

Runs each query through the full RAG pipeline and scores responses with a judge LLM.

```bash
VECTORDB_PATH="../qdrant_local_data" \
VECTORDB_COLLECTION_NAME="mealie_recipes_local" \
uv run evaluate.py \
  --limit 5 \
  --judge-temperature 0.5 \
  --experiment-name "my_experiment"
```
