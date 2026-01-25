# Mealie RAG CLI

A CLI tool and library for the Mealie RAG system.

## Backlog

- Use logging instead of print statements
- Add tests
- ...

## Features

- **Ingestion**: Fetches recipes from Mealie, generates embeddings, and stores them in Qdrant.
- **Q&A**: Provides both a CLI and Web UI (Gradio) to chat with your recipe collection.

## Configuration

The application is configured via environment variables:

- `MEALIE_API_URL`: URL to your Mealie API (e.g., `http://localhost:9000/api/recipes`).
- `MEALIE_TOKEN`: Your Mealie API token.
- `VECTORDB_URL`: URL to Qdrant (default: `http://localhost:6333`).
- `OLLAMA_BASE_URL`: URL to Ollama (default: `http://localhost:11434`).

## Usage

Run commands using the `mealierag` entrypoint.

### Ingest Recipes
Fetch recipes from Mealie and index them into Qdrant:
```bash
uv run mealierag ingest
```

### Start Web UI
Launch the Gradio-based chat interface:
```bash
uv run mealierag qa-ui
```

### CLI Chat
Run a quick chat session in the terminal:
```bash
uv run mealierag qa-cli
```

### Debug Fetch
Print fetched recipes to stdout (for debugging):
```bash
uv run mealierag fetch
```
