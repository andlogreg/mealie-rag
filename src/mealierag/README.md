# Mealie RAG CLI

A CLI tool and library for the Mealie RAG system.

## Features

- **Ingestion**: Fetches recipes from Mealie, generates embeddings, and stores them in Qdrant.
- **Q&A**: Provides both a CLI and Web UI (Gradio) to chat with your recipe collection.
- **Observability**: Integrated with **Langfuse** for full trace visibility and prompt management.
- **Flexible LLM Support**: Supports both local (**Ollama**) and cloud (**OpenAI**) models.

## Configuration

The application is configured via environment variables (or `.env` file):

### Core
- `MEALIE_API_URL`: URL to your Mealie API (e.g., `http://localhost:9000/api/recipes`).
- `MEALIE_TOKEN`: Your Mealie API token.
- `VECTORDB_URL`: URL to Qdrant (default: `http://localhost:6333`).

### LLM & Embeddings
- `LLM_PROVIDER`: `ollama` (default) or `openai`.
- `LLM_MODEL`: Model name (e.g., `llama3.2` or `gpt-4o`).
- `LLM_BASE_URL`: Base URL for the provider (default: `http://localhost:11434` for Ollama).
- `LLM_API_KEY`: API Key (required for OpenAI).

### Observability (Langfuse)
- `TRACING_ENABLED`: `true` to enable tracing.
- `TRACING_BASE_URL`: URL for Langfuse (e.g. `https://cloud.langfuse.com`).
- `TRACING_PUBLIC_KEY`: Your Langfuse Public Key.
- `TRACING_SECRET_KEY`: Your Langfuse Secret Key.

### Application
- `LOG_LEVEL`: Application log level (default: `INFO`).
- `DEPENDENCY_LOG_LEVEL`: Log level for libraries (default: `WARNING`).

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
