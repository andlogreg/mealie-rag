# Mealie RAG

<img src="assets/screenshot.png" alt="Mealie RAG Screenshot" width="600">

## Overview

**Mealie RAG** is a Retrieval-Augmented Generation (RAG) system built as a companion to [Mealie](https://mealie.io/). It acts as a culinary assistant ("MealieChef"), allowing users to interact with their personal recipe collection using natural language!

I use this project as an **engineering sandbox** to explore the challenges of operationalizing GenAI systems. It moves beyond simple prototypes to explore the engineering challenges of building observable, reliable, and structured RAG pipelines. As such... ⚠️ Things may break or pivot significantly without notice!

## Current Architecture

<img src="assets/arch.png" alt="Mealie RAG Architecture" width="800">

## Engineering Highlights

*   **Advanced Retrieval Strategies**: Implements **Structured Query Extraction**, **Multi-Query Expansion**, and **Reciprocal Rank Fusion (RRF)** to filter and rank results.
*   **LLM-Driven Ingestion Enrichment**: At ingest time, a structured LLM pipeline is used to normalise ingredients and infers missing recipe properties (`categories`, `tags`, `cooking method`, `is_healthy`, `total_time_minutes`), making filtering and retrieval significantly more reliable.
*   **End-to-End Observability**: Integration with **Langfuse** for tracing execution paths, managing prompt versions, and tracking model usage/costs. (Prometheus metrics to be added in the future...)
*   **Offline Evaluation Pipeline**: A reproducible evaluation subsystem using **Ragas** (generation quality) and custom retrieval metrics (Precision@K, Recall@K, MRR, nDCG@K, Hit Rate). Results can be stored as **Langfuse Experiments** for tracking across runs. Eval datasets are versioned with **DVC**.
*   **Data Infrastructure**: Integration with **Qdrant** for efficient vector operations (server and local embedded modes).
*   **Reproducibility & Standards**:
    *   **Modern Tooling**: managed via `uv` (dependency resolution) and `mise` (tool versioning).
    *   **Code Quality**: Enforced via strict typing, `ruff` linting/formatting, and pre-commit hooks.

## Technology Stack

*   **Core Application**: Python, Typer (CLI), Gradio (Web UI)
*   **Inference & Storage**: Qdrant, OpenAI SDK / Ollama (Code-level abstraction for Ollama/OpenAI)
*   **Observability & Evaluation**: Langfuse, Ragas
*   **Infrastructure**: Docker Compose, GitHub Actions, DVC
