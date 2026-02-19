"""
RAG evaluation.
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel, SecretStr
from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import NumericMetric
from ragas.metrics.collections import Faithfulness

from mealierag.config import LLMProvider, settings
from mealierag.prompts import LangfusePromptManager, PromptType
from mealierag.service import MealieRAGService, create_mealie_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT_DIR = ".."
DEFAULT_DATASET_NAME = f"{datetime.now().strftime('%Y%m%d')}_queries"


class EvaluationConfig(BaseModel):
    """Runtime configuration for a single evaluation experiment."""

    dataset_name: str
    dataset_root_dir: str
    judge_model: str
    judge_temperature: float
    judge_provider: LLMProvider
    judge_base_url: str
    judge_api_key: SecretStr | None
    embedding_model: str
    experiment_name: str | None
    limit: int | None


def format_ragas_contexts(hits: list) -> tuple[str, list[str]]:
    """Build a formatted context string and a list of recipe names from Qdrant hits."""
    contexts = []
    recipe_names = []
    for hit in hits:
        payload = hit.payload
        lines = [
            "[RECIPE START]",
            f"Name: {payload.get('name', 'Unknown')}",
            f"Rating: {payload.get('rating', 'Unknown')}",
            f"Description: {payload.get('description', '')}",
            "Ingredients:",
            *("- " + i for i in payload.get("ingredients", [])),
            "Method:",
            *("- " + s for s in payload.get("method", [])),
            "[RECIPE END]",
        ]
        contexts.append("\n".join(lines))
        recipe_names.append(payload.get("name", "Unknown"))

    return "\n".join(contexts), recipe_names


@experiment()
async def evaluate_service(
    row: dict[str, Any],
    service: MealieRAGService,
    relevancy_metric: NumericMetric,
    faithfulness_metric: Faithfulness,
    llm: Any,
    settings_str: str,
    evaluation_config_str: str,
) -> dict[str, Any]:
    """Evaluate a single query row through the full RAG pipeline."""
    query = row["question"]
    logger.info("Evaluating: %s", query)

    error = None
    answer = ""
    relevancy_result = None
    faithfulness_result = None
    recipe_names: list[str] = []

    try:
        # RAG pipeline: query extraction → retrieval → generation
        extraction = service.generate_queries(query)
        hits = service.retrieve_recipes(extraction)
        contexts, recipe_names = format_ragas_contexts(hits)
        messages = service.populate_messages(query, hits)
        answer = "".join(service.chat(messages))

        # Score with judge LLM
        relevancy_result = relevancy_metric.score(query=query, answer=answer, llm=llm)
        logger.info("Answer Relevancy Score: %s", relevancy_result.value)

        faithfulness_result = faithfulness_metric.score(
            query=query, context=contexts, answer=answer, llm=llm
        )
        logger.info("Answer Faithfulness Score: %s", faithfulness_result.value)

    except Exception as e:
        logger.error("Error evaluating query '%s': %s", query, e)
        error = str(e)

    return {
        **row,
        "response": answer,
        "relevancy_score": None if relevancy_result is None else relevancy_result.value,
        "relevancy_reasoning": None
        if relevancy_result is None
        else relevancy_result.reason,
        "faithfulness_score": None
        if faithfulness_result is None
        else faithfulness_result.value,
        "faithfulness_reasoning": None
        if faithfulness_result is None
        else faithfulness_result.reason,
        "recipe_context": "\n".join("- " + name for name in recipe_names),
        "settings": settings_str,
        "evaluation_config": evaluation_config_str,
        "success": error is None,
        "error": error,
        "trace_id": None,
        "trace_url": None,
    }


async def run_experiment(config: EvaluationConfig) -> None:
    """Orchestrate the full evaluation experiment."""
    prompt_manager = LangfusePromptManager()

    # 1. Load input dataset
    logger.info(
        "Loading dataset '%s' from %s...", config.dataset_name, config.dataset_root_dir
    )
    dataset = Dataset.load(
        name=config.dataset_name,
        backend="local/csv",
        root_dir=config.dataset_root_dir,
    )
    backend = dataset.backend
    if config.limit:
        dataset = dataset[: config.limit]

    # 2. Initialise the RAG service
    service = create_mealie_rag_service(settings)

    # 3. Set up the judge LLM and embeddings
    if config.judge_provider == LLMProvider.OLLAMA:
        raise ValueError(
            "OLLAMA is not supported as a judge provider. "
            "Use an OpenAI-compatible endpoint instead."
        )
    if config.judge_provider == LLMProvider.OPENAI:
        openai_client = OpenAI(
            api_key=config.judge_api_key.get_secret_value()
            if config.judge_api_key
            else None,
            base_url=config.judge_base_url,
        )
    else:
        raise ValueError(f"Unsupported judge provider: {config.judge_provider}")

    llm = llm_factory(
        config.judge_model,
        client=openai_client,
        temperature=config.judge_temperature,
        top_p=None,
        adapter="instructor",
        mode=Mode.JSON_SCHEMA,
    )

    # 4. Build metrics using Langfuse-managed prompts
    relevancy_metric = NumericMetric(
        name="relevancy",
        prompt=prompt_manager.get_prompt(
            PromptType.METRIC_RELEVANCY, label="production"
        ).compile(),
        allowed_values=(1, 5),
    )
    faithfulness_metric = Faithfulness(
        name="faithfulness",
        prompt=prompt_manager.get_prompt(
            PromptType.METRIC_FAITHFULNESS, label="production"
        ).compile(),
        allowed_values=["hallucination", "faithful"],
    )

    # 5. Run the experiment row-by-row
    settings_str = settings.model_dump_json()
    evaluation_config_str = config.model_dump_json()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_experiment_name = (
        f"{timestamp}_{config.experiment_name}" if config.experiment_name else timestamp
    )

    experiment_results = await evaluate_service.arun(
        dataset,
        service=service,
        relevancy_metric=relevancy_metric,
        faithfulness_metric=faithfulness_metric,
        llm=llm,
        settings_str=settings_str,
        evaluation_config_str=evaluation_config_str,
        backend=backend,
        name=full_experiment_name,
    )

    # 6. Aggregate and log summary metrics
    faithful_count = sum(
        1 for r in experiment_results if r.get("faithfulness_score") == "faithful"
    )
    total_faithfulness = sum(
        1 for r in experiment_results if r.get("faithfulness_score") is not None
    )
    faithfulness_rate = (
        (faithful_count / total_faithfulness * 100) if total_faithfulness > 0 else 0
    )

    valid_relevancy = [
        r["relevancy_score"]
        for r in experiment_results
        if r.get("relevancy_score") is not None
    ]
    relevancy_score = (
        sum(valid_relevancy) / len(valid_relevancy) if valid_relevancy else 0
    )

    logger.info(
        "Faithfulness Rate: %.1f%% (%d / %d)",
        faithfulness_rate,
        faithful_count,
        total_faithfulness,
    )
    logger.info("Mean Relevancy Score: %.1f", relevancy_score)

    # Append a summary row so the CSV is self-contained
    experiment_results.append(
        {
            "id": -1,
            "response": "",
            "relevancy_score": relevancy_score,
            "relevancy_reasoning": "summary: mean relevancy_score",
            "faithfulness_score": faithfulness_rate,
            "faithfulness_reasoning": "summary: faithfulness_rate (%)",
            "recipe_context": "",
            "settings": settings_str,
            "evaluation_config": evaluation_config_str,
            "success": True,
            "error": None,
            "trace_id": None,
            "trace_url": None,
        }
    )
    experiment_results.save()

    csv_path = (
        Path(config.dataset_root_dir) / "experiments" / f"{experiment_results.name}.csv"
    )
    logger.info("Experiment results saved to: %s", csv_path.resolve())


def main() -> None:
    """Parse CLI arguments and launch the evaluation experiment."""
    parser = argparse.ArgumentParser(
        description="Run End-to-End RAG Evaluation with Ragas"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the Ragas dataset to load.",
    )
    parser.add_argument(
        "--dataset-root-dir",
        type=str,
        default=DEFAULT_DATASET_ROOT_DIR,
        help="Root directory containing the dataset (default: '..').",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=settings.llm_model,
        help="Model identifier for the Ragas judge LLM.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=settings.llm_provider,
        help="Provider for the judge LLM (openai only).",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default=settings.llm_base_url,
        help="Base URL for the judge LLM API.",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default=settings.llm_api_key.get_secret_value()
        if settings.llm_api_key
        else None,
        help="API key for the judge LLM.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=settings.llm_temperature,
        help="Sampling temperature for the judge LLM.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=settings.embedding_model,
        help="Embedding model used by Ragas.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional label appended to the auto-generated timestamp experiment name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of dataset rows evaluated (useful for smoke tests).",
    )

    args = parser.parse_args()

    evaluation_config = EvaluationConfig(
        dataset_name=args.dataset_name,
        dataset_root_dir=args.dataset_root_dir,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_provider=args.judge_provider,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        embedding_model=args.embedding_model,
        experiment_name=args.experiment_name,
        limit=args.limit,
    )

    asyncio.run(run_experiment(evaluation_config))


if __name__ == "__main__":
    main()
