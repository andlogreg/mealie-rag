"""
RAG evaluation — local / CSV backend.
"""

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any

from eval_core import (
    DEFAULT_DATASET_NAME,
    BaseEvaluationConfig,
    RetrievalMetrics,
    build_ground_truth_filters,
    build_judge_llm,
    build_ragas_metrics,
    compute_retrieval_metrics,
    format_ragas_contexts,
    get_relevant_ids,
    log_retrieval_metrics,
    make_experiment_name,
    mean,
    parse_expected_properties,
)
from ragas import Dataset, experiment

from mealierag.config import settings
from mealierag.prompts import LangfusePromptManager
from mealierag.service import MealieRAGService, create_mealie_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_ROOT_DIR = ".."


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseEvaluationConfig):
    """Runtime configuration for a local/CSV evaluation experiment."""

    dataset_name: str
    dataset_root_dir: str
    embedding_model: str


# ---------------------------------------------------------------------------
# RAGAS experiment
# ---------------------------------------------------------------------------


@experiment()
async def evaluate_service(
    row: dict[str, Any],
    service: MealieRAGService,
    relevancy_metric: Any,
    faithfulness_metric: Any,
    llm: Any,
    settings_str: str,
    evaluation_config_str: str,
    qdrant_client: Any,
    collection_name: str,
) -> dict[str, Any]:
    """Evaluate a single query row through the full RAG pipeline."""
    query = row["question"]
    logger.info("Evaluating: %s", query)

    error = None
    answer = ""
    relevancy_result = None
    faithfulness_result = None
    extraction = None
    recipe_names: list[str] = []
    retrieval: RetrievalMetrics | None = None

    try:
        # RAG pipeline: query extraction → retrieval → generation
        extraction = service.generate_queries(query)
        hits = service.retrieve_recipes(extraction)
        contexts, recipe_names = format_ragas_contexts(hits)
        messages = service.populate_messages(query, hits)
        answer = "".join(service.chat(messages))

        # ── Retrieval metrics ────────────────────────────────────────────
        expected = parse_expected_properties(row.get("expected_properties"))
        gt_filter = build_ground_truth_filters(expected)
        if gt_filter is None:
            logger.warning(
                "No expected_properties for query '%s' — retrieval metrics skipped.",
                query,
            )
        else:
            relevant_ids = get_relevant_ids(qdrant_client, collection_name, gt_filter)
            retrieved_ids = [str(h.id) for h in hits]
            retrieval = compute_retrieval_metrics(retrieved_ids, relevant_ids)
            log_retrieval_metrics(retrieval)

        # ── Generation metrics ───────────────────────────────────────────
        relevancy_result = relevancy_metric.score(query=query, answer=answer, llm=llm)
        logger.info("Answer Relevancy Score: %s", relevancy_result.value)

        faithfulness_result = faithfulness_metric.score(
            query=query, context=contexts, answer=answer, llm=llm
        )
        logger.info("Answer Faithfulness Score: %s", faithfulness_result.value)

    except Exception as e:
        logger.exception("Error evaluating query '%s'", query)
        error = str(e)

    return {
        **row,
        "response": answer,
        # generation
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
        # retrieval
        "retrieval_precision": None if retrieval is None else retrieval.precision,
        "retrieval_recall": None if retrieval is None else retrieval.recall,
        "retrieval_recall_capped": None
        if retrieval is None
        else retrieval.recall_capped,
        "retrieval_mrr": None if retrieval is None else retrieval.mrr,
        "retrieval_ndcg": None if retrieval is None else retrieval.ndcg,
        "retrieval_hit": None if retrieval is None else retrieval.hit,
        "retrieval_relevant_count": None
        if retrieval is None
        else retrieval.relevant_count,
        # metadata
        "recipe_context": "\n".join("- " + name for name in recipe_names),
        "query_extraction": None
        if extraction is None
        else extraction.model_dump_json(),
        "settings": settings_str,
        "evaluation_config": evaluation_config_str,
        "success": error is None,
        "error": error,
        "trace_id": None,
        "trace_url": None,
    }


# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------


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

    # 3. Set up the judge LLM and RAGAS metrics
    llm = build_judge_llm(config)
    relevancy_metric, faithfulness_metric = build_ragas_metrics(prompt_manager)

    # 4. Run the experiment row-by-row
    settings_str = settings.model_dump_json()
    evaluation_config_str = config.model_dump_json()

    experiment_results = await evaluate_service.arun(
        dataset,
        service=service,
        relevancy_metric=relevancy_metric,
        faithfulness_metric=faithfulness_metric,
        llm=llm,
        settings_str=settings_str,
        evaluation_config_str=evaluation_config_str,
        qdrant_client=service.vector_db_client,
        collection_name=settings.vectordb_collection_name,
        backend=backend,
        name=make_experiment_name(config.experiment_name),
    )

    # 5. Aggregate and log summary metrics
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
    mean_relevancy = mean(valid_relevancy)

    def _mean_retrieval(key: str) -> float:
        return mean([r[key] for r in experiment_results if r.get(key) is not None])

    mean_precision = _mean_retrieval("retrieval_precision")
    mean_recall = _mean_retrieval("retrieval_recall")
    mean_recall_capped = _mean_retrieval("retrieval_recall_capped")
    mean_mrr = _mean_retrieval("retrieval_mrr")
    mean_ndcg = _mean_retrieval("retrieval_ndcg")
    hit_rate = _mean_retrieval("retrieval_hit") * 100

    logger.info(
        "Faithfulness Rate: %.1f%% (%d / %d)",
        faithfulness_rate,
        faithful_count,
        total_faithfulness,
    )
    logger.info("Mean Relevancy Score: %.2f", mean_relevancy)
    logger.info(
        "Retrieval — P@K=%.3f  R@K=%.3f  R_capped@K=%.3f  MRR=%.3f  nDCG=%.3f  HitRate=%.1f%%",
        mean_precision,
        mean_recall,
        mean_recall_capped,
        mean_mrr,
        mean_ndcg,
        hit_rate,
    )

    # Append a summary row so the CSV is self-contained
    experiment_results.append(
        {
            "id": -1,
            "response": "",
            # generation summary
            "relevancy_score": mean_relevancy,
            "relevancy_reasoning": "summary: mean relevancy_score",
            "faithfulness_score": faithfulness_rate,
            "faithfulness_reasoning": "summary: faithfulness_rate (%)",
            # retrieval summary
            "retrieval_precision": mean_precision,
            "retrieval_recall": mean_recall,
            "retrieval_recall_capped": mean_recall_capped,
            "retrieval_mrr": mean_mrr,
            "retrieval_ndcg": mean_ndcg,
            "retrieval_hit": hit_rate,
            "retrieval_relevant_count": None,
            # metadata
            "recipe_context": "",
            "query_extraction": "",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
