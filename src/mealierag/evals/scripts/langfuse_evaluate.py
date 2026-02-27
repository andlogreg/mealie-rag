"""
RAG evaluation — Langfuse backend.
"""

import argparse
import logging
from typing import Any

from eval_core import (
    DEFAULT_DATASET_NAME,
    BaseEvaluationConfig,
    build_ground_truth_filters,
    build_judge_llm,
    build_ragas_metrics,
    compute_retrieval_metrics,
    format_ragas_contexts,
    get_relevant_ids,
    log_retrieval_metrics,
    make_experiment_name,
    parse_expected_properties,
)
from langfuse import Evaluation, get_client
from langfuse.api import ScoreDataType

from mealierag.config import settings
from mealierag.prompts import LangfusePromptManager
from mealierag.service import create_mealie_rag_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class EvaluationConfig(BaseEvaluationConfig):
    """Runtime configuration for a Langfuse evaluation experiment."""

    dataset_name: str
    include_item_results: bool = False


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def run_experiment(config: EvaluationConfig) -> None:
    """Orchestrate the full evaluation experiment via Langfuse SDK."""
    langfuse = get_client()
    prompt_manager = LangfusePromptManager()

    # 1. Load the Langfuse dataset
    logger.info("Loading Langfuse dataset '%s'...", config.dataset_name)
    dataset = langfuse.get_dataset(config.dataset_name)
    logger.info("Dataset loaded with %d items.", len(dataset.items))

    # 2. Initialise the RAG service
    service = create_mealie_rag_service(settings)

    # 3. Set up the judge LLM and RAGAS metrics
    llm = build_judge_llm(config)
    relevancy_metric, faithfulness_metric = build_ragas_metrics(prompt_manager)

    # Qdrant client + collection for retrieval metrics
    qdrant_client = service.vector_db_client
    collection_name = settings.vectordb_collection_name

    # Serialised config strings attached as experiment metadata
    settings_str = settings.model_dump_json()
    evaluation_config_str = config.model_dump_json()

    # ── Task function ─────────────────────────────────────────────────────
    def evaluate_item(*, item: Any, **kwargs: Any) -> dict:
        """Run the full RAG pipeline for a single dataset item."""
        query = item.input["question"]
        logger.info("Evaluating: %s", query)

        error = None
        answer = ""
        extraction = None
        recipe_names: list[str] = []
        contexts = ""
        retrieved_ids: list[str] = []

        try:
            extraction = service.generate_queries(query)
            hits = service.retrieve_recipes(extraction)
            retrieved_ids = [str(h.id) for h in hits]
            contexts, recipe_names = format_ragas_contexts(hits)
            messages = service.populate_messages(query, hits)
            answer = "".join(service.chat(messages))
        except Exception as e:
            logger.exception("Error in RAG pipeline for query '%s'", query)
            error = str(e)

        return {
            "question": query,
            "response": answer,
            "contexts": contexts,
            "recipe_names": recipe_names,
            "retrieved_ids": retrieved_ids,
            "query_extraction": None
            if extraction is None
            else extraction.model_dump_json(),
            "success": error is None,
            "error": error,
        }

    # ── Item-level evaluators ─────────────────────────────────────────────
    def relevancy_evaluator(*, output: dict, **kwargs: Any) -> Evaluation | None:
        """LLM-as-judge relevancy score (1–5)."""
        if not output.get("success"):
            return None
        result = relevancy_metric.score(
            query=output["question"], answer=output["response"], llm=llm
        )
        logger.info("Relevancy Score: %s", result.value)
        return Evaluation(name="relevancy", value=result.value, comment=result.reason)

    def faithfulness_evaluator(*, output: dict, **kwargs: Any) -> Evaluation | None:
        """LLM-as-judge faithfulness (faithful / hallucination)."""
        if not output.get("success"):
            return None
        result = faithfulness_metric.score(
            query=output["question"],
            context=output["contexts"],
            answer=output["response"],
            llm=llm,
        )
        logger.info("Faithfulness Score: %s", result.value)
        return Evaluation(
            name="faithfulness",
            value=result.value,
            comment=result.reason,
            data_type=ScoreDataType.CATEGORICAL,
        )

    def retrieval_evaluator(
        *, input: dict, output: dict, expected_output: dict | None, **kwargs: Any
    ) -> list[Evaluation]:
        """Compute retrieval metrics against Qdrant ground-truth."""
        if not output.get("success") or not expected_output:
            logger.warning(
                "Query '%s' failed or no expected_output — retrieval metrics skipped.",
                output["question"],
            )
            return []

        expected = parse_expected_properties(expected_output.get("expected_properties"))
        gt_filter = build_ground_truth_filters(expected)
        if gt_filter is None:
            logger.warning(
                "No expected_properties for query '%s' — retrieval metrics skipped.",
                output["question"],
            )
            return []

        relevant_id_to_slug = get_relevant_ids(
            qdrant_client, collection_name, gt_filter
        )
        relevant_ids = set(relevant_id_to_slug.keys())
        retrieved_ids = output.get("retrieved_ids", [])
        metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids)
        log_retrieval_metrics(metrics)

        sorted_pairs = sorted(relevant_id_to_slug.items())
        sorted_relevant_ids = [id_ for id_, _ in sorted_pairs]
        sorted_relevant_slugs = [slug for _, slug in sorted_pairs]

        return [
            Evaluation(
                name="retrieval_precision",
                value=metrics.precision,
                metadata={
                    "retrieved_and_relevant_ids": sorted(
                        list(metrics.retrieved_and_relevant_ids)
                    ),
                    "retrieved_and_not_relevant_ids": sorted(
                        list(metrics.retrieved_and_not_relevant_ids)
                    ),
                },
            ),
            Evaluation(name="retrieval_recall", value=metrics.recall),
            Evaluation(name="retrieval_recall_capped", value=metrics.recall_capped),
            Evaluation(name="retrieval_mrr", value=metrics.mrr),
            Evaluation(name="retrieval_ndcg", value=metrics.ndcg),
            Evaluation(
                name="retrieval_hit",
                value=metrics.hit,
                comment=f"relevant_count={metrics.relevant_count}",
                metadata={
                    "relevant_ids": sorted_relevant_ids,
                    "relevant_slugs": sorted_relevant_slugs,
                },
                data_type=ScoreDataType.BOOLEAN,
            ),
        ]

    # ── Run-level evaluators ──────────────────────────────────────────────
    def aggregate_metrics(*, item_results: list, **kwargs: Any) -> list[Evaluation]:
        """Compute aggregate metrics over the full experiment run."""

        def _mean_eval(name: str) -> float | None:
            vals = [
                e.value
                for r in item_results
                for e in r.evaluations
                if e.name == name and e.value is not None
            ]
            return sum(vals) / len(vals) if vals else None

        aggs: list[Evaluation] = []

        # Faithfulness rate
        faithful_vals = [
            e.value
            for r in item_results
            for e in r.evaluations
            if e.name == "faithfulness" and e.value is not None
        ]
        if faithful_vals:
            faithful_count = sum(1 for v in faithful_vals if v == "faithful")
            rate = faithful_count / len(faithful_vals) * 100
            aggs.append(
                Evaluation(
                    name="faithfulness_rate",
                    value=rate,
                    comment=f"{faithful_count}/{len(faithful_vals)} faithful ({rate:.1f}%)",
                )
            )

        # Mean retrieval hit rate
        mean_hit = _mean_eval("retrieval_hit")
        if mean_hit is not None:
            aggs.append(
                Evaluation(
                    name="mean_retrieval_hit",
                    value=mean_hit,
                    comment=f"Mean retrieval_hit: {mean_hit:.3f}",
                )
            )

        return aggs

    # 4. Run the experiment
    run_name = make_experiment_name(config.experiment_name)
    experiment_kwargs = {
        "name": run_name,
        "description": f"RAG evaluation | judge={config.judge_model}",
        "task": evaluate_item,
        "evaluators": [
            relevancy_evaluator,
            faithfulness_evaluator,
            retrieval_evaluator,
        ],
        "run_evaluators": [aggregate_metrics],
        "max_concurrency": 1,  # sequential — keeps logs readable
        "metadata": {
            "settings": settings_str,
            "evaluation_config": evaluation_config_str,
        },
    }

    if config.limit:
        items = dataset.items[: config.limit]
        logger.info(
            "Starting experiment '%s' on %d / %d items from '%s'...",
            run_name,
            len(items),
            len(dataset.items),
            config.dataset_name,
        )
        result = langfuse.run_experiment(data=items, **experiment_kwargs)
    else:
        logger.info(
            "Starting experiment '%s' on dataset '%s' (%d items)...",
            run_name,
            config.dataset_name,
            len(dataset.items),
        )
        result = dataset.run_experiment(**experiment_kwargs)

    logger.info("Experiment complete!")
    # NOTE: result.format() in current Langfuse SDK (3.12.1) has literal "\n"
    # instead of real newlines in many places.
    summary = result.format(include_item_results=config.include_item_results)
    print(summary.replace("\\n", "\n"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and launch the Langfuse evaluation experiment."""
    parser = argparse.ArgumentParser(
        description="Run End-to-End RAG Evaluation via Langfuse Experiments"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Name of the Langfuse dataset to evaluate.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=settings.llm_model,
        help="Model identifier for the judge LLM.",
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
        "--experiment-name",
        type=str,
        default=None,
        help="Optional label appended to the auto-generated timestamp experiment name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of items to evaluate.",
    )
    parser.add_argument(
        "--include-item-results",
        action="store_true",
        help="Include individual item evaluations in the terminal summary.",
    )

    args = parser.parse_args()

    config = EvaluationConfig(
        dataset_name=args.dataset_name,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_provider=args.judge_provider,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        experiment_name=args.experiment_name,
        limit=args.limit,
        include_item_results=args.include_item_results,
    )

    run_experiment(config)


if __name__ == "__main__":
    main()
