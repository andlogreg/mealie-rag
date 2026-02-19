"""
Shared primitives for eval scripts.
"""

from __future__ import annotations

import ast
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from instructor import Mode
from openai import OpenAI
from pydantic import BaseModel, SecretStr
from qdrant_client import models as qdrant_models
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric, NumericMetric

from mealierag.config import LLMProvider
from mealierag.prompts import LangfusePromptManager, PromptType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DATASET_NAME = f"{datetime.now().strftime('%Y%m%d')}_queries"

# Ingredient category expansion (e.g. "meat" → ["chicken", "beef", …])
# Used when building ground-truth Qdrant filters.
INGREDIENT_CATEGORIES: dict[str, list[str]] = {
    "meat": [
        "chicken",
        "beef",
        "pork",
        "lamb",
        "steak",
        "turkey",
        "salami",
        "ham",
        "bacon",
        "sausage",
        "duck",
        "mince",
    ],
    "fish": ["salmon", "tuna", "cod", "hake", "bass", "trout", "sardine", "mackerel"],
    "seafood": [
        "shrimp",
        "prawn",
        "crab",
        "lobster",
        "mussel",
        "clam",
        "squid",
        "octopus",
    ],
    "vegetable": [
        "carrot",
        "leek",
        "onion",
        "pepper",
        "broccoli",
        "cauliflower",
        "spinach",
        "cabbage",
        "zucchini",
        "aubergine",
        "eggplant",
        "potato",
        "tomato",
        "cucumber",
        "lettuce",
        "pea",
        "bean",
    ],
}


# ---------------------------------------------------------------------------
# Ground-truth filter builder
# ---------------------------------------------------------------------------


def _expand_ingredient(name: str) -> list[str]:
    """Return the ingredient name plus any category synonyms."""
    lower = name.lower()
    return [lower, *INGREDIENT_CATEGORIES.get(lower, [])]


def build_ground_truth_filters(
    expected: dict[str, Any],
) -> qdrant_models.Filter | None:
    """Translate an ``expected_properties`` dict into a Qdrant Filter.

    Returns ``None`` when *expected* is empty, meaning no filter should be applied.
    """
    if not expected:
        return None

    must: list[qdrant_models.Condition] = []
    must_not: list[qdrant_models.Condition] = []

    for key, value in expected.items():
        # ── must-have ingredients ────────────────────────────────────────
        if key == "must_have_ingredients":
            for req in value:
                candidates = _expand_ingredient(req)
                # Each required ingredient: at least one synonym must match
                must.append(
                    qdrant_models.Filter(
                        should=[
                            qdrant_models.FieldCondition(
                                key="normalized_ingredients",
                                match=qdrant_models.MatchText(text=c),
                            )
                            for c in candidates
                        ]
                    )
                )

        # ── must-not-have ingredients ────────────────────────────────────
        elif key == "must_not_have_ingredients":
            for forb in value:
                for candidate in _expand_ingredient(forb):
                    must_not.append(
                        qdrant_models.FieldCondition(
                            key="normalized_ingredients",
                            match=qdrant_models.MatchText(text=candidate),
                        )
                    )

        # ── boolean / rating / time ──────────────────────────────────────
        elif key == "is_healthy":
            must.append(
                qdrant_models.FieldCondition(
                    key="is_healthy",
                    match=qdrant_models.MatchValue(value=value),
                )
            )

        elif key == "min_rating":
            must.append(
                qdrant_models.FieldCondition(
                    key="rating",
                    range=qdrant_models.Range(gte=value),
                )
            )

        elif key == "max_total_time_minutes":
            must.append(
                qdrant_models.FieldCondition(
                    key="total_time_minutes",
                    range=qdrant_models.Range(lte=value),
                )
            )

        elif key == "max_ingredient_count":
            must.append(
                qdrant_models.FieldCondition(
                    key="ingredient_count",
                    range=qdrant_models.Range(lte=value),
                )
            )

        # ── list payload fields (stored lowercase) ───────────────────────
        elif key == "tags":
            # Ignored for now — tag matching not yet implemented in Qdrant payload
            pass
            # must.append(
            #     qdrant_models.FieldCondition(
            #         key="tags",
            #         match=qdrant_models.MatchAny(any=[t.lower() for t in value]),
            #     )
            # )

        elif key == "tools":
            must.append(
                qdrant_models.FieldCondition(
                    key="tools",
                    match=qdrant_models.MatchAny(any=[t.lower() for t in value]),
                )
            )

        elif key == "method":
            must.append(
                qdrant_models.FieldCondition(
                    key="method",
                    match=qdrant_models.MatchAny(any=[m.lower() for m in value]),
                )
            )

        elif key == "recipeCategory":
            must.append(
                qdrant_models.FieldCondition(
                    key="category",
                    match=qdrant_models.MatchAny(any=[c.lower() for c in value]),
                )
            )

        # ── skip non-filter keys ─────────────────────────────────────────
        # Ignored for now
        elif key == "limit":
            pass

        else:
            logger.warning("Unknown expected_properties key '%s' — skipped.", key)

    return qdrant_models.Filter(
        must=must or None,
        must_not=must_not or None,
    )


def get_relevant_ids(
    client: Any,
    collection_name: str,
    gt_filter: qdrant_models.Filter,
) -> set[str]:
    """Scroll *collection_name* and return the set of all matching recipe IDs."""
    relevant: set[str] = set()
    offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=gt_filter,
            limit=1000,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points:
            relevant.add(str(point.id))

        if next_offset is None:
            break
        offset = next_offset

    return relevant


# ---------------------------------------------------------------------------
# Retrieval metric computation
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Retrieval quality scores for a single query."""

    precision: float
    recall: float
    recall_capped: float
    mrr: float
    ndcg: float
    hit: bool
    relevant_count: int


def _calculate_ndcg(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalised Discounted Cumulative Gain at *k*."""
    dcg = sum(
        1 / math.log2(i + 2)
        for i, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    num_relevant = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 2) for i in range(num_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def compute_retrieval_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
) -> RetrievalMetrics:
    """Compute Precision@K, Recall@K, MRR, nDCG@K, and Hit Rate."""
    k = len(retrieved_ids)
    relevant_retrieved = sum(1 for rid in retrieved_ids if rid in relevant_ids)

    precision = relevant_retrieved / k if k > 0 else 0.0
    recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0.0
    recall_capped = (
        relevant_retrieved / min(k, len(relevant_ids))
        if (relevant_ids and k > 0)
        else 0.0
    )
    hit = any(rid in relevant_ids for rid in retrieved_ids)

    mrr = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            mrr = 1 / (i + 1)
            break

    ndcg = _calculate_ndcg(retrieved_ids, relevant_ids, k)

    return RetrievalMetrics(
        precision=precision,
        recall=recall,
        recall_capped=recall_capped,
        mrr=mrr,
        ndcg=ndcg,
        hit=hit,
        relevant_count=len(relevant_ids),
    )


def log_retrieval_metrics(metrics: RetrievalMetrics) -> None:
    """Emit a single INFO log line with all retrieval scores."""
    logger.info(
        "Retrieval — P@K=%.3f R@K=%.3f R_capped@K=%.3f MRR=%.3f nDCG=%.3f Hit=%s (relevant=%d)",
        metrics.precision,
        metrics.recall,
        metrics.recall_capped,
        metrics.mrr,
        metrics.ndcg,
        metrics.hit,
        metrics.relevant_count,
    )


# ---------------------------------------------------------------------------
# Context helper
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------


class BaseEvaluationConfig(BaseModel):
    """Fields shared by all evaluation backends."""

    judge_model: str
    judge_temperature: float
    judge_provider: LLMProvider
    judge_base_url: str
    judge_api_key: SecretStr | None
    experiment_name: str | None
    limit: int | None


# ---------------------------------------------------------------------------
# Setup factories
# ---------------------------------------------------------------------------


def build_judge_llm(config: BaseEvaluationConfig) -> Any:
    """Construct and return a RAGAS-compatible judge LLM from *config*.

    Raises ``ValueError`` for unsupported providers.
    """
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

    return llm_factory(
        config.judge_model,
        client=openai_client,
        temperature=config.judge_temperature,
        top_p=None,
        adapter="instructor",
        mode=Mode.JSON_SCHEMA,
    )


def build_ragas_metrics(
    prompt_manager: LangfusePromptManager,
) -> tuple[NumericMetric, DiscreteMetric]:
    """Build and return the RAGAS relevancy and faithfulness metrics."""
    relevancy_metric = NumericMetric(
        name="relevancy",
        prompt=prompt_manager.get_prompt(
            PromptType.METRIC_RELEVANCY, label="production"
        ).compile(),
        allowed_values=(1, 5),
    )
    faithfulness_metric = DiscreteMetric(
        name="faithfulness",
        prompt=prompt_manager.get_prompt(
            PromptType.METRIC_FAITHFULNESS, label="production"
        ).compile(),
        allowed_values=["hallucination", "faithful"],
    )
    return relevancy_metric, faithfulness_metric


# ---------------------------------------------------------------------------
# Generic utils
# ---------------------------------------------------------------------------


def mean(values: list[float]) -> float:
    """Return the arithmetic mean of *values*, or ``0.0`` if the list is empty."""
    return sum(values) / len(values) if values else 0.0


def parse_expected_properties(raw: str | dict | None) -> dict[str, Any]:
    """Parse an `expected_properties` value.

    Accepts:
    - `None` / empty string → empty dict
    - `dict` → returned as-is
    - `str` → parsed with `ast.literal_eval`
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        result = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse expected_properties: {raw!r}") from exc
    if not isinstance(result, dict):
        raise ValueError(
            f"expected_properties must be a dict, got {type(result).__name__}: {raw!r}"
        )
    return result


# ---------------------------------------------------------------------------
# Experiment name helper
# ---------------------------------------------------------------------------


def make_experiment_name(label: str | None) -> str:
    """Return a timestamped experiment name, optionally suffixed with *label*."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{label}" if label else timestamp
