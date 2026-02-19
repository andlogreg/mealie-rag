"""
Vector database client for Qdrant.
"""

import logging

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint

from .models import QueryExtraction

logger = logging.getLogger(__name__)


def get_vector_db_client(
    url: str | None = None, path: str | None = None
) -> QdrantClient:
    """
    Get a Qdrant client instance.
    Prioritizes local path if provided, otherwise uses URL.

    Args:
        url: The URL of the Qdrant service.
        path: The local path for Qdrant persistence.

    Returns:
        QdrantClient: Configured Qdrant client.
    """
    if path:
        logger.info(f"Initializing Qdrant client with local path: {path}")
        return QdrantClient(path=path)

    if url:
        return QdrantClient(url=url)

    raise ValueError("Either url or path must be provided to initialize QdrantClient")


def _build_filters(query_extraction: QueryExtraction | None) -> models.Filter | None:
    """
    Build Qdrant filters from the extracted query parameters.

    Args:
        query_extraction: Extracted query parameters.

    Returns:
        models.Filter or None: Qdrant filter object or None if no filters are needed.
    """
    if not query_extraction:
        return None

    must_not_conditions = []
    must_conditions = []

    # Negative Ingredients
    if query_extraction.negative_ingredients:
        for ing in query_extraction.negative_ingredients:
            must_not_conditions.append(
                models.FieldCondition(
                    key="normalized_ingredients",
                    match=models.MatchText(text=ing),
                )
            )

    # Ratings Range
    if (
        query_extraction.min_rating is not None
        or query_extraction.max_rating is not None
    ):
        must_conditions.append(
            models.FieldCondition(
                key="rating",
                range=models.Range(
                    gte=query_extraction.min_rating,
                    lt=query_extraction.max_rating,
                ),
            )
        )

    if not must_not_conditions and not must_conditions:
        return None

    return models.Filter(
        must=must_conditions if must_conditions else None,
        must_not=must_not_conditions if must_not_conditions else None,
    )


def retrieve_results_simple(
    query_vectors: list[list[float]],
    client: QdrantClient,
    collection_name: str,
    k: int = 3,
    query_extraction: QueryExtraction | None = None,
) -> list[ScoredPoint]:
    """
    Retrieve search results from Qdrant using a single query vector.
    """
    if len(query_vectors) != 1:
        error_msg = f"Simple retrieval supports exactly one query vector, got {len(query_vectors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    query_filter = _build_filters(query_extraction)

    logger.debug(
        "Executing simple vector search",
        extra={
            "collection": collection_name,
            "k": k,
            "query_extraction": query_extraction.model_dump()
            if query_extraction
            else None,
            "query_filter": query_filter.model_dump() if query_filter else None,
        },
    )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vectors[0],
        limit=k,
        query_filter=query_filter,
    )
    return results.points


def retrieve_results_rrf(
    query_vectors: list[list[float]],
    client: QdrantClient,
    collection_name: str,
    k: int = 3,
    query_extraction: QueryExtraction | None = None,
) -> list[ScoredPoint]:
    """
    Retrieve search results from Qdrant using Reciprocal Rank Fusion (RRF).
    """

    query_filter = _build_filters(query_extraction)
    logger.debug(
        f"Executing RRF search with {len(query_vectors)} vectors",
        extra={
            "collection": collection_name,
            "k": k,
            "query_extraction": query_extraction.model_dump()
            if query_extraction
            else None,
            "query_filter": query_filter.model_dump() if query_filter else None,
        },
    )

    prefetch = [
        models.Prefetch(
            query=query_vector,
            limit=k,
            filter=query_filter,
        )
        for query_vector in query_vectors
    ]

    results = client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=k,
    )
    return results.points
