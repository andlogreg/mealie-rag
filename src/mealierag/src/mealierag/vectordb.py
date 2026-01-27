"""
Vector database client for Qdrant.
"""

import logging

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import ScoredPoint

logger = logging.getLogger(__name__)


def get_vector_db_client(url: str) -> QdrantClient:
    """
    Get a Qdrant client instance.

    Args:
        url: The URL of the Qdrant service.

    Returns:
        QdrantClient: Configured Qdrant client.
    """
    return QdrantClient(url=url)


def retrieve_results_simple(
    query_vectors: list[list[float]],
    client: QdrantClient,
    collection_name: str,
    k: int = 3,
) -> list[ScoredPoint]:
    """
    Retrieve search results from Qdrant using a single query vector.

    Args:
        query_vectors: A list containing a single query vector.
        client: The Qdrant client.
        collection_name: The name of the collection to search.
        k: The number of results to return.

    Returns:
        List[ScoredPoint]: The search results.

    Raises:
        ValueError: If more than one query vector is provided.
    """
    if len(query_vectors) != 1:
        error_msg = f"Simple retrieval supports exactly one query vector, got {len(query_vectors)}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(
        "Executing simple vector search",
        extra={"collection": collection_name, "k": k},
    )

    results = client.query_points(
        collection_name=collection_name, query=query_vectors[0], limit=k
    )
    return results.points


def retrieve_results_rrf(
    query_vectors: list[list[float]],
    client: QdrantClient,
    collection_name: str,
    k: int = 3,
) -> list[ScoredPoint]:
    """
    Retrieve search results from Qdrant using Reciprocal Rank Fusion (RRF).

    Args:
        query_vectors: A list of query vectors.
        client: The Qdrant client.
        collection_name: The name of the collection to search.
        k: The number of results to return.

    Returns:
        List[ScoredPoint]: The search results.
    """
    logger.debug(
        f"Executing RRF search with {len(query_vectors)} vectors",
        extra={"collection": collection_name, "k": k},
    )

    prefetch = [
        models.Prefetch(
            query=query_vector,
            limit=k,
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
