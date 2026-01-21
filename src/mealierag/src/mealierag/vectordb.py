"""
Vector database client for Qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint


def get_vector_db_client(url: str) -> QdrantClient:
    """Get a Qdrant client"""
    return QdrantClient(url=url)


def retrieve_results(
    query_vector, client: QdrantClient, collection_name: str, k: int = 3
) -> list[ScoredPoint]:
    """Retrieve search results from Qdrant"""
    results = client.query_points(
        collection_name=collection_name, query=query_vector, limit=k
    )
    return results.points
