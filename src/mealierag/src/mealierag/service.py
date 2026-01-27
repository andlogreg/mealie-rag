import logging
from collections.abc import Generator
from typing import Any

from qdrant_client.http.models import ScoredPoint

from .chat import populate_messages
from .config import SearchStrategy, settings
from .embeddings import get_embedding
from .llm_client import OllamaClient
from .query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder
from .vectordb import (
    get_vector_db_client,
    retrieve_results_rrf,
    retrieve_results_simple,
)

logger = logging.getLogger(__name__)


class MealieRAGService:
    def __init__(self):
        self.ollama_client = OllamaClient(settings.ollama_base_url)
        self.vector_db_client = get_vector_db_client(settings.vectordb_url)

        if settings.search_strategy == SearchStrategy.MULTIQUERY:
            self.query_builder = MultiQueryQueryBuilder(
                ollama_client=self.ollama_client,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                seed=settings.llm_seed,
            )
            self._retrieve_results = retrieve_results_rrf
        else:
            self.query_builder = DefaultQueryBuilder()
            self._retrieve_results = retrieve_results_simple

    def generate_queries(self, user_input: str) -> list[str]:
        """
        Generate search queries based on user input.
        """
        logger.debug("Generating queries", extra={"user_input": user_input})
        return self.query_builder(user_input)

    def retrieve_recipes(self, queries: list[str]) -> list[ScoredPoint]:
        """
        Retrieve relevant recipes using the provided queries.
        """
        logger.debug("Retrieving recipes", extra={"queries_count": len(queries)})
        query_vectors = get_embedding(queries, self.ollama_client, settings)

        if not query_vectors:
            logger.warning("No embeddings generated for queries")
            return []

        return self._retrieve_results(
            query_vectors,
            self.vector_db_client,
            settings.vectordb_collection_name,
            k=settings.vectordb_k,
        )

    def populate_messages(
        self, user_input: str, hits: list[ScoredPoint]
    ) -> list[dict[str, str]]:
        """
        Populate messages with user input and retrieved recipes.
        """
        logger.debug(
            "Populating messages",
            extra={"user_input": user_input, "hits_count": len(hits)},
        )
        return populate_messages(user_input, hits)

    def chat(
        self, messages: list[dict[str, str]]
    ) -> Generator[dict[str, Any], None, None]:
        """
        Stream chat response from LLM.
        """
        logger.debug(
            "Generating chat response",
            extra={"messages_count": len(messages), "messages": messages},
        )
        return self.ollama_client.streaming_chat(
            messages=messages,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            seed=settings.llm_seed,
        )

    def check_health(self) -> bool:
        """Check if service is healthy."""
        healthy = self.vector_db_client.collection_exists(
            settings.vectordb_collection_name
        )
        if not healthy:
            logger.error(f"Collection '{settings.vectordb_collection_name}' not found.")
        return healthy
