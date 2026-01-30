import logging
from collections.abc import Generator
from typing import Any, Callable

from qdrant_client.http.models import ScoredPoint

from .chat import populate_messages
from .config import SearchStrategy, settings
from .embeddings import get_embedding
from .llm_client import OllamaClient
from .prompts import LangfusePromptManager
from .query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder, QueryBuilder
from .vectordb import (
    get_vector_db_client,
    retrieve_results_rrf,
    retrieve_results_simple,
)

logger = logging.getLogger(__name__)


class MealieRAGService:
    def __init__(
        self,
        ollama_client: OllamaClient,
        vector_db_client: Any,
        prompt_manager: LangfusePromptManager,
        query_builder: QueryBuilder,
        retrieve_results_fn: Callable,
    ):
        self.ollama_client = ollama_client
        self.vector_db_client = vector_db_client
        self.prompt_manager = prompt_manager
        self.query_builder = query_builder
        self._retrieve_results = retrieve_results_fn

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
        return populate_messages(user_input, hits, self.prompt_manager)

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


def create_mealie_rag_service(settings_obj=settings) -> MealieRAGService:
    """
    Factory function to create a MealieRAGService instance with all dependencies.
    """
    ollama_client = OllamaClient(settings_obj.ollama_base_url)
    vector_db_client = get_vector_db_client(settings_obj.vectordb_url)
    prompt_manager = LangfusePromptManager()

    if settings_obj.search_strategy == SearchStrategy.MULTIQUERY:
        query_builder = MultiQueryQueryBuilder(
            ollama_client=ollama_client,
            model=settings_obj.llm_model,
            temperature=settings_obj.llm_temperature,
            seed=settings_obj.llm_seed,
        )
        retrieve_results_fn = retrieve_results_rrf
    else:
        query_builder = DefaultQueryBuilder()
        retrieve_results_fn = retrieve_results_simple

    return MealieRAGService(
        ollama_client=ollama_client,
        vector_db_client=vector_db_client,
        prompt_manager=prompt_manager,
        query_builder=query_builder,
        retrieve_results_fn=retrieve_results_fn,
    )
