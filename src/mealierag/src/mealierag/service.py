import logging
from collections.abc import Generator
from typing import Any, Callable

from qdrant_client.http.models import ScoredPoint

from .api import ChatMessages
from .chat import populate_messages
from .config import LLMProvider, SearchStrategy, settings
from .embeddings import get_embedding
from .llm_client import LLMClient, OllamaClient, OpenAIClient
from .models import QueryExtraction
from .prompts import LangfusePromptManager
from .query_builder import DefaultQueryBuilder, MultiQueryQueryBuilder, QueryBuilder
from .tracing import tracer
from .vectordb import (
    get_vector_db_client,
    retrieve_results_rrf,
    retrieve_results_simple,
)

logger = logging.getLogger(__name__)


class MealieRAGService:
    def __init__(
        self,
        llm_client: LLMClient,
        vector_db_client: Any,
        prompt_manager: LangfusePromptManager,
        query_builder: QueryBuilder,
        retrieve_results_fn: Callable,
    ):
        self.llm_client = llm_client
        self.vector_db_client = vector_db_client
        self.prompt_manager = prompt_manager
        self.query_builder = query_builder
        self._retrieve_results = retrieve_results_fn

    @tracer.observe(name="service_generate_queries", as_type="span")
    def generate_queries(self, user_input: str) -> QueryExtraction:
        """
        Generate search queries and negative constraints based on user input.
        """
        logger.debug("Generating queries", extra={"user_input": user_input})
        return self.query_builder(user_input)

    @tracer.observe(name="service_retrieve_recipes", as_type="retriever")
    def retrieve_recipes(self, query_extraction: QueryExtraction) -> list[ScoredPoint]:
        """
        Retrieve relevant recipes using the provided queries and constraints.
        """
        logger.debug(
            "Retrieving recipes",
            extra={
                "queries_count": len(query_extraction.expanded_queries),
                "query_extraction": query_extraction.model_dump_json(),
            },
        )
        query_vectors = get_embedding(
            query_extraction.expanded_queries, self.llm_client, settings
        )

        if not query_vectors:
            logger.warning("No embeddings generated for queries")
            return []

        return self._retrieve_results(
            query_vectors,
            self.vector_db_client,
            settings.vectordb_collection_name,
            k=settings.vectordb_k,
            query_extraction=query_extraction,
        )

    def populate_messages(
        self, user_input: str, hits: list[ScoredPoint]
    ) -> ChatMessages:
        """
        Populate messages with user input and retrieved recipes.
        """
        logger.debug(
            "Populating messages",
            extra={"user_input": user_input, "hits_count": len(hits)},
        )
        return populate_messages(user_input, hits, self.prompt_manager)

    @tracer.observe(name="service_chat", as_type="span")
    def chat(self, chat_messages: ChatMessages) -> Generator[str, None, None]:
        """
        Stream chat response from LLM.
        """
        logger.debug(
            "Generating chat response",
            extra={
                "messages_count": chat_messages.messages_count,
                "messages": chat_messages.messages,
            },
        )
        return self.llm_client.streaming_chat(
            chat_messages=chat_messages,
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
    vector_db_client = get_vector_db_client(
        url=settings_obj.vectordb_url, path=settings_obj.vectordb_path
    )
    prompt_manager = LangfusePromptManager()

    if settings_obj.llm_provider == LLMProvider.OLLAMA:
        llm_client = OllamaClient(base_url=settings_obj.llm_base_url)
    elif settings_obj.llm_provider == LLMProvider.OPENAI:
        llm_client = OpenAIClient(
            api_key=settings_obj.llm_api_key.get_secret_value(),
            base_url=settings_obj.llm_base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {settings_obj.llm_provider}")

    if settings_obj.search_strategy == SearchStrategy.MULTIQUERY:
        query_builder = MultiQueryQueryBuilder(
            llm_client=llm_client,
            model=settings_obj.llm_model,
            temperature=settings_obj.llm_temperature,
            seed=settings_obj.llm_seed,
            prompt_manager=prompt_manager,
        )
        retrieve_results_fn = retrieve_results_rrf
    else:
        query_builder = DefaultQueryBuilder()
        retrieve_results_fn = retrieve_results_simple

    return MealieRAGService(
        llm_client=llm_client,
        vector_db_client=vector_db_client,
        prompt_manager=prompt_manager,
        query_builder=query_builder,
        retrieve_results_fn=retrieve_results_fn,
    )
