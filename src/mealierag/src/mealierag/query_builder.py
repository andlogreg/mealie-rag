import logging
from abc import ABC, abstractmethod

from .api import ChatMessages
from .llm_client import LLMClient
from .models import QueryExtraction
from .prompts import PromptManager, PromptType

logger = logging.getLogger(__name__)


class QueryBuilder(ABC):
    """
    Abstract base class for query builders.

    Query builders are responsible for taking a user's raw input and transforming it
    into one or more search queries.
    """

    @abstractmethod
    def build(self, user_input: str) -> QueryExtraction:
        """
        Build query from the user input.

        Args:
            user_input: The raw query string from the user.

        Returns:
            QueryExtraction: The generated query.
        """
        pass

    def __call__(self, user_input: str) -> QueryExtraction:
        """
        Allow the instance to be called directly to generate queries.
        """
        return self.build(user_input)


class DefaultQueryBuilder(QueryBuilder):
    """
    A simple query builder that uses the user's input as the single search query.
    """

    def build(self, user_input: str) -> QueryExtraction:
        """
        Returns the user input as a single-item list wrapped in QueryExtraction.
        """
        return QueryExtraction(expanded_queries=[user_input])


class MultiQueryQueryBuilder(QueryBuilder):
    """
    Uses an LLM to generate multiple variations of the user's query.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        model: str,
        temperature: float,
        seed: int,
        prompt_manager: PromptManager,
        enable_expand: bool = True,
        enable_culinary_brainstorm: bool = True,
    ):
        """
        Initialize the MultiQueryQueryBuilder.

        Args:
            llm_client: Client for interacting with the LLM API.
            model: The name of the LLM model to use.
            temperature: Sampling temperature for the LLM.
            seed: Random seed for reproducibility.
            prompt_manager: Manager for retrieving prompts.
            enable_expand: Expand the query into multiple variations.
                If False, user input is used as the single query.
            enable_culinary_brainstorm: Rewrite each expanded query as
                a short cooking-instruction sentence.
                If False, raw expanded queries are used.
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.prompt_manager = prompt_manager
        self.enable_expand = enable_expand
        self.enable_culinary_brainstorm = enable_culinary_brainstorm

    def build(self, user_input: str) -> QueryExtraction:
        """
        Generate multiple search queries and extract negative constraints.

        Args:
            user_input: The raw user query.

        Returns:
            QueryExtraction: The generated queries and negative constraints.
        """
        if self.enable_expand:
            prompt = self.prompt_manager.get_prompt(PromptType.MULTI_QUERY_BUILDER)
            messages = prompt.compile(user_input=user_input)
            chat_messages = ChatMessages(messages=messages, prompt=prompt)

            logger.debug(
                "Generating multi-query variations",
                extra={"user_input": user_input, "messages": messages},
            )

            response = self.llm_client.chat(
                chat_messages=chat_messages,
                model=self.model,
                temperature=self.temperature,
                seed=self.seed,
                response_model=QueryExtraction,
            )
            logger.debug("Generated queries", extra={"response": response})
        else:
            logger.debug(
                "Multi-query expansion disabled - using raw user input",
                extra={"user_input": user_input},
            )
            response = QueryExtraction(expanded_queries=[user_input])

        if self.enable_culinary_brainstorm:
            culinary_brainstorm_prompt = self.prompt_manager.get_prompt(
                PromptType.CULINARY_BRAINSTORM
            )
            for idx, query in enumerate(response.expanded_queries):
                messages = culinary_brainstorm_prompt.compile(user_input=query)
                chat_messages = ChatMessages(
                    messages=messages, prompt=culinary_brainstorm_prompt
                )
                culinary_brainstorm_response = self.llm_client.chat(
                    chat_messages=chat_messages,
                    model=self.model,
                    temperature=self.temperature,
                    seed=self.seed,
                )
                logger.debug(
                    "Generated culinary brainstorm",
                    extra={"response": culinary_brainstorm_response},
                )
                response.expanded_queries[idx] = culinary_brainstorm_response
        else:
            logger.debug(
                "Culinary brainstorm disabled - using raw expanded queries",
                extra={"queries": response.expanded_queries},
            )

        logger.debug("Final Generated queries", extra={"response": response})

        return response
