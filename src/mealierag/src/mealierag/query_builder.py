import logging
import re
from abc import ABC, abstractmethod

from .llm_client import OllamaClient

logger = logging.getLogger(__name__)

MULTI_QUERY_PROMPT = """
You are a culinary search optimization expert. Your goal is to expand a user's recipe request into five distinct search queries to ensure the most relevant recipes are retrieved from a vector database.

## INSTRUCTIONS
Generate 5 variations of the original user question. Focus on different culinary dimensions such as:
1.  **Direct Synonyms:** Use different names for the same dish or key ingredients.
2.  **Ingredient-Based:** Focus on the primary flavors or components.
3.  **Technique/Method:** Reference how the dish is prepared (e.g., "slow cooked," "one-pot," "grilled").
4.  **Occasion/Category:** Relate it to meal types (e.g., "quick weeknight dinner," "healthy breakfast").
5.  **Descriptive/Sensory:** Use adjectives describing texture or flavor profiles (e.g., "spicy," "crunchy," "comfort food").
6. **General -> Specific:** If present, translate generic terms that one can use for search but typically are not mentioned in recipes (eg, "meat") to related more specific terms that could actually be mentioned in recipes (eg, "Chicken", "beef", etc)

## OUTPUT FORMAT
Provide exclusively the variations separated by newlines. Do not include numbering or any introductory text or notes.
"""


class QueryBuilder(ABC):
    """
    Abstract base class for query builders.

    Query builders are responsible for taking a user's raw input and transforming it
    into one or more search queries.
    """

    @abstractmethod
    def build(self, user_input: str) -> list[str]:
        """
        Build a list of search queries from the user input.

        Args:
            user_input: The raw query string from the user.

        Returns:
            A list of query strings to be used for retrieval.
        """
        pass

    def __call__(self, user_input: str) -> list[str]:
        """
        Allow the instance to be called directly to generate queries.
        """
        return self.build(user_input)


class DefaultQueryBuilder(QueryBuilder):
    """
    A simple query builder that uses the user's input as the single search query.
    """

    def build(self, user_input: str) -> list[str]:
        """
        Returns the user input as a single-item list.
        """
        return [user_input]


class MultiQueryQueryBuilder(QueryBuilder):
    """
    Uses an LLM to generate multiple variations of the user's query.
    """

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: str,
        temperature: float,
        seed: int,
    ):
        """
        Initialize the MultiQueryQueryBuilder.

        Args:
            ollama_client: Client for interacting with the Ollama API.
            model: The name of the LLM model to use.
            temperature: Sampling temperature for the LLM.
            seed: Random seed for reproducibility.
        """
        self.system_prompt = MULTI_QUERY_PROMPT
        self.ollama_client = ollama_client
        self.model = model
        self.temperature = temperature
        self.seed = seed

    def build(self, user_input: str) -> list[str]:
        """
        Generate multiple search queries.

        Args:
            user_input: The raw user query.

        Returns:
            A list of generated search queries.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input},
        ]

        logger.debug(
            "Generating multi-query variations", extra={"user_input": user_input}
        )

        response = self.ollama_client.chat(
            messages=messages,
            model=self.model,
            temperature=self.temperature,
            seed=self.seed,
        )
        logger.debug("Generated queries", extra={"response": response})
        queries = self._parse_response(response)
        logger.debug("Parsed queries", extra={"queries": queries})
        return queries

    def _parse_response(self, response: str) -> list[str]:
        """
        Parse and clean the LLM response into a list of queries.
        Handles potentially malformed output like numbered lists.
        """
        queries = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue

            # Remove leading numbers/bullets (e.g., "1. ", "- ", "* ")
            # This regex matches optional whitespace, optional number/bullet, and checks for the rest
            cleaned_line = re.sub(r"^[\d\-\*\.]+\s+", "", line)

            if cleaned_line:
                queries.append(cleaned_line)

        return queries
