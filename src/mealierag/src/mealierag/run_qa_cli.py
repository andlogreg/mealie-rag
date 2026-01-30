"""
Run the QA Rag interface in the terminal.
"""

import logging
import sys

from qdrant_client.http.models import ScoredPoint

from .service import create_mealie_rag_service

logger = logging.getLogger(__name__)


def print_hits(hits: list[ScoredPoint]):
    for hit in hits:
        print(
            f"**Name:** {hit.payload['name']} **Rating:** {hit.payload['rating']} **Tags:** {hit.payload['tags']} **Category:** {hit.payload['category']}"
        )


def main():
    print("Welcome to Mealie QA! (Type 'exit' to quit)")

    service = create_mealie_rag_service()

    # Initial check
    if not service.check_health():
        logger.error("Service not healthy. Exiting...")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n👤 You: ")
            logger.debug("Received user input", extra={"user_input": user_input})
            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input.strip():
                continue

            print(" 👾 Consulting the digital oracles...")
            queries = service.generate_queries(user_input)

            print(" 🔍 Finding relevant recipes...")

            hits = service.retrieve_recipes(queries)

            if not hits:
                print("No relevant recipes found.")
                continue
            print_hits(hits)

            # Populate messages
            messages = service.populate_messages(user_input, hits)

            # Generate response
            print("\nThinking...\n", end="", flush=True)
            try:
                response_stream = service.chat(messages)
                print("\r🤖 MealieChef: ", end="")
                for chunk in response_stream:
                    content = chunk["message"]["content"]
                    print(content, end="", flush=True)
                print("\n")
            except Exception as e:
                logger.error(f"Error generating response: {e}", exc_info=True)
                print("Sorry, I encountered an error talking to the AI.")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
