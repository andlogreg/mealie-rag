"""
Run the QA Rag interface in the terminal.
"""

import logging
import sys

from qdrant_client.http.models import ScoredPoint

from .service import create_mealie_rag_service
from .tracing import TraceContext, tracer

# Initialize service
trace_context = TraceContext()
service = create_mealie_rag_service()

logger = logging.getLogger(__name__)


def print_hits(hits: list[ScoredPoint]):
    for hit in hits:
        print(
            f"**Name:** {hit.payload['name']} **Rating:** {hit.payload['rating']} **Tags:** {hit.payload['tags']} **Category:** {hit.payload['category']} **Score:** {hit.score}"
        )


def transform_fn(inputs):
    """Helper function to disable automatic output tracing"""
    return None


@tracer.observe(transform_to_string=transform_fn)
def process_input(user_input: str):
    trace_context.set_trace_id(tracer.get_current_trace_id())
    tracer.update_current_trace(
        name="qa_cli_trace",
        session_id=trace_context.session_id,
        input=user_input,
    )
    tracer.update_current_span(
        name="qa_cli",
        input=user_input,
    )
    print(" 👾 Consulting the digital oracles...")
    query_extraction = service.generate_queries(user_input)

    print(" 🔍 Finding relevant recipes...")

    hits = service.retrieve_recipes(query_extraction)

    if not hits:
        print("No relevant recipes found.")
        return
    print_hits(hits)

    # Populate messages
    messages = service.populate_messages(user_input, hits)

    # Generate response
    print("\nThinking...\n", end="", flush=True)
    full_response = ""
    try:
        response_stream = service.chat(messages)
        print("\r🤖 MealieChef: ", end="")
        for chunk in response_stream:
            full_response += chunk
            print(chunk, end="", flush=True)
        print("\n")
    except Exception as e:
        logger.error(f"Error generating response: {e}", exc_info=True)
        print("Sorry, I encountered an error talking to the AI.")

    tracer.update_current_span(
        output=full_response,
    )
    tracer.update_current_trace(
        output=full_response,
    )

    return full_response


def main():
    print("Welcome to Mealie QA! (Type 'exit' to quit)")

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

            process_input(user_input)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
