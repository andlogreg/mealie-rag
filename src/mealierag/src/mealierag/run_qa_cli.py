"""
Run the QA Rag interface in the terminal.
"""

import sys
from typing import Dict, List

import ollama
from qdrant_client.http.models import ScoredPoint

from .chat import populate_messages
from .config import settings
from .embeddings import get_embedding
from .vectordb import get_vector_db_client, retrieve_results

# Client initialization
ollama_client = ollama.Client(host=settings.ollama_base_url)
vector_db_client = get_vector_db_client(settings.qdrant_url)


def get_hits(query) -> List[ScoredPoint]:
    query_vector = get_embedding(query, ollama_client, settings)
    if not query_vector:
        return []
    return retrieve_results(query_vector, vector_db_client, settings.collection_name)


def print_hits(hits: List[ScoredPoint]):
    for hit in hits:
        print(
            f"**Name:** {hit.payload['name']} **Rating:** {hit.payload['rating']} **Tags:** {hit.payload['tags']} **Category:** {hit.payload['category']}"
        )


def generate_response(messages: List[Dict[str, str]]):
    print("\nThinking...", end="", flush=True)
    try:
        response = ollama_client.chat(
            model=settings.llm_model, messages=messages, stream=True
        )
        print("\r🤖 MealieChef: ", end="")
        full_response = ""
        for chunk in response:
            content = chunk["message"]["content"]
            print(content, end="", flush=True)
            full_response += content
        print("\n")
        return full_response
    except Exception as e:
        print(f"\nError generating response: {e}")
        return "Sorry, I encountered an error talking to the AI."


def main():
    print("Welcome to Mealie QA! (Type 'exit' to quit)")

    # Initial check if collection exists
    # TODO: Abstract this to vectordb.py
    if not vector_db_client.collection_exists(settings.collection_name):
        print("Error: Recipe collection not found. Did you run 'scripts/ingest.py'?")
        sys.exit(1)

    while True:
        try:
            user_input = input("\n👤 You: ")
            if user_input.lower() in ["exit", "quit"]:
                break

            if not user_input.strip():
                continue

            # Get hits
            hits = get_hits(user_input)
            if not hits:
                print("No relevant recipes found.")
                continue
            print_hits(hits)

            # Populate messages
            messages = populate_messages(user_input, hits)

            # Generate response
            generate_response(messages)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
