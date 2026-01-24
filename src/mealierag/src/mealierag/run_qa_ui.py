"""
Serve the QA Rag interface in the browser.
"""

import logging
from typing import List

import gradio as gr
import ollama
from qdrant_client.http.models import ScoredPoint

from .chat import populate_messages
from .config import settings
from .embeddings import get_embedding
from .vectordb import get_vector_db_client, retrieve_results

logger = logging.getLogger(__name__)


def get_hits(query, ollama_client, vector_db_client):
    query_vector = get_embedding(query, ollama_client, settings)
    if not query_vector:
        return []
    return retrieve_results(query_vector, vector_db_client, settings.collection_name)


def print_hits(hits: List[ScoredPoint]):
    hits_str = ""
    for hit in hits:
        hits_str += f"**{hit.payload['name']}** Rating: {hit.payload['rating']} Tags: {hit.payload['tags']} Category: {hit.payload['category']}\n"
    logger.info(hits_str)
    return hits_str


def chat_fn(message: str, history: List[List[str]]):
    # TODO: re-use client instances
    ollama_client = ollama.Client(host=settings.ollama_base_url)
    vector_db_client = get_vector_db_client(settings.qdrant_url)

    hits = get_hits(message, ollama_client, vector_db_client)
    if not hits:
        yield "I couldn't find any relevant recipes."
        return
    hit_str = print_hits(hits)

    messages = populate_messages(message, hits)

    response = ollama_client.chat(
        model=settings.llm_model, messages=messages, stream=True
    )

    partial = "**🤖 MealieChef:**\n"
    for chunk in response:
        partial += chunk["message"]["content"]
        yield partial

    partial += (
        "\n\n" + "### 🐛 Recipes context used for the above answer: ###\n" + hit_str
    )
    yield partial


with gr.Blocks(title="🍳 MealieChef") as demo:
    gr.Markdown("# 🍳 MealieChef\nYour personal recipe assistant")

    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Ask me about your recipes...", scale=7),
    )


def main():
    demo.launch(server_name="0.0.0.0", server_port=settings.ui_port)


if __name__ == "__main__":
    main()
