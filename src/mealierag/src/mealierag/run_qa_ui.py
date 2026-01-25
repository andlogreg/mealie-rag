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
    return retrieve_results(
        query_vector,
        vector_db_client,
        settings.vectordb_collection_name,
        k=settings.vectordb_k,
    )


def print_hits(hits: List[ScoredPoint]):
    hits_table = "| Name | Rating | Tags | Category |\n|---|---|---|---|\n"
    for hit in hits:
        tags = hit.payload.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)
        hits_table += f"| {hit.payload.get('name', 'N/A')} | {hit.payload.get('rating', 'N/A')} | {tags} | {hit.payload.get('category', 'N/A')} |\n"
    logger.info(hits_table)
    return hits_table


def chat_fn(message: str, history: List[List[str]]):
    # TODO: re-use client instances
    ollama_client = ollama.Client(host=settings.ollama_base_url)
    vector_db_client = get_vector_db_client(settings.vectordb_url)

    partial = " 🔍 Finding relevant recipes..."
    yield partial
    hits = get_hits(message, ollama_client, vector_db_client)
    if not hits:
        yield "I couldn't find any relevant recipes."
        return
    hit_str = print_hits(hits)

    partial += "\n 🤔 Done! Processing your request..."
    yield partial
    messages = populate_messages(message, hits)

    logger.info("Generating response...", extra={"messages": messages})
    options = {
        "temperature": settings.llm_temperature,
    }
    if settings.llm_seed is not None:
        options["seed"] = settings.llm_seed
    response = ollama_client.chat(
        model=settings.llm_model, messages=messages, stream=True, options=options
    )

    partial = "**🤖 MealieChef:**\n"
    for chunk in response:
        partial += chunk["message"]["content"]
        yield partial

    logger.info("Response generation finished.", extra={"partial": partial})

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
    logout_button = gr.Button("Logout", link="/logout")


def main():
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        auth=(settings.ui_username, settings.ui_password.get_secret_value()),
    )


if __name__ == "__main__":
    main()
