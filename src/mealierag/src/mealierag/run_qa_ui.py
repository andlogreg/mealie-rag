"""
Serve the QA Rag interface in the browser.
"""

import logging

import gradio as gr
from qdrant_client.http.models import ScoredPoint

from .config import settings
from .service import MealieRAGService

logger = logging.getLogger(__name__)

# Initialize service
service = MealieRAGService()


def print_hits(hits: list[ScoredPoint]):
    hits_table = "| Name | Rating | Tags | Category |\n|---|---|---|---|\n"
    for hit in hits:
        tags = hit.payload.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)
        hits_table += f"| {hit.payload.get('name', 'N/A')} | {hit.payload.get('rating', 'N/A')} | {tags} | {hit.payload.get('category', 'N/A')} |\n"
    logger.debug("Hits table", extra={"hits_table": hits_table})
    return hits_table


def chat_fn(message: str, history: list[list[str]]):
    partial = " 👾 Consulting the digital oracles..."
    yield partial

    queries = service.generate_queries(message)

    partial += "\n 🔍 Finding relevant recipes..."
    yield partial

    hits = service.retrieve_recipes(queries)

    if not hits:
        yield "I couldn't find any relevant recipes."
        return

    hit_str = print_hits(hits)

    partial += "\n 🤔 Done! Processing your request..."
    yield partial

    messages = service.populate_messages(message, hits)

    logger.debug("Generating response...")
    response_stream = service.chat(messages)

    partial = "**🤖 MealieChef:**\n"
    for chunk in response_stream:
        partial += chunk["message"]["content"]
        yield partial

    logger.debug("Response generation finished.")

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
