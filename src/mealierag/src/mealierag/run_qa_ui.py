"""
Serve the QA Rag interface in the browser.
"""

import logging

import gradio as gr
from qdrant_client.http.models import ScoredPoint

from .config import settings
from .service import create_mealie_rag_service
from .tracing import TraceContext, tracer

logger = logging.getLogger(__name__)

# Initialize service
trace_context = TraceContext()
service = create_mealie_rag_service()


def print_hits(hits: list[ScoredPoint]):
    hits_table = "| Name | Rating | Tags | Category | Score |\n|---|---|---|---|---|\n"
    for hit in hits:
        recipe_url = (
            f"{settings.mealie_external_url}/g/home/r/{hit.payload['recipe_id']}"
        )
        tags = hit.payload.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)
        hits_table += f"| [{hit.payload.get('name', 'N/A')}]({recipe_url}) | {hit.payload.get('rating', 'N/A')} | {tags} | {hit.payload.get('category', 'N/A')} | {hit.score} |\n"
    logger.debug("Hits table", extra={"hits_table": hits_table})
    return hits_table


def transform_fn(inputs):
    """Helper function to disable automatic output tracing"""
    return None


@tracer.observe(transform_to_string=transform_fn)
def process_input(user_input: str):
    """Process the user input and generate a response."""
    trace_context.set_trace_id(tracer.get_current_trace_id())
    tracer.update_current_trace(
        name="qa_ui_trace",
        session_id=trace_context.session_id,
        input=user_input,
    )
    tracer.update_current_span(
        name="qa_ui",
        input=user_input,
    )
    debug_info = "\n\n[View trace](%s)" % tracer.get_trace_url(trace_context.trace_id)

    partial = " 👾 Consulting the digital oracles..."
    yield partial, gr.Markdown(value=debug_info)

    query_extraction = service.generate_queries(user_input)

    partial += "\n 🔍 Finding relevant recipes..."
    yield partial, gr.Markdown(value=debug_info)

    hits = service.retrieve_recipes(query_extraction)

    if not hits:
        yield "I couldn't find any relevant recipes.", gr.Markdown(value=debug_info)
        return

    hit_str = print_hits(hits)
    debug_info = (
        "### 🤓 Recipes context for the above answer: ###\n"
        + hit_str
        + "\n\n[View trace](%s)" % tracer.get_trace_url(trace_context.trace_id)
    )

    partial += "\n 🤔 Done! Processing your request..."
    yield partial, gr.Markdown(value=debug_info)

    messages = service.populate_messages(user_input, hits)

    logger.debug("Generating response...")
    response_stream = service.chat(messages)

    partial = "**🤖 MealieChef:**\n"
    response = ""
    for chunk in response_stream:
        partial += chunk
        response += chunk
        yield partial, gr.Markdown(value=debug_info)

    logger.debug("Response generation finished.")

    yield partial, gr.Markdown(value=debug_info)

    tracer.update_current_span(
        output=response,
    )
    tracer.update_current_trace(
        output=partial + "\n\n" + debug_info,
    )


def chat_fn(message: str, history: list[list[str]]):
    yield from process_input(message)


def handle_like(data: gr.LikeData):
    """Handle the like event from the user."""
    if data.liked:
        tracer.create_score(
            value=1, name="user-feedback", trace_id=trace_context.trace_id
        )
    else:
        tracer.create_score(
            value=0, name="user-feedback", trace_id=trace_context.trace_id
        )


with gr.Blocks() as demo:
    debug_output = gr.Markdown(label="Debug Output", render=False, min_height=300)
    logout_button = gr.Button("Logout", link="/logout", size="sm", variant="secondary")
    chatbot = gr.Chatbot(height=500)
    chatbot.like(handle_like, None, None)
    chatbot.clear(trace_context.create_new_session_id, None, None)
    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
        title="🍳 MealieChef",
        description="Your personal recipe assistant",
        additional_outputs=[debug_output],
    )
    debug_output.render()


def main():
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        auth=(settings.ui_username, settings.ui_password.get_secret_value()),
        share=False,
        max_threads=5,
        enable_monitoring=False,
    )


if __name__ == "__main__":
    main()
