"""
Serve the QA Rag interface in the browser.
"""

import logging

import gradio as gr

from .config import settings
from .models import RecipeHit
from .qa_ui_core import _TABLE_CSS, build_demo, print_hits
from .service import create_mealie_rag_service
from .tracing import TraceContext, tracer

logger = logging.getLogger(__name__)

service = create_mealie_rag_service()


# ---------------------------------------------------------------------------
# Standalone backend callbacks
# ---------------------------------------------------------------------------


@tracer.observe(transform_to_string=lambda _: None)
def process_input(user_input: str, ctx: TraceContext):
    """Process the user input and generate a response."""
    ctx.set_trace_id(tracer.get_current_trace_id())
    tracer.update_current_trace(
        name="qa_ui_trace",
        session_id=ctx.session_id,
        input=user_input,
    )
    tracer.update_current_span(name="qa_ui", input=user_input)

    trace_link = "\n\n[View trace](%s)" % tracer.get_trace_url(ctx.trace_id)
    partial = " 👾 Consulting the digital oracles..."
    yield partial, gr.Markdown(value=trace_link), ctx

    query_extraction = service.generate_queries(user_input)

    partial += "\n 🔍 Finding relevant recipes..."
    yield partial, gr.Markdown(value=trace_link), ctx

    hits = service.retrieve_recipes(query_extraction)

    if not hits:
        yield (
            "I couldn't find any relevant recipes.",
            gr.Markdown(value=trace_link),
            ctx,
        )
        return

    recipe_hits = [RecipeHit.from_scored_point(h) for h in hits]

    debug_info = (
        "### 🤓 Recipes context for the above answer: ###\n"
        + print_hits(recipe_hits)
        + "\n\n[View trace](%s)" % tracer.get_trace_url(ctx.trace_id)
    )

    partial += "\n 🤔 Done! Processing your request..."
    yield partial, gr.Markdown(value=debug_info), ctx

    messages = service.populate_messages(user_input, hits)

    logger.debug("Generating response...")
    partial = "**🤖 MealieChef:**\n"
    response = ""
    for chunk in service.chat(messages):
        partial += chunk
        response += chunk
        yield partial, gr.Markdown(value=debug_info), ctx

    logger.debug("Response generation finished.")
    yield partial, gr.Markdown(value=debug_info), ctx

    tracer.update_current_span(output=response)
    tracer.update_current_trace(output=partial + "\n\n" + debug_info)


def chat_fn(message: str, history: list[list[str]], ctx: TraceContext):
    yield from process_input(message, ctx)


def submit_feedback(comment: str, pending: dict) -> tuple[dict, gr.Row, str]:
    """Send the score (with an optional comment) to Langfuse and hide the row."""
    kwargs: dict = {
        "value": pending["value"],
        "name": "user-feedback",
        "trace_id": pending["trace_id"],
    }
    if comment and comment.strip():
        kwargs["comment"] = comment.strip()
    tracer.create_score(**kwargs)
    return {}, gr.Row(visible=False), ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    demo = build_demo(chat_fn=chat_fn, submit_feedback_fn=submit_feedback)
    demo.launch(
        server_name="0.0.0.0",
        server_port=settings.ui_port,
        auth=(settings.ui_username, settings.ui_password.get_secret_value()),
        share=False,
        max_threads=5,
        enable_monitoring=False,
        css=_TABLE_CSS,
    )


if __name__ == "__main__":
    main()
