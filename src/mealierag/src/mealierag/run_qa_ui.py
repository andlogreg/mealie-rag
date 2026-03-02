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

service = create_mealie_rag_service()

_TABLE_CSS = (
    "#debug-output table { display: block; overflow-x: auto; white-space: nowrap; }"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_hits(hits: list[ScoredPoint]) -> str:
    """Render retrieved recipes as a Markdown table."""
    rows = (
        "| Name | Rating | Time (min) | Tools | Method | Ingredients | Tags | Category | Score |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for hit in hits:
        recipe_url = (
            f"{settings.mealie_external_url}/g/home/r/{hit.payload['recipe_id']}"
        )
        tags = hit.payload.get("tags", [])
        if isinstance(tags, list):
            tags = ", ".join(tags)
        tools = hit.payload.get("tools", [])
        if isinstance(tools, list):
            tools = ", ".join(tools)
        method = hit.payload.get("method", [])
        if isinstance(method, list):
            method = ", ".join(method)
        rows += (
            f"| [{hit.payload.get('name', 'N/A')}]({recipe_url})"
            f" | {hit.payload.get('rating', 'N/A')}"
            f" | {hit.payload.get('total_time_minutes', 'N/A')}"
            f" | {tools or 'N/A'}"
            f" | {method or 'N/A'}"
            f" | {hit.payload.get('ingredient_count', 'N/A')}"
            f" | {tags}"
            f" | {hit.payload.get('category', 'N/A')}"
            f" | {hit.score} |\n"
        )
    logger.debug("Hits table", extra={"hits_table": rows})
    return rows


# ---------------------------------------------------------------------------
# Core chat pipeline
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

    debug_info = (
        "### 🤓 Recipes context for the above answer: ###\n"
        + print_hits(hits)
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


# ---------------------------------------------------------------------------
# Feedback handlers
# ---------------------------------------------------------------------------


def handle_like(data: gr.LikeData, ctx: TraceContext) -> tuple[dict, gr.Row]:
    """Store the reaction in pending state and reveal the comment row."""
    pending = {"value": 1 if data.liked else 0, "trace_id": ctx.trace_id}
    return pending, gr.Row(visible=True)


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


def reset_session(ctx: TraceContext) -> TraceContext:
    """Create a fresh session ID when the chat is cleared."""
    ctx.create_new_session_id()
    return ctx


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    # Per-session state — each browser tab gets its own TraceContext copy.
    trace_context = gr.State(TraceContext)
    pending_feedback = gr.State(value={})

    debug_output = gr.Markdown(
        label="Debug Output",
        render=False,
        min_height=300,
        elem_id="debug-output",
    )

    gr.Button("Logout", link="/logout", size="sm", variant="secondary")

    chatbot = gr.Chatbot(height=500)
    chat = gr.ChatInterface(
        fn=chat_fn,
        chatbot=chatbot,
        title="🍳 MealieChef",
        description="Your personal recipe assistant",
        additional_inputs=[trace_context],
        additional_outputs=[debug_output, trace_context],
    )

    with gr.Row(visible=False, elem_id="feedback_row") as feedback_row:
        feedback_comment = gr.Textbox(
            placeholder="Optional: tell us why…",
            label="Feedback comment",
            scale=4,
            lines=1,
        )
        feedback_submit = gr.Button("Submit", variant="primary", scale=1)
        feedback_skip = gr.Button("Skip", variant="secondary", scale=1)

    debug_output.render()

    chatbot.like(handle_like, [trace_context], [pending_feedback, feedback_row])
    chatbot.clear(reset_session, [trace_context], [trace_context])
    feedback_submit.click(
        submit_feedback,
        [feedback_comment, pending_feedback],
        [pending_feedback, feedback_row, feedback_comment],
    )
    feedback_skip.click(
        submit_feedback,
        [gr.State(""), pending_feedback],
        [pending_feedback, feedback_row, feedback_comment],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
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
