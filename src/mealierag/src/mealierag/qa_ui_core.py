"""
Shared Gradio UI layout and helpers for MealieRAG.
"""

import logging
from collections.abc import Callable

import gradio as gr

from .models import RecipeHit
from .tracing import TraceContext

logger = logging.getLogger(__name__)

_TABLE_CSS = (
    "#debug-output table { display: block; overflow-x: auto; white-space: nowrap; }"
)


# ---------------------------------------------------------------------------
# Helpers (shared by standalone & API-client modes)
# ---------------------------------------------------------------------------


def print_hits(hits: list[RecipeHit]) -> str:
    """Render retrieved recipes as a Markdown table."""
    rows = (
        "| Name | Rating | Time (min) | Tools | Method | Ingredients | Tags | Category | Score |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    for hit in hits:
        tags = ", ".join(hit.tags) if isinstance(hit.tags, list) else str(hit.tags)
        tools = ", ".join(hit.tools) if isinstance(hit.tools, list) else str(hit.tools)
        method = (
            ", ".join(hit.method) if isinstance(hit.method, list) else str(hit.method)
        )

        rating_str = str(hit.rating) if hit.rating is not None else "N/A"
        time_str = (
            str(hit.total_time_minutes) if hit.total_time_minutes is not None else "N/A"
        )
        ing_count_str = (
            str(hit.ingredient_count) if hit.ingredient_count is not None else "N/A"
        )

        if isinstance(hit.category, list):
            category_str = ", ".join(hit.category)
        else:
            category_str = str(hit.category) if hit.category is not None else "N/A"

        rows += (
            f"| [{hit.name}]({hit.url})"
            f" | {rating_str}"
            f" | {time_str}"
            f" | {tools or 'N/A'}"
            f" | {method or 'N/A'}"
            f" | {ing_count_str}"
            f" | {tags or 'N/A'}"
            f" | {category_str}"
            f" | {hit.score} |\n"
        )
    logger.debug("Hits table", extra={"hits_table": rows})
    return rows


def handle_like(data: gr.LikeData, ctx: TraceContext) -> tuple[dict, gr.Row]:
    """Store the reaction in pending state and reveal the comment row."""
    pending = {"value": 1 if data.liked else 0, "trace_id": ctx.trace_id}
    return pending, gr.Row(visible=True)


def reset_session(ctx: TraceContext) -> TraceContext:
    """Create a fresh session ID when the chat is cleared."""
    ctx.create_new_session_id()
    return ctx


# ---------------------------------------------------------------------------
# Gradio layout builder
# ---------------------------------------------------------------------------


def build_demo(
    chat_fn: Callable,
    submit_feedback_fn: Callable,
) -> gr.Blocks:
    """Build the Gradio Blocks UI, wired to the provided callbacks.

    Parameters
    ----------
    chat_fn:
        ``(message, history, ctx) -> Generator[(text, gr.Markdown, ctx)]``
    submit_feedback_fn:
        ``(comment, pending) -> (state, gr.Row, str)``
    """
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
        gr.ChatInterface(
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
            submit_feedback_fn,
            [feedback_comment, pending_feedback],
            [pending_feedback, feedback_row, feedback_comment],
        )
        feedback_skip.click(
            submit_feedback_fn,
            [gr.State(""), pending_feedback],
            [pending_feedback, feedback_row, feedback_comment],
        )

    return demo
