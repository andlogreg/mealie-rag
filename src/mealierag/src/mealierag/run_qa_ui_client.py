"""
Serve the QA Rag interface in the browser.
"""

import json
import logging

import gradio as gr
import httpx
from httpx_sse import connect_sse

from .config import settings
from .models import RecipeHit
from .qa_ui_core import _TABLE_CSS, build_demo, print_hits
from .tracing import TraceContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API-client backend callbacks
# ---------------------------------------------------------------------------


def _make_client(api_url: str, api_key: str) -> httpx.Client:
    """Create a pre-configured httpx client."""
    return httpx.Client(
        base_url=api_url,
        headers={"X-API-Key": api_key},
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0),
    )


def create_process_input(client: httpx.Client):
    """Return a ``process_input`` generator wired to the API."""

    def process_input(user_input: str, ctx: TraceContext):
        partial = ""
        debug_info = ""
        yield partial, gr.Markdown(value=""), ctx

        try:
            with connect_sse(
                client,
                "POST",
                "/api/v1/chat",
                json={"message": user_input, "session_id": ctx.session_id},
            ) as event_source:
                for sse in event_source.iter_sse():
                    if sse.event == "status":
                        partial += f"\n{sse.data}"
                        yield partial, gr.Markdown(value=debug_info), ctx

                    elif sse.event == "recipes":
                        raw_recipes = json.loads(sse.data)
                        if raw_recipes:
                            recipes = [RecipeHit(**r) for r in raw_recipes]
                            debug_info = (
                                "### 🤓 Recipes context for the above answer: ###\n"
                                + print_hits(recipes)
                            )
                        yield partial, gr.Markdown(value=debug_info), ctx

                    elif sse.event == "token":
                        if "MealieChef" not in partial:
                            # First token, add the header
                            partial = "**🤖 MealieChef:**\n"
                        partial += sse.data
                        yield partial, gr.Markdown(value=debug_info), ctx

                    elif sse.event == "done":
                        done_data = json.loads(sse.data)
                        trace_id = done_data.get("trace_id")
                        trace_url = done_data.get("trace_url")
                        if trace_id:
                            ctx.trace_id = trace_id
                        message = done_data.get("message")
                        if message:
                            partial = message
                        if trace_url:
                            debug_info += f"\n[View trace]({trace_url})"
                        yield partial, gr.Markdown(value=debug_info), ctx

                    elif sse.event == "error":
                        partial = f"⚠️ Error: {sse.data}"
                        yield partial, gr.Markdown(value=debug_info), ctx

        except httpx.HTTPError as exc:
            logger.error("API request failed", exc_info=True)
            partial = f"⚠️ Could not reach the API: {exc}"
            yield partial, gr.Markdown(value=""), ctx

    return process_input


def create_submit_feedback(client: httpx.Client):
    """Return a ``submit_feedback`` callable wired to the API."""

    def submit_feedback(comment: str, pending: dict) -> tuple[dict, gr.Row, str]:
        """Send feedback to the API and hide the row."""
        payload: dict = {
            "trace_id": pending["trace_id"],
            "value": pending["value"],
        }
        if comment and comment.strip():
            payload["comment"] = comment.strip()
        try:
            client.post("/api/v1/feedback", json=payload)
        except httpx.HTTPError:
            logger.error("Failed to submit feedback via API", exc_info=True)
        return {}, gr.Row(visible=False), ""

    return submit_feedback


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(api_url: str = "http://localhost:8000"):
    """Launch the Gradio UI in API-client mode."""
    client = _make_client(api_url, settings.api_key.get_secret_value())

    process_input = create_process_input(client)
    submit_feedback = create_submit_feedback(client)

    def chat_fn(message: str, history: list[list[str]], ctx: TraceContext):
        yield from process_input(message, ctx)

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
