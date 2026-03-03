from unittest.mock import MagicMock

import gradio as gr

from mealierag.models import RecipeHit
from mealierag.qa_ui_core import handle_like, print_hits, reset_session
from mealierag.tracing import TraceContext


def test_print_hits():
    """Test hits table formatting."""
    hits = [
        RecipeHit(
            name="Recipe 1",
            url="http://mock-url/g/home/r/123",
            recipe_id="123",
            rating=5.0,
            total_time_minutes=30,
            tools=["oven"],
            method=["bake"],
            ingredient_count=7,
            tags=["t1"],
            category=["c1"],
            score=1.0,
        )
    ]

    table = print_hits(hits)
    # Markdown link is present
    assert "[Recipe 1]" in table
    assert "(http" in table
    assert "123)" in table
    assert "| 5.0 |" in table  # rating
    assert "| 30 |" in table  # total_time_minutes
    assert "| oven |" in table  # tools
    assert "| bake |" in table  # method
    assert "| 7 |" in table  # ingredient_count
    assert "| t1 |" in table  # tags
    assert "| c1 |" in table  # category


# ---------------------------------------------------------------------------
# handle_like
# ---------------------------------------------------------------------------


def test_handle_like_stores_pending_and_shows_row(mocker):
    """handle_like should store the reaction value and show the feedback row."""
    mocker.patch(
        "mealierag.tracing.tracer.get_trace_url", return_value="http://mock-url"
    )
    ctx = TraceContext()
    ctx.set_trace_id("trace-123")

    like_data = MagicMock(spec=gr.LikeData)
    like_data.liked = True
    pending, row_update = handle_like(like_data, ctx)

    assert pending == {"value": 1, "trace_id": "trace-123"}
    assert row_update.visible is True

    like_data.liked = False
    pending, _ = handle_like(like_data, ctx)
    assert pending["value"] == 0


# ---------------------------------------------------------------------------
# reset_session
# ---------------------------------------------------------------------------


def test_reset_session():
    """reset_session should generate a new session_id on the context."""
    ctx = TraceContext()
    original_session_id = ctx.session_id

    returned_ctx = reset_session(ctx)

    assert returned_ctx is ctx
    assert ctx.session_id != original_session_id
