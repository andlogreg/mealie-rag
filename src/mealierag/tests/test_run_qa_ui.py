from unittest.mock import MagicMock

import gradio as gr
from qdrant_client.http.models import ScoredPoint

from mealierag.models import QueryExtraction
from mealierag.run_qa_ui import (
    chat_fn,
    handle_like,
    print_hits,
    reset_session,
    submit_feedback,
)
from mealierag.tracing import TraceContext


def test_print_hits():
    """Test hits table formatting."""
    hits = [
        ScoredPoint(
            id=1,
            version=1,
            score=1.0,
            payload={
                "name": "Recipe 1",
                "rating": 5,
                "tags": ["t1"],
                "category": "c1",
                "recipe_id": "123",
                "total_time_minutes": 30,
                "tools": ["oven"],
                "method": ["bake"],
                "ingredient_count": 7,
            },
        )
    ]

    table = print_hits(hits)
    # Markdown link is present
    assert "[Recipe 1]" in table
    assert "(http" in table
    assert "123)" in table
    assert "| 5 |" in table  # rating
    assert "| 30 |" in table  # total_time_minutes
    assert "| oven |" in table  # tools
    assert "| bake |" in table  # method
    assert "| 7 |" in table  # ingredient_count
    assert "| t1 |" in table  # tags
    assert "| c1 |" in table  # category


def test_chat_fn(mocker):
    """Test chat generator function."""
    mock_service = MagicMock()
    mocker.patch("mealierag.run_qa_ui.service", mock_service)

    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service.retrieve_recipes.return_value = [
        ScoredPoint(
            id=1, version=1, score=1.0, payload={"name": "Recipe 1", "recipe_id": "123"}
        )
    ]
    mock_service.populate_messages.return_value = []
    mock_service.chat.return_value = iter(["Hello"])

    ctx = TraceContext()
    generator = chat_fn("test message", [], ctx)
    responses = list(generator)

    # Each yield is (text, Markdown, ctx)
    texts = [r[0] for r in responses]
    assert any("Consulting" in t for t in texts)
    assert any("Finding" in t for t in texts)
    assert any("Hello" in t for t in texts)
    assert "Recipe 1" in responses[-1][1].value

    mock_service.generate_queries.assert_called_once()
    mock_service.retrieve_recipes.assert_called_once()
    mock_service.chat.assert_called_once()


def test_chat_fn_no_results(mocker):
    """Test chat function with no results."""
    mock_service = MagicMock()
    mocker.patch("mealierag.run_qa_ui.service", mock_service)

    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service.retrieve_recipes.return_value = []

    ctx = TraceContext()
    responses = list(chat_fn("test message", [], ctx))
    assert any("couldn't find" in r[0] for r in responses)


# ---------------------------------------------------------------------------
# handle_like
# ---------------------------------------------------------------------------


def test_handle_like_stores_pending_and_shows_row():
    """handle_like should store the reaction value and show the feedback row."""
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
# submit_feedback (covers both submit and skip paths)
# ---------------------------------------------------------------------------


def test_submit_feedback_with_comment(mocker):
    """submit_feedback should forward the comment to create_score."""
    mock_tracer = mocker.patch("mealierag.run_qa_ui.tracer")
    pending = {"value": 1, "trace_id": "trace-456"}

    state, row_update, text_clear = submit_feedback("Great answer!", pending)

    mock_tracer.create_score.assert_called_once_with(
        value=1, name="user-feedback", trace_id="trace-456", comment="Great answer!"
    )
    assert state == {}
    assert row_update.visible is False
    assert text_clear == ""


def test_submit_feedback_without_comment(mocker):
    """submit_feedback with blank text should omit the comment kwarg (skip path)."""
    mock_tracer = mocker.patch("mealierag.run_qa_ui.tracer")
    pending = {"value": 0, "trace_id": "trace-789"}

    submit_feedback("  ", pending)

    mock_tracer.create_score.assert_called_once_with(
        value=0, name="user-feedback", trace_id="trace-789"
    )


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
