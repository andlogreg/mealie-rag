from unittest.mock import MagicMock

from qdrant_client.http.models import ScoredPoint

from mealierag.models import QueryExtraction
from mealierag.run_qa_ui import (
    chat_fn,
    submit_feedback,
)
from mealierag.tracing import TraceContext


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
# submit_feedback
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
