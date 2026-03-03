import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.models import ScoredPoint

from mealierag.api_server import app
from mealierag.models import QueryExtraction

API_KEY = "mealie-rag-dev-key"
AUTH_HEADERS = {"X-API-Key": API_KEY}


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture()
def mock_service(mocker):
    svc = MagicMock()
    mocker.patch("mealierag.api_server.service", svc)
    return svc


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health_ok(client, mock_service):
    """Healthy collection should return 200."""
    mock_service.check_health.return_value = True
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_health_unhealthy(client, mock_service):
    """Missing collection should return 503."""
    mock_service.check_health.return_value = False
    resp = client.get("/api/v1/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "unhealthy"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_auth_missing_key(client):
    """Chat without API key should return 401."""
    resp = client.post("/api/v1/chat", json={"message": "hello"})
    assert resp.status_code == 401


def test_auth_invalid_key(client):
    """Chat with wrong API key should return 401."""
    resp = client.post(
        "/api/v1/chat",
        json={"message": "hello"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


def _parse_sse_events(text: str) -> list[dict]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events = []
    current_event = None
    current_data = []

    for line in text.split("\n"):
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
        elif line.startswith("data: "):
            current_data.append(line[len("data: ") :])
        elif line == "" and current_event is not None:
            events.append({"event": current_event, "data": "\n".join(current_data)})
            current_event = None
            current_data = []

    return events


def test_chat_streaming(client, mock_service):
    """Full SSE chat flow: status → recipes → tokens → done."""
    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["pasta query"]
    )
    mock_service.retrieve_recipes.return_value = [
        ScoredPoint(
            id=1,
            version=1,
            score=0.95,
            payload={
                "name": "Pasta",
                "recipe_id": "r1",
                "rating": 5,
                "tags": ["italian"],
                "category": ["Dinner"],
                "total_time_minutes": 30,
                "tools": ["pot"],
                "method": ["boil"],
                "ingredient_count": 5,
            },
        )
    ]
    mock_service.populate_messages.return_value = MagicMock()
    mock_service.chat.return_value = iter(["Boil ", "pasta."])

    resp = client.post(
        "/api/v1/chat",
        json={"message": "quick pasta"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    events = _parse_sse_events(resp.text)
    event_types = [e["event"] for e in events]

    assert "status" in event_types
    assert "recipes" in event_types
    assert "token" in event_types
    assert "done" in event_types

    # Verify recipes payload
    recipes_event = next(e for e in events if e["event"] == "recipes")
    recipes = json.loads(recipes_event["data"])
    assert len(recipes) == 1
    assert recipes[0]["name"] == "Pasta"

    # Verify tokens
    token_events = [e for e in events if e["event"] == "token"]
    full_response = "".join(e["data"] for e in token_events)
    assert full_response == "Boil pasta."

    # Verify done has trace_id
    done_event = next(e for e in events if e["event"] == "done")
    done_data = json.loads(done_event["data"])
    assert "trace_id" in done_data


def test_chat_with_session_id(client, mock_service, mocker):
    """Client-provided session_id should be used for trace grouping."""
    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["query"]
    )
    mock_service.retrieve_recipes.return_value = []

    mock_ctx_class = mocker.patch("mealierag.api_server.TraceContext")
    mock_ctx = MagicMock()
    mock_ctx.trace_id = "t-1"
    mock_ctx_class.return_value = mock_ctx

    resp = client.post(
        "/api/v1/chat",
        json={"message": "hello", "session_id": "client-session-abc"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200

    # Verify the client's session_id was applied
    assert mock_ctx.session_id == "client-session-abc"


def test_chat_no_hits(client, mock_service):
    """No recipes found should still return a done event."""
    mock_service.generate_queries.return_value = QueryExtraction(
        expanded_queries=["nothing"]
    )
    mock_service.retrieve_recipes.return_value = []

    resp = client.post(
        "/api/v1/chat",
        json={"message": "find me unicorn soup"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200

    events = _parse_sse_events(resp.text)
    event_types = [e["event"] for e in events]

    assert "done" in event_types
    assert "token" not in event_types

    done_event = next(e for e in events if e["event"] == "done")
    done_data = json.loads(done_event["data"])
    assert "No relevant recipes" in done_data.get("message", "")


def test_chat_missing_message(client):
    """Empty or missing message should return 422."""
    resp = client.post("/api/v1/chat", json={}, headers=AUTH_HEADERS)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------


def test_feedback_submit(client, mocker):
    """Feedback should be forwarded to tracer.create_score."""
    mock_tracer = mocker.patch("mealierag.api_server.tracer")

    resp = client.post(
        "/api/v1/feedback",
        json={"trace_id": "t-123", "value": 1, "comment": "Great!"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    mock_tracer.create_score.assert_called_once_with(
        value=1, name="user-feedback", trace_id="t-123", comment="Great!"
    )


def test_feedback_without_comment(client, mocker):
    """Feedback without comment should omit the comment kwarg."""
    mock_tracer = mocker.patch("mealierag.api_server.tracer")

    resp = client.post(
        "/api/v1/feedback",
        json={"trace_id": "t-456", "value": 0},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200

    mock_tracer.create_score.assert_called_once_with(
        value=0, name="user-feedback", trace_id="t-456"
    )
