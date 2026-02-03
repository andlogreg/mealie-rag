from unittest.mock import MagicMock

import pytest

from mealierag.tracing import TraceContext, Tracer


@pytest.fixture
def mock_langfuse(mocker):
    return mocker.patch("mealierag.tracing.Langfuse")


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.tracing_base_url = "http://test-langfuse"
    config.tracing_public_key.get_secret_value.return_value = "pk"
    config.tracing_secret_key.get_secret_value.return_value = "sk"
    config.tracing_environment = "dev"
    config.tracing_enabled = True
    return config


def test_tracer_init(mock_langfuse, mock_config, mocker):
    mocker.patch("mealierag.tracing.version", return_value="1.2.3")
    Tracer(mock_config)
    mock_langfuse.assert_called_with(
        release="1.2.3",
        base_url="http://test-langfuse",
        public_key="pk",
        secret_key="sk",
        environment="dev",
        tracing_enabled=True,
    )


def test_tracer_methods(mock_langfuse, mock_config):
    tracer = Tracer(mock_config)

    tracer.get_current_trace_id()
    tracer.langfuse.get_current_trace_id.assert_called_once()

    tracer.get_current_observation_id()
    tracer.langfuse.get_current_observation_id.assert_called_once()

    tracer.update_current_span(name="span")
    tracer.langfuse.update_current_span.assert_called_with(name="span")

    tracer.score(name="score", value=1)
    tracer.langfuse.score.assert_called_with(name="score", value=1)


def test_trace_context(mocker):
    mocker.patch("mealierag.tracing.tracer")

    # Mock uuid
    mocker.patch("uuid.uuid4", return_value="test-uuid")

    context = TraceContext()

    assert context.session_id == "test-uuid"
    assert context.trace_id is None

    context.set_trace_id("test-trace")
    assert context.trace_id == "test-trace"
