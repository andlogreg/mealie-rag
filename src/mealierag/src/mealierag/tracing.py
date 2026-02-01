import logging
import uuid
from importlib.metadata import version

from langfuse import Langfuse, observe

from .config import Settings, settings

logger = logging.getLogger(__name__)


class Tracer:
    def __init__(self, config: Settings):
        self.release = version("mealierag")
        self.langfuse = Langfuse(
            release=self.release,
            base_url=config.tracing_base_url,
            public_key=config.tracing_public_key.get_secret_value(),
            secret_key=config.tracing_secret_key.get_secret_value(),
            environment=config.tracing_environment,
            tracing_enabled=config.tracing_enabled,
        )
        self.observe = observe

    def get_current_trace_id(self):
        return self.langfuse.get_current_trace_id()

    def get_current_observation_id(self):
        return self.langfuse.get_current_observation_id()

    def update_current_span(self, **kwargs):
        self.langfuse.update_current_span(**kwargs)

    def update_current_trace(self, **kwargs):
        self.langfuse.update_current_trace(**kwargs)

    def score(self, **kwargs):
        self.langfuse.score(**kwargs)

    def create_score(self, **kwargs):
        self.langfuse.create_score(**kwargs)

    def get_trace_url(self, trace_id: str | None = None):
        return self.langfuse.get_trace_url(trace_id=trace_id)


tracer = Tracer(settings)


class TraceContext:
    def __init__(self):
        self.trace_id = None
        self.session_id = None

        self.create_new_session_id()

    def create_new_session_id(self):
        self.session_id = str(uuid.uuid4())
        logger.info("New session id", extra={"session_id": self.session_id})

    def set_trace_id(self, trace_id: str):
        self.trace_id = trace_id
        logger.info(
            "Trace id set",
            extra={"trace_id": trace_id, "trace_url": tracer.get_trace_url(trace_id)},
        )
