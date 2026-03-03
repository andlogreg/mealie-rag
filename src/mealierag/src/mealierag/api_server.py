"""
REST API server for MealieRAG.

HTTP with SSE streaming for chat responses.
"""

import json
import logging
from importlib.metadata import version

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .config import settings
from .models import RecipeHit
from .service import create_mealie_rag_service
from .tracing import TraceContext, tracer

logger = logging.getLogger(__name__)

service = create_mealie_rag_service()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MealieRAG API",
    description="Chat with your recipes!",
    version=version("mealierag"),
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip() for o in settings.api_cors_origins.split(",") if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Simple Auth
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key")


def verify_api_key(api_key: str = Depends(_api_key_header)) -> None:
    """Validate the X-API-Key header against the configured key."""
    if api_key != settings.api_key.get_secret_value():
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User question")
    session_id: str | None = Field(
        None, description="Optional session ID for grouping multi-turn conversations"
    )


class FeedbackRequest(BaseModel):
    trace_id: str = Field(..., description="Langfuse trace ID")
    value: int = Field(..., ge=0, le=1, description="Feedback score (0 or 1)")
    comment: str | None = Field(None, description="Optional feedback comment")


class HealthResponse(BaseModel):
    status: str
    collection: str


def _format_sse(data: str | list | dict, event: str) -> str:
    """Format an SSE message. Uses split('\\n') to preserve trailing newlines."""
    # TODO: Use fastapi.sse.format_sse_event instead
    # With v0.135.1 a single '\n' was not preserved
    if not isinstance(data, str):
        data = json.dumps(data)

    lines = data.split("\n")
    out = "".join(f"data: {line}\n" for line in lines)
    if event:
        out += f"event: {event}\n"
    return out + "\n"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
def health():
    """Check whether the Qdrant collection is available."""
    healthy = service.check_health()
    resp = HealthResponse(
        status="ok" if healthy else "unhealthy",
        collection=settings.vectordb_collection_name,
    )
    if not healthy:
        return JSONResponse(status_code=503, content=resp.model_dump())
    return resp


@app.post(
    "/api/v1/chat",
    response_class=StreamingResponse,
    dependencies=[Depends(verify_api_key)],
)
def chat(body: ChatRequest) -> StreamingResponse:
    """Stream chat response as SSE events."""
    # NOTE: TraceContext probably not needed anymore when using API
    ctx = TraceContext()
    if body.session_id:
        ctx.session_id = body.session_id
    trace_url = None

    @tracer.observe(transform_to_string=lambda _: None)
    def _traced_pipeline():
        ctx.set_trace_id(tracer.get_current_trace_id())
        trace_url = tracer.get_trace_url(ctx.trace_id)
        tracer.update_current_trace(
            name="qa_api_trace",
            session_id=ctx.session_id,
            input=body.message,
        )
        tracer.update_current_span(name="qa_api", input=body.message)

        yield _format_sse(data="👾 Consulting the digital oracles...", event="status")

        query_extraction = service.generate_queries(body.message)

        yield _format_sse(data="🔍 Finding relevant recipes...", event="status")

        hits = service.retrieve_recipes(query_extraction)

        if not hits:
            yield _format_sse(data=[], event="recipes")
            yield _format_sse(
                data={
                    "trace_id": ctx.trace_id,
                    "trace_url": trace_url,
                    "message": "No relevant recipes found.",
                },
                event="done",
            )
            return

        yield _format_sse(
            data=[RecipeHit.from_scored_point(h).model_dump() for h in hits],
            event="recipes",
        )

        yield _format_sse(data="⌛ Generating answer...", event="status")

        messages = service.populate_messages(body.message, hits)

        full_response = ""
        for chunk in service.chat(messages):
            full_response += chunk
            yield _format_sse(data=chunk, event="token")

        tracer.update_current_span(output=full_response)
        tracer.update_current_trace(output=full_response)

        yield _format_sse(
            data={
                "trace_id": ctx.trace_id,
                "trace_url": trace_url,
            },
            event="done",
        )

    def _safe_generator():
        try:
            yield from _traced_pipeline()
        except Exception:
            logger.exception("Error during chat pipeline")
            yield _format_sse(
                data="Internal server error. Trace available at: " + trace_url,
                event="error",
            )

    return StreamingResponse(_safe_generator(), media_type="text/event-stream")


@app.post("/api/v1/feedback", dependencies=[Depends(verify_api_key)])
def feedback(body: FeedbackRequest):
    """Submit user feedback for a trace to Langfuse."""
    kwargs: dict = {
        "value": body.value,
        "name": "user-feedback",
        "trace_id": body.trace_id,
    }
    if body.comment and body.comment.strip():
        kwargs["comment"] = body.comment.strip()
    tracer.create_score(**kwargs)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Start the API server via uvicorn."""
    import uvicorn

    uvicorn.run(
        "mealierag.api_server:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_level="info",
    )
