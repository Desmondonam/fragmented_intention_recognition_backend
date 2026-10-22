"""
FastAPI Application — Intent & Emotion Escalation API
=====================================================
Endpoints:
  POST /analyze        → Run inference on a single utterance
  POST /session/reset  → Reset a conversation's state
  GET  /health         → Health check
  GET  /labels         → Return all known intent/emotion labels

Configuration (environment variables):
  MODEL_DIR / MODEL_REPO_ID / MODEL_REVISION   see app/model_store.py
  ALLOWED_ORIGINS     comma-separated CORS origins (default "*")
  TOP_K_INTENTS       number of intents returned (default 3)
  W_INTENT, W_EMOTION, THRESHOLD_REVIEW, THRESHOLD_ESCALATE   escalation tuning
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.inference import MTLInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_pipeline_from_env() -> MTLInferencePipeline:
    from app.model_store import load_model_and_tokenizer, resolve_model_dir

    model, tokenizer = load_model_and_tokenizer(resolve_model_dir())
    return MTLInferencePipeline(
        model=model,
        tokenizer=tokenizer,
        top_k_intents=int(os.getenv("TOP_K_INTENTS", "3")),
        escalation_weights={
            "w_intent": float(os.getenv("W_INTENT", "0.4")),
            "w_emotion": float(os.getenv("W_EMOTION", "0.6")),
            "threshold_review": float(os.getenv("THRESHOLD_REVIEW", "0.4")),
            "threshold_escalate": float(os.getenv("THRESHOLD_ESCALATE", "0.7")),
        },
    )


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512,
                      examples=["My card was stolen and someone is using it — I need it blocked NOW!"])
    session_id: str = Field("default", min_length=1, max_length=64)

class ResetRequest(BaseModel):
    session_id: str = Field("default", min_length=1, max_length=64)

class IntentOut(BaseModel):
    label: str
    probability: float

class EmotionOut(BaseModel):
    label: str
    probability: float

class EscalationOut(BaseModel):
    urgency_score: float
    action: str
    message: str
    emotion_drift: bool

class AnalyzeResponse(BaseModel):
    raw_text: str
    session_id: str
    conversation_turn: int
    intents: list[IntentOut]
    emotion: EmotionOut
    escalation: EscalationOut
    latency_ms: float


# ─────────────────────────────────────────────
# App Factory
# ─────────────────────────────────────────────

def create_app(pipeline_factory: Callable[[], MTLInferencePipeline] = load_pipeline_from_env) -> FastAPI:
    """Build the app. Tests pass a factory that returns a lightweight pipeline."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading MTL model...")
        app.state.pipeline = pipeline_factory()
        logger.info(f"Model loaded. Device: {app.state.pipeline.device}")
        yield
        logger.info("Shutting down.")

    app = FastAPI(
        title="Intent & Emotion Escalation API",
        description="Multi-task RoBERTa: Banking77 intent + GoEmotions emotion, with emotion-aware escalation",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.pipeline = None

    origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def get_pipeline(request: Request) -> MTLInferencePipeline:
        pipeline: Optional[MTLInferencePipeline] = request.app.state.pipeline
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded yet.")
        return pipeline

    @app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
    def analyze(body: AnalyzeRequest, request: Request):
        """
        Analyze a user utterance for intent and emotion, then apply
        the escalation controller to decide how to respond.
        """
        pipeline = get_pipeline(request)
        t0 = time.perf_counter()
        result = pipeline.predict(body.text, session_id=body.session_id)
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return AnalyzeResponse(
            raw_text=result.raw_text,
            session_id=result.session_id,
            conversation_turn=result.conversation_turn,
            intents=[IntentOut(label=i.label, probability=i.probability) for i in result.intents],
            emotion=EmotionOut(label=result.emotion.label, probability=result.emotion.probability),
            escalation=EscalationOut(**result.escalation.__dict__),
            latency_ms=latency_ms,
        )

    @app.post("/session/reset", tags=["Session"])
    def reset_session(request: Request, body: Optional[ResetRequest] = None):
        """Reset a conversation's state (emotion history, turn counter)."""
        session_id = body.session_id if body else "default"
        get_pipeline(request).reset_session(session_id)
        return {"status": "ok", "session_id": session_id}

    @app.get("/labels", tags=["Meta"])
    def get_labels(request: Request):
        """Return all supported intent and emotion labels."""
        pipeline = get_pipeline(request)
        return {"intents": pipeline.intent_labels, "emotions": pipeline.emotion_labels}

    @app.get("/health", tags=["Meta"])
    def health(request: Request):
        pipeline = request.app.state.pipeline
        return {
            "status": "healthy",
            "model_loaded": pipeline is not None,
            "device": pipeline.device if pipeline else None,
        }

    return app


app = create_app()
