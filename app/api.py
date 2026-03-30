"""
FastAPI Application — MTL Intent & Emotion Demo
================================================
Endpoints:
  POST /analyze        → Run inference on a single utterance
  POST /session/reset  → Reset conversation state
  GET  /health         → Health check
  GET  /labels         → Return all known intent/emotion labels
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Local imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.mtl_model import MTLIntentEmotionModel, INTENT_LABELS, EMOTION_LABELS
from app.inference import MTLInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# App Startup: Load Model Once
# ─────────────────────────────────────────────

pipeline: MTLInferencePipeline = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("Loading MTL model...")

    model = MTLIntentEmotionModel(
        encoder_name="roberta-base",
        freeze_encoder_layers=8,
    )
    pipeline = MTLInferencePipeline(
        model=model,
        encoder_name="roberta-base",
        intent_threshold=0.35,
        escalation_weights={
            "w_intent": 0.4,
            "w_emotion": 0.6,
            "threshold_review": 0.4,
            "threshold_escalate": 0.7,
        },
    )
    logger.info(f"Model loaded. Device: {pipeline.device}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="MTL Intent & Emotion API",
    description="Multi-Task Learning demo: Joint Intent Recognition + Emotion-Aware Escalation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response Schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512, example="I've been waiting for 30 minutes — this is ridiculous, I need to cancel my card NOW!")

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

class AnalyzeResponse(BaseModel):
    raw_text: str
    conversation_turn: int
    intents: list[IntentOut]
    emotion: EmotionOut
    escalation: EscalationOut
    latency_ms: float


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse, tags=["Inference"])
async def analyze(request: AnalyzeRequest):
    """
    Analyze a user utterance for intents and emotion, then apply
    the escalation controller to decide how to respond.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    t0 = time.perf_counter()
    result = pipeline.predict(request.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return AnalyzeResponse(
        raw_text=result.raw_text,
        conversation_turn=result.conversation_turn,
        intents=[IntentOut(label=i.label, probability=i.probability) for i in result.intents],
        emotion=EmotionOut(label=result.emotion.label, probability=result.emotion.probability),
        escalation=EscalationOut(
            urgency_score=result.escalation.urgency_score,
            action=result.escalation.action,
            message=result.escalation.message,
        ),
        latency_ms=latency_ms,
    )


@app.post("/session/reset", tags=["Session"])
async def reset_session():
    """Reset conversation state (emotion history, turn counter)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    pipeline.reset_session()
    return {"status": "ok", "message": "Session reset."}


@app.get("/labels", tags=["Meta"])
async def get_labels():
    """Return all supported intent and emotion labels."""
    return {
        "intents": INTENT_LABELS,
        "emotions": EMOTION_LABELS,
    }


@app.get("/health", tags=["Meta"])
async def health():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": pipeline.device if pipeline else None,
    }
