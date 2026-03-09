"""
FastAPI Application — MTL Intent & Emotion Demo (Production-ready)
==================================================================
Set these environment variables in your hosting platform:
  ALLOWED_ORIGINS   = https://your-app.vercel.app   (comma-separated)
  PORT              = 8000  (auto-set by Render/Railway)
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.mtl_model import MTLIntentEmotionModel, INTENT_LABELS, EMOTION_LABELS
from app.inference import MTLInferencePipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── CORS origins from environment ─────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── Global pipeline ───────────────────────────────────────────────────────
pipeline: MTLInferencePipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    encoder = os.getenv("ENCODER_NAME", "roberta-base")
    logger.info(f"Loading MTL model (encoder={encoder})…")

    model = MTLIntentEmotionModel(encoder_name=encoder, freeze_encoder_layers=8)
    pipeline = MTLInferencePipeline(
        model=model,
        encoder_name=encoder,
        intent_threshold=float(os.getenv("INTENT_THRESHOLD", "0.35")),
        escalation_weights={
            "w_intent": float(os.getenv("W_INTENT", "0.4")),
            "w_emotion": float(os.getenv("W_EMOTION", "0.6")),
            "threshold_review":   float(os.getenv("THRESHOLD_REVIEW",   "0.4")),
            "threshold_escalate": float(os.getenv("THRESHOLD_ESCALATE", "0.7")),
        },
    )
    logger.info(f"Ready. Device={pipeline.device} | Origins={ALLOWED_ORIGINS}")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="MTL Intent & Emotion API",
    description="Multi-Task Learning: Joint Intent Recognition + Emotion-Aware Escalation",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Schemas ────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512)

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

# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    if pipeline is None:
        raise HTTPException(503, "Model not loaded.")
    t0 = time.perf_counter()
    result = pipeline.predict(request.text)
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return AnalyzeResponse(
        raw_text=result.raw_text,
        conversation_turn=result.conversation_turn,
        intents=[IntentOut(label=i.label, probability=i.probability) for i in result.intents],
        emotion=EmotionOut(label=result.emotion.label, probability=result.emotion.probability),
        escalation=EscalationOut(**result.escalation.__dict__),
        latency_ms=latency_ms,
    )

@app.post("/session/reset")
async def reset():
    if pipeline is None:
        raise HTTPException(503, "Model not loaded.")
    pipeline.reset_session()
    return {"status": "ok"}

@app.get("/labels")
async def labels():
    return {"intents": INTENT_LABELS, "emotions": EMOTION_LABELS}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": pipeline is not None,
            "device": pipeline.device if pipeline else None}
