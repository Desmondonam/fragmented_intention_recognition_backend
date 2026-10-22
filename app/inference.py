"""
Inference Engine + Escalation Controller
==========================================
Handles tokenization, model inference, urgency scoring,
and escalation decision logic.
"""

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from model.mtl_model import MTLIntentEmotionModel

# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class IntentResult:
    label: str
    probability: float

@dataclass
class EmotionResult:
    label: str
    probability: float

@dataclass
class EscalationDecision:
    urgency_score: float
    action: str          # "autonomous" | "flag_for_review" | "escalate_now"
    message: str         # Bot response message
    emotion_drift: bool = False

@dataclass
class InferenceResult:
    raw_text: str
    intents: List[IntentResult]
    emotion: EmotionResult
    escalation: EscalationDecision
    conversation_turn: int = 0
    session_id: str = "default"


# ─────────────────────────────────────────────
# Escalation Controller
# ─────────────────────────────────────────────

# Banking77 intents that signal fraud, loss of funds or loss of access —
# the requests where a mishandled bot reply is most costly.
HIGH_VALUE_INTENTS = {
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "compromised_card",
    "card_swallowed",
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "transaction_charged_twice",
    "wrong_amount_of_cash_received",
    "transfer_not_received_by_recipient",
    "pin_blocked",
}

# Ekman emotions that drive urgency
HIGH_URGENCY_EMOTIONS = {"anger", "disgust", "fear", "sadness"}

DRIFT_WINDOW = 3
DRIFT_BOOST = 0.15


class EscalationController:
    """
    Computes the Urgency Score:
        U = (w_i * P_intent) + (w_e * P_emotion)

    where P_intent is the probability mass the model puts on high-value
    intents and P_emotion the mass on high-urgency emotions. Using the full
    distribution (rather than only the argmax label) keeps U stable when the
    model is split between two related classes, e.g. compromised_card vs
    lost_or_stolen_card.
    """

    def __init__(
        self,
        w_intent: float = 0.4,
        w_emotion: float = 0.6,
        threshold_review: float = 0.4,
        threshold_escalate: float = 0.7,
    ):
        self.w_intent = w_intent
        self.w_emotion = w_emotion
        self.threshold_review = threshold_review
        self.threshold_escalate = threshold_escalate

    @staticmethod
    def compute_p_intent(intent_probs: Dict[str, float]) -> float:
        return sum(p for label, p in intent_probs.items() if label in HIGH_VALUE_INTENTS)

    @staticmethod
    def compute_p_emotion(emotion_probs: Dict[str, float]) -> float:
        return sum(p for label, p in emotion_probs.items() if label in HIGH_URGENCY_EMOTIONS)

    def decide(
        self,
        intent_probs: Dict[str, float],
        emotion_probs: Dict[str, float],
        emotion_drift: bool = False,
    ) -> EscalationDecision:
        p_i = self.compute_p_intent(intent_probs)
        p_e = self.compute_p_emotion(emotion_probs)
        # Persistent negative emotion across turns raises urgency even when
        # each individual turn is borderline.
        if emotion_drift:
            p_e = min(p_e + DRIFT_BOOST, 1.0)

        U = (self.w_intent * p_i) + (self.w_emotion * p_e)
        U = round(min(U, 1.0), 4)

        if U > self.threshold_escalate:
            action = "escalate_now"
            message = (
                "I can hear that this is urgent and I want to make sure you get the right help. "
                "I'm connecting you to a specialist right now — they'll have full context of our conversation."
            )
        elif U > self.threshold_review:
            action = "flag_for_review"
            message = (
                "I've noted your concern and I'm handling this for you. "
                "A supervisor has also been alerted to review our conversation shortly."
            )
        else:
            action = "autonomous"
            message = "Got it — I can help you with that right away."

        return EscalationDecision(urgency_score=U, action=action, message=message, emotion_drift=emotion_drift)


# ─────────────────────────────────────────────
# Per-session conversation state
# ─────────────────────────────────────────────

@dataclass
class SessionState:
    turn: int = 0
    emotion_history: List[str] = field(default_factory=list)

    def has_emotion_drift(self) -> bool:
        """True if the last DRIFT_WINDOW turns were all high-urgency emotions."""
        recent = self.emotion_history[-DRIFT_WINDOW:]
        return len(recent) == DRIFT_WINDOW and all(e in HIGH_URGENCY_EMOTIONS for e in recent)


class SessionStore:
    """In-memory LRU of conversation state, bounded so memory can't grow without limit."""

    def __init__(self, max_sessions: int = 10_000):
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, SessionState]" = OrderedDict()
        # FastAPI runs sync endpoints in a threadpool, so guard the dict.
        self._lock = threading.Lock()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            state = self._sessions.pop(session_id, None) or SessionState()
            self._sessions[session_id] = state
            if len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return state

    def reset(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)


# ─────────────────────────────────────────────
# Inference Pipeline
# ─────────────────────────────────────────────

class MTLInferencePipeline:
    """
    End-to-end inference: raw text → structured result with escalation decision.
    """

    def __init__(
        self,
        model: MTLIntentEmotionModel,
        tokenizer,
        top_k_intents: int = 3,
        device: Optional[str] = None,
        escalation_weights: Optional[Dict] = None,
        max_length: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = tokenizer
        self.top_k_intents = top_k_intents
        self.max_length = max_length
        self.intent_labels = model.intent_labels
        self.emotion_labels = model.emotion_labels

        self.escalation_controller = EscalationController(**(escalation_weights or {}))
        self.sessions = SessionStore()

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        return {k: encoded[k].to(self.device) for k in ("input_ids", "attention_mask")}

    @torch.no_grad()
    def predict(self, text: str, session_id: str = "default") -> InferenceResult:
        state = self.sessions.get(session_id)
        state.turn += 1

        intent_logits, emotion_logits = self.model(**self._tokenize(text))
        intent_probs = dict(zip(self.intent_labels, F.softmax(intent_logits[0], dim=-1).tolist(), strict=True))
        emotion_probs = dict(zip(self.emotion_labels, F.softmax(emotion_logits[0], dim=-1).tolist(), strict=True))

        top_intents = sorted(intent_probs.items(), key=lambda kv: kv[1], reverse=True)[: self.top_k_intents]
        intents = [IntentResult(label=k, probability=round(p, 4)) for k, p in top_intents]
        emotion_label = max(emotion_probs, key=emotion_probs.get)
        emotion = EmotionResult(label=emotion_label, probability=round(emotion_probs[emotion_label], 4))

        state.emotion_history.append(emotion_label)
        escalation = self.escalation_controller.decide(
            intent_probs, emotion_probs, emotion_drift=state.has_emotion_drift()
        )

        return InferenceResult(
            raw_text=text,
            intents=intents,
            emotion=emotion,
            escalation=escalation,
            conversation_turn=state.turn,
            session_id=session_id,
        )

    def reset_session(self, session_id: str = "default"):
        """Call this when a user session ends or restarts."""
        self.sessions.reset(session_id)
