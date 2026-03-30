"""
Inference Engine + Escalation Controller
==========================================
Handles tokenization, model inference, urgency scoring,
and escalation decision logic.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from model.mtl_model import MTLIntentEmotionModel, INTENT_LABELS, EMOTION_LABELS


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

@dataclass
class InferenceResult:
    raw_text: str
    intents: List[IntentResult]
    emotion: EmotionResult
    escalation: EscalationDecision
    conversation_turn: int = 0


# ─────────────────────────────────────────────
# Escalation Controller
# ─────────────────────────────────────────────

# Which intents are "high complexity / high value" for intent scoring
HIGH_VALUE_INTENTS = {"report_theft", "request_emergency_cash", "speak_to_agent"}

# Which emotions drive urgency
HIGH_URGENCY_EMOTIONS = {"Frustration", "Sarcasm", "Urgent"}


class EscalationController:
    """
    Computes the Urgency Score:
        U = (w_i * P_intent) + (w_e * P_emotion)

    and maps it to an action tier.
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

    def compute_p_intent(self, intents: List[IntentResult]) -> float:
        """
        P_intent = max probability among high-value intents that fired.
        If none fired, use 0.
        """
        high_value_probs = [i.probability for i in intents if i.label in HIGH_VALUE_INTENTS]
        return max(high_value_probs) if high_value_probs else 0.0

    def compute_p_emotion(self, emotion: EmotionResult) -> float:
        """
        P_emotion = emotion probability if it's a high-urgency emotion, else 0.
        """
        return emotion.probability if emotion.label in HIGH_URGENCY_EMOTIONS else 0.0

    def decide(self, intents: List[IntentResult], emotion: EmotionResult) -> EscalationDecision:
        p_i = self.compute_p_intent(intents)
        p_e = self.compute_p_emotion(emotion)

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

        return EscalationDecision(urgency_score=U, action=action, message=message)


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
        encoder_name: str = "roberta-base",
        intent_threshold: float = 0.35,  # Sigmoid threshold for multi-label intents
        device: Optional[str] = None,
        escalation_weights: Optional[Dict] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.intent_threshold = intent_threshold

        esc_kwargs = escalation_weights or {}
        self.escalation_controller = EscalationController(**esc_kwargs)

        # Track emotion history for multi-turn drift detection
        self._turn_counter: int = 0
        self._emotion_history: List[str] = []

    def _tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        return {k: v.to(self.device) for k, v in encoded.items()}

    def _decode_intents(self, logits: torch.Tensor) -> List[IntentResult]:
        probs = torch.sigmoid(logits).squeeze(0).tolist()
        results = [
            IntentResult(label=label, probability=round(p, 4))
            for label, p in zip(INTENT_LABELS, probs)
            if p >= self.intent_threshold
        ]
        # Sort descending by probability
        return sorted(results, key=lambda x: x.probability, reverse=True)

    def _decode_emotion(self, logits: torch.Tensor) -> EmotionResult:
        probs = F.softmax(logits.squeeze(0), dim=-1).tolist()
        max_idx = probs.index(max(probs))
        return EmotionResult(
            label=EMOTION_LABELS[max_idx],
            probability=round(probs[max_idx], 4),
        )

    def _check_emotion_drift(self) -> bool:
        """
        Returns True if the last 3 turns have all been high-urgency.
        This lets us escalate even when a single turn's score is borderline.
        """
        if len(self._emotion_history) < 3:
            return False
        return all(e in HIGH_URGENCY_EMOTIONS for e in self._emotion_history[-3:])

    @torch.no_grad()
    def predict(self, text: str) -> InferenceResult:
        self._turn_counter += 1
        inputs = self._tokenize(text)

        intent_logits, emotion_logits = self.model(**inputs)

        intents = self._decode_intents(intent_logits)
        emotion = self._decode_emotion(emotion_logits)

        self._emotion_history.append(emotion.label)

        # Boost urgency if persistent negative emotion drift
        if self._check_emotion_drift():
            emotion = EmotionResult(label=emotion.label, probability=min(emotion.probability + 0.15, 1.0))

        escalation = self.escalation_controller.decide(intents, emotion)

        return InferenceResult(
            raw_text=text,
            intents=intents,
            emotion=emotion,
            escalation=escalation,
            conversation_turn=self._turn_counter,
        )

    def reset_session(self):
        """Call this when a new user session starts."""
        self._turn_counter = 0
        self._emotion_history = []
