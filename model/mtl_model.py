"""
Multi-Task Learning (MTL) Model
================================
Shared Encoder (RoBERTa) → Intent Head + Emotion Head

Architecture:
  [Raw Text]
      │
  [RoBERTa Encoder]  ← Shared backbone (frozen or fine-tuned)
      │
  [Pooled CLS Embedding]
      ├──► [Intent Head]   → Multi-label Sigmoid  → e.g. ["report_theft", "request_emergency_cash"]
      └──► [Emotion Head]  → Softmax classifier   → e.g. "Frustration" (0.82)
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────
# Label Definitions
# ─────────────────────────────────────────────

INTENT_LABELS = [
    "report_theft",
    "request_emergency_cash",
    "cancel_card",
    "check_balance",
    "speak_to_agent",
    "general_inquiry",
    "update_details",
]

EMOTION_LABELS = [
    "Neutral",
    "Frustration",
    "Sarcasm",
    "Urgent",
]


# ─────────────────────────────────────────────
# MTL Model
# ─────────────────────────────────────────────

class MTLIntentEmotionModel(nn.Module):
    """
    Multi-Task Learning model with a shared RoBERTa encoder
    and two task-specific heads:
      1. Multi-label Intent Classifier (Sigmoid)
      2. Multi-class Emotion Classifier (Softmax)
    """

    def __init__(
        self,
        encoder_name: str = "roberta-base",
        num_intents: int = len(INTENT_LABELS),
        num_emotions: int = len(EMOTION_LABELS),
        dropout_rate: float = 0.3,
        freeze_encoder_layers: int = 8,  # freeze bottom N layers for speed
    ):
        super().__init__()

        # ── Shared Encoder ───────────────────────────────────────────────
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size  # 768 for roberta-base

        # Optionally freeze early layers (speeds up fine-tuning significantly)
        self._freeze_encoder_layers(freeze_encoder_layers)

        # ── Shared Dropout ────────────────────────────────────────────────
        self.dropout = nn.Dropout(dropout_rate)

        # ── Intent Head (Multi-Label) ─────────────────────────────────────
        # Uses Sigmoid so multiple intents can fire simultaneously
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_intents),
            # No activation here — raw logits for BCEWithLogitsLoss during training
            # At inference we apply Sigmoid externally
        )

        # ── Emotion Head (Multi-Class) ────────────────────────────────────
        # Uses Softmax — exactly one dominant emotion per utterance
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_emotions),
            # Softmax applied externally at inference
        )

    def _freeze_encoder_layers(self, n: int):
        """Freeze the first N transformer layers of the encoder."""
        if n == 0:
            return
        modules_to_freeze = [self.encoder.embeddings]
        if hasattr(self.encoder, "encoder"):
            modules_to_freeze += list(self.encoder.encoder.layer[:n])
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            intent_logits  : (batch, num_intents)  — raw logits (apply Sigmoid for probs)
            emotion_logits : (batch, num_emotions) — raw logits (apply Softmax for probs)
        """
        encoder_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)

        # CLS token embedding → shared representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        cls_embedding = self.dropout(cls_embedding)

        intent_logits = self.intent_head(cls_embedding)
        emotion_logits = self.emotion_head(cls_embedding)

        return intent_logits, emotion_logits


# ─────────────────────────────────────────────
# Loss Function for Training
# ─────────────────────────────────────────────

class MTLLoss(nn.Module):
    """
    Combined loss for both tasks.
    alpha controls how much each task contributes to total loss.
    """

    def __init__(self, alpha_intent: float = 0.5, alpha_emotion: float = 0.5):
        super().__init__()
        self.alpha_intent = alpha_intent
        self.alpha_emotion = alpha_emotion
        self.intent_loss_fn = nn.BCEWithLogitsLoss()   # multi-label
        self.emotion_loss_fn = nn.CrossEntropyLoss()   # multi-class

    def forward(
        self,
        intent_logits: torch.Tensor,
        emotion_logits: torch.Tensor,
        intent_targets: torch.Tensor,   # float tensor, shape (batch, num_intents)
        emotion_targets: torch.Tensor,  # long tensor,  shape (batch,)
    ) -> Dict[str, torch.Tensor]:

        intent_loss = self.intent_loss_fn(intent_logits, intent_targets)
        emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_targets)
        total_loss = self.alpha_intent * intent_loss + self.alpha_emotion * emotion_loss

        return {
            "total": total_loss,
            "intent": intent_loss,
            "emotion": emotion_loss,
        }
