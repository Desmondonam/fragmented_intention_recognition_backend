"""
Multi-Task Learning (MTL) Model
================================
Shared Encoder (RoBERTa family) → Intent Head + Emotion Head

Architecture:
  [Raw Text]
      │
  [RoBERTa Encoder]  ← Shared backbone (bottom layers frozen, top layers fine-tuned)
      │
  [Pooled CLS Embedding]
      ├──► [Intent Head]   → Softmax over 77 Banking77 intents   e.g. "lost_or_stolen_card" (0.93)
      └──► [Emotion Head]  → Softmax over 7 Ekman emotions       e.g. "anger" (0.71)

Both tasks are single-label multi-class problems, so both heads are trained
with cross-entropy. A checkpoint directory is self-describing: it holds the
encoder config, the label lists and the tokenizer, so inference never needs
the training code or the original base weights.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from model.labels import EMOTION_LABELS, INTENT_LABELS

WEIGHTS_NAME = "model.pt"
CONFIG_NAME = "mtl_config.json"


class MTLIntentEmotionModel(nn.Module):
    """
    Multi-Task Learning model with a shared transformer encoder and
    two task-specific classification heads.
    """

    def __init__(
        self,
        encoder_name: str = "distilroberta-base",
        intent_labels: List[str] = INTENT_LABELS,
        emotion_labels: List[str] = EMOTION_LABELS,
        dropout_rate: float = 0.1,
        freeze_encoder_layers: int = 0,
        encoder_config: Optional[AutoConfig] = None,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.intent_labels = list(intent_labels)
        self.emotion_labels = list(emotion_labels)
        self.dropout_rate = dropout_rate

        # ── Shared Encoder ───────────────────────────────────────────────
        # With an explicit config we build the architecture only (weights come
        # from a checkpoint); otherwise we start from pretrained weights.
        if encoder_config is not None:
            self.encoder = AutoModel.from_config(encoder_config)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self._freeze_encoder_layers(freeze_encoder_layers)

        self.dropout = nn.Dropout(dropout_rate)

        # ── Task Heads ────────────────────────────────────────────────────
        # Raw logits out; softmax is applied at inference time.
        self.intent_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, len(self.intent_labels)),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, len(self.emotion_labels)),
        )

    def _freeze_encoder_layers(self, n: int):
        """Freeze the embeddings and the first N transformer layers of the encoder."""
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
            intent_logits  : (batch, num_intents)
            emotion_logits : (batch, num_emotions)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token embedding → shared representation
        cls_embedding = self.dropout(outputs.last_hidden_state[:, 0, :])

        return self.intent_head(cls_embedding), self.emotion_head(cls_embedding)

    # ─────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────

    def save_pretrained(self, output_dir: str, extra: Optional[Dict] = None):
        """Write weights + a JSON config that is enough to rebuild the model."""
        os.makedirs(output_dir, exist_ok=True)
        config = {
            "encoder_name": self.encoder_name,
            "intent_labels": self.intent_labels,
            "emotion_labels": self.emotion_labels,
            "dropout_rate": self.dropout_rate,
            "encoder_config": self.encoder.config.to_dict(),
            **(extra or {}),
        }
        with open(os.path.join(output_dir, CONFIG_NAME), "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

    @classmethod
    def from_pretrained(cls, model_dir: str, map_location: str = "cpu") -> "MTLIntentEmotionModel":
        with open(os.path.join(model_dir, CONFIG_NAME)) as f:
            config = json.load(f)
        encoder_config = AutoConfig.for_model(**config["encoder_config"])
        model = cls(
            encoder_name=config["encoder_name"],
            intent_labels=config["intent_labels"],
            emotion_labels=config["emotion_labels"],
            dropout_rate=config.get("dropout_rate", 0.1),
            encoder_config=encoder_config,
        )
        state_dict = torch.load(
            os.path.join(model_dir, WEIGHTS_NAME), map_location=map_location, weights_only=True
        )
        model.load_state_dict(state_dict)
        return model


# ─────────────────────────────────────────────
# Loss Function for Training
# ─────────────────────────────────────────────

class MTLLoss(nn.Module):
    """
    Weighted sum of per-task cross-entropy losses.

    The two tasks come from different datasets (Banking77, GoEmotions), so a
    batch normally carries labels for only one task. A missing target (None)
    simply contributes nothing to the total.
    """

    def __init__(self, alpha_intent: float = 1.0, alpha_emotion: float = 1.0):
        super().__init__()
        self.alpha_intent = alpha_intent
        self.alpha_emotion = alpha_emotion
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        intent_logits: torch.Tensor,
        emotion_logits: torch.Tensor,
        intent_targets: Optional[torch.Tensor] = None,   # long tensor, shape (batch,)
        emotion_targets: Optional[torch.Tensor] = None,  # long tensor, shape (batch,)
    ) -> Dict[str, torch.Tensor]:
        zero = intent_logits.new_zeros(())
        intent_loss = self.ce(intent_logits, intent_targets) if intent_targets is not None else zero
        emotion_loss = self.ce(emotion_logits, emotion_targets) if emotion_targets is not None else zero
        total_loss = self.alpha_intent * intent_loss + self.alpha_emotion * emotion_loss

        return {
            "total": total_loss,
            "intent": intent_loss,
            "emotion": emotion_loss,
        }
