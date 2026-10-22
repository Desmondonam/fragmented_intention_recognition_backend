"""
Shared fixtures. Everything here is offline: the encoder is a tiny randomly
initialised RoBERTa built from a config, and the tokenizer is a stub, so the
suite runs in CI in seconds without downloading weights.
"""

import pytest
import torch
from transformers import AutoConfig

from app.inference import MTLInferencePipeline
from model.mtl_model import MTLIntentEmotionModel

VOCAB_SIZE = 100


class StubTokenizer:
    """Maps characters to ids — deterministic, dependency-free stand-in for a real tokenizer."""

    def __call__(self, texts, return_tensors="pt", truncation=True, max_length=64, padding=False):
        single = isinstance(texts, str)
        texts = [texts] if single else texts
        ids = [[0] + [3 + ord(c) % (VOCAB_SIZE - 3) for c in t][: max_length - 2] + [2] for t in texts]
        width = max(len(x) for x in ids)
        input_ids = torch.ones(len(ids), width, dtype=torch.long)
        attention_mask = torch.zeros(len(ids), width, dtype=torch.long)
        for i, row in enumerate(ids):
            input_ids[i, : len(row)] = torch.tensor(row)
            attention_mask[i, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def tiny_encoder_config():
    return AutoConfig.for_model(
        "roberta", vocab_size=VOCAB_SIZE, hidden_size=32, num_hidden_layers=2,
        num_attention_heads=2, intermediate_size=64, max_position_embeddings=80,
    )


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return MTLIntentEmotionModel(encoder_name="tiny-roberta", encoder_config=tiny_encoder_config())


@pytest.fixture
def pipeline(tiny_model):
    return MTLInferencePipeline(model=tiny_model, tokenizer=StubTokenizer(), device="cpu")
