import torch

from model.labels import EMOTION_LABELS, GOEMOTIONS_TO_EKMAN, INTENT_LABELS
from model.mtl_model import MTLIntentEmotionModel, MTLLoss
from tests.conftest import StubTokenizer, tiny_encoder_config
from training.data import goemotions_to_ekman_label


def test_forward_shapes(tiny_model):
    batch = StubTokenizer()(["my card is gone", "hi"])
    intent_logits, emotion_logits = tiny_model(**batch)
    assert intent_logits.shape == (2, len(INTENT_LABELS))
    assert emotion_logits.shape == (2, len(EMOTION_LABELS))


def test_save_load_roundtrip(tiny_model, tmp_path):
    tiny_model.save_pretrained(tmp_path)
    restored = MTLIntentEmotionModel.from_pretrained(tmp_path)
    tiny_model.eval()
    restored.eval()
    batch = StubTokenizer()("where is my refund")
    for a, b in zip(tiny_model(**batch), restored(**batch), strict=True):
        assert torch.allclose(a, b)
    assert restored.intent_labels == INTENT_LABELS


def test_loss_ignores_missing_task():
    loss_fn = MTLLoss()
    intent_logits, emotion_logits = torch.randn(4, 77), torch.randn(4, 7)
    out = loss_fn(intent_logits, emotion_logits, intent_targets=torch.tensor([0, 1, 2, 3]))
    assert out["emotion"].item() == 0.0
    assert out["total"].item() == out["intent"].item() > 0


def test_freezing_bottom_layers():
    model = MTLIntentEmotionModel(encoder_config=tiny_encoder_config(), freeze_encoder_layers=1)
    assert not any(p.requires_grad for p in model.encoder.embeddings.parameters())
    assert not any(p.requires_grad for p in model.encoder.encoder.layer[0].parameters())
    assert all(p.requires_grad for p in model.encoder.encoder.layer[1].parameters())


def test_label_spaces():
    assert len(INTENT_LABELS) == len(set(INTENT_LABELS)) == 77
    assert set(GOEMOTIONS_TO_EKMAN.values()) == set(EMOTION_LABELS)
    assert len(GOEMOTIONS_TO_EKMAN) == 28  # 27 emotions + neutral


def test_high_value_intents_exist_in_banking77():
    from app.inference import HIGH_URGENCY_EMOTIONS, HIGH_VALUE_INTENTS

    assert HIGH_VALUE_INTENTS <= set(INTENT_LABELS)
    assert HIGH_URGENCY_EMOTIONS <= set(EMOTION_LABELS)


def test_goemotions_folding():
    names = list(GOEMOTIONS_TO_EKMAN)
    idx = {n: i for i, n in enumerate(names)}
    assert goemotions_to_ekman_label([idx["annoyance"]], names) == EMOTION_LABELS.index("anger")
    # annoyance + anger both fold to anger → kept
    assert goemotions_to_ekman_label([idx["annoyance"], idx["anger"]], names) == EMOTION_LABELS.index("anger")
    # joy + sadness disagree → dropped
    assert goemotions_to_ekman_label([idx["joy"], idx["grief"]], names) is None
