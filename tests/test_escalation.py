import pytest

from app.inference import DRIFT_BOOST, EscalationController, SessionState, SessionStore
from model.labels import EMOTION_LABELS, INTENT_LABELS


def uniform(labels, **overrides):
    """A distribution with the given label probabilities and the rest spread evenly."""
    rest = (1.0 - sum(overrides.values())) / (len(labels) - len(overrides))
    return {label: overrides.get(label, rest) for label in labels}


@pytest.fixture
def controller():
    return EscalationController(w_intent=0.4, w_emotion=0.6, threshold_review=0.4, threshold_escalate=0.7)


def test_calm_routine_request_is_autonomous(controller):
    decision = controller.decide(
        uniform(INTENT_LABELS, check_balance=0.0, card_arrival=0.95),
        uniform(EMOTION_LABELS, neutral=0.9, joy=0.1 - 1e-9),
    )
    assert decision.action == "autonomous"
    assert decision.urgency_score < 0.4


def test_angry_theft_report_escalates(controller):
    decision = controller.decide(
        uniform(INTENT_LABELS, lost_or_stolen_card=0.9),
        uniform(EMOTION_LABELS, anger=0.8, fear=0.1),
    )
    assert decision.action == "escalate_now"
    assert decision.urgency_score > 0.7


def test_angry_but_routine_request_is_flagged(controller):
    # 0.6 * ~0.95 urgent-emotion mass, near-zero high-value intent mass
    decision = controller.decide(
        uniform(INTENT_LABELS, card_arrival=0.99),
        uniform(EMOTION_LABELS, anger=0.85, disgust=0.1),
    )
    assert decision.action == "flag_for_review"


def test_probability_mass_is_summed_across_related_classes(controller):
    # Split between two high-value intents: argmax alone would see only 0.45.
    intents = dict.fromkeys(INTENT_LABELS, 0.0)
    intents.update(compromised_card=0.45, lost_or_stolen_card=0.45, card_arrival=0.1)
    assert controller.compute_p_intent(intents) == pytest.approx(0.9)


def test_drift_boost_raises_urgency(controller):
    intents = uniform(INTENT_LABELS, card_arrival=0.99)
    emotions = uniform(EMOTION_LABELS, anger=0.5)
    base = controller.decide(intents, emotions)
    boosted = controller.decide(intents, emotions, emotion_drift=True)
    assert boosted.urgency_score == pytest.approx(base.urgency_score + 0.6 * DRIFT_BOOST, abs=1e-3)
    assert boosted.emotion_drift


def test_urgency_is_capped_at_one():
    controller = EscalationController(w_intent=1.0, w_emotion=1.0)
    decision = controller.decide(
        uniform(INTENT_LABELS, lost_or_stolen_card=1.0 - 1e-9),
        uniform(EMOTION_LABELS, anger=1.0 - 1e-9),
        emotion_drift=True,
    )
    assert decision.urgency_score == 1.0


def test_drift_requires_three_consecutive_urgent_turns():
    state = SessionState(emotion_history=["anger", "fear"])
    assert not state.has_emotion_drift()
    state.emotion_history.append("sadness")
    assert state.has_emotion_drift()
    state.emotion_history.append("neutral")
    assert not state.has_emotion_drift()


def test_session_store_isolates_and_evicts():
    store = SessionStore(max_sessions=2)
    store.get("a").turn = 5
    assert store.get("b").turn == 0
    store.get("c")  # evicts least-recently-used "a"
    assert store.get("a").turn == 0
