import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from model.labels import EMOTION_LABELS, INTENT_LABELS


@pytest.fixture
def client(pipeline):
    with TestClient(create_app(pipeline_factory=lambda: pipeline)) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body == {"status": "healthy", "model_loaded": True, "device": "cpu"}


def test_labels(client):
    body = client.get("/labels").json()
    assert body["intents"] == INTENT_LABELS
    assert body["emotions"] == EMOTION_LABELS


def test_analyze_response_shape(client):
    res = client.post("/analyze", json={"text": "My card was stolen, block it now!"})
    assert res.status_code == 200
    body = res.json()
    assert body["conversation_turn"] == 1
    assert len(body["intents"]) == 3
    probs = [i["probability"] for i in body["intents"]]
    assert probs == sorted(probs, reverse=True)
    assert body["intents"][0]["label"] in INTENT_LABELS
    assert body["emotion"]["label"] in EMOTION_LABELS
    assert 0.0 <= body["escalation"]["urgency_score"] <= 1.0
    assert body["escalation"]["action"] in {"autonomous", "flag_for_review", "escalate_now"}
    assert body["latency_ms"] >= 0


def test_sessions_are_isolated_and_resettable(client):
    for _ in range(2):
        client.post("/analyze", json={"text": "hello", "session_id": "alice"})
    bob = client.post("/analyze", json={"text": "hello", "session_id": "bob"}).json()
    assert bob["conversation_turn"] == 1

    client.post("/session/reset", json={"session_id": "alice"})
    alice = client.post("/analyze", json={"text": "hello", "session_id": "alice"}).json()
    assert alice["conversation_turn"] == 1


def test_reset_without_body_resets_default_session(client):
    client.post("/analyze", json={"text": "hello"})
    assert client.post("/session/reset").json()["session_id"] == "default"
    assert client.post("/analyze", json={"text": "hello"}).json()["conversation_turn"] == 1


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "x" * 513}, {}])
def test_invalid_input_rejected(client, payload):
    assert client.post("/analyze", json=payload).status_code == 422


def test_503_before_model_loads(pipeline):
    app = create_app(pipeline_factory=lambda: pipeline)
    client = TestClient(app)  # not used as a context manager → lifespan never runs
    assert client.post("/analyze", json={"text": "hi"}).status_code == 503
    assert client.get("/health").json()["model_loaded"] is False
