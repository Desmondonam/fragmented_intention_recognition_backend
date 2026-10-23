# Intent & Emotion Escalation API

[![CI](https://github.com/Desmondonam/intent-emotion-escalation-api/actions/workflows/ci.yml/badge.svg)](https://github.com/Desmondonam/intent-emotion-escalation-api/actions/workflows/ci.yml)

A multi-task RoBERTa model that reads a customer message once and predicts both
**what they want** (77 banking intents) and **how they feel** (7 emotions). A
small escalation controller then decides whether the bot answers alone, flags
the conversation for a supervisor, or hands off to a human immediately.

Served with FastAPI; weights live on the Hugging Face Hub; a Next.js UI lives in [`frontend/`](frontend/).

---

## Results

Trained and evaluated on public benchmarks. All numbers are on the **official test splits**.

| Task | Metric | Single-task | Multi-task | Δ |
|------|--------|-------------|------------|---|
| Intent (Banking77, 77 classes) | Accuracy | 0.8760 | 0.8714 | -0.0046 |
| Intent (Banking77, 77 classes) | Macro-F1 | 0.8740 | 0.8687 | -0.0053 |
| Emotion (GoEmotions → Ekman, 7 classes) | Accuracy | 0.6826 | 0.6739 | -0.0087 |
| Emotion (GoEmotions → Ekman, 7 classes) | Macro-F1 | 0.5928 | 0.5641 | -0.0287 |

<sub>Test sets: intent n=3,080, emotion n=4,968. distilroberta-base, embeddings + bottom 3 of 6 layers
frozen, 9,000 training examples per task, 3 epochs, lr 5e-5, batch 32, seed 42. Best epoch picked on
validation macro-F1. Raw metrics and per-epoch history: [`results/`](results/).</sub>

**What the comparison shows.** At this budget, sharing the encoder did *not* help
either task. Intent is essentially unchanged (−0.5 pt, the size of seed-to-seed
noise with one seed), and emotion macro-F1 drops by 2.9 pt. Most of that loss is
in the rare classes: disgust F1 falls from 0.359 to 0.245 (76 test examples) and
fear from 0.654 to 0.622 (80), while the common classes move by under 2 pt. This is mild negative transfer: the
two tasks come from very different text (short banking queries vs Reddit
comments), and with half the encoder frozen only three layers are shared, so
the tasks compete for the same capacity.

**What the shared model buys.** One encoder pass returns both predictions. Two
single-task models need two passes and twice the memory. Measured below, that's
the trade: about 0.5 pt of intent accuracy and 3 pt of emotion macro-F1, in
exchange for roughly half the serving compute.

**Context.** All three runs were still improving at epoch 3; the budget was set
by CPU-only training, not convergence. For reference, the Banking77 paper
reports ~93% accuracy for fully fine-tuned models on the full training set, and
the GoEmotions paper reports 0.64 Ekman-level macro-F1 for BERT-base. That is
multi-label on all comments, so it's not directly comparable to the single-label
subset used here.

### Latency

Single request (batch size 1), 500 utterances sampled from the two test sets,
after 20 warm-up requests. Measured on a laptop CPU (Intel Core, 4 PyTorch
threads, Windows 11, torch 2.14 CPU). Raw output: [`results/latency.json`](results/latency.json).

| Path | p50 | p95 | p99 |
|------|-----|-----|-----|
| Model only (tokenize + forward + escalation) | 52 ms | 120 ms | 225 ms |
| Full HTTP `POST /analyze` (in-process client, no network) | 54 ms | 107 ms | 276 ms |
| One multi-task forward pass | 48 ms | 116 ms | 222 ms |
| Two single-task forward passes (the alternative) | 97 ms | 215 ms | 483 ms |

The <200 ms goal is met at the median and p95 on this laptop, but **not at p99**.
The tail comes from CPU frequency scaling on a mobile chip, and a server CPU or
ONNX Runtime export should tighten it. Two single-task models would roughly
double latency and blow the p95 budget.

---

## Architecture

```
[Customer message]
        │
[distilroberta-base encoder]   ← shared; bottom 3 layers frozen
        │
   [CLS embedding]
        ├──► Intent head  → softmax over 77 Banking77 intents
        └──► Emotion head → softmax over 7 Ekman emotions
                    │
          [Escalation controller]
     U = w_i · P(high-value intent) + w_e · P(urgent emotion)
                    │
      ┌─────────────┼──────────────┐
   U ≤ 0.4      0.4 < U ≤ 0.7     U > 0.7
 autonomous   flag_for_review   escalate_now
```

- **P(high-value intent)** is the probability mass on intents where a wrong bot
  answer is costly: `lost_or_stolen_card`, `compromised_card`,
  `card_payment_not_recognised`, `transaction_charged_twice`, and so on (see
  [`app/inference.py`](app/inference.py)).
- **P(urgent emotion)** is the mass on anger, disgust, fear and sadness.
- Using the full distribution rather than the top label keeps U stable when the
  model is split between related classes.
- **Emotion drift.** If a session's last three turns were all urgent emotions,
  P(urgent emotion) is boosted by 0.15, so sustained frustration escalates even
  when each message alone is borderline.
- Weights and thresholds are environment variables. They're hand-set business
  rules, not learned or evaluated against labelled escalations (see Limitations).

### Training setup

The two datasets are disjoint, so each training batch comes from one task and
only that head's loss is used. Batches from the two tasks are interleaved
randomly within an epoch. The single-task baselines use the same script,
architecture, hyperparameters and per-task data. The only difference is which
loss is active, so the comparison is like-for-like.

- **Banking77**: 13,083 customer-service queries labelled with 77 intents.
  10% of train is held out (stratified) for validation.
- **GoEmotions**: 58k Reddit comments with 27 emotions + neutral, folded into
  Ekman's 7 classes using the dataset authors' mapping. Comments whose labels
  disagree at the Ekman level are dropped, leaving a single-label task.

---

## Repository layout

```
├── model/          MTLIntentEmotionModel, loss, label spaces
├── training/       dataset loading + training/evaluation loop
├── app/            FastAPI service, inference pipeline, escalation, Hub loading
├── scripts/        run_experiments.sh, summarize_results, benchmark_latency, push_to_hub
├── results/        committed metrics from the runs reported above
├── tests/          offline pytest suite (no downloads needed)
├── frontend/       Next.js UI that calls the API
└── Dockerfile      CPU image for the API
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

# Run the API. Weights download from the Hugging Face Hub on first start.
uvicorn app.api:app --port 8000
# ...or point it at a local training run:
MODEL_DIR=runs/mtl uvicorn app.api:app --port 8000
```

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Someone used my card in another country, block it right now!", "session_id": "abc123"}'
```

Real response from the trained multi-task model:

```json
{
  "raw_text": "Someone used my card in another country, block it right now!",
  "session_id": "abc123",
  "conversation_turn": 1,
  "intents": [
    { "label": "compromised_card",            "probability": 0.2935 },
    { "label": "card_payment_not_recognised", "probability": 0.1216 },
    { "label": "terminate_account",           "probability": 0.0882 }
  ],
  "emotion": { "label": "anger", "probability": 0.3443 },
  "escalation": {
    "urgency_score": 0.6333,
    "action": "flag_for_review",
    "message": "I've noted your concern and I'm handling this for you. A supervisor has also been alerted to review our conversation shortly.",
    "emotion_drift": false
  },
  "latency_ms": 64.32
}
```

The probabilities are modest: the top intent is right but holds only 0.29 of the
mass, and the rest is spread over related fraud intents. That's why urgency is
computed from probability mass rather than the top label.

Interactive docs: http://localhost:8000/docs

### Reproduce the results

```bash
bash scripts/run_experiments.sh        # trains MTL + both baselines, writes results/
MODEL_DIR=runs/mtl python -m scripts.benchmark_latency \
    --single_task_dirs runs/stl_intent runs/stl_emotion
```

### Tests

```bash
ruff check . && pytest -q
```

CI runs lint and tests for the API and builds the frontend on every push and PR.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | `{text, session_id?}` → top-3 intents, emotion, escalation decision |
| `POST` | `/session/reset` | `{session_id?}` → clear that conversation's turn count and emotion history |
| `GET`  | `/labels` | All intent and emotion labels |
| `GET`  | `/health` | Liveness + whether the model is loaded |

Configuration is all environment variables (`MODEL_REPO_ID`, `MODEL_REVISION`,
`MODEL_DIR`, `ALLOWED_ORIGINS`, `W_INTENT`, `W_EMOTION`, `THRESHOLD_REVIEW`,
`THRESHOLD_ESCALATE`, `TOP_K_INTENTS`). Deployment steps are in [DEPLOY.md](DEPLOY.md).

## Limitations

- **Escalation is not evaluated end to end.** There's no public dataset of
  support conversations labelled with "should have escalated", so the urgency
  weights and thresholds are hand-set. Intent and emotion accuracy are measured;
  escalation quality is not.
- **Domain shift on emotion.** GoEmotions is Reddit, not support chat. Expect
  emotion accuracy on real customer messages to differ from the test score.
- **Single seed, CPU budget.** Differences under ~1 pt shouldn't be read as
  significant. Multiple seeds and full fine-tuning on a GPU are the obvious next
  step.
- **Session state is in-process memory.** Fine for one replica; use Redis or
  similar before scaling out.

## Roadmap

- [ ] Multiple seeds + full fine-tuning on GPU; try uncertainty-weighted task losses to reduce negative transfer
- [ ] ONNX Runtime / int8 export and re-benchmark
- [ ] Redis-backed sessions for multi-replica deployments
- [ ] Label a small set of real escalation decisions to calibrate the thresholds
