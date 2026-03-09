# MTL Intent & Emotion — Demo Application

A Multi-Task Learning (MTL) system that jointly recognises **user intent** and **emotion** from conversational text, then applies a weighted **Urgency Score** to decide whether to auto-handle, flag, or immediately escalate to a human agent.

---

## Architecture at a Glance

```
[Raw Text]
    │
[RoBERTa Encoder]   ← shared backbone
    │
[CLS Embedding]
    ├──► [Intent Head]  → Sigmoid  → multi-label  e.g. ["report_theft", "cancel_card"]
    └──► [Emotion Head] → Softmax  → single label e.g. "Frustration" (0.87)
                                         │
                              [Escalation Controller]
                               U = (w_i · P_intent) + (w_e · P_emotion)
                                         │
                    ┌────────────────────┼──────────────────────┐
                 U < 0.4             0.4–0.7                U > 0.7
              Autonomous           Flag for review         Escalate NOW
```

---

## Project Structure

```
mtl_demo/
├── model/
│   └── mtl_model.py      # MTLIntentEmotionModel + MTLLoss
├── app/
│   ├── inference.py      # MTLInferencePipeline + EscalationController
│   └── api.py            # FastAPI REST application
├── train.py              # Training loop with synthetic data
├── demo.py               # Standalone terminal demo (no server)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the terminal demo (fastest way to test)
```bash
python demo.py
```
This simulates a 5-turn escalating conversation and prints intent, emotion, and escalation decisions to the terminal. Uses random model weights — outputs will be random, but the full pipeline is exercised.

### 3. Train on synthetic data
```bash
python train.py --epochs 5 --batch_size 4
# Saves checkpoint to ./checkpoints/mtl_model.pt
```
Swap `SYNTHETIC_DATA` in `train.py` for your real labeled dataset.

### 4. Start the API server
```bash
uvicorn app.api:app --reload --port 8000
```

**Then try it:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I need to cancel my card, it was stolen and nobody is helping me!"}'
```

**Or open the interactive docs:**  
http://localhost:8000/docs

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/analyze` | Analyze text → intents + emotion + escalation |
| `POST` | `/session/reset` | Reset multi-turn conversation state |
| `GET`  | `/labels` | List all intent and emotion labels |
| `GET`  | `/health` | Health check |

### Example Response `/analyze`

```json
{
  "raw_text": "I've been waiting 20 minutes — I need to speak to someone NOW.",
  "conversation_turn": 3,
  "intents": [
    { "label": "speak_to_agent", "probability": 0.91 },
    { "label": "report_theft",   "probability": 0.47 }
  ],
  "emotion": {
    "label": "Frustration",
    "probability": 0.84
  },
  "escalation": {
    "urgency_score": 0.87,
    "action": "escalate_now",
    "message": "I can hear that this is urgent..."
  },
  "latency_ms": 43.2
}
```

---

## Escalation Thresholds

| Score (U) | Action | Description |
|-----------|--------|-------------|
| 0.0 – 0.4 | `autonomous` | Bot handles independently |
| 0.4 – 0.7 | `flag_for_review` | Bot responds; supervisor alerted |
| > 0.7 | `escalate_now` | Immediate handoff to human agent |

Weights `w_intent` and `w_emotion` are configurable per business vertical (e.g. healthcare may weight emotion higher).

---

## Next Steps (Production Roadmap)

- [ ] Replace synthetic data with real labeled conversations
- [ ] Add Slot Filling / NER head for entity extraction (Location, Object, etc.)
- [ ] Add Cross-Attention Interaction Layer (emotion boosts intent weights)
- [ ] Export model to ONNX for <200ms inference via ONNX Runtime
- [ ] Add Redis for multi-turn session state persistence
- [ ] Instrument with Weights & Biases for sarcasm detection monitoring
