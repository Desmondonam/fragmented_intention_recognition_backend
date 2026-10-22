# Deployment Guide

## Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────┐        ┌──────────────────────┐
│   FRONTEND (Next.js)        │  HTTP   │   API (FastAPI, Docker)     │  once  │  Hugging Face Hub    │
│   frontend/ on Vercel       │ ──────► │   Render / HF Spaces        │ ─────► │  model weights       │
└─────────────────────────────┘         └─────────────────────────────┘        └──────────────────────┘
                                          downloads + caches weights on startup
```

Model weights are never committed to git. They are published to the Hugging
Face Hub and the API downloads them on startup (`app/model_store.py`).

---

## 1. Publish the model

```bash
pip install -r requirements-dev.txt
python -m training.train --tasks intent,emotion --output_dir runs/mtl
huggingface-cli login
python -m scripts.push_to_hub --run_dir runs/mtl --repo_id <hf-user>/intent-emotion-mtl-distilroberta
```

The script prints the commit sha. Use it as `MODEL_REVISION` so a deploy is
pinned to an exact checkpoint.

## 2. Deploy the API (Render, Docker)

1. **New → Web Service**, connect this repo, Runtime **Docker**, port `8000`.
2. Environment variables:

| Key | Value |
|-----|-------|
| `MODEL_REPO_ID` | `<hf-user>/intent-emotion-mtl-distilroberta` |
| `MODEL_REVISION` | commit sha from step 1 |
| `HF_TOKEN` | only if the model repo is private |
| `ALLOWED_ORIGINS` | `https://<your-app>.vercel.app` |
| `W_INTENT` / `W_EMOTION` | `0.4` / `0.6` |
| `THRESHOLD_REVIEW` / `THRESHOLD_ESCALATE` | `0.4` / `0.7` |

The image is CPU-only; distilroberta-base fits comfortably in 1 GB RAM.

> Free Render instances sleep after 15 minutes idle; the first request after
> that waits for the container to start and the model to load.

### Alternative: Hugging Face Spaces (Docker SDK)

Create a Docker Space, push this repo to it, and add to the Space README:

```yaml
---
title: Intent Emotion Escalation API
sdk: docker
app_port: 8000
---
```

## 3. Deploy the frontend (Vercel)

1. Import the repo on Vercel and set **Root Directory** to `frontend`.
2. Set `NEXT_PUBLIC_API_URL` to the API URL from step 2.
3. Add the Vercel URL to the API's `ALLOWED_ORIGINS` and redeploy the API.

---

## Local development

```bash
# Terminal 1 — API (uses a local run instead of the Hub)
MODEL_DIR=runs/mtl uvicorn app.api:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm install && npm run dev   # http://localhost:3000
```

## Production checklist

- [ ] `ALLOWED_ORIGINS` set to the exact Vercel URL (not `*`)
- [ ] `MODEL_REVISION` pinned to a commit sha, not `main`
- [ ] Rate limiting (e.g. `slowapi`) in front of `/analyze`
- [ ] Session state moved to Redis if running more than one replica
- [ ] Uptime monitoring on `/health`
