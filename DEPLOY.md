# Deployment Guide — MTL Demo

## Architecture

```
┌─────────────────────────────┐         ┌─────────────────────────────┐
│   FRONTEND (Next.js)        │  HTTP   │   BACKEND (FastAPI)         │
│   Deployed on Vercel        │ ──────► │   Deployed on Render        │
│   mtl-demo.vercel.app       │         │   mtl-api.onrender.com      │
└─────────────────────────────┘         └─────────────────────────────┘
         ↑ Free tier OK                          ↑ Needs 1GB+ RAM for PyTorch
```

---

## Part 1 — Deploy the Backend (FastAPI) on Render

Render is the easiest free option for ML workloads. It supports Docker and
gives you enough RAM/CPU for a RoBERTa-base model.

### Step 1 — Push your backend to GitHub

```bash
cd mtl_demo
git init
git add .
git commit -m "Initial MTL backend"
gh repo create mtl-backend --public --push
```

### Step 2 — Create a Render Web Service

1. Go to https://render.com → **New** → **Web Service**
2. Connect your GitHub repo (`mtl-backend`)
3. Set these options:
   - **Runtime**: Docker
   - **Instance Type**: Starter (1GB RAM) — free tier works for demo
   - **Port**: 8000

### Step 3 — Set Environment Variables on Render

In the Render dashboard → Environment, add:

| Key | Value |
|-----|-------|
| `ALLOWED_ORIGINS` | `https://your-app.vercel.app` (set after Vercel deploy) |
| `ENCODER_NAME` | `roberta-base` |
| `W_INTENT` | `0.4` |
| `W_EMOTION` | `0.6` |
| `THRESHOLD_REVIEW` | `0.4` |
| `THRESHOLD_ESCALATE` | `0.7` |

### Step 4 — Deploy

Click **Create Web Service**. Render will build your Docker image and deploy.
Your API URL will be: `https://mtl-backend.onrender.com`

> **Note on free tier**: Render free services spin down after 15 minutes of
> inactivity. First request after sleep takes ~30s (model reload). Upgrade to
> a paid instance ($7/mo) to keep it warm.

---

## Part 2 — Deploy the Frontend (Next.js) on Vercel

### Step 1 — Push your frontend to GitHub

```bash
cd mtl-ui
git init
git add .
git commit -m "Initial MTL frontend"
gh repo create mtl-frontend --public --push
```

### Step 2 — Import on Vercel

1. Go to https://vercel.com → **Add New Project**
2. Import your `mtl-frontend` GitHub repo
3. Framework: **Next.js** (auto-detected)

### Step 3 — Set Environment Variables on Vercel

In the Vercel project settings → **Environment Variables**, add:

| Key | Value |
|-----|-------|
| `NEXT_PUBLIC_API_URL` | `https://mtl-backend.onrender.com` |

### Step 4 — Deploy

Click **Deploy**. Vercel builds and deploys automatically.
Your app will be live at: `https://mtl-frontend.vercel.app`

### Step 5 — Update CORS on Render

Go back to Render → Environment Variables and update:
```
ALLOWED_ORIGINS = https://mtl-frontend.vercel.app
```
Trigger a redeploy on Render.

---

## Alternative Backend Platforms

| Platform | Free Tier | Notes |
|----------|-----------|-------|
| **Render** | 750h/month | Easiest, Docker support |
| **Railway** | $5 credit | Faster cold starts |
| **Hugging Face Spaces** | Free | Best for ML demos, built-in model caching |
| **Modal** | $30/month free | Serverless GPU, scales to zero |
| **Fly.io** | 3 free VMs | Good for always-on deployments |

### Deploy to Hugging Face Spaces (Recommended for demos)

HF Spaces natively supports FastAPI and pre-caches HuggingFace models:

1. Create a Space at https://huggingface.co/new-space
2. Select **Docker** as the SDK
3. Upload your `mtl_demo/` directory
4. Add a `README.md` with:
   ```yaml
   ---
   title: MTL Intent Emotion API
   sdk: docker
   app_port: 8000
   ---
   ```
5. Your API is at: `https://USERNAME-mtl-backend.hf.space`

---

## Local Development

```bash
# Terminal 1 — Backend
cd mtl_demo
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000

# Terminal 2 — Frontend
cd mtl-ui
npm install
npm run dev
# Open http://localhost:3000
```

---

## Production Checklist

- [ ] Set `ALLOWED_ORIGINS` to your exact Vercel URL (not `*`)
- [ ] Switch to `app/api_production.py` for the production server
- [ ] Load a trained checkpoint via `model.load_state_dict(torch.load(...))`
- [ ] Add rate limiting (e.g. `slowapi`) to the FastAPI app
- [ ] Set up uptime monitoring (Better Uptime, UptimeRobot)
- [ ] Enable Render auto-deploy on `main` branch push
- [ ] Enable Vercel auto-deploy on `main` branch push
