# Frontend — Intent & Emotion Escalation UI

Next.js (pages router) client for the FastAPI service in the repo root. Every
prediction shown here comes from the trained model via `POST /analyze`; there is
no client-side heuristic fallback.

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev   # http://localhost:3000
```

## Deploying on Vercel

1. Import the repo and set **Root Directory** to `frontend`.
2. Add `NEXT_PUBLIC_API_URL` (your deployed API URL) under Environment Variables.
3. Add the Vercel URL to the API's `ALLOWED_ORIGINS`.
