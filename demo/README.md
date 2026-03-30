# Intention Recognition MVP

**AI-powered multi-intent parsing, real-time emotion detection, and intelligent escalation for conversational systems.**

Intention Recognition is a next-generation conversational AI engine that closes the gap between what users *say* and what they actually *mean*. It detects sarcasm, parses multiple intents from a single message, tracks frustration over time, and automatically escalates high-emotion conversations to human agents with a full sentiment brief.

---

## Features

- **Multi-intent detection** — parses compound requests into individual, trackable intents with confidence scores
- **Sarcasm recognition** — identifies sarcastic tone and inverts sentiment polarity accordingly
- **Real-time emotion tracking** — detects frustration, confusion, urgency, resignation, and satisfaction across the conversation timeline
- **Automatic escalation** — triggers handoff to a human agent when cumulative frustration exceeds a configurable threshold, with a generated sentiment summary
- **Agent dashboard** — live frustration timeline, intent inventory, and escalation briefs for human agents
- **Pre-built scenarios** — demo frustrated customers, multi-intent queries, and sarcasm detection out of the box

---

## Tech Stack

- **React 18** (functional components + hooks)
- **Next.js 14** (App Router)
- **CSS-in-JS** (inline styles, zero external dependencies)
- **Google Fonts** (Space Grotesk, DM Mono, Instrument Sans)

---

## Prerequisites

Make sure you have the following installed:

- **Node.js** — version 18.17 or later ([download here](https://nodejs.org/))
- **npm** — comes with Node.js (or use yarn/pnpm if you prefer)
- **Git** — for version control and Vercel deployment

Verify your installation:

```bash
node --version    # should print v18.17.0 or higher
npm --version     # should print 9.x or higher
```

---

## Project Setup

### 1. Create a new Next.js project

Open your terminal and run:

```bash
npx create-next-app@latest intention-recognition --app --src-dir --tailwind=no --eslint --import-alias="@/*"
```

When prompted, select:
- TypeScript: **No**
- Tailwind CSS: **No**
- App Router: **Yes**

Then navigate into the project:

```bash
cd intention-recognition
```

### 2. Add the MVP component

Copy the downloaded `intention-recognition-mvp.jsx` file into your project's components directory:

```bash
mkdir -p src/components
cp /path/to/your/downloaded/intention-recognition-mvp.jsx src/components/IntentionRecognition.jsx
```

### 3. Set up the main page

Replace the contents of `src/app/page.js` with:

```jsx
"use client";

import IntentionRecognition from "@/components/IntentionRecognition";

export default function Home() {
  return <IntentionRecognition />;
}
```

### 4. Clean up default styles

Replace the contents of `src/app/globals.css` with:

```css
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html,
body {
  height: 100%;
  overflow: hidden;
}
```

### 5. Update the layout metadata (optional)

In `src/app/layout.js`, update the metadata:

```jsx
export const metadata = {
  title: "Intention Recognition — AI Emotion & Intent Engine",
  description:
    "Multi-intent parsing, sarcasm detection, and real-time emotion tracking for conversational AI.",
};
```

---

## Running Locally

### Install dependencies

```bash
npm install
```

### Start the development server

```bash
npm run dev
```

Open your browser and go to:

```
http://localhost:3000
```

You should see the full Intention Recognition MVP with the chat panel, analysis panel, and agent dashboard.

### Test it out

1. Click one of the **scenario buttons** (Frustrated Customer, Multi-Intent Query, Sarcasm Detection) to watch the system process a full conversation automatically.
2. Type your own messages in the input field — try mixing sarcasm, multiple requests, and frustrated language.
3. Switch between the **Chat**, **Analysis**, and **Dashboard** tabs to see real-time intent parsing, emotion detection, and the agent sentiment view.

---

## Deploying to Vercel

### Option A: Deploy via GitHub (recommended)

This is the simplest and most maintainable approach. Every push to your repo will trigger an automatic redeployment.

**Step 1 — Push your project to GitHub:**

```bash
git init
git add .
git commit -m "Initial commit — Intention Recognition MVP"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/intention-recognition.git
git push -u origin main
```

**Step 2 — Connect to Vercel:**

1. Go to [vercel.com](https://vercel.com) and sign in (or create an account) with your GitHub account.
2. Click **"Add New..."** then **"Project"**.
3. Find and select your `intention-recognition` repository from the list.
4. Vercel will auto-detect that it is a Next.js project. Leave all settings as default.
5. Click **"Deploy"**.

**Step 3 — Done.**

Vercel will build and deploy your app. You will get a live URL like:

```
https://intention-recognition-XXXX.vercel.app
```

Every time you push to `main`, Vercel redeploys automatically.

### Option B: Deploy via Vercel CLI

If you prefer deploying from the command line without a GitHub repo:

**Step 1 — Install the Vercel CLI:**

```bash
npm install -g vercel
```

**Step 2 — Log in:**

```bash
vercel login
```

Follow the prompts to authenticate via your browser.

**Step 3 — Deploy:**

From inside your project directory:

```bash
vercel
```

The CLI will ask a few questions:

- **Set up and deploy?** Yes
- **Which scope?** Select your account
- **Link to existing project?** No (create new)
- **Project name?** `intention-recognition` (or whatever you prefer)
- **Directory with your code?** `./` (current directory)
- **Override settings?** No

Vercel will build and deploy. You will get a preview URL immediately.

**Step 4 — Deploy to production:**

```bash
vercel --prod
```

This pushes to your production URL.

---

## Project Structure

After setup, your project should look like this:

```
intention-recognition/
├── src/
│   ├── app/
│   │   ├── globals.css          # Minimal global styles
│   │   ├── layout.js            # Root layout with metadata
│   │   └── page.js              # Main page (loads the MVP component)
│   └── components/
│       └── IntentionRecognition.jsx   # The full MVP component
├── public/
├── package.json
├── next.config.mjs
└── README.md
```

---

## Configuration

The escalation threshold and analysis engine are configurable inside `IntentionRecognition.jsx`:

| Constant | Default | Description |
|---|---|---|
| `FRUSTRATION_THRESHOLD` | `7` | Cumulative frustration score (0–10) that triggers automatic escalation to a human agent |

To adjust, open the component file and change the value at the top:

```jsx
const FRUSTRATION_THRESHOLD = 7; // Lower = more sensitive escalation
```

---

## Extending the MVP

This MVP uses a rule-based analysis engine (regex + heuristics) designed as a drop-in placeholder. Here is how to evolve it toward production.

### Replace the analysis engine with a real ML model

The entire analysis pipeline lives in a single function: `analyzeMessage()`. To integrate a real NLP backend:

1. Create an API route at `src/app/api/analyze/route.js`:

```jsx
export async function POST(request) {
  const { text, history } = await request.json();

  // Call your ML model / external API here
  const analysis = await yourMLService.analyze(text, history);

  return Response.json(analysis);
}
```

2. Update the component to call this API instead of the local function.

### Add persistent storage

The current MVP holds all state in memory. To persist conversations, connect a database (Supabase, PlanetScale, or Vercel KV) via an API route.

### Add authentication

Use NextAuth.js or Clerk to add user authentication, so agents and customers see different views.

### Add WebSocket support

For true real-time agent dashboards with multiple concurrent conversations, add a WebSocket layer (e.g., Pusher, Ably, or Socket.io).

---

## Troubleshooting

**"Module not found" error on import**

Make sure the component file is at `src/components/IntentionRecognition.jsx` and that your import path uses the alias:

```jsx
import IntentionRecognition from "@/components/IntentionRecognition";
```

**Fonts not loading**

The component imports Google Fonts via a CSS `@import` rule. If fonts don't render, check that your network allows requests to `fonts.googleapis.com`. In production on Vercel, this works automatically.

**Vercel build fails**

If the build fails with a hydration error, make sure you have `"use client"` at the top of `src/app/page.js`. The MVP component uses React hooks and browser APIs that require client-side rendering.

**Port 3000 already in use**

Run on a different port:

```bash
npm run dev -- -p 3001
```

---

## License

MIT — use freely, modify as needed, build something great.
