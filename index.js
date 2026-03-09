import { useState, useRef, useEffect } from 'react';
import Head from 'next/head';
import styles from '../styles/Home.module.css';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// ── Preset example utterances ────────────────────────────────────────────
const PRESETS = [
  "What's my current account balance?",
  "My card was stolen at the airport — I need to cancel it immediately.",
  "Oh sure, your system is SO helpful. I've only been waiting 40 minutes.",
  "I need emergency cash and to report a theft, please help me now.",
  "I'VE CALLED THREE TIMES. Nobody is fixing this. I need a real human RIGHT NOW.",
];

// ── Colour maps ──────────────────────────────────────────────────────────
const EMOTION_COLORS = {
  Neutral:     '#00e5ff',
  Frustration: '#ff3d5a',
  Sarcasm:     '#ffd600',
  Urgent:      '#ff6b35',
};

const ACTION_CONFIG = {
  autonomous:      { label: 'AUTONOMOUS',      color: '#00ff88', bg: 'rgba(0,255,136,0.08)' },
  flag_for_review: { label: 'FLAG FOR REVIEW', color: '#ffd600', bg: 'rgba(255,214,0,0.08)'  },
  escalate_now:    { label: 'ESCALATE NOW',    color: '#ff3d5a', bg: 'rgba(255,61,90,0.08)'  },
};

// ── Sub-components ───────────────────────────────────────────────────────

function ProbBar({ value, color, label, sublabel }) {
  return (
    <div className={styles.barRow}>
      <div className={styles.barLabel}>
        <span className={styles.barName}>{label}</span>
        {sublabel && <span className={styles.barSub}>{sublabel}</span>}
      </div>
      <div className={styles.barTrack}>
        <div
          className={styles.barFill}
          style={{ width: `${Math.round(value * 100)}%`, background: color }}
        />
      </div>
      <span className={styles.barPct} style={{ color }}>{(value * 100).toFixed(0)}%</span>
    </div>
  );
}

function EscalationBadge({ action, urgency }) {
  const cfg = ACTION_CONFIG[action] || ACTION_CONFIG.autonomous;
  return (
    <div className={styles.escalationBadge} style={{ background: cfg.bg, borderColor: cfg.color }}>
      <div className={styles.escalationTop}>
        <span className={styles.escalationDot} style={{ background: cfg.color }} />
        <span className={styles.escalationLabel} style={{ color: cfg.color }}>{cfg.label}</span>
        <span className={styles.urgencyScore}>U = {urgency.toFixed(4)}</span>
      </div>
      <div className={styles.urgencyBarTrack}>
        <div
          className={styles.urgencyBarFill}
          style={{
            width: `${Math.round(urgency * 100)}%`,
            background: `linear-gradient(90deg, #00ff88, #ffd600 50%, #ff3d5a)`,
          }}
        />
        <div className={styles.urgencyThreshold} style={{ left: '40%' }} title="Review threshold 0.4" />
        <div className={styles.urgencyThreshold} style={{ left: '70%' }} title="Escalate threshold 0.7" />
      </div>
    </div>
  );
}

function TurnBubble({ turn, isLatest }) {
  const cfg = ACTION_CONFIG[turn.escalation.action] || ACTION_CONFIG.autonomous;
  return (
    <div className={`${styles.turnBubble} ${isLatest ? styles.turnLatest : ''}`}>
      <div className={styles.turnHeader}>
        <span className={styles.turnNum}>Turn {turn.conversation_turn}</span>
        <span className={styles.turnAction} style={{ color: cfg.color }}>● {cfg.label}</span>
        <span className={styles.turnLatency}>{turn.latency_ms}ms</span>
      </div>
      <p className={styles.turnText}>"{turn.raw_text}"</p>
      <div className={styles.turnTags}>
        {turn.intents.map(i => (
          <span key={i.label} className={styles.intentTag}>{i.label} {(i.probability * 100).toFixed(0)}%</span>
        ))}
        <span className={styles.emotionTag} style={{ borderColor: EMOTION_COLORS[turn.emotion.label] || '#00e5ff' }}>
          {turn.emotion.label} {(turn.emotion.probability * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

// ── Main Page ────────────────────────────────────────────────────────────

export default function Home() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking'); // 'checking' | 'online' | 'offline'
  const textareaRef = useRef(null);

  // ── Check API health on mount ──────────────────────────────────────────
  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? setApiStatus('online') : setApiStatus('offline'))
      .catch(() => setApiStatus('offline'));
  }, []);

  // ── Submit ─────────────────────────────────────────────────────────────
  async function handleAnalyze() {
    if (!text.trim() || loading) return;
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      setHistory(prev => [data, ...prev]);
      setText('');
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleReset() {
    await fetch(`${API_URL}/session/reset`, { method: 'POST' }).catch(() => {});
    setHistory([]);
    setResult(null);
    setError(null);
    setText('');
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleAnalyze();
  }

  const emotionColor = result ? (EMOTION_COLORS[result.emotion.label] || '#00e5ff') : '#00e5ff';

  return (
    <>
      <Head>
        <title>MTL Intent & Emotion — Demo</title>
        <meta name="description" content="Multi-Task Learning: Joint Intent Recognition + Emotion-Aware Escalation" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>" />
      </Head>

      <div className={styles.layout}>
        {/* ── Header ── */}
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <div className={styles.logoMark}>
              <span className={styles.logoIcon}>⬡</span>
              <span className={styles.logoText}>MTL<span className={styles.logoDim}>/demo</span></span>
            </div>
            <p className={styles.headerSub}>Multi-Task Learning · Intent + Emotion · Escalation Engine</p>
          </div>
          <div className={styles.headerRight}>
            <div className={`${styles.statusPill} ${styles[apiStatus]}`}>
              <span className={styles.statusDot} />
              <span className={styles.statusLabel}>
                {apiStatus === 'checking' ? 'CONNECTING…' : apiStatus === 'online' ? 'API ONLINE' : 'API OFFLINE'}
              </span>
            </div>
            {history.length > 0 && (
              <button className={styles.resetBtn} onClick={handleReset}>
                ↺ RESET SESSION
              </button>
            )}
          </div>
        </header>

        <main className={styles.main}>
          {/* ── Left column: Input + Results ── */}
          <section className={styles.colLeft}>

            {/* Input panel */}
            <div className={styles.inputPanel}>
              <div className={styles.inputHeader}>
                <span className={styles.inputLabel}>USER UTTERANCE</span>
                <span className={styles.inputHint}>⌘↵ to analyze</span>
              </div>
              <textarea
                ref={textareaRef}
                className={styles.textarea}
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type a customer message…"
                rows={4}
              />
              <div className={styles.inputFooter}>
                <div className={styles.presets}>
                  {PRESETS.map((p, i) => (
                    <button key={i} className={styles.presetBtn} onClick={() => setText(p)}>
                      {p.length > 42 ? p.slice(0, 42) + '…' : p}
                    </button>
                  ))}
                </div>
                <button
                  className={`${styles.analyzeBtn} ${loading ? styles.analyzeBtnLoading : ''}`}
                  onClick={handleAnalyze}
                  disabled={!text.trim() || loading || apiStatus === 'offline'}
                >
                  {loading ? <span className={styles.spinner} /> : '▶ ANALYZE'}
                </button>
              </div>
            </div>

            {error && (
              <div className={styles.errorBanner}>
                ⚠ {error}
              </div>
            )}

            {/* Results panel */}
            {result && (
              <div className={styles.resultsPanel} key={result.conversation_turn}>

                {/* Escalation */}
                <div className={styles.section}>
                  <div className={styles.sectionTitle}>ESCALATION DECISION</div>
                  <EscalationBadge action={result.escalation.action} urgency={result.escalation.urgency_score} />
                  <div className={styles.botResponse}>
                    <span className={styles.botLabel}>BOT RESPONSE</span>
                    <p className={styles.botMessage}>"{result.escalation.message}"</p>
                  </div>
                </div>

                {/* Intents */}
                <div className={styles.section}>
                  <div className={styles.sectionTitle}>DETECTED INTENTS <span className={styles.sectionNote}>(multi-label · sigmoid)</span></div>
                  {result.intents.length > 0
                    ? result.intents.map(i => (
                        <ProbBar key={i.label} value={i.probability} color="var(--accent-cyan)" label={i.label} />
                      ))
                    : <p className={styles.emptyNote}>No intents above threshold.</p>
                  }
                </div>

                {/* Emotion */}
                <div className={styles.section}>
                  <div className={styles.sectionTitle}>EMOTION STATE <span className={styles.sectionNote}>(softmax)</span></div>
                  <ProbBar
                    value={result.emotion.probability}
                    color={emotionColor}
                    label={result.emotion.label}
                  />
                </div>

                <div className={styles.metaRow}>
                  <span>Turn <strong>{result.conversation_turn}</strong></span>
                  <span>Latency <strong>{result.latency_ms}ms</strong></span>
                </div>
              </div>
            )}

            {!result && !loading && (
              <div className={styles.emptyState}>
                <div className={styles.emptyIcon}>◈</div>
                <p>Enter a message above or pick a preset to run inference.</p>
              </div>
            )}
          </section>

          {/* ── Right column: Conversation history ── */}
          <section className={styles.colRight}>
            <div className={styles.historyHeader}>
              <span className={styles.historyTitle}>CONVERSATION HISTORY</span>
              <span className={styles.historyCount}>{history.length} turns</span>
            </div>

            {history.length === 0 ? (
              <div className={styles.historyEmpty}>
                <p>Turns will appear here as you analyze utterances.</p>
                <p className={styles.historyEmptyNote}>Emotion drift detection activates after 3 consecutive high-urgency turns.</p>
              </div>
            ) : (
              <div className={styles.historyList}>
                {history.map((turn, i) => (
                  <TurnBubble key={`${turn.conversation_turn}-${i}`} turn={turn} isLatest={i === 0} />
                ))}
              </div>
            )}

            {/* Architecture legend */}
            <div className={styles.legend}>
              <div className={styles.legendTitle}>URGENCY SCORE FORMULA</div>
              <div className={styles.legendFormula}>
                U = (w<sub>i</sub> · P<sub>intent</sub>) + (w<sub>e</sub> · P<sub>emotion</sub>)
              </div>
              <div className={styles.legendThresholds}>
                <span style={{ color: 'var(--autonomous)' }}>● 0.0–0.4 Autonomous</span>
                <span style={{ color: 'var(--flag-for-review)' }}>● 0.4–0.7 Review</span>
                <span style={{ color: 'var(--escalate-now)' }}>● &gt;0.7 Escalate</span>
              </div>
            </div>
          </section>
        </main>
      </div>
    </>
  );
}
