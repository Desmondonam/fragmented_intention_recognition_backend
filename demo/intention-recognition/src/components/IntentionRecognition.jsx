import { useState, useEffect, useRef, useCallback } from "react";

// ─── CONFIG ───
const FRUSTRATION_THRESHOLD = 7;

// ─── MOCK ANALYSIS ENGINE (Replace with real API calls in production) ───
function analyzeMessage(text, conversationHistory) {
  const lower = text.toLowerCase();

  // Sarcasm detection heuristics
  const sarcasmPatterns = [
    /oh (great|wonderful|fantastic|perfect|sure|brilliant)/i,
    /yeah,? right/i,
    /how (wonderful|lovely|nice|great)/i,
    /thanks for nothing/i,
    /exactly what i (needed|wanted)/i,
    /oh joy/i,
    /what a surprise/i,
    /clearly/i,
    /obviously/i,
  ];
  const hasSarcasm = sarcasmPatterns.some((p) => p.test(text));

  // Emotion detection
  const frustrationWords = [
    "frustrated", "annoying", "useless", "broken", "terrible", "awful",
    "ridiculous", "waste", "stupid", "hate", "worst", "unacceptable",
    "pathetic", "garbage", "trash", "horrible", "disgusting", "angry",
    "furious", "fed up", "sick of", "tired of", "enough", "seriously",
    "again", "still", "never works", "doesn't work", "can't believe",
  ];
  const confusionWords = [
    "confused", "don't understand", "makes no sense", "unclear", "lost",
    "what do you mean", "how does", "why is", "help me understand",
  ];
  const urgencyWords = [
    "urgent", "asap", "immediately", "right now", "emergency", "critical",
    "deadline", "hurry", "quickly", "can't wait",
  ];
  const positiveWords = [
    "thanks", "great", "awesome", "perfect", "love", "excellent",
    "helpful", "appreciate", "good", "nice", "wonderful",
  ];

  let emotions = {};
  let frustrationScore = 0;

  const wordCount = (words) =>
    words.filter((w) => lower.includes(w)).length;

  const frustCount = wordCount(frustrationWords);
  const confCount = wordCount(confusionWords);
  const urgCount = wordCount(urgencyWords);
  const posCount = wordCount(positiveWords);

  if (hasSarcasm) {
    emotions.sarcasm = 0.85;
    frustrationScore += 3;
  }
  if (frustCount > 0) {
    emotions.frustration = Math.min(0.4 + frustCount * 0.2, 1.0);
    frustrationScore += frustCount * 2;
  }
  if (confCount > 0) {
    emotions.confusion = Math.min(0.3 + confCount * 0.25, 1.0);
    frustrationScore += 1;
  }
  if (urgCount > 0) {
    emotions.urgency = Math.min(0.4 + urgCount * 0.3, 1.0);
    frustrationScore += urgCount;
  }
  if (posCount > 0 && !hasSarcasm) {
    emotions.satisfaction = Math.min(0.3 + posCount * 0.2, 1.0);
    frustrationScore = Math.max(0, frustrationScore - 2);
  }

  // Short, clipped messages in a long conversation = frustration
  if (text.length < 15 && conversationHistory.length > 4) {
    frustrationScore += 1;
    emotions.resignation = 0.4;
  }
  // ALL CAPS
  if (text === text.toUpperCase() && text.length > 5) {
    frustrationScore += 2;
    emotions.frustration = Math.min((emotions.frustration || 0) + 0.3, 1.0);
  }
  // Repeated punctuation
  if (/[!?]{2,}/.test(text)) {
    frustrationScore += 1;
  }

  if (Object.keys(emotions).length === 0) {
    emotions.neutral = 0.7;
  }

  // Multi-intent detection
  const intents = [];
  const intentPatterns = [
    { pattern: /cancel|unsubscribe|stop|end my/i, label: "cancellation", icon: "⊘" },
    { pattern: /refund|money back|charge|billing|payment|charged/i, label: "billing_inquiry", icon: "₿" },
    { pattern: /reset|change|update my (password|email|account|address|phone)/i, label: "account_update", icon: "⟲" },
    { pattern: /how (do|can|to)|explain|help me|what is|tell me about/i, label: "information_request", icon: "?" },
    { pattern: /bug|error|crash|broken|not working|glitch|issue|problem|fail/i, label: "technical_issue", icon: "⚡" },
    { pattern: /speak|human|agent|manager|supervisor|escalat|someone/i, label: "escalation_request", icon: "↑" },
    { pattern: /pricing|plan|upgrade|downgrade|subscription|cost/i, label: "pricing_inquiry", icon: "◈" },
    { pattern: /ship|deliver|track|order|package|arrival/i, label: "order_tracking", icon: "◱" },
  ];

  intentPatterns.forEach(({ pattern, label, icon }) => {
    if (pattern.test(text)) {
      intents.push({
        label,
        icon,
        confidence: 0.7 + Math.random() * 0.25,
        resolved: false,
      });
    }
  });

  if (intents.length === 0) {
    intents.push({
      label: "general_query",
      icon: "◉",
      confidence: 0.6,
      resolved: false,
    });
  }

  // Accumulate frustration from history
  const historicalFrustration = conversationHistory.reduce(
    (sum, msg) => sum + (msg.analysis?.frustrationScore || 0),
    0
  );
  const cumulativeFrustration = Math.min(
    frustrationScore + Math.floor(historicalFrustration * 0.3),
    10
  );

  return {
    emotions,
    intents,
    frustrationScore: cumulativeFrustration,
    hasSarcasm,
    shouldEscalate: cumulativeFrustration >= FRUSTRATION_THRESHOLD,
    timestamp: Date.now(),
  };
}

function generateResponse(analysis) {
  if (analysis.shouldEscalate) {
    return {
      text: "I can see this experience has been frustrating. I'm connecting you with a specialist who will have the full context of our conversation and can give this the attention it deserves.",
      isEscalation: true,
    };
  }
  if (analysis.hasSarcasm) {
    return {
      text: "I hear you — that's clearly not the experience you expected. Let me look into what's actually going wrong here and get you a real answer.",
      isEscalation: false,
    };
  }
  if (analysis.emotions.frustration > 0.6) {
    return {
      text: "I understand this is frustrating. Let me focus on getting this resolved for you as directly as possible.",
      isEscalation: false,
    };
  }
  if (analysis.emotions.confusion) {
    return {
      text: "Let me break this down more clearly. Which part would you like me to start with?",
      isEscalation: false,
    };
  }
  if (analysis.intents.length > 1) {
    const labels = analysis.intents.map((i) => i.label.replace(/_/g, " "));
    return {
      text: `I've picked up ${analysis.intents.length} things you need help with: ${labels.join(" and ")}. Let me address each one.`,
      isEscalation: false,
    };
  }
  return {
    text: "Got it. Let me look into that for you.",
    isEscalation: false,
  };
}

// ─── COMPONENTS ───

function EmotionBar({ label, value, color }) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2, fontFamily: "'DM Mono', monospace", textTransform: "uppercase", letterSpacing: "0.05em", color: "var(--text-secondary)" }}>
        <span>{label}</span>
        <span>{(value * 100).toFixed(0)}%</span>
      </div>
      <div style={{ height: 6, background: "var(--surface-2)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{
          height: "100%",
          width: `${value * 100}%`,
          background: color,
          borderRadius: 3,
          transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
        }} />
      </div>
    </div>
  );
}

function IntentTag({ intent }) {
  return (
    <div style={{
      display: "inline-flex",
      alignItems: "center",
      gap: 6,
      padding: "4px 10px",
      background: "var(--surface-2)",
      borderRadius: 4,
      fontSize: 11,
      fontFamily: "'DM Mono', monospace",
      color: "var(--text-primary)",
      border: "1px solid var(--border)",
    }}>
      <span>{intent.icon}</span>
      <span>{intent.label.replace(/_/g, " ")}</span>
      <span style={{ color: "var(--accent)", fontWeight: 600 }}>
        {(intent.confidence * 100).toFixed(0)}%
      </span>
    </div>
  );
}

function FrustrationGauge({ score }) {
  const percentage = (score / 10) * 100;
  const getColor = () => {
    if (score <= 3) return "#22c55e";
    if (score <= 5) return "#eab308";
    if (score <= 7) return "#f97316";
    return "#ef4444";
  };
  const getLabel = () => {
    if (score <= 2) return "CALM";
    if (score <= 4) return "MILD";
    if (score <= 6) return "RISING";
    if (score <= 8) return "HIGH";
    return "CRITICAL";
  };

  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ position: "relative", width: 120, height: 120, margin: "0 auto" }}>
        <svg viewBox="0 0 120 120" style={{ transform: "rotate(-90deg)" }}>
          <circle cx="60" cy="60" r="50" fill="none" stroke="var(--surface-2)" strokeWidth="8" />
          <circle cx="60" cy="60" r="50" fill="none" stroke={getColor()} strokeWidth="8"
            strokeDasharray={`${percentage * 3.14} ${314 - percentage * 3.14}`}
            strokeLinecap="round"
            style={{ transition: "stroke-dasharray 0.8s cubic-bezier(0.22, 1, 0.36, 1), stroke 0.4s" }}
          />
        </svg>
        <div style={{
          position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)",
          fontFamily: "'Space Grotesk', sans-serif", fontSize: 28, fontWeight: 700, color: getColor(),
          transition: "color 0.4s",
        }}>
          {score}
        </div>
      </div>
      <div style={{
        marginTop: 8,
        fontSize: 10,
        fontFamily: "'DM Mono', monospace",
        letterSpacing: "0.15em",
        color: getColor(),
        fontWeight: 600,
        transition: "color 0.4s",
      }}>
        {getLabel()}
      </div>
    </div>
  );
}

function MessageBubble({ message, isUser }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: isUser ? "flex-end" : "flex-start",
      marginBottom: 12,
      animation: "fadeSlideUp 0.3s ease-out",
    }}>
      <div style={{
        maxWidth: "75%",
        padding: "10px 14px",
        borderRadius: isUser ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
        background: isUser ? "var(--accent)" : "var(--surface-2)",
        color: isUser ? "#fff" : "var(--text-primary)",
        fontSize: 13.5,
        lineHeight: 1.5,
        fontFamily: "'Instrument Sans', sans-serif",
        ...(message.isEscalation && !isUser ? {
          border: "1px solid #ef4444",
          background: "rgba(239,68,68,0.08)",
        } : {}),
      }}>
        {message.isEscalation && !isUser && (
          <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "#ef4444", marginBottom: 6, letterSpacing: "0.1em", fontWeight: 600 }}>
            ↑ ESCALATION TRIGGERED
          </div>
        )}
        {message.text}
      </div>
    </div>
  );
}

// ─── SAMPLE SCENARIOS ───
const SCENARIOS = [
  {
    name: "Frustrated Customer",
    messages: [
      "I've been waiting 3 weeks for my order and nobody can tell me where it is",
      "I already tried that. It still says processing. This is ridiculous.",
      "Oh wonderful, another suggestion that doesn't work. Thanks for nothing.",
      "I want a refund AND I want to cancel my subscription. This is unacceptable.",
    ],
  },
  {
    name: "Multi-Intent Query",
    messages: [
      "Can you help me reset my password and also explain why I was charged twice last month?",
      "Also while you're at it, what plans do you have that include priority support?",
    ],
  },
  {
    name: "Sarcasm Detection",
    messages: [
      "Oh great, another chatbot. This will definitely solve my problems.",
      "Yeah right, because the last 3 bots were SO helpful.",
      "Clearly your system is working perfectly. That's why I'm here for the 4th time.",
    ],
  },
];

// ─── MAIN APP ───
export default function IntentionRecognitionMVP() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [conversationHistory, setConversationHistory] = useState([]);
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [allAnalyses, setAllAnalyses] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeTab, setActiveTab] = useState("chat");
  const [escalated, setEscalated] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const processMessage = useCallback(
    async (text) => {
      if (!text.trim() || escalated) return;
      setIsProcessing(true);

      const userMsg = { text, isUser: true, timestamp: Date.now() };
      setMessages((prev) => [...prev, userMsg]);

      await new Promise((r) => setTimeout(r, 400 + Math.random() * 400));

      const analysis = analyzeMessage(text, conversationHistory);
      const response = generateResponse(analysis);

      const botMsg = { text: response.text, isUser: false, isEscalation: response.isEscalation, timestamp: Date.now() };
      setMessages((prev) => [...prev, botMsg]);
      setCurrentAnalysis(analysis);
      setAllAnalyses((prev) => [...prev, analysis]);
      setConversationHistory((prev) => [...prev, { text, analysis }]);

      if (analysis.shouldEscalate) setEscalated(true);
      setIsProcessing(false);
    },
    [conversationHistory, escalated]
  );

  const handleSend = () => {
    processMessage(input);
    setInput("");
  };

  const runScenario = async (scenario) => {
    setMessages([]);
    setConversationHistory([]);
    setAllAnalyses([]);
    setCurrentAnalysis(null);
    setEscalated(false);
    for (const msg of scenario.messages) {
      await new Promise((r) => setTimeout(r, 800));
      await processMessage(msg);
    }
  };

  const resetChat = () => {
    setMessages([]);
    setConversationHistory([]);
    setAllAnalyses([]);
    setCurrentAnalysis(null);
    setEscalated(false);
    setInput("");
  };

  const emotionColors = {
    frustration: "#ef4444",
    sarcasm: "#a855f7",
    confusion: "#f59e0b",
    urgency: "#f97316",
    satisfaction: "#22c55e",
    resignation: "#6b7280",
    neutral: "#94a3b8",
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=DM+Mono:wght@400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

        :root {
          --bg: #0a0a0b;
          --surface-1: #111113;
          --surface-2: #1a1a1f;
          --surface-3: #242429;
          --border: #2a2a32;
          --border-light: #333340;
          --text-primary: #e8e8ed;
          --text-secondary: #7a7a88;
          --accent: #6366f1;
          --accent-glow: rgba(99, 102, 241, 0.15);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        @keyframes fadeSlideUp {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>

      <div style={{
        minHeight: "100vh",
        background: "var(--bg)",
        color: "var(--text-primary)",
        fontFamily: "'Instrument Sans', sans-serif",
        display: "flex",
        flexDirection: "column",
      }}>
        {/* Header */}
        <header style={{
          padding: "16px 24px",
          borderBottom: "1px solid var(--border)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "var(--surface-1)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16,
            }}>◎</div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, fontFamily: "'Space Grotesk', sans-serif", letterSpacing: "-0.02em" }}>
                Intention Recognition
              </div>
              <div style={{ fontSize: 10, color: "var(--text-secondary)", fontFamily: "'DM Mono', monospace", letterSpacing: "0.08em" }}>
                MVP v0.1 — MULTI-INTENT · EMOTION-AWARE · ESCALATION
              </div>
            </div>
          </div>
          <button onClick={resetChat} style={{
            padding: "6px 14px", fontSize: 11, fontFamily: "'DM Mono', monospace",
            background: "var(--surface-2)", border: "1px solid var(--border)",
            color: "var(--text-secondary)", borderRadius: 6, cursor: "pointer",
            letterSpacing: "0.05em",
          }}>
            RESET
          </button>
        </header>

        {/* Mobile Tabs */}
        <div style={{
          display: "flex",
          borderBottom: "1px solid var(--border)",
          background: "var(--surface-1)",
        }}>
          {["chat", "analysis", "dashboard"].map((tab) => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              flex: 1, padding: "10px 0", fontSize: 11, fontFamily: "'DM Mono', monospace",
              letterSpacing: "0.1em", textTransform: "uppercase",
              background: "transparent", border: "none", cursor: "pointer",
              color: activeTab === tab ? "var(--accent)" : "var(--text-secondary)",
              borderBottom: activeTab === tab ? "2px solid var(--accent)" : "2px solid transparent",
            }}>
              {tab}
            </button>
          ))}
        </div>

        {/* Main Content */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          {/* ─── CHAT PANEL ─── */}
          <div style={{
            flex: 1,
            display: activeTab === "chat" ? "flex" : "none",
            flexDirection: "column",
            minWidth: 0,
          }}>
            {/* Scenario Buttons */}
            <div style={{
              padding: "12px 16px",
              borderBottom: "1px solid var(--border)",
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              background: "var(--surface-1)",
            }}>
              <span style={{ fontSize: 10, color: "var(--text-secondary)", fontFamily: "'DM Mono', monospace", alignSelf: "center", letterSpacing: "0.08em" }}>
                SCENARIOS:
              </span>
              {SCENARIOS.map((s, i) => (
                <button key={i} onClick={() => runScenario(s)} style={{
                  padding: "5px 10px", fontSize: 11, fontFamily: "'DM Mono', monospace",
                  background: "var(--surface-2)", border: "1px solid var(--border)",
                  color: "var(--text-primary)", borderRadius: 4, cursor: "pointer",
                  transition: "border-color 0.2s",
                }}>
                  {s.name}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div style={{
              flex: 1, overflowY: "auto", padding: 16,
              display: "flex", flexDirection: "column",
            }}>
              {messages.length === 0 && (
                <div style={{
                  flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
                  flexDirection: "column", gap: 12, color: "var(--text-secondary)",
                }}>
                  <div style={{ fontSize: 40, opacity: 0.3 }}>◎</div>
                  <div style={{ fontSize: 13, fontFamily: "'DM Mono', monospace", textAlign: "center", lineHeight: 1.8 }}>
                    Type a message or run a scenario<br />
                    <span style={{ fontSize: 11, opacity: 0.6 }}>Try sarcasm, frustration, or multi-intent queries</span>
                  </div>
                </div>
              )}
              {messages.map((msg, i) => (
                <MessageBubble key={i} message={msg} isUser={msg.isUser} />
              ))}
              {isProcessing && (
                <div style={{ display: "flex", gap: 4, padding: "8px 14px" }}>
                  {[0, 1, 2].map((i) => (
                    <div key={i} style={{
                      width: 6, height: 6, borderRadius: "50%",
                      background: "var(--accent)",
                      animation: `pulse 1s ease-in-out ${i * 0.15}s infinite`,
                    }} />
                  ))}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <div style={{
              padding: 16,
              borderTop: "1px solid var(--border)",
              background: "var(--surface-1)",
            }}>
              {escalated ? (
                <div style={{
                  textAlign: "center", padding: 12,
                  color: "#ef4444", fontSize: 12,
                  fontFamily: "'DM Mono', monospace",
                  border: "1px solid rgba(239,68,68,0.3)",
                  borderRadius: 8,
                  background: "rgba(239,68,68,0.05)",
                }}>
                  ↑ CONVERSATION ESCALATED TO HUMAN AGENT — Chat paused
                </div>
              ) : (
                <div style={{ display: "flex", gap: 8 }}>
                  <input
                    type="text"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSend()}
                    placeholder="Type a message... (try sarcasm or multiple requests)"
                    style={{
                      flex: 1, padding: "10px 14px", fontSize: 13,
                      fontFamily: "'Instrument Sans', sans-serif",
                      background: "var(--surface-2)", border: "1px solid var(--border)",
                      borderRadius: 8, color: "var(--text-primary)",
                      outline: "none",
                    }}
                  />
                  <button onClick={handleSend} disabled={isProcessing || !input.trim()} style={{
                    padding: "10px 20px", fontSize: 12,
                    fontFamily: "'DM Mono', monospace",
                    background: "var(--accent)", border: "none",
                    borderRadius: 8, color: "#fff", cursor: "pointer",
                    opacity: isProcessing || !input.trim() ? 0.4 : 1,
                    letterSpacing: "0.05em",
                  }}>
                    SEND
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* ─── ANALYSIS PANEL ─── */}
          <div style={{
            width: activeTab === "analysis" ? "100%" : 0,
            display: activeTab === "analysis" ? "block" : "none",
            overflowY: "auto",
            padding: 20,
            background: "var(--surface-1)",
            borderLeft: "1px solid var(--border)",
          }}>
            {currentAnalysis ? (
              <div style={{ animation: "fadeSlideUp 0.3s ease-out" }}>
                {/* Frustration Gauge */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10, marginBottom: 16,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 16 }}>
                    FRUSTRATION LEVEL
                  </div>
                  <FrustrationGauge score={currentAnalysis.frustrationScore} />
                  {currentAnalysis.frustrationScore >= FRUSTRATION_THRESHOLD && (
                    <div style={{
                      marginTop: 12, padding: "8px 12px",
                      background: "rgba(239,68,68,0.1)",
                      border: "1px solid rgba(239,68,68,0.3)",
                      borderRadius: 6, fontSize: 11,
                      fontFamily: "'DM Mono', monospace",
                      color: "#ef4444", textAlign: "center",
                    }}>
                      ⚠ THRESHOLD EXCEEDED — ESCALATION TRIGGERED
                    </div>
                  )}
                </div>

                {/* Detected Emotions */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10, marginBottom: 16,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 14 }}>
                    DETECTED EMOTIONS
                  </div>
                  {Object.entries(currentAnalysis.emotions).map(([emotion, value]) => (
                    <EmotionBar
                      key={emotion}
                      label={emotion}
                      value={value}
                      color={emotionColors[emotion] || "#6366f1"}
                    />
                  ))}
                </div>

                {/* Detected Intents */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10, marginBottom: 16,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 14 }}>
                    DETECTED INTENTS ({currentAnalysis.intents.length})
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {currentAnalysis.intents.map((intent, i) => (
                      <IntentTag key={i} intent={intent} />
                    ))}
                  </div>
                </div>

                {/* Flags */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 14 }}>
                    FLAGS
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    {[
                      { label: "SARCASM DETECTED", active: currentAnalysis.hasSarcasm, color: "#a855f7" },
                      { label: "MULTI-INTENT", active: currentAnalysis.intents.length > 1, color: "#6366f1" },
                      { label: "ESCALATION", active: currentAnalysis.shouldEscalate, color: "#ef4444" },
                    ].map((flag) => (
                      <div key={flag.label} style={{
                        display: "flex", alignItems: "center", gap: 8,
                        fontSize: 11, fontFamily: "'DM Mono', monospace",
                        color: flag.active ? flag.color : "var(--text-secondary)",
                        opacity: flag.active ? 1 : 0.4,
                      }}>
                        <div style={{
                          width: 8, height: 8, borderRadius: "50%",
                          background: flag.active ? flag.color : "var(--surface-3)",
                          boxShadow: flag.active ? `0 0 8px ${flag.color}` : "none",
                        }} />
                        {flag.label}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div style={{
                height: "100%", display: "flex", alignItems: "center", justifyContent: "center",
                color: "var(--text-secondary)", fontSize: 12,
                fontFamily: "'DM Mono', monospace",
              }}>
                Send a message to see analysis
              </div>
            )}
          </div>

          {/* ─── DASHBOARD PANEL (Agent View) ─── */}
          <div style={{
            width: activeTab === "dashboard" ? "100%" : 0,
            display: activeTab === "dashboard" ? "block" : "none",
            overflowY: "auto",
            padding: 20,
            background: "var(--surface-1)",
          }}>
            <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 20 }}>
              AGENT SENTIMENT DASHBOARD
            </div>

            {allAnalyses.length > 0 ? (
              <>
                {/* Sentiment Timeline */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10, marginBottom: 16,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 16 }}>
                    FRUSTRATION TIMELINE
                  </div>
                  <div style={{ display: "flex", alignItems: "flex-end", gap: 4, height: 80 }}>
                    {allAnalyses.map((a, i) => {
                      const h = Math.max(8, (a.frustrationScore / 10) * 80);
                      const color = a.frustrationScore <= 3 ? "#22c55e" : a.frustrationScore <= 6 ? "#eab308" : "#ef4444";
                      return (
                        <div key={i} style={{
                          flex: 1, height: h,
                          background: color,
                          borderRadius: "3px 3px 0 0",
                          transition: "height 0.4s",
                          minWidth: 16,
                          maxWidth: 40,
                          opacity: 0.85,
                        }} title={`Message ${i + 1}: Frustration ${a.frustrationScore}/10`} />
                      );
                    })}
                  </div>
                  <div style={{
                    marginTop: 8, fontSize: 10,
                    fontFamily: "'DM Mono', monospace",
                    color: "var(--text-secondary)",
                    display: "flex", justifyContent: "space-between",
                  }}>
                    <span>msg 1</span>
                    <span>msg {allAnalyses.length}</span>
                  </div>
                </div>

                {/* Escalation Summary */}
                {escalated && (
                  <div style={{
                    padding: 20, background: "rgba(239,68,68,0.05)",
                    borderRadius: 10, marginBottom: 16,
                    border: "1px solid rgba(239,68,68,0.3)",
                  }}>
                    <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "#ef4444", letterSpacing: "0.12em", marginBottom: 12 }}>
                      ⚠ ESCALATION BRIEF FOR AGENT
                    </div>
                    <div style={{ fontSize: 13, lineHeight: 1.7, color: "var(--text-primary)" }}>
                      <strong>Summary:</strong> Customer frustration exceeded threshold after {allAnalyses.length} messages.
                      Peak frustration: {Math.max(...allAnalyses.map((a) => a.frustrationScore))}/10.
                      {allAnalyses.some((a) => a.hasSarcasm) && " Sarcasm was detected in the conversation."}
                      {" "}Total intents detected: {[...new Set(allAnalyses.flatMap((a) => a.intents.map((i) => i.label)))].join(", ").replace(/_/g, " ")}.
                    </div>
                  </div>
                )}

                {/* All Intents Collected */}
                <div style={{
                  padding: 20, background: "var(--surface-2)",
                  borderRadius: 10,
                  border: "1px solid var(--border)",
                }}>
                  <div style={{ fontSize: 10, fontFamily: "'DM Mono', monospace", color: "var(--text-secondary)", letterSpacing: "0.12em", marginBottom: 14 }}>
                    ALL DETECTED INTENTS
                  </div>
                  <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                    {[...new Map(
                      allAnalyses
                        .flatMap((a) => a.intents)
                        .map((i) => [i.label, i])
                    ).values()].map((intent, i) => (
                      <IntentTag key={i} intent={intent} />
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div style={{
                height: 300, display: "flex", alignItems: "center", justifyContent: "center",
                color: "var(--text-secondary)", fontSize: 12,
                fontFamily: "'DM Mono', monospace",
              }}>
                No conversation data yet
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
