import React, { useEffect, useRef, useState } from 'react';
import styles from './AgentPanel.module.css';
import { chunkText, buildIndex, queryIndex, DocChunk } from '../utils/rag';

interface AgentPanelProps {
  classifier: string;
  compare: boolean;
  speedScale: number;
  theme: ThemeShape;
  isOpen: boolean;
  onClose: () => void;
}

type ThemeShape = {
  controlBg?: string;
  text?: string;
  accent?: string;
  shadow?: string;
};

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
  sources?: (DocChunk & { score?: number })[];
  isTyping?: boolean;
}

// ── Rule-based fallback ───────────────────────────────────────────────────────

function getClassifierInfo(type: string) {
  const info: Record<string, {
    description: string;
    strengths: string[];
    weaknesses: string[];
    useCases: string[];
  }> = {
    linear: {
      description: 'A linear perceptron that finds a straight line to separate data into two classes.',
      strengths: ['Simple and fast', 'Works well with linearly separable data', 'Easy to interpret'],
      weaknesses: ['Cannot model curved boundaries', 'Sensitive to outliers', 'Underfits nonlinear patterns'],
      useCases: ['Linearly separable data', 'Baseline comparison', 'Teaching the perceptron update rule'],
    },
    poly: {
      description: 'A polynomial perceptron with degree-2 feature expansion: [x, y, x², y², xy].',
      strengths: ['Handles curved decision boundaries', 'Simple extension of linear', 'No extra tuning for degree'],
      weaknesses: ['Fixed degree (2)', 'Feature space grows fast at higher degrees', 'Can overfit small datasets'],
      useCases: ['Circular or elliptic patterns', 'Non-linear 2D classification', 'Feature engineering demo'],
    },
    mlp: {
      description: 'A Multi-Layer Perceptron (neural network) with configurable hidden layers, activations, and optimisers.',
      strengths: ['Very flexible — learns any decision boundary', 'Multiple activations & optimisers', 'Handles spirals, moons, XOR'],
      weaknesses: ['Requires tuning (layers, LR, epochs)', 'Slower to train', 'Less interpretable'],
      useCases: ['Complex nonlinear patterns', 'Deep learning demonstration', 'Showing Adam vs SGD'],
    },
    knn: {
      description: 'K-Nearest Neighbors: classifies by majority vote of the k closest training points.',
      strengths: ['No training phase', 'Intuitive — directly uses the data', 'Non-parametric'],
      weaknesses: ['Slow prediction on large datasets (O(n))', 'Sensitive to noisy neighbours', 'k must be tuned'],
      useCases: ['Instance-based learning', 'Small datasets', 'Understanding distance-based classification'],
    },
    logreg: {
      description: 'Logistic Regression: P(y=1|x) = σ(w·x + b). Trained with mini-batch gradient descent and L2 regularisation.',
      strengths: ['Smooth differentiable boundary', 'Outputs calibrated probabilities', 'Regularisation prevents overfitting'],
      weaknesses: ['Still linear decision boundary', 'Cannot separate XOR, moons, spirals'],
      useCases: ['Binary classification with probability output', 'Seeing gradient descent live', 'Comparing to perceptron'],
    },
    dtree: {
      description: 'Decision Tree (CART): grows axis-aligned splits by maximising Gini information gain at each node.',
      strengths: ['Interpretable', 'Handles XOR and moons with enough depth', 'No feature scaling needed'],
      weaknesses: ['Overfits easily at high depth', 'Axis-aligned splits only (no diagonal)', 'Unstable — small data changes shift splits'],
      useCases: ['Nonlinear classification', 'Visualising bias-variance tradeoff', 'Showing overfitting vs underfitting'],
    },
  };
  return info[type] ?? {
    description: 'An ML classifier in the visualizer.',
    strengths: [], weaknesses: [], useCases: [],
  };
}

function formatInfo(type: string): string {
  const { description, strengths, weaknesses, useCases } = getClassifierInfo(type);
  const title = type === 'logreg' ? 'LOGISTIC REGRESSION'
    : type === 'dtree' ? 'DECISION TREE'
    : type.toUpperCase();
  return [
    `**${title}**`,
    description,
    '',
    'Strengths:',
    ...strengths.map(s => `• ${s}`),
    '',
    'Limitations:',
    ...weaknesses.map(w => `• ${w}`),
    '',
    'Best for:',
    ...useCases.map(u => `• ${u}`),
  ].join('\n');
}

function generateFallbackResponse(userInput: string, classifier: string, compare: boolean, speedScale: number): string {
  const q = userInput.toLowerCase().trim();

  if (q.includes('equation') || q.includes('formula') || q.includes('boundary')) {
    if (compare) return 'In Compare mode there are two classifiers running — turn it off to get the current equation.';
    if (classifier === 'linear' || classifier === 'poly') {
      const s = (window as unknown as { mlvStatus?: { classifier?: string; equation?: string | null } }).mlvStatus;
      if (s?.classifier === classifier && s.equation) return `Current decision boundary:\n${s.equation}`;
      return 'Train the model first, then ask again — I\'ll show the live equation.';
    }
    if (classifier === 'mlp') return 'The MLP has no closed-form equation — it\'s a composition of learned nonlinear layers.';
    if (classifier === 'knn') return 'KNN has no equation; it classifies by the labels of the k nearest neighbours.';
    if (classifier === 'logreg') return 'Logistic regression boundary: w₀x + w₁y + b = 0, equivalently y = −(w₀/w₁)x − b/w₁ (the dashed line on the canvas).';
    if (classifier === 'dtree') return 'Decision trees use a series of axis-aligned rules, not a single equation. Watch the partition lines animate — each one is one split.';
  }

  const classifierTypes = ['linear', 'poly', 'mlp', 'knn', 'logreg', 'dtree'];
  if (q.includes('what') && (q.includes('classifier') || q.includes('algorithm') || q.includes('model'))) {
    const mentioned = classifierTypes.find(t =>
      q.includes(t) ||
      (t === 'linear' && q.includes('perceptron')) ||
      (t === 'poly' && q.includes('polynomial')) ||
      (t === 'mlp' && (q.includes('neural') || q.includes('network'))) ||
      (t === 'knn' && q.includes('nearest')) ||
      (t === 'logreg' && q.includes('logistic')) ||
      (t === 'dtree' && (q.includes('decision') || q.includes('tree')))
    );
    return formatInfo(mentioned ?? classifier);
  }

  if (q.includes('advantage') || q.includes('strength') || q.includes('pro') ||
      q.includes('disadvantage') || q.includes('weakness') || q.includes('con') ||
      q.includes('limitation') || q.includes('tradeoff') || q.includes('trade-off')) {
    const mentioned = classifierTypes.find(t =>
      q.includes(t) ||
      (t === 'linear' && q.includes('perceptron')) ||
      (t === 'poly' && q.includes('polynomial')) ||
      (t === 'mlp' && (q.includes('neural') || q.includes('network'))) ||
      (t === 'knn' && q.includes('nearest')) ||
      (t === 'logreg' && q.includes('logistic')) ||
      (t === 'dtree' && (q.includes('decision') || q.includes('tree')))
    );
    return formatInfo(mentioned ?? classifier);
  }

  if (q.includes('speed') || q.includes('slow') || q.includes('fast')) {
    return `Current speed: ${speedScale}x. Adjust with the slider in the bottom bar. Try 0.5× to watch individual updates, or 3× to race to convergence.`;
  }

  if (q.includes('compare')) {
    return `Compare mode shows two classifiers side-by-side on the same dataset. ${compare ? 'It\'s on now.' : 'Toggle it in the bottom bar.'} Supports Linear, Poly, and MLP.`;
  }

  if (q.includes('data') || q.includes('point') || q.includes('click') || q.includes('add')) {
    return 'Left-click the canvas for class A (red), right-click for class B (blue). On touch devices, use the Touch tap selector in the controls.';
  }

  if (q.includes('dataset') || q.includes('xor') || q.includes('moon') || q.includes('circle') || q.includes('spiral') || q.includes('blob')) {
    return 'Logistic Regression and Decision Tree have preset dataset buttons: Blobs (easy), Moons, Circles, XOR, and Spirals. Spirals are the hardest — only a deep tree or MLP can learn them.';
  }

  if (q.includes('status') || q.includes('current') || q.includes('state')) {
    return `Current: classifier=${classifier.toUpperCase()}, compare=${compare ? 'on' : 'off'}, speed=${speedScale}×.`;
  }

  if (q.includes('hello') || q.includes('hi') || q.includes('hey')) {
    return 'Hello! Ask me anything about the algorithms, controls, or the maths behind what you see.';
  }

  if (q.includes('thank')) {
    return 'Happy learning!';
  }

  return 'I can explain any classifier (Linear, Poly, MLP, KNN, Logistic Regression, Decision Tree), describe controls, or walk through the maths. What would you like to know?';
}

// ── LLM call ─────────────────────────────────────────────────────────────────

interface AppState {
  classifier: string;
  compare: boolean;
  speedScale: number;
  accuracy?: number | null;
  epoch?: number | null;
}

async function callLLM(
  message: string,
  ragContext: string,
  appState: AppState
): Promise<string | null> {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, ragContext, appState }),
    signal: AbortSignal.timeout(15000),
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({})) as Record<string, unknown>;
    // 503 with NO_API_KEY means the key isn't set — fall back silently
    if (res.status === 503 && body.code === 'NO_API_KEY') return null;
    throw new Error(`HTTP ${res.status}`);
  }

  const data = await res.json() as { response?: string };
  return typeof data.response === 'string' ? data.response : null;
}

// ── Component ─────────────────────────────────────────────────────────────────

const AgentPanel: React.FC<AgentPanelProps> = ({
  classifier,
  compare,
  speedScale,
  theme,
  isOpen,
  onClose,
}) => {
  const isDark = theme?.text === '#f7fafc';
  const panelBg  = theme?.controlBg  ?? 'white';
  const headerBg = isDark ? 'rgba(45,55,72,0.98)' : '#f8fafc';
  const borderColor = isDark ? 'rgba(255,255,255,0.10)' : '#e2e8f0';
  const textColor   = theme?.text ?? '#1a202c';
  const mutedText   = isDark ? '#a0aec0' : '#718096';
  const inputBg     = isDark ? 'rgba(30,40,55,0.9)' : 'white';
  const botBubbleBg = isDark ? 'rgba(45,55,72,0.8)' : '#f1f5f9';
  const botBubbleColor = isDark ? '#e2e8f0' : '#334155';
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm your ML Assistant. Ask me about any algorithm, the maths behind it, or how to use the visualizer.",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const indexRef = useRef<ReturnType<typeof buildIndex> | null>(null);
  const [indexReady, setIndexReady] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`);

  const isDev = () => {
    try {
      const p = (globalThis as unknown as { process?: { env?: { NODE_ENV?: string } } }).process;
      return !!(p?.env?.NODE_ENV !== 'production');
    } catch { return false; }
  };

  // Build TF-IDF index from assistant context doc
  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch('/docs/assistant_context.md');
        if (!res.ok) throw new Error('no docs');
        const md = await res.text();
        indexRef.current = buildIndex(chunkText(md, 'assistant_context'));
        setIndexReady(true);
      } catch (err) {
        indexRef.current = null;
        if (isDev()) console.debug('agentpanel: index load failed', err);
      }
    };
    load();
  }, []);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const saveQuestionToDB = async (question: string) => {
    try {
      await fetch('/api/questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, classifier, compareMode: compare, speedScale, sessionId }),
      });
    } catch (err) {
      if (isDev()) console.debug('agentpanel: DB save failed', err);
    }
  };

  const handleSendMessage = async () => {
    const text = inputValue.trim();
    if (!text || isLoading) return;
    setInputValue('');

    const userMsg: Message = { id: Date.now(), text, isUser: true, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);

    // Save to DB (non-blocking)
    saveQuestionToDB(text).catch(() => {});

    setIsLoading(true);

    // ── Build RAG context ──────────────────────────────────────────────────
    let ragContext = '';
    let sources: (DocChunk & { score?: number })[] | undefined;
    try {
      if (indexRef.current) {
        const hits = queryIndex(indexRef.current, text, 3);
        sources = hits as (DocChunk & { score?: number })[];
        ragContext = hits.map(h => h.text).join('\n\n---\n\n');
      }
    } catch (err) {
      if (isDev()) console.debug('agentpanel: RAG query failed', err);
    }

    // ── Get live training state ────────────────────────────────────────────
    const mlvStatus = (window as unknown as { mlvStatus?: Record<string, unknown> }).mlvStatus;
    const appState: AppState = {
      classifier,
      compare,
      speedScale,
      accuracy: typeof mlvStatus?.accuracy === 'number' ? mlvStatus.accuracy : null,
      epoch: typeof mlvStatus?.epoch === 'number' ? mlvStatus.epoch : null,
    };

    // ── Try LLM, fall back to rule-based ───────────────────────────────────
    let responseText: string;
    try {
      const llmResponse = await callLLM(text, ragContext, appState);
      responseText = llmResponse ?? generateFallbackResponse(text, classifier, compare, speedScale);
    } catch (err) {
      if (isDev()) console.debug('agentpanel: LLM call error', err);
      responseText = generateFallbackResponse(text, classifier, compare, speedScale);
    }

    setIsLoading(false);
    const botMsg: Message = {
      id: Date.now() + 1,
      text: responseText,
      isUser: false,
      timestamp: new Date(),
      sources,
    };
    setMessages(prev => [...prev, botMsg]);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className={`${styles.agentSidebar} ${styles.open}`}
      style={{ background: panelBg, borderLeft: `1px solid ${borderColor}` }}
    >
      <div
        className={styles.agentHeader}
        style={{ background: headerBg, borderBottom: `1px solid ${borderColor}` }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <h3 style={{ margin: 0, color: textColor }}>ML Assistant</h3>
          {indexReady && (
            <span style={{
              fontSize: 10,
              background: isDark ? 'rgba(39,103,73,0.4)' : '#c6f6d5',
              color: isDark ? '#68d391' : '#276749',
              padding: '2px 6px',
              borderRadius: 4,
              fontWeight: 600,
            }}>
              RAG ready
            </span>
          )}
        </div>
        <button
          onClick={onClose}
          className={styles.closeButton}
          title="Close Assistant"
          style={{ color: mutedText }}
        >✕</button>
      </div>

      <div className={styles.agentContent}>
        <div className={styles.chatContainer}>
          <div className={styles.messages}>
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`${styles.message} ${msg.isUser ? styles.userMessage : styles.botMessage}`}
                style={!msg.isUser ? { background: botBubbleBg, color: botBubbleColor, borderColor } : undefined}
              >
                <div className={styles.messageText}>
                  {msg.text.split('\n').map((line, i) => (
                    <div key={i}>{line || <br />}</div>
                  ))}
                </div>
                <div className={styles.messageTime}>
                  {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            ))}

            {/* Animated typing indicator */}
            {isLoading && (
              <div
                className={`${styles.message} ${styles.botMessage}`}
                style={{ background: botBubbleBg, color: botBubbleColor, borderColor }}
              >
                <div className={styles.typingIndicator}>
                  <span className={styles.typingDot} />
                  <span className={styles.typingDot} />
                  <span className={styles.typingDot} />
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          <div
            className={styles.inputContainer}
            style={{ background: headerBg, borderTop: `1px solid ${borderColor}` }}
          >
            <div style={{ display: 'flex', gap: 10 }}>
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about any algorithm…"
                className={styles.chatInput}
                disabled={isLoading}
                style={{
                  background: inputBg,
                  color: textColor,
                  borderColor: isDark ? 'rgba(255,255,255,0.15)' : '#e2e8f0',
                }}
              />
              <button
                onClick={handleSendMessage}
                className={styles.sendButton}
                disabled={isLoading || !inputValue.trim()}
              >
                {isLoading ? '…' : 'Send'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentPanel;
