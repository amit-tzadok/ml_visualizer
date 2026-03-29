'use strict';

/**
 * /api/chat — LLM-powered ML assistant endpoint.
 *
 * POST body:
 *   { message: string, ragContext?: string, appState?: { classifier, compare, speedScale, accuracy, epoch } }
 *
 * Returns:
 *   200  { response: string, model: string }
 *   400  { error: 'Invalid message' }
 *   503  { error: 'LLM not configured', code: 'NO_API_KEY' }  ← frontend falls back gracefully
 *   502  { error: 'LLM request failed' }
 *
 * The ANTHROPIC_API_KEY env var must be set (Vercel secret or .env locally).
 * The key is NEVER sent to the client.
 */

const SYSTEM_PROMPT_BASE = `\
You are an ML education assistant embedded inside ML Visualizer — an interactive \
browser-based tool for learning classic machine learning algorithms. Users watch \
algorithms learn in real time by clicking to add training points.

Algorithms in the app:
• Linear Perceptron     — linear boundary, online learning, hinge loss, averaged variant
• Polynomial Perceptron — degree-2 expansion [x, y, x², y², xy], curved boundary
• Neural Network (MLP)  — configurable layers, sigmoid/tanh/ReLU/LeakyReLU, SGD/Momentum/Adam
• K-Nearest Neighbors   — distance-based, interactive k, no training phase
• Logistic Regression   — sigmoid output, mini-batch GD, L2 regularisation, live loss curve
• Decision Tree         — CART/Gini impurity, max-depth control, animated split growth

Dataset presets (Logistic Regression & Decision Tree):
  Blobs (linearly separable) | Moons | Circles | XOR | Spirals

UI controls:
• Bottom bar  — Algorithm selector, Compare toggle, Speed slider, training status
• Canvas      — left-click = class A (red), right-click = class B (blue)
• 🤖 Agent    — this assistant panel
• ⌨️ button  — keyboard shortcuts

Guidelines:
• Be concise (2–4 sentences). Expand only when asked.
• Relate maths to what the user can see on the canvas.
• Never invent features the app doesn't have.
• Use plain notation (e.g. w·x + b) without LaTeX delimiters.`;

module.exports = async (req, res) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // ── API key guard ────────────────────────────────────────────────────────
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    // Signal to the client to fall back to the rule-based system
    return res.status(503).json({ error: 'LLM not configured', code: 'NO_API_KEY' });
  }

  // ── Input validation ─────────────────────────────────────────────────────
  const body = req.body || {};
  const { message, ragContext, appState } = body;

  if (!message || typeof message !== 'string') {
    return res.status(400).json({ error: 'Invalid message: must be a non-empty string' });
  }
  if (message.length > 2000) {
    return res.status(400).json({ error: 'Message too long (max 2000 characters)' });
  }

  // ── Build system prompt ──────────────────────────────────────────────────
  const parts = [SYSTEM_PROMPT_BASE];

  if (appState && typeof appState === 'object') {
    const s = appState;
    const lines = [
      `\nCurrent app state:`,
      `  classifier = ${s.classifier ?? 'unknown'}`,
      `  compare mode = ${s.compare ? 'on' : 'off'}`,
      `  speed = ${s.speedScale ?? 1}×`,
    ];
    if (s.accuracy != null) lines.push(`  training accuracy = ${(Number(s.accuracy) * 100).toFixed(1)}%`);
    if (s.epoch != null)    lines.push(`  epoch = ${s.epoch}`);
    parts.push(lines.join('\n'));
  }

  if (ragContext && typeof ragContext === 'string' && ragContext.trim()) {
    parts.push(`\nRelevant documentation retrieved for this query:\n${ragContext.trim()}`);
  }

  const systemPrompt = parts.join('\n');

  // ── Call Claude Haiku ────────────────────────────────────────────────────
  try {
    // Require inside the function so the module still loads when the SDK is
    // not installed (server returns 503 instead of crashing).
    const { default: Anthropic } = require('@anthropic-ai/sdk');
    const client = new Anthropic({ apiKey });

    const msg = await client.messages.create({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 400,
      system: systemPrompt,
      messages: [{ role: 'user', content: message }],
    });

    const text =
      Array.isArray(msg.content) && msg.content[0]?.type === 'text'
        ? msg.content[0].text
        : '';

    return res.status(200).json({ response: text, model: msg.model });
  } catch (err) {
    const detail = err instanceof Error ? err.message : String(err);
    console.error('[chat] LLM request failed:', detail);
    return res.status(502).json({ error: 'LLM request failed', detail });
  }
};
