#!/usr/bin/env node

/**
 * Local development server for API endpoints
 * Simulates Vercel's serverless functions locally
 *
 * Usage: node api-server.cjs
 */

const express = require('express');
const questionsHandler = require('./api/questions.cjs');

const app = express();
const PORT = 3001;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

function wrapHandler(handler) {
  return async (req, res) => {
    try {
      await handler(req, res);
    } catch (error) {
      console.error('API Error:', error);
      if (!res.headersSent) {
        res.status(500).json({ error: 'Internal server error' });
      }
    }
  };
}

// API Routes
app.post('/api/questions', wrapHandler(questionsHandler));
app.get('/api/questions', wrapHandler(questionsHandler));
app.options('/api/questions', wrapHandler(questionsHandler));

// chat.js is ESM — load via dynamic import inside handler
app.post('/api/chat', async (req, res) => {
  const { default: chatHandler } = await import('./api/chat.js');
  await wrapHandler(chatHandler)(req, res);
});
app.options('/api/chat', async (req, res) => {
  const { default: chatHandler } = await import('./api/chat.js');
  await wrapHandler(chatHandler)(req, res);
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.listen(PORT, () => {
  console.log(`✅ API server running on http://localhost:${PORT}`);
  console.log(`   Endpoints:`);
  console.log(`   - POST /api/questions (save question)`);
  console.log(`   - GET  /api/questions (retrieve questions)`);
  console.log(`   - POST /api/chat      (LLM assistant — requires ANTHROPIC_API_KEY)`);
  console.log(`   - GET  /api/health (health check)`);
  console.log(``);
  console.log(`💡 Run "npm run dev" in another terminal to start the Vite dev server`);
});
