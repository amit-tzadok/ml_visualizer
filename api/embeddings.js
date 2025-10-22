// Vercel-style serverless function (Node.js) to proxy embedding requests to OpenAI
// Expects process.env.OPENAI_API_KEY to be set in the deployment environment.
const fetch = require('node-fetch');

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const { inputs, model } = req.body || {};
  if (!inputs || !Array.isArray(inputs)) {
    res.status(400).json({ error: 'Request must include `inputs` array' });
    return;
  }

  const key = process.env.OPENAI_API_KEY;
  if (!key) {
    res.status(500).json({ error: 'Server misconfigured: missing OPENAI_API_KEY' });
    return;
  }

  try {
    const body = JSON.stringify({ model: model || 'text-embedding-3-small', input: inputs });
    const r = await fetch('https://api.openai.com/v1/embeddings', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${key}`,
      },
      body,
    });
    const j = await r.text();
    // forward status and body
    res.status(r.status).send(j);
  } catch (err) {
    console.error('embedding proxy error', err);
    res.status(502).json({ error: 'Failed to fetch embeddings' });
  }
};
