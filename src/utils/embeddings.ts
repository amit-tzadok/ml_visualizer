export async function getOpenAIEmbedding(texts: string[], apiKey?: string): Promise<number[][]> {
  // Prefer calling a local/proxied serverless endpoint (e.g. Vercel/Netlify) at /api/embeddings.
  // This keeps API keys off the client. If the proxy isn't available, fall back to a direct
  // call using `VITE_OPENAI_API_KEY` (for local dev only).

  try {
    // If running in a browser and the proxy endpoint resolves, use it.
    if (typeof window !== 'undefined') {
      // attempt a quick OPTIONS probe to see if /api/embeddings exists (small, fast)
      const probe = await fetch('/api/embeddings', { method: 'OPTIONS' }).catch(() => null);
      if (probe && probe.status < 400) {
        const res = await fetch('/api/embeddings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ inputs: texts, model: 'text-embedding-3-small' }),
        });
        if (!res.ok) {
          const txt = await res.text();
          throw new Error(`Proxy embedding request failed: ${txt}`);
        }
        const j = await res.json();
        const embeddings = j.data.map((d: any) => d.embedding as number[]);
        return embeddings;
      }
    }

    // Fallback: direct call to OpenAI using VITE_OPENAI_API_KEY (local dev only)
    const key = apiKey || (import.meta.env.VITE_OPENAI_API_KEY as string | undefined);
    if (!key) throw new Error('OpenAI API key not provided (no proxy available)');

    const url = 'https://api.openai.com/v1/embeddings';
    const body = JSON.stringify({ model: 'text-embedding-3-small', input: texts });
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
      body,
    });
    if (!res.ok) {
      const err = await res.text();
      throw new Error(`Embedding request failed: ${err}`);
    }
    const j = await res.json();
    const embeddings = j.data.map((d: any) => d.embedding as number[]);
    return embeddings;
  } catch (err) {
    // bubble up errors to the caller for handling; keep message informative
    throw err;
  }
}
