import React, { useEffect, useState } from 'react';

type AgentMessage = { id: number; role: 'user' | 'assistant' | 'system'; text: string };

const AgentPanel: React.FC<{ open?: boolean; onClose?: () => void }> = ({ open = false, onClose }) => {
  const [contextText, setContextText] = useState<string>('');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // fetch the local docs file we created
    fetch('/docs/assistant_context.md')
      .then((r) => r.text())
      .then((t) => setContextText(t))
      .catch(() => setContextText(''));
  }, []);

  const systemPrompt = `You are an assistant for the ML Visualizer app. Use the provided context to answer questions. Be concise and reference files when relevant.`;

  const send = async () => {
    if (!input.trim()) return;
    const userText = input.trim();
    setInput('');
    const uid = Date.now();
    setMessages((m) => [...m, { id: uid, role: 'user', text: userText }]);

    // very small client-side "retrieval": pick lines from the context with query keywords
    setLoading(true);
    await new Promise((r) => setTimeout(r, 300));

    const kws = userText.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
    const lines = contextText.split('\n').filter((l) => {
      const ll = l.toLowerCase();
      return kws.some((k) => ll.includes(k));
    }).slice(0, 6);

    const reply = lines.length
      ? `Based on docs:\n${lines.join('\n')}`
      : `I couldn't find an exact match in the local docs. Quick tip: Ask about where UI controls are (e.g., 'Where is Model Controls?') or ask for the file name like 'Where is MLP code?'.`;

    setMessages((m) => [...m, { id: uid + 1, role: 'assistant', text: reply }]);
    setLoading(false);
  };

  if (!open) return null;
  return (
    <div style={{ position: 'fixed', right: 12, top: 90, width: 360, bottom: 12, background: 'white', boxShadow: '0 12px 40px rgba(0,0,0,0.2)', borderRadius: 12, zIndex: 1400, display: 'flex', flexDirection: 'column' }}>
      <div style={{ padding: 12, borderBottom: '1px solid rgba(0,0,0,0.06)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <strong>Assistant</strong>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={() => { navigator.clipboard?.writeText(contextText).catch(() => {}); }}>Copy Context</button>
          <button onClick={() => onClose?.()}>Close</button>
        </div>
      </div>

      <div style={{ padding: 12, overflow: 'auto', flex: 1 }}>
        {messages.length === 0 && (
          <div style={{ color: '#555' }}>
            Ask questions about the app or code. The assistant uses a local doc and simple retrieval.
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} style={{ marginBottom: 10 }}>
            <div style={{ fontSize: 12, color: m.role === 'user' ? '#333' : '#0b6', fontWeight: 600 }}>{m.role}</div>
            <div style={{ marginTop: 6, whiteSpace: 'pre-wrap' }}>{m.text}</div>
          </div>
        ))}
        {loading && <div style={{ color: '#888' }}>Thinking…</div>}
      </div>

      <div style={{ padding: 12, borderTop: '1px solid rgba(0,0,0,0.06)', display: 'flex', gap: 8 }}>
        <input value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && send()} style={{ flex: 1, padding: '8px 10px', borderRadius: 8, border: '1px solid #ddd' }} placeholder="Ask about the app or code..." />
        <button onClick={send} disabled={loading} style={{ padding: '8px 12px' }}>{loading ? '…' : 'Ask'}</button>
      </div>
    </div>
  );
};

export default AgentPanel;
