// Lightweight client-side RAG prototype using TF-IDF cosine similarity.
// Not a replacement for proper embeddings, but useful for demos and provenance.

export type DocChunk = {
  id: string;
  source: string;
  text: string;
};

type Index = {
  docs: DocChunk[];
  idf: Record<string, number>;
  docTfs: Record<string, Record<string, number>>; // docId -> term -> tf
  docNorms: Record<string, number>;
};

const TOKEN_RE = /[a-z0-9]+/g;
const STOPWORDS = new Set(["the","and","is","in","to","of","a","for","with","on","that","this","it","as","are","be","or"]);

function tokenize(text: string) {
  const toks = (text || "").toLowerCase().match(TOKEN_RE) || [];
  return toks.filter((t) => !STOPWORDS.has(t));
}

export function chunkText(text: string, source = "doc", chunkSize = 800, overlap = 200): DocChunk[] {
  const out: DocChunk[] = [];
  let i = 0;
  while (i < text.length) {
    const chunk = text.slice(i, i + chunkSize);
    const id = `${source}#${i}`;
    out.push({ id, source, text: chunk.trim() });
    i += chunkSize - overlap;
  }
  return out;
}

export function buildIndex(docs: DocChunk[]): Index {
  const DF: Record<string, number> = {};
  const docTfs: Record<string, Record<string, number>> = {};
  const docsTokens: Record<string, string[]> = {};

  for (const d of docs) {
    const toks = tokenize(d.text);
    docsTokens[d.id] = toks;
    const tf: Record<string, number> = {};
    for (const t of toks) tf[t] = (tf[t] || 0) + 1;
    docTfs[d.id] = tf;
    const seen = new Set<string>();
    for (const t of Object.keys(tf)) {
      if (!seen.has(t)) {
        DF[t] = (DF[t] || 0) + 1;
        seen.add(t);
      }
    }
  }

  const N = docs.length || 1;
  const idf: Record<string, number> = {};
  for (const term of Object.keys(DF)) idf[term] = Math.log(1 + N / DF[term]);

  const docNorms: Record<string, number> = {};
  for (const d of docs) {
    const tf = docTfs[d.id] || {};
    let sum = 0;
    for (const [t, v] of Object.entries(tf)) {
      const w = (v || 0) * (idf[t] || 0);
      sum += w * w;
    }
    docNorms[d.id] = Math.sqrt(sum) || 1;
  }

  return { docs, idf, docTfs, docNorms };
}

export function queryIndex(index: Index, query: string, k = 4) {
  const qToks = tokenize(query);
  const qTf: Record<string, number> = {};
  for (const t of qToks) qTf[t] = (qTf[t] || 0) + 1;

  const qVec: Record<string, number> = {};
  for (const [t, v] of Object.entries(qTf)) {
    qVec[t] = v * (index.idf[t] || 0);
  }
  let qNorm = 0;
  for (const v of Object.values(qVec)) qNorm += v * v;
  qNorm = Math.sqrt(qNorm) || 1;

  const scores: { id: string; score: number }[] = [];
  for (const d of index.docs) {
    const tf = index.docTfs[d.id] || {};
    let dot = 0;
    for (const [t, qv] of Object.entries(qVec)) {
      const dv = (tf[t] || 0) * (index.idf[t] || 0);
      dot += qv * dv;
    }
    const denom = (index.docNorms[d.id] || 1) * qNorm;
    const score = denom ? dot / denom : 0;
    scores.push({ id: d.id, score });
  }

  scores.sort((a, b) => b.score - a.score);
  const results = scores.slice(0, k).map((s) => {
    const doc = index.docs.find((d) => d.id === s.id)!;
    return { ...doc, score: s.score };
  });
  return results;
}

// Embedding-based index utilities
export type EmbeddingIndex = {
  docs: DocChunk[];
  embeddings: number[][]; // same order as docs
};

export function buildEmbeddingIndexFromVectors(docs: DocChunk[], embeddings: number[][]): EmbeddingIndex {
  return { docs, embeddings };
}

export function queryEmbeddingIndex(index: EmbeddingIndex, qEmbedding: number[], k = 4) {
  const scores: { id: string; score: number }[] = [];
  for (let i = 0; i < index.docs.length; i++) {
    const e = index.embeddings[i] || [];
    let dot = 0;
    let ne = 0;
    let nq = 0;
    for (let j = 0; j < Math.min(e.length, qEmbedding.length); j++) {
      dot += e[j] * (qEmbedding[j] || 0);
      ne += e[j] * e[j];
      nq += (qEmbedding[j] || 0) * (qEmbedding[j] || 0);
    }
    const denom = Math.sqrt(ne) * Math.sqrt(nq) || 1;
    const score = denom ? dot / denom : 0;
    scores.push({ id: index.docs[i].id, score });
  }
  scores.sort((a, b) => b.score - a.score);
  return scores.slice(0, k).map((s) => ({ ...index.docs.find((d) => d.id === s.id)!, score: s.score }));
}
