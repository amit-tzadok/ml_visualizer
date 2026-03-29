# ML Visualizer — Project Memory

## Stack
- React 19 + TypeScript 5.9 + Vite 7
- p5.js (existing demos) + HTML Canvas 2D API (LogReg + DecisionTree demos)
- Vitest + @vitest/coverage-v8 for tests
- GitHub Actions CI (lint → tsc --noEmit → build → test:coverage)
- GitHub Pages auto-deploy on main push (deploy-gh-pages.yml)

## Key paths
- `/src/utils/` — all algorithm implementations + tests (no p5 dependency)
- `/src/components/` — React demo components (lazy-loaded in App.tsx)
- `/vitest.config.ts` — includes coverage config with thresholds (70%/55%)

## Algorithm list (as of 2026-03)
- linear / poly → PerceptronDemo.tsx
- mlp → MlpDemo.tsx
- knn → KnnDemo.tsx
- logreg → LogisticRegressionDemo.tsx
- dtree → DecisionTreeDemo.tsx
- compare → CompareDemo.tsx (supports linear/poly/mlp only)

## LLM Agent (api/chat.cjs)
- POST /api/chat — calls Claude Haiku with system prompt + RAG context + live appState
- Requires ANTHROPIC_API_KEY env var; returns 503 {code:'NO_API_KEY'} when missing
- AgentPanel.tsx calls it, falls back to generateFallbackResponse() on 503 or network error
- Local dev: add route to api-server.cjs (already done)
- RAG: TF-IDF index built from public/docs/assistant_context.md (fetched at runtime)

## Test status
- 95 tests, 8 files, all passing
- Coverage: ~80% lines, ~92% functions, ~64% branches

## Patterns
- New Canvas-based demos use `useRef + requestAnimationFrame` (no p5 needed)
- Coordinate helper convention: `toPx(v, size)` / `fromPx(px, size)` with RANGE=1.3
- `reset()` must call `stopAnimation()` first then reinit refs, then rAF render
- `npm test` = `vitest --run`; `npm run test:coverage` for coverage
