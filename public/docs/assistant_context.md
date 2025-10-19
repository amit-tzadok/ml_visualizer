ML Visualizer — Assistant Context

Overview

ML Visualizer is a small client-side React + p5 app that demonstrates simple binary classification algorithms and lets users interactively add training points and visualize decision boundaries.

Key demos and locations

- Linear Perceptron demo: `src/components/PerceptronDemo.jsx`
- Polynomial Perceptron demo: `src/components/CompareDemo.jsx` (and polynomial classifier logic in `src/utils/polynomialClassifier.js`)
- MLP demo: `src/components/MlpDemo.tsx` and model: `src/utils/mlp.ts`
- KNN demo: `src/components/KnnDemo.jsx` and helper `src/utils/knn.js`

Common controls and UI

- Algorithm dropdown (label: "Algorithm") switches demos.
- Add points: click canvas (left click = class A / red, right click = class B / blue). On touch devices use the "Touch tap adds" selector in Model Controls.
- Floating controls:
  - Speed: bottom-left
  - Algorithm + Compare toggle: bottom-center
  - Model Controls (MLP) panel: bottom-right
  - Help/Controls box: top-right

Dev notes

- Run locally: `npm install` then `npm run dev` (Vite dev server).
- Type-check: `npx tsc --noEmit`.
- Tests: `npm test` (Vitest).

Useful files

- App shell: `src/App.tsx`
- Entry: `src/main.tsx`
- Types: `src/types/*` (p5 shims and vite env)

Example Q/A style

Q: How to retrain the MLP with higher LR?
A: Open Model Controls (bottom-right) -> set LR -> click Train at top-left of the canvas.

Q: Where is the MLP code?
A: See `src/utils/mlp.ts` for the training loops and `MLPClassifier.fit` implementation.

---

If you need deeper code-aware answers, consider enabling a retrieval pipeline (RAG) that indexes `src/` and `docs/` and runs a vector search before calling an LLM. This app contains a `docs/assistant_context.md` file to start with.
