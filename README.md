# ML Visualizer

**Interactive, real-time visualizations of classic machine learning algorithms — built with React, TypeScript, and p5.js.**

[![CI](https://github.com/amitt-codework/ml_visualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/amitt-codework/ml_visualizer/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-95%20passing-brightgreen)](#testing)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue?logo=typescript)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb?logo=react)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-7-646cff?logo=vite)](https://vitejs.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **[Live Demo →](https://amitt-codework.github.io/ml_visualizer/)**
> Watch algorithms learn in real time. Click to add training points, adjust hyperparameters, and compare classifiers side by side.

---

## What is this?

ML Visualizer lets you interact with machine learning algorithms the way you'd interact with a physics simulation — by touching the data and watching the model react instantly. It's designed to make the math *click* through visual intuition.

Every algorithm runs entirely in the browser with zero external dependencies. The implementations are from scratch in TypeScript (no scikit-learn, no TensorFlow).

---

## Algorithms

| Algorithm | Type | Key Features |
|-----------|------|-------------|
| **Linear Perceptron** | Binary classifier | Online learning, hinge loss, averaged variant |
| **Polynomial Perceptron** | Nonlinear classifier | Degree-2 feature expansion (x, y, x², y², xy) |
| **Neural Network (MLP)** | Deep binary classifier | Configurable layers, sigmoid/tanh/ReLU/LeakyReLU, SGD/Momentum/Adam |
| **K-Nearest Neighbors** | Non-parametric | Interactive k, Euclidean distance, neighbor highlighting |
| **Logistic Regression** | Binary classifier | Gradient descent, L2 regularization, animated loss curve |
| **Decision Tree** | Binary classifier | CART/Gini, configurable depth, animated split growth |

### Benchmark Datasets

All new demos include a dataset picker with:

| Dataset | Shape | Challenge |
|---------|-------|-----------|
| **Blobs** | Two Gaussian clusters | Linearly separable baseline |
| **Moons** | Two interleaved crescents | Needs nonlinear boundary |
| **Circles** | Concentric rings | Requires radial feature |
| **XOR** | Four-quadrant alternating | Classic nonlinear problem |
| **Spirals** | Two interleaved spirals | Hardest — needs deep tree or MLP |

---

## Screenshots

> _The decision tree demo growing splits depth-by-depth, the logistic regression demo showing gradient descent with a live loss curve, and the MLP demo visualizing the neural network architecture._

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI Framework** | React 19 (with Suspense + lazy loading) |
| **Language** | TypeScript 5.9 |
| **Build** | Vite 7 (p5.js externalized in prod for fast initial load) |
| **Canvas rendering** | p5.js (existing demos) + HTML Canvas 2D API (new demos) |
| **Math rendering** | KaTeX |
| **AI Assistant** | RAG with TF-IDF cosine similarity |
| **Backend** | Express + better-sqlite3 (question tracking) |
| **Tests** | Vitest + `@vitest/coverage-v8` |
| **CI/CD** | GitHub Actions (lint → typecheck → build → test on Node 18 & 20) |
| **Deployment** | GitHub Pages (auto-deploy on push to `main`) |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/amitt-codework/ml_visualizer.git
cd ml_visualizer
npm ci

# 2. Start development server
npm run dev

# 3. Run tests
npm test

# 4. Check coverage
npm run test:coverage

# 5. Lint + type-check + build
npm run lint && npx tsc --noEmit && npm run build
```

---

## Testing

95 tests across 8 test files covering every algorithm utility:

```
src/utils/perceptron.test.ts          3 tests  — fitOnline, averaged, pocket
src/utils/mlp.test.ts                 2 tests  — forward pass, accuracy
src/utils/mlp.loss.test.ts            1 test   — decreasing loss
src/utils/knn.test.ts                14 tests  — predict, majority vote, edge cases
src/utils/logisticRegression.test.ts 19 tests  — sigmoid, BCE loss, accuracy, L2 reg
src/utils/decisionTree.test.ts       19 tests  — Gini, XOR, depth limits, reset
src/utils/datasets.test.ts           24 tests  — all 5 generators, toXy
src/utils/polynomialClassifier.test.ts 13 tests — transform, train, losses
```

Run with coverage:

```bash
npm run test:coverage
```

Coverage is enforced at the CI level (≥70% lines/functions on utils).

---

## Architecture

```
src/
├── components/          React UI components
│   ├── PerceptronDemo.tsx       – p5.js Perceptron + Polynomial visualizer
│   ├── MlpDemo.tsx              – p5.js MLP with network diagram
│   ├── KnnDemo.tsx              – p5.js interactive KNN
│   ├── CompareDemo.tsx          – side-by-side comparison mode
│   ├── LogisticRegressionDemo.tsx  – Canvas 2D logistic regression
│   ├── DecisionTreeDemo.tsx        – Canvas 2D decision tree
│   └── AgentPanel.tsx           – in-app ML assistant (RAG)
├── utils/               Algorithm implementations (no dependencies)
│   ├── perceptron.ts    – Perceptron (online, averaged, pocket, hinge SGD)
│   ├── mlp.ts           – MLP (SGD / Momentum / Adam)
│   ├── knn.ts           – KNNClassifier + predictKNN helper
│   ├── logisticRegression.ts  – LogisticRegression (mini-batch GD, L2)
│   ├── decisionTree.ts  – DecisionTree (CART, Gini impurity)
│   ├── datasets.ts      – makeXOR / makeMoons / makeCircles / makeBlobs / makeSpirals
│   └── polynomialClassifier.ts  – degree-2 polynomial feature wrapper
└── App.tsx              – top-level layout, theme, algorithm selector
```

---

## CI/CD Pipeline

Every push and pull request runs:

1. **Lint** — ESLint + TypeScript-ESLint
2. **Type-check** — `tsc --noEmit`
3. **Build** — Vite production build
4. **Tests** — Vitest on Node 18 and 20

Merges to `main` auto-deploy to GitHub Pages via `peaceiris/actions-gh-pages`.

---

## Contributing

Pull requests are welcome. Please:

1. Run `npm test` and ensure all tests pass
2. Add tests for any new algorithm or utility
3. Follow the existing TypeScript style (no `any` where avoidable)

See `CONTRIBUTING.md` for more details.

---

## License

MIT — see [LICENSE](LICENSE).
