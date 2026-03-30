# ML Visualizer — Assistant Context

## What the app does

ML Visualizer is a browser-based interactive tool for learning machine learning algorithms. Users add 2D data points by clicking the canvas, then watch classifiers train and draw decision boundaries in real time. All algorithm implementations are from scratch in TypeScript — no external ML libraries.

---

## Algorithms

### Linear Perceptron (`src/utils/perceptron.ts`, demo: `PerceptronDemo.tsx`)

The simplest binary classifier. Finds a straight-line decision boundary.

- **Update rule**: w ← w + α(y - ŷ)x, b ← b + α(y - ŷ)
- **Variants**: standard online, averaged perceptron, pocket algorithm (handles noisy data), hinge-loss SGD
- **Limitation**: only works when classes are linearly separable (or nearly so)
- **Decision boundary**: a single line `w₀x + w₁y + b = 0`

### Polynomial Perceptron (`src/utils/polynomialClassifier.ts`, demo: `PerceptronDemo.tsx` with type="poly")

Wraps the Perceptron with degree-2 feature expansion.

- **Feature map**: [x, y] → [x, y, x², y², xy]  (5 features)
- Can learn curved (elliptic, hyperbolic) decision boundaries
- Runs a standard perceptron in the expanded space

### Neural Network — MLP (`src/utils/mlp.ts`, demo: `MlpDemo.tsx`)

Multi-layer perceptron for complex nonlinear classification.

- **Activations**: sigmoid, tanh, ReLU, Leaky ReLU
- **Optimisers**: SGD, SGD with momentum, Adam
- **Initialization**: Xavier normal
- **Stability**: gradient clipping (clip=5), weight clamping (±1000)
- **Loss**: binary cross-entropy with sigmoid output layer
- Supports configurable hidden layers (e.g. [16, 16])

### K-Nearest Neighbors (`src/utils/knn.ts`, demo: `KnnDemo.tsx`)

Non-parametric; no training phase. Classifies by majority vote of k nearest neighbours.

- **Distance**: Euclidean
- **k**: user-adjustable; odd values avoid ties
- **Limitation**: every prediction requires scanning all training points (O(n))

### Logistic Regression (`src/utils/logisticRegression.ts`, demo: `LogisticRegressionDemo.tsx`)

Smooth differentiable binary classifier using the sigmoid function.

- **Model**: P(y=1|x) = σ(w·x + b) where σ(z) = 1/(1+e^−z)
- **Loss**: binary cross-entropy L = −[y·log(p) + (1−y)·log(1−p)]
- **Update**: mini-batch gradient descent, gradient of BCE w.r.t. z is simply (p − y)
- **Regularisation**: L2 penalty λ/2·||w||², keeps weights bounded
- **Extras**: learning rate slider, live loss curve, `fitStep()` for animation
- **Decision boundary**: straight line (same as linear perceptron, but trained differently with softer gradients)
- **Dataset presets**: Blobs, Moons, Circles, XOR

### Decision Tree (`src/utils/decisionTree.ts`, demo: `DecisionTreeDemo.tsx`)

Axis-aligned recursive splits using Gini impurity (CART algorithm).

- **Split criterion**: maximise information gain IG = Gini(parent) − wL·Gini(L) − wR·Gini(R)
- **Gini impurity**: Gini(S) = 1 − Σ pᵢ²
- **Parameters**: max depth (1–8), min samples per split (4), min samples per leaf (2)
- **Visualisation**: splits appear animated depth-by-depth, each depth has its own colour
- **Insight**: shallow trees underfit; deep trees overfit (perfect training accuracy, complex jagged regions)
- **Dataset presets**: Blobs, Moons, Circles, XOR, Spirals

---

## Benchmark datasets (`src/utils/datasets.ts`)

| Dataset | Shape | Difficulty |
|---------|-------|-----------|
| Blobs | Two Gaussian clusters | Easy — linearly separable |
| Moons | Two interleaved crescents | Medium — needs curved boundary |
| Circles | Concentric rings | Medium — radial boundary |
| XOR | Four quadrants alternating | Hard — nonlinear, needs polynomial/tree/MLP |
| Spirals | Two interleaved spirals | Very hard — only deep tree or MLP succeeds |

---

## UI controls

- **Bottom bar**: Algorithm dropdown, Compare toggle, Speed slider, training status badge
- **Canvas interaction**: left-click = class A, right-click = class B (or use Touch tap selector on mobile)
- **🤖 Agent** (top-right): this chat assistant
- **⌨️ button** (top-right): keyboard shortcuts
- **ℹ️ Info** (top-right): algorithm overview
- **🏠 Home** (top-left): return to welcome screen

## Compare mode

Shows two classifiers side-by-side on the same dataset. Supports Linear, Poly, and MLP. KNN, Logistic Regression, and Decision Tree are standalone-only.

---

## Development

```bash
npm ci                # install deps
npm run dev           # Vite dev server (port 5173)
node api-server.cjs   # local API server (port 3001) — optional
npm test              # Vitest (95 tests, all passing)
npm run test:coverage # with coverage report
npx tsc --noEmit      # type-check
npm run build         # production build
```

Key files:
- `src/App.tsx` — top-level layout, algorithm routing, theme
- `src/utils/` — all algorithm implementations + tests (no dependencies)
- `src/components/` — React demo components (lazy-loaded)
- `api/chat.cjs` — LLM assistant endpoint (requires ANTHROPIC_API_KEY)
- `api/questions.cjs` — question logging endpoint (SQLite)
