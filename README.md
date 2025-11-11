# ml_visualizer

Interactive machine-learning visualizations built with React, TypeScript and p5.js.

![Embeddings: disabled](https://img.shields.io/badge/Embeddings-disabled-red)

Note: Embeddings disabled — this repository is intentionally configured to not use external
embedding APIs (e.g. OpenAI). The client helper and serverless endpoint for embeddings are
disabled to avoid storing API keys in the repo. If you want to enable embeddings later, see
the "Enabling embeddings" section below.

This repository contains small demos (KNN, Perceptron, MLP) intended for teaching and experimentation.

Quick start

1. Install dependencies

```bash
npm install
```

2. Run a dev server

```bash
npm run dev
```

3. Run lint, tests and build

```bash
npm run lint
npm test
npm run build
```

Contributing

See `CONTRIBUTING.md` for how to open issues and PRs. If you'd like to help, please run the test suite and follow the repository style.

License

This project is licensed under the MIT License — see the `LICENSE` file for details.

# React + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.

## CopilotKit Demo

This repo includes a small CopilotKit integration for demo and submission purposes.

Quick start:

1. Install dependencies:

# ml_visualizer

Interactive visualizations and demos for simple machine learning models (Perceptron, MLP, KNN) built with React + Vite + p5.js.

This repository contains interactive demos used for teaching and experimentation. It includes the visualizers, algorithms, and a small in-app assistant.

Quick start

1. Install dependencies

```bash
npm ci
```

2. Start the dev server

```bash
npm run dev
```

3. Run tests

```bash
npm run test
```

4. Build for production

```bash
npm run build
```

Repository notes

- License: MIT (see `LICENSE`).
- Tests: Vitest is configured; CI will run tests and build on PRs.

Contributing

See `CONTRIBUTING.md` for contributing guidelines and how to run tests locally.

Enabling embeddings

If you'd like to enable embeddings (not recommended for public repos unless you use a secure secret store):

- Implement or restore a server-side proxy that holds your API key (do not store keys in client code).
- Set server-side environment variable `OPENAI_API_KEY` in your deployment provider (e.g., Vercel) and, for local dev only, optionally set `VITE_OPENAI_API_KEY` in a local `.env` file.
- Re-enable the proxy implementation in `api/embeddings.js` and restore the client helper in `src/utils/embeddings.ts`, or set `ENABLE_VERCEL_PROXY=true` if using the previous guarded implementation.
- Never commit real keys. Keep `.env` ignored and use `.env.example` for placeholders.

If you only want local/demo embeddings without external APIs, consider using the included demo helpers and mock data instead of enabling external keys.
