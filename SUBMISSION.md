# CopilotKit Submission Checklist — ml_visualizer

This document lists the artifacts and steps to prepare `ml_visualizer` as a candidate project for the CopilotKit internship application.

Required artifacts

- Public repository (this repo) with a clear README highlighting CopilotKit integration.
- A short demo (2–3 min) showing CopilotKit in action inside the app (Screencast.mp4 or hosted link).
- A technical write-up (800–1200 words) describing architecture, retrieval strategy, and experiment ideas.
- A small PR or contribution to CopilotKit (docs or example) showing familiarity with the codebase.
- Basic analytics screenshots showing event tracking for assistant opens/queries.

Checklist

1. README updates

   - Add a top-level "CopilotKit Demo" section describing how to open the assistant and example queries.
   - Include a direct link to the CopilotKit popup or any demo endpoints.

2. Demo recording

   - Record a 2–3 minute screencast (see `Screencast.md`) that shows: opening the assistant, asking codebase questions, showing retrieved sources, and leaving feedback.

3. RAG/Context integration

   - Ensure `public/docs/assistant_context.md` is up-to-date.
   - Implement a small retrieval helper (client-side or serverless) that the CopilotKit integration can call.

4. Analytics

   - Add simple event tracking for: assistant_open, query_submitted, reply_feedback.
   - Provide a screenshot of events over a short run or logs exported as CSV.

5. Blog/Tutorial

   - Write an article explaining how you built the demo, components used (CopilotKit, any retriever), and growth experiments to run.

6. PR to CopilotKit

   - Identify a small improvement (doc fix, example) and open a PR. Link the PR in this repo.

7. QA & final checks
   - Run `npm ci`, `npx tsc --noEmit`, `npm run lint`, and `npm test` — confirm green.
   - Ensure the demo runs locally and CopilotKit popup opens.

Submission package

- `README.md` (updated)
- `SUBMISSION.md` (this file)
- `Screencast.mp4` (or hosted link in README)
- `docs/` (tutorial draft and architecture diagram)
- `demo/` (if any serverless functions or retrieval scripts)

Notes

- Keep the CopilotKit public key and any secrets out of the repo. Use `VITE_COPILOTKIT_PUBLIC_KEY` for local demos.
- If you need help implementing the retrieval pipeline (LlamaIndex / LangGraph), I can scaffold a minimal client-side RAG that indexes `public/docs/assistant_context.md`.
