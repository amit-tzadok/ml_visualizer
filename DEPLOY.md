# Deploying to Vercel

This document describes the minimal steps to deploy this Vite + React app with the small serverless API in `api/` to Vercel.

Prerequisites

- A GitHub (or GitLab/Bitbucket) repository connected to Vercel
- Vercel account (free tier is fine for small projects)

1. Connect the repository

- Go to https://vercel.com/new and import the `ml_visualizer` repository.

2. Build settings

- Framework Preset: Vite
- Build Command: `npm run build` (the project already has this script)
- Output Directory: `dist`

3. Environment variables / secrets

- This project does not require any third-party API keys by default. No `OPENAI_API_KEY` or similar secrets are needed for the app to run.
- Do NOT commit any real keys to the repository. Use the `.env.example` as a template if you later add secrets.

4. Serverless functions

- Files in the `api/` directory will be deployed as Vercel Serverless Functions automatically. No extra configuration required for basic use.
- Keep function execution limits in mind: serverless functions have time and memory limits on Vercel's free/paid plans.

5. SPA routing (history fallback)

- Client-side routing needs a fallback to `index.html` for unknown routes. A `vercel.json` with a rewrite is included in the repo to ensure refresh/navigation works on nested routes.

6. Local testing

- Build locally:
  ```bash
  npm install
  npm run build
  npm run preview
  # open http://localhost:4173 (vite preview default)
  ```
- Note: the `api/embeddings` serverless endpoint is disabled in this repository (the project is configured not to use external embedding APIs).

7. Optional: Vercel CLI

- Install the Vercel CLI to deploy from your machine or inspect deployments:
  ```bash
  npm i -g vercel
  vercel login
  vercel --confirm
  ```

8. Post-deploy checklist

- Verify preview deployment for a PR (Vercel creates preview URLs automatically).
- Confirm the `api/embeddings` endpoint responds on the deployed domain.
- Add uptime monitoring and Sentry (recommended) after first deploy.

If you want, I can also add a GitHub Actions workflow that runs tests and typechecks on PRs before Vercel deploys.
