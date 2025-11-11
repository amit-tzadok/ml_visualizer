# Contributing

Thanks for wanting to contribute! A few simple guidelines to make reviews fast and easy:

- Fork the repository and open a branch for your change.
- Run `npm install` and make sure `npm run lint` and `npm test` pass.
- Provide a concise PR title and description explaining the change and why it's useful.
- Keep changes small and focused. Add tests for new behavior where practical.

Coding standards

- The project uses TypeScript and ESLint. Please run `npm run lint` before sending a PR.
- Follow existing patterns for p5 sketches (avoid recreating expensive sketches in render loops).

Security

If you discover a security issue, please follow `SECURITY.md` to report it privately.

# Contributing

Thanks for your interest in contributing to ml_visualizer — contributions are very welcome.

Quick guidelines:

- Fork the repo and create a branch for your work (feature/bugfix).
- Keep changes focused and use clear commit messages (imperative, one-line summary).
- Add tests for bug fixes and new features where appropriate. Run `npm run test`.
- Keep formatting consistent. This project uses TypeScript, ESLint, and Prettier conventions.

Local development:

1. Install dependencies

```bash
npm ci
```

2. Start dev server

```bash
npm run dev
```

3. Run tests

```bash
npm run test
```

Pull requests:

- Open a PR against `main` with a clear description and link to any relevant issue.
- CI will run tests and build; keep PRs small and reviewable.
- For major changes, open an issue first to discuss the approach.

## Code of Conduct

This project follows the Contributor Covenant Code of Conduct. By participating you agree to abide by its terms.
