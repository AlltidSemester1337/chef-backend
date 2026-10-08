# CLAUDE.md

Guidance for Claude Code when working in this repository.

**At the start of every session, read `PROGRESS.md`.** After each checkpoint, add an entry to it (date, done, next steps, learnings). `PROGRESS.md` is gitignored and stays local because it may describe security work that is not yet released.

## Security rules (non-negotiable)

This repository is **public**. Treat everything committed or pushed as published forever.

- **Never commit, push or publish secrets or private configuration.** That includes API keys, tokens, passwords, service-account key files, `.env` files, `google-services.json` / `GoogleService-Info.plist`, Firebase debug/App Check tokens, OAuth client secrets, signing keys and keystores.
- **Never commit private data**: database exports, backups, user data, chat logs or real user IDs/emails. Use synthetic fixtures in tests.
- **No real project-specific configuration in the repo either**: GCP project IDs, bucket names, service-account emails, internal URLs. Use placeholders (`your-gcp-project-id`) in examples and inject real values via environment variables.
- Secrets live in **Google Secret Manager** and reach Cloud Run as env vars. Google APIs are called via **ADC** (the Cloud Run service account; locally `gcloud auth application-default login`). Never create or download service-account key files.
- Service accounts get **least privilege**, scoped to specific resources where possible (e.g. `secretAccessor` per secret).
- Before every commit: check `git diff --cached` for anything that looks like a secret or private config. If something leaks, stop and tell the user immediately: rotating the secret is required, deleting the commit is not enough.
- Do not describe unfixed vulnerabilities in commits, PRs, issues or public docs. Keep that in the local `PROGRESS.md` until the fix has shipped.
- Never paste secret values into chat, logs, test output, PR descriptions or issue trackers.
- CI runs gitleaks on every push; do not weaken or bypass it.

## Cloud resources and cost

- Budget is about **200 SEK/month** for the whole GCP project, with a budget alert in place.
- **Always ask before creating or changing cloud resources** (Cloud Run, Secret Manager, IAM, Artifact Registry, Firestore, APIs, quotas) and show the estimated monthly cost.
- Prefer scale-to-zero (`min-instances=0`) and set a `max-instances` cap as a cost guard.
- Region: `europe-north2` (Stockholm), matching Firestore.

## Working agreements

- Respond in Swedish; code, comments, commit messages and docs in English.
- **Learning mode**: briefly explain new concepts. The user writes key parts (e.g. auth middleware, MCP tool definitions) and Claude reviews. Ask which parts the user wants to write when starting a new feature.
- Work in checkpoints: small, finishable steps that each leave the repo green and are logged in `PROGRESS.md`. Session length varies, so one session may cover several checkpoints or only part of one.
- Tests are written together with the implementation, never deferred.

## Project overview

Backend for **Chef**, an AI-first personal recipe assistant (Android app: `AlltidSemester1337/chef`). Planned web and iOS clients will use the same backend.

**Architecture (hybrid):**
- Clients (Android, web, later iOS) do plain CRUD directly against **Cloud Firestore** under its security rules (rules and schema live in the Android repo: `firestore.rules`, `.ai/firestore-schema.md`).
- This backend owns what must not live in a client: **AI calls, secrets, quota enforcement, system prompts, domain logic** (week planning, shopping lists) and the **MCP server** (mounted at `/mcp` in the same FastAPI app).
- The AI proxy accepts a *purpose* (e.g. `chat`, `extract_recipes`), never client-supplied system prompts or model parameters.
- Requests are authenticated with a Firebase ID token plus a Firebase App Check token.
- The backend uses the Firebase Admin SDK, which **bypasses security rules**, so ownership checks must be enforced in backend code.

## Commands

```bash
uv sync                        # install dependencies (creates .venv)
uv run pytest                  # tests
uv run ruff check . && uv run ruff format --check .   # lint and format check
uv run ruff format .           # format
uv run pyright                 # type check (strict for src/)
uv run uvicorn chef_backend.main:app --reload --port 8080   # run locally
```

Docker is not installed locally; the image is built in CI (`docker-build` job) and later by Cloud Build.

## Layout

- `src/chef_backend/main.py`: app factory (`create_app`) and routes
- `src/chef_backend/config.py`: settings from env vars (prefix `CHEF_`)
- `tests/`: pytest tests, using FastAPI `TestClient` and fakes rather than real cloud services
