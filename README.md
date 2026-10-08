# chef-backend

Backend API and MCP server for **Chef**, an AI-first personal recipe assistant.
The Android app lives in [AlltidSemester1337/chef](https://github.com/AlltidSemester1337/chef).

**Status:** early development. Current focus: an authenticated AI proxy for the Chef clients.

## Stack

- Python 3.13, FastAPI, uv
- Cloud Run (scale to zero), Cloud Firestore, Secret Manager
- MCP server (Streamable HTTP) via the official MCP Python SDK (planned)
- Tracing with OpenTelemetry/OpenInference to Arize Phoenix (planned)

## Getting started

```bash
uv sync
cp .env.example .env            # placeholders only; never commit .env
gcloud auth application-default login   # Google credentials via ADC, no key files
uv run uvicorn chef_backend.main:app --reload --port 8080
curl localhost:8080/healthz
```

Run checks:

```bash
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run pyright
```

## Security

No secrets or private configuration are stored in this repository. Secrets are kept in
Google Secret Manager and Google APIs are reached via Application Default Credentials.
CI runs [gitleaks](https://github.com/gitleaks/gitleaks) on every push.

## License

Apache-2.0
