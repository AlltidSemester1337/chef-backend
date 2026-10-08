# Build stage: resolve dependencies from uv.lock into a virtualenv.
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

# Runtime stage: only the virtualenv and source, running as a non-root user.
FROM python:3.13-slim
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
# Cloud Run sets PORT; default to 8080 for local runs.
CMD ["sh", "-c", "exec uvicorn chef_backend.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
