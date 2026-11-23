# One image for the store API, the orders sub-agent and the chat API.
# Which one runs is decided by the `command` in docker-compose.yml.

# --- build stage: compilers and pip caches stay behind ---------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv

# Dependencies first, so changing application code does not reinstall them.
COPY requirements.lock.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements.lock.txt

COPY pyproject.toml README.md ./
COPY common ./common
COPY store_api ./store_api
COPY orders_agent ./orders_agent
COPY orchestrator ./orchestrator
COPY evals ./evals
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

# --- runtime stage -----------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0

USER app
WORKDIR /home/app

# PORT is set per service. The slim image has no curl, so Python does the check.
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=12 \
    CMD ["python", "-c", "import os, urllib.request as u; u.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/healthz', timeout=2)"]
