# ShopMind AI — ShopHub order-support assistant

Multi-tenant support assistant for ShopHub merchant staff: a LangGraph orchestrator that
talks to an orders sub-agent over JSON-RPC 2.0, which calls a mock store API with the
caller's own bearer token.

> Work in progress. Full run, test and eval instructions are added as each phase lands.

## Quick start (local, fake-LLM mode — no API key needed)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # Linux/macOS: cp .env.example .env
pytest
```

## Layout

| Path | Purpose |
|---|---|
| `store_api/` | R0 mock store API (port 8002) |
| `orders_agent/` | R2 orders sub-agent, JSON-RPC + agent card (port 8001) |
| `orchestrator/` | LangGraph orchestrator + SSE chat API (port 8000) |
| `common/` | Shared configuration |
| `evals/` | R9 evaluation harness |
| `tests/` | pytest suite (runs offline) |
