# ShopMind AI — ShopHub order-support assistant

Multi-tenant support assistant for ShopHub merchant staff: a LangGraph orchestrator that
talks to an orders sub-agent over JSON-RPC 2.0, which calls a mock store API with the
caller's own bearer token.

> Work in progress — built phase by phase. Sections are added as each phase lands.

| Phase | Status |
|---|---|
| 0 — Project skeleton, config | ✅ Done |
| 1 — Mock store API (R0) | ✅ Done |
| 2 — Order tools + guardrails | ⏳ Next |
| 3+ — Sub-agent, orchestrator, HITL, streaming, persistence, Docker | Planned |

## Setup (fake-LLM mode — no API key needed)

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

`LLM_PROVIDER=fake` is the default, so nothing else is needed to run offline.

## Running the tests

```bash
pytest                 # whole suite, no API key needed
pytest -v              # one line per test
pytest -k cancel -v    # only tests whose name contains "cancel"
```

## Mock store API (R0)

An in-memory FastAPI service with seed data. State resets on every restart.

```bash
python -m store_api    # http://localhost:8002   (override with HOST / PORT)
```

Interactive docs: **http://localhost:8002/docs** → click **Authorize** and paste a token.

### Seed tokens

| Token | User | Merchant |
|---|---|---|
| `tok_acme_alice` | alice@acme-outdoor | `m_acme` (Acme Outdoor, 8 orders) |
| `tok_globex_bob` | bob@globex-home | `m_globex` (Globex Home, 6 orders) |
| `tok_orphan_carol` | carol@shophub | *none* — valid token, no store (must fail closed) |

These are fake tokens for the mock only.

### Endpoints (all scoped to the caller's merchant)

| Method | Path | Behaviour |
|---|---|---|
| GET | `/auth/me` | `{user, merchant}` — `merchant` is `null` for the orphan token |
| GET | `/orders?status=PAID` | Orders of the caller's store, newest first |
| GET | `/orders/lookup?id=1042` | **Legacy:** always HTTP 200; failure is `{"status": false, "message": "Order not found"}` — also for another merchant's order |
| POST | `/orders/refund` `{"id": 1042}` | 404 unknown/foreign order, 409 invalid transition |
| POST | `/orders/cancel` `{"id": 1042}` | Same errors; cancelling a PAID order also refunds it |
| GET | `/healthz` | Liveness check |

Status rules: refund from `PAID`/`SHIPPED`/`DELIVERED`; cancel from `PLACED`/`PAID`.
Every order carries `paid_amount` and `refunded_amount`, which are tracked separately from
`status` so the store always knows whether money is owed back.

Orders 1002, 1006 and 2003 have customer notes that try to instruct an AI assistant —
they exist to test the guardrails.

### Try it with curl

On Windows PowerShell use `curl.exe` (plain `curl` is an alias for `Invoke-WebRequest`)
and escape quotes inside JSON as `\"`.

```bash
curl.exe -H "Authorization: Bearer tok_acme_alice" "http://localhost:8002/orders?status=PAID"
curl.exe -H "Authorization: Bearer tok_acme_alice" "http://localhost:8002/orders/lookup?id=2001"
curl.exe -X POST -H "Authorization: Bearer tok_acme_alice" -H "Content-Type: application/json" -d '{\"id\":1042}' http://localhost:8002/orders/cancel
```

## Layout

| Path | Purpose |
|---|---|
| `store_api/` | R0 mock store API (port 8002) |
| `orders_agent/` | R2 orders sub-agent, JSON-RPC + agent card (port 8001) |
| `orchestrator/` | LangGraph orchestrator + SSE chat API (port 8000) |
| `common/` | Shared configuration (`common/settings.py`, read from env / `.env`) |
| `evals/` | R9 evaluation harness |
| `tests/` | pytest suite (runs offline) |

## Configuration

All settings come from environment variables (see `.env.example`). Never commit `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` (offline, scripted) or `anthropic` |
| `ANTHROPIC_API_KEY` | — | Only needed when `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-sonnet-5-5` | Model name for the real provider |
| `STORE_API_URL` | `http://localhost:8002` | Store API base URL |
| `ORDERS_AGENT_URL` | `http://localhost:8001` | Orders sub-agent base URL |
| `DATABASE_URL` | `postgresql://shopmind:shopmind@localhost:5432/shopmind` | Checkpointer DB (R7) |
