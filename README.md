# ShopMind AI — ShopHub order-support assistant

Multi-tenant support assistant for ShopHub merchant staff: a LangGraph orchestrator that
talks to an orders sub-agent over JSON-RPC 2.0, which calls a mock store API with the
caller's own bearer token.

> Work in progress — built phase by phase. Sections are added as each phase lands.

| Phase | Status |
|---|---|
| 0 — Project skeleton, config | ✅ Done |
| 1 — Mock store API (R0) | ✅ Done |
| 2 — Order tools + guardrails (R1 envelope, R4a/b) | ✅ Done |
| 3 — Orders sub-agent over JSON-RPC + agent card (R2, R3) | ✅ Done |
| 4 — LLM factory + fake LLM | ✅ Done |
| 5 — LangGraph orchestrator (R1) | ✅ Done |
| 6 — Human approval for refunds and cancels (R5) | ✅ Done |
| 7 — Chat API and SSE streaming (R6, R4c) | ✅ Done |
| 8 — PostgreSQL persistence (R7) | ✅ Done |
| 9 — Test review (R8) | ⏳ Next |
| 10+ — Evals, Docker, DESIGN.md | Planned |

## Setup (fake-LLM mode — no API key needed)

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -e ".[dev]"
copy .env.example .env          # Linux/macOS: cp .env.example .env
```

`LLM_PROVIDER=fake` is the default, so nothing else is needed to run offline.

## Running the services locally

Each service is its own process, so use one terminal per service and activate the
virtual environment in each. Start the store first, because the sub-agent calls it.

| Terminal | Command | URL |
|---|---|---|
| 1 — Store API | `python -m store_api` | http://localhost:8002 |
| 2 — Orders sub-agent | `python -m orders_agent` | http://localhost:8001 |
| 3 — Chat API | `python -m orchestrator` | http://localhost:8000 |

Each one prints `Uvicorn running on http://127.0.0.1:<port>` when ready; stop it with
`Ctrl+C`. Then check them from a third terminal:

```bash
curl.exe http://localhost:8002/healthz
curl.exe http://localhost:8001/healthz
curl.exe http://localhost:8000/healthz
curl.exe http://localhost:8001/.well-known/agent-card.json
```

The agent card is served by the running sub-agent and built from the code in
`orders_agent/skills.py` — it is a URL, not a file in the repository.

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

## Orders sub-agent (R2)

Owns the order tools and the guardrails. Reached over JSON-RPC 2.0; the orchestrator
discovers it from its agent card and never imports its code. Needs the store API running
(viewing the agent card does not).

```bash
python -m orders_agent   # http://localhost:8001   (override with HOST / PORT)
```

All skills go through the single `POST /` endpoint, so use curl rather than the `/docs` page
to try them.

| Method | Path | Purpose |
|---|---|---|
| GET | `/.well-known/agent-card.json` | Agent card: name, endpoint, skills with JSON Schemas (public, no token) |
| POST | `/` | JSON-RPC 2.0 — `method` is a skill id, `params` an object |

| Skill | Mutating | Purpose |
|---|---|---|
| `list_orders` | no | Filter by `status`, `customer_name`, `placed_more_than_days_ago` |
| `get_order` | no | One order by `order_id` (legacy envelope handled) |
| `preview_action` | no | Proposal for `refund`/`cancel`: what changes and the amount returned |
| `refund_order` | **yes** | Only after human approval |
| `cancel_order` | **yes** | Only after human approval |

Every order is sanitized before it is returned: emails/phones redacted, instruction-like
customer notes withheld (`note_flagged: true`). Business failures (not found, 409, store
down) come back as `result: {"ok": false, "error": ...}`; protocol failures (bad JSON,
unknown method, invalid params, missing token) as JSON-RPC `error` objects.

```bash
curl.exe -s http://localhost:8001/.well-known/agent-card.json
curl.exe -s -X POST http://localhost:8001/ -H "Authorization: Bearer tok_acme_alice" -H "Content-Type: application/json" -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"preview_action\",\"params\":{\"action\":\"refund\",\"order_id\":1042}}'
```

## Choosing the model

All model creation goes through `get_chat_model()` in `orchestrator/llm.py`.

| `LLM_PROVIDER` | Model | Needs |
|---|---|---|
| `fake` (default) | `ScriptedChatModel`: fixed rules, real `tool_calls`, repeatable output | Nothing |
| `google` | `ChatGoogleGenerativeAI` (Gemini), using `GOOGLE_MODEL` | `pip install -e ".[google]"` and `GOOGLE_API_KEY` |

The fake model is deliberately gullible: if injection text reaches it, it obeys and calls
`refund_order`, so a guardrail failure fails a test. It proves the system around the model,
not how well a real model chooses tools or resists injection.

## Orchestrator (R1)

A LangGraph loop: the `agent` node asks the model what to do, the `tools` node runs any tool
calls, and the results go back to the model until it answers in text.

- **Tools come from the agent card.** The orchestrator never imports the sub-agent's code.
- **Only read-only skills become tools.** The model has no way to call `refund_order` or
  `cancel_order`.
- **The token travels in LangGraph's run context** (`RunContext`). It is not a tool argument,
  so the model never sees it. It is not in `configurable` either, because those values are
  copied into checkpoint metadata and would end up in the database.

### Human approval (R5)

A refund or cancel never runs straight away:

1. The model calls the read-only `preview_action`.
2. If the action is possible, the graph pauses with LangGraph's `interrupt()` and returns the
   proposal: order, merchant, what changes and the amount.
3. Resuming with `Command(resume="approve")` runs the change. Any other value changes nothing.

After a decision the outcome is written by code, not the model, so what happened to the money
is never paraphrased. The token is passed again on resume, because it is never saved, so the
approver's own merchant is what the store checks.

### Chat API (R6)

Needs the store and the sub-agent running first, and PostgreSQL (see below).

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | `{"message": "...", "thread_id": "..."}` and an optional `"model"`; the reply streams over SSE |
| GET | `/models` | The models that can be selected, default first |
| POST | `/chat/{thread_id}/approve` | Approve the waiting proposal; the result streams |
| POST | `/chat/{thread_id}/reject` | Reject it; the result streams |
| GET | `/healthz` | Liveness check |

Stream events: `token`, `tool_started`, `tool_finished`, `proposal`, `done`, `error`.

| Status | Meaning |
|---|---|
| 401 | Missing or invalid token |
| 403 | Valid token with no merchant (refused before the model runs) |
| 409 | A proposal is already waiting, or there is nothing to decide |
| 503 | The store could not confirm who you are |

Each thread belongs to the caller's merchant: the real key is `merchant:thread_id`, so the
same `thread_id` from another merchant is a separate conversation.

Chat in the browser at **http://localhost:8000/**. Pick a demo account at the top; refunds and
cancels appear as a card with Approve and Reject. With `LLM_PROVIDER=google` a model picker
also appears, so you can switch model in the middle of a conversation if one reaches its
usage limit.

Or chat from the terminal, answering `a` or `r` when a proposal appears:

```bash
python -m orchestrator.client --token tok_acme_alice
```

With curl: in Windows PowerShell 5.1 keep the message to one word, because spaces break
the escaped JSON:

```bash
curl.exe -N -X POST http://localhost:8000/chat -H "Authorization: Bearer tok_acme_alice" -H "Content-Type: application/json" -d '{\"message\":\"1042\",\"thread_id\":\"demo\"}'
```

### Persistence (R7)

Conversations and waiting proposals are saved in PostgreSQL, so they survive a restart of the
chat API. Create the database once, as a PostgreSQL admin (for example in pgAdmin):

```sql
CREATE USER shopmind WITH PASSWORD 'shopmind';
CREATE DATABASE shopmind OWNER shopmind;
```

The tables are created at startup. If PostgreSQL cannot be reached the chat API stops with a
clear error. To run without it, set `DATABASE_URL=` (empty); conversations are then lost on
restart.

To see it work: ask for a refund, stop `python -m orchestrator` with `Ctrl+C`, start it again,
then approve the same proposal. Restart only the chat API, because the mock store keeps its
orders in memory.

The bearer token is never saved. Tests that need a database skip themselves if there is none.

## Layout

| Path | Purpose |
|---|---|
| `store_api/` | R0 mock store API (port 8002) |
| `orders_agent/` | R2 orders sub-agent, JSON-RPC + agent card (port 8001) |
| `orchestrator/` | LangGraph orchestrator + SSE chat API (port 8000) |
| `common/` | Shared configuration (`common/settings.py`, read from env / `.env`) and the untrusted-data marker |
| `evals/` | R9 evaluation harness |
| `tests/` | pytest suite (runs offline) |

## Configuration

All settings come from environment variables (see `.env.example`). Never commit `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` (offline, scripted) or `google` |
| `GOOGLE_API_KEY` | — | Only needed when `LLM_PROVIDER=google` |
| `GOOGLE_MODEL` | `gemini-3.1-flash-lite` | Default model for the real provider |
| `GOOGLE_EXTRA_MODELS` | `gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-3.6-flash` | Other models the web page can switch to |
| `STORE_API_URL` | `http://localhost:8002` | Store API base URL |
| `ORDERS_AGENT_URL` | `http://localhost:8001` | Orders sub-agent base URL (also the endpoint published in its card) |
| `HTTP_TIMEOUT_SECONDS` | `10` | Timeout for each call between services |
| `DATABASE_URL` | `postgresql://shopmind:shopmind@localhost:5432/shopmind` | Checkpointer DB (R7) |
