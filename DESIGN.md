# ShopMind AI: Design

A multi-tenant order-support assistant for ShopHub merchant staff.

**Built with an AI coding assistant.** I used Claude Code to design and write this project
phase by phase, and reviewed each phase so I can explain and change every line.

**Not finished:** Docker (R10) is written but has never run. Docker Desktop could not start on my
machine because WSL was missing. The compose file validates and the lock file was resolved for
Linux, but no image has been built, and the code has only ever run on Windows. What remains is
one `docker compose up` from a clean clone, plus fixing whatever that reveals.

## Architecture

```text
 Browser / terminal client
        | HTTP + SSE, Authorization: Bearer <user token>
        v
 +-------------------------------+         +------------------+
 | Chat API + orchestrator :8000 |-------->| PostgreSQL       |
 | - /auth/me check, fail closed |         | graph checkpoints|
 | - thread = merchant:thread_id |         | (never the token)|
 | - LangGraph agent/tools/      |         +------------------+
 |   approval                    |
 +-------------------------------+-------> Gemini / scripted fake
        | Google A2A 1.0 (a2a-sdk), JSON-RPC, same bearer token
        v
 +-------------------------------+
 | Orders sub-agent :8001        |
 | - agent card, skills          |
 | - PII redaction, injection    |
 +-------------------------------+
        | HTTP, same bearer token
        v
 +-------------------------------+
 | Mock store API :8002          |
 | - tenancy enforced in data    |
 +-------------------------------+
```

## One turn

Example: "Please refund order 1042".

1. `POST /chat` arrives. The API calls the store's `/auth/me`. No token gives 401, and a token
   with no merchant gives 403 before the model runs.
2. The thread key becomes `merchant:thread_id`. The graph runs with the token in LangGraph's
   runtime context, never in state.
3. The `agent` node calls the model, which sees only read-only tools and picks `preview_action`.
4. The `tools` node sends an A2A `SendMessage` carrying `{"skill", "params"}` and the user's token.
5. The sub-agent reads the order from the store, redacts it, and builds a proposal. Output is
   wrapped in `<untrusted_data>` tags before the model can see it.
6. A successful preview routes to the `approval` node, which calls `interrupt()`. State is saved
   and a `proposal` SSE event ends the stream. Nothing has changed yet.
7. Later, possibly after a restart, `POST /chat/{thread}/approve` resumes the graph with the
   approver's own token. Code calls `refund_order`, the store applies it, and code writes the
   outcome message. The model is not involved.

## Key trade-offs

**A2A with immediate replies, not the Task lifecycle.** The mail made Google's A2A mandatory, so
both sides use the official SDK. The sub-agent is deterministic and each skill finishes at once,
so it replies with an A2A `Message`. Standard skills have no field for argument schemas or "this
changes data", so the card publishes them through an A2A extension. The cost is that a generic
A2A client would need to know our `{"skill", "params"}` convention.

**The model cannot change anything.** Mutating skills are never turned into tools, so injection
cannot reach a refund even if every other layer fails. Weakness: the sub-agent trusts the
orchestrator for approval, so anyone holding a valid token could call `refund_order` directly.
Production needs a signed, single-use approval token checked by the sub-agent.

**The token lives in runtime context.** My first plan used `config["configurable"]`. I tested it
and found LangGraph copies those values into checkpoint metadata, which would have stored every
token in Postgres. A test now searches the real tables for it.

**Injection defence is layered and imperfect.** Notes that look like instructions are withheld
by regex, tool output is marked as data, and refunds need a human. Regex misses paraphrases and
can withhold harmless notes. It is one layer, not the defence.

**Failures are data.** The legacy `200 + status:false` and sub-agent outages become results the
model explains, instead of crashes or false success.

**The fake model is deliberately gullible.** It obeys injection text if any reaches it, so a
guardrail failure fails a test. It proves the system around the model, not the model's judgement.

**Other choices.** Date arithmetic runs in code because models are unreliable at it. Money uses
`float` in the mock; production needs `Decimal`. The double-click guard is an in-process lock, so
with several replicas the store's atomic check is what prevents a second refund.

## What I cut, and why

| Cut | Why |
|---|---|
| Docker verification | See the top of this document |
| Stretch goals S1 to S4 | R1 to R11 came first, and the brief says a finished small thing beats a large broken one |
| Sub-agent streaming and A2A Tasks | Skills finish immediately; token streaming happens at the chat API |
| Reloading a conversation after a page refresh | R7 is about process restarts, and a web UI is not graded |
| Durable store data | The brief only asks for a mock |
| Audit log of approvals | Needed in production, not asked for here |

The small web page is a demo convenience, not a requirement.

Known limits I left in:

- A disconnect in the middle of an approval is untested.
- If a preview and another tool arrive in the same reply, the other result is not summarized.
- A crash just after a refund but before the save would rerun the node. The store should then
  return 409 and the user would see "could not confirm". I reasoned this through but did not test it.

## Rough cost and latency per turn

Measured on Gemini 3.5 Flash-Lite (paid tier: $0.30 input, $2.50 output per million tokens):

| Turn | Model calls | Input tokens | Output tokens | Cost | Graph time |
|---|---|---|---|---|---|
| Read one order | 2 | 1,370 | 156 | about $0.0008 | 3.8 s |
| Search orders | 2 | 1,702 | 295 | about $0.0013 | 3.4 s |

So a turn costs roughly **$0.001 and 4 to 8 seconds** through the chat API. Cost grows with the
conversation, because the full history is sent each time. A proposal needs one model call, and an
approval needs none. A slower model such as Gemini 3.6 Flash took 5 to 7 seconds in the graph
alone.

Part of the latency is mine: each service call creates a new HTTP client, about 0.75 s each on
this Windows machine. Shared clients would remove most of that. The free tier is limited to 20
requests per model per day, and Google may use free-tier content to improve its products, so
production needs the paid tier.

## Testing

336 tests run offline with only the model scripted, using the real store and sub-agent in-process.
Five need PostgreSQL and one needs the Gemini package; those skip cleanly. A fresh install with no
key, `.env` or database passes. I also injected 65 bugs on purpose to check that tests fail when
the code is wrong, which found real gaps that I then closed.

All 17 eval cases pass on the fake model. The 15 that reach a model passed on real Gemini before
the move to the A2A SDK, and I re-ran only 2 of them afterwards because of free-tier limits.
Injection results show the guardrail works, not that Gemini would resist unaided.

## With another week

1. Run Docker from a clean clone on Linux, and add CI that does the same.
2. Signed, single-use approval tokens enforced by the sub-agent.
3. A correlation ID from the chat API through A2A to the store, with structured logs.
4. Per-merchant rate limits and per-turn token accounting.
5. Shared HTTP clients, and a better timeout for models that hang.
6. Reliable concurrency across replicas, using a Postgres advisory lock per thread.
7. `Decimal` money and an audit log of who approved what.
8. Adversarial evals against real models with guardrails disabled, to measure the model itself.
