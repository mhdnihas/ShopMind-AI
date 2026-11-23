# Architecture

Trade-offs and reasoning are in [DESIGN.md](DESIGN.md).

## Minimal architecture

```mermaid
flowchart TD
    card["BEFORE ANY CHAT<br/>Orchestrator reads the sub-agent's agent card with the A2A SDK<br/>and learns its skills: list orders, get order, refund, cancel"]

    s1["Step 1: User asks a question<br/>and sends a bearer token"]
    s2["Step 2: Orchestrator receives it"]
    s3["Step 3: LLM decides which skill to use"]
    d{"Is it a refund<br/>or a cancel?"}
    ok["A person must click Approve"]
    s4["Step 4: Orchestrator calls the skill<br/>on the sub-agent, using the A2A SDK<br/>and the same bearer token"]
    s5["Step 5: Sub-agent calls the Store API<br/>with the same bearer token"]
    s6["Step 6: Store API returns only<br/>that merchant's orders"]
    s7["Step 7: Sub-agent runs guardrails<br/>hides email and phone, removes injected notes"]
    s8["Step 8: Safe result goes back to the orchestrator<br/>and the LLM writes the answer"]
    s9["Step 9: User sees the answer"]

    card -.-> s3
    s1 --> s2 --> s3 --> d
    d -->|"No"| s4
    d -->|"Yes"| ok --> s4
    s4 --> s5 --> s6 --> s7 --> s8 --> s9

    classDef person fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef orchestrator fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef subagent fill:#fef3c7,stroke:#d97706,color:#111827
    classDef store fill:#f3e8ff,stroke:#9333ea,color:#111827

    class s1,s9,ok person
    class card,s2,s3,d,s4,s8 orchestrator
    class s5,s7 subagent
    class s6 store
```

Colours: blue is the person, green is the orchestrator, yellow is the sub-agent, purple is the
store.

## Overall architecture

```mermaid
flowchart TB
    user["Merchant staff<br/>web page or terminal client"]
    chat["Chat API and LangGraph orchestrator<br/>port 8000"]
    llm["Gemini, or the scripted fake model"]
    db[("PostgreSQL<br/>saved conversations")]
    agent["Orders sub-agent<br/>port 8001"]
    store["Mock store API<br/>port 8002"]

    user -->|"HTTP and SSE<br/>user's bearer token"| chat
    chat -->|"prompt and tools<br/>never the token"| llm
    chat -->|"checkpoints<br/>never the token"| db
    chat -->|"Google A2A 1.0<br/>same bearer token"| agent
    agent -->|"HTTP<br/>same bearer token"| store
```

## Full architecture

```mermaid
flowchart TB
    subgraph clients["Clients"]
        web["Web page<br/>orchestrator/static"]
        cli["Terminal client<br/>orchestrator/client.py"]
    end

    subgraph chatproc["Chat API and orchestrator, port 8000"]
        direction TB

        subgraph api["FastAPI, orchestrator/api.py"]
            routes["POST /chat<br/>POST /chat/id/approve<br/>POST /chat/id/reject<br/>GET /models"]
            auth["authenticate<br/>401, 403 or 503 before the model runs"]
            scope["Thread key: merchant:thread_id<br/>409 if the proposal state is wrong"]
            lock["Per-conversation lock<br/>guards double clicks"]
            sse["SSE events: token, tool_started,<br/>tool_finished, proposal, done, error"]
        end

        subgraph langgraph["LangGraph, orchestrator/graph.py"]
            start((START))
            agentnode["agent node<br/>system prompt + model"]
            toolsnode["tools node<br/>read-only tools only"]
            approval["approval node<br/>interrupt, waits for a person"]
            finish((END))
        end

        context["RunContext<br/>token, never saved"]
        toolsfile["orchestrator/tools.py<br/>tools built from the agent card<br/>output marked untrusted"]
        factory["get_chat_model<br/>orchestrator/llm.py"]
        fake["ScriptedChatModel<br/>default, offline"]
        a2aclient["A2AClient, a2a-sdk<br/>orchestrator/a2a_client.py"]
        persist["open_checkpointer<br/>orchestrator/persistence.py"]
    end

    gemini["Google Gemini API"]
    db[("PostgreSQL<br/>checkpoints")]

    subgraph agentproc["Orders sub-agent, port 8001"]
        direction TB
        guardmw["Middleware<br/>401 without a token<br/>JSON-RPC notifications not run"]
        card["GET /.well-known/agent-card.json<br/>A2A card + skill schemas extension"]
        handler["POST / SendMessage<br/>a2a-sdk DefaultRequestHandler"]
        executor["OrdersExecutor<br/>validates skill and params"]
        skills["Skills<br/>list_orders, get_order, preview_action<br/>refund_order, cancel_order"]
        storeclient["StoreClient<br/>handles legacy 200 + status:false<br/>errors returned as data"]
        guardrails["Guardrails<br/>redact email and phone<br/>withhold injected notes"]
    end

    subgraph storeproc["Mock store API, port 8002"]
        direction TB
        storeauth["Bearer token to merchant"]
        endpoints["/auth/me, /orders<br/>/orders/lookup legacy<br/>/orders/refund, /orders/cancel"]
        data[("In-memory orders<br/>scoped to merchant<br/>locked state changes")]
    end

    web -->|"fetch + SSE"| routes
    cli -->|"HTTP + SSE"| routes
    routes --> auth
    auth -->|"GET /auth/me"| storeauth
    auth --> scope --> lock --> start
    sse -.->|"stream"| web
    sse -.->|"stream"| cli

    start --> agentnode
    agentnode -->|"tool calls"| toolsnode
    agentnode -->|"text answer"| finish
    toolsnode -->|"tool results"| agentnode
    toolsnode -->|"successful preview"| approval
    approval -->|"approve or reject"| finish
    langgraph -.-> sse

    agentnode --> factory
    factory --> fake
    factory --> gemini
    toolsnode --> toolsfile --> a2aclient
    approval -->|"approved: refund or cancel"| a2aclient
    context -.->|"token"| a2aclient
    langgraph <-->|"save and resume"| persist
    persist --> db

    a2aclient -->|"discovers skills at startup"| card
    a2aclient -->|"A2A SendMessage<br/>user's bearer token"| guardmw
    guardmw --> handler --> executor --> skills
    skills --> storeclient
    storeclient -->|"same bearer token"| storeauth
    storeclient --> guardrails
    storeauth --> endpoints --> data
```

Three rules the diagram shows:

- **Only the approval node can change an order.** The model's tools are read-only, and refunds
  and cancels run only after a person approves.
- **The bearer token is passed along and never stored.** It travels from the client to the store
  unchanged, and it is not given to the model or saved in PostgreSQL.
- **The store enforces tenancy.** Every order lookup and change is scoped to the token's merchant.
