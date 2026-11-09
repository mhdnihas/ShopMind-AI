"""R7 — conversations and waiting proposals survive a restart, in real PostgreSQL.

These need a PostgreSQL server, so they skip themselves when there is none. Set
TEST_DATABASE_URL to use a specific server; otherwise DATABASE_URL is used.
"""

import os
import uuid

import httpx
import psycopg
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from common.settings import get_settings
from orchestrator.a2a_client import A2AClient
from orchestrator.api import create_app
from orchestrator.client import parse_events
from orchestrator.fake_llm import ScriptedChatModel
from orchestrator.graph import APPROVE, build_graph
from orchestrator.persistence import CheckpointStoreError, open_checkpointer
from orchestrator.tools import RunContext
from orders_agent.server import create_app as create_agent
from store_api.main import create_app as create_store

AGENT = "http://agent.test"
STORE = "http://store.test"
ACME = "tok_acme_alice"
DB_URL = os.getenv("TEST_DATABASE_URL") or get_settings().database_url


@pytest.fixture
async def database():
    try:
        connection = await psycopg.AsyncConnection.connect(DB_URL, connect_timeout=3)
    except psycopg.Error:
        pytest.skip("PostgreSQL is not available")
    await connection.close()
    return DB_URL


@pytest.fixture
async def thread(database):
    thread_id = uuid.uuid4().hex
    yield thread_id
    async with open_checkpointer(database) as saver:
        await saver.adelete_thread(thread_id)
        await saver.adelete_thread(f"m_acme:{thread_id}")


@pytest.fixture
def store():
    return httpx.ASGITransport(app=create_store())


@pytest.fixture
async def a2a(store):
    app = create_agent(store_url=STORE, public_url=AGENT, store_transport=store)
    return await A2AClient.discover(AGENT, transport=httpx.ASGITransport(app=app))


def config_for(thread_id):
    return {"configurable": {"thread_id": thread_id}}


async def ask(graph, thread_id, text):
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=text)]},
        config_for(thread_id),
        context=RunContext(auth_token=ACME),
    )


async def status_of(a2a, order_id):
    return (await a2a.call("get_order", {"order_id": order_id}, ACME))["data"]["status"]


# --- surviving a restart -------------------------------------------------------------


async def test_waiting_proposal_can_be_approved_after_a_restart(database, thread, a2a):
    async with open_checkpointer(database) as saver:
        first = build_graph(ScriptedChatModel(), a2a, saver)
        paused = await ask(first, thread, "Please refund order 1042")
    assert paused["__interrupt__"][0].value["order_id"] == 1042
    assert await status_of(a2a, 1042) == "PAID"

    # Everything from before is gone: a new connection pool and a new graph.
    async with open_checkpointer(database) as saver:
        after_restart = build_graph(ScriptedChatModel(), a2a, saver)
        assert (await after_restart.aget_state(config_for(thread))).next == ("approval",)
        result = await after_restart.ainvoke(
            Command(resume=APPROVE), config_for(thread), context=RunContext(auth_token=ACME)
        )

    assert "is now REFUNDED" in result["messages"][-1].content
    assert await status_of(a2a, 1042) == "REFUNDED"


async def test_conversation_history_survives_a_restart(database, thread, a2a):
    async with open_checkpointer(database) as saver:
        await ask(build_graph(ScriptedChatModel(), a2a, saver), thread, "Hello there")

    async with open_checkpointer(database) as saver:
        result = await ask(build_graph(ScriptedChatModel(), a2a, saver), thread, "Show me order 1")

    assert [m.type for m in result["messages"][:3]] == ["human", "ai", "human"]
    assert result["messages"][0].content == "Hello there"


async def test_approve_over_http_after_the_api_restarts(database, thread, a2a, store):
    def api(saver):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(
                app=create_app(
                    graph=build_graph(ScriptedChatModel(), a2a, saver),
                    store_url=STORE,
                    store_transport=store,
                )
            ),
            base_url="http://chat.test",
        )

    headers = {"Authorization": f"Bearer {ACME}"}
    body = {"message": "Please refund order 1042", "thread_id": thread}
    async with open_checkpointer(database) as saver:
        response = await api(saver).post("/chat", json=body, headers=headers)
    assert list(parse_events(iter(response.text.splitlines())))[-1][0] == "proposal"

    async with open_checkpointer(database) as saver:
        http = api(saver)
        # The waiting proposal still blocks a new message after the restart.
        again = await http.post("/chat", json=body, headers=headers)
        assert again.status_code == 409
        response = await http.post(f"/chat/{thread}/approve", headers=headers)
    assert "is now REFUNDED" in response.text


# --- what is stored --------------------------------------------------------------------


async def stored_count(connection, thread_id, needle):
    text = f"%{needle}%"
    data = needle.encode()
    query = """
        select
          (select count(*) from checkpoints where thread_id = %(t)s
             and (checkpoint::text like %(text)s or metadata::text like %(text)s))
        + (select count(*) from checkpoint_blobs where thread_id = %(t)s
             and position(%(data)s::bytea in coalesce(blob, ''::bytea)) > 0)
        + (select count(*) from checkpoint_writes where thread_id = %(t)s
             and position(%(data)s::bytea in blob) > 0)
    """
    cursor = await connection.execute(query, {"t": thread_id, "text": text, "data": data})
    return (await cursor.fetchone())[0]


async def test_token_is_never_written_to_the_database(database, thread, a2a):
    async with open_checkpointer(database) as saver:
        graph = build_graph(ScriptedChatModel(), a2a, saver)
        await ask(graph, thread, "Please refund order 1042")
        await graph.ainvoke(
            Command(resume=APPROVE), config_for(thread), context=RunContext(auth_token=ACME)
        )

    async with await psycopg.AsyncConnection.connect(database) as connection:
        # Control: proves the search can find text that really is stored.
        assert await stored_count(connection, thread, "Maria Lopez") > 0
        assert await stored_count(connection, thread, ACME) == 0


async def test_customer_contact_details_are_never_written_to_the_database(database, thread, a2a):
    async with open_checkpointer(database) as saver:
        await ask(build_graph(ScriptedChatModel(), a2a, saver), thread, "Show me order 1042")

    async with await psycopg.AsyncConnection.connect(database) as connection:
        assert await stored_count(connection, thread, "Maria Lopez") > 0
        assert await stored_count(connection, thread, "maria.lopez@example.com") == 0
        assert await stored_count(connection, thread, "612 345 678") == 0


# --- startup ----------------------------------------------------------------------------


async def test_unreachable_database_stops_startup_without_leaking_the_url():
    with pytest.raises(CheckpointStoreError) as error:
        async with open_checkpointer("postgresql://nobody:hunter2@127.0.0.1:1/x", timeout=1):
            pass
    assert "hunter2" not in str(error.value)
    assert error.value.__suppress_context__


async def test_empty_url_explicitly_uses_memory():
    async with open_checkpointer("") as saver:
        assert isinstance(saver, InMemorySaver)
