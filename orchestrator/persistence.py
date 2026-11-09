"""R7 — Where conversations are saved, so they survive a restart.

A paused refund is just saved graph state, so with PostgreSQL behind the graph a person
can approve it after the chat API has restarted.

If DATABASE_URL is set but PostgreSQL cannot be reached, startup fails. Quietly falling
back to memory would lose conversations with no warning. Leaving DATABASE_URL empty is
the explicit way to choose memory, for quick local runs.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

log = logging.getLogger("orchestrator")

POOL_MAX_SIZE = 10
CONNECT_TIMEOUT_SECONDS = 10.0


class CheckpointStoreError(RuntimeError):
    """PostgreSQL could not be used. Raised at startup, not mid-conversation."""


@asynccontextmanager
async def open_checkpointer(
    database_url: str, *, timeout: float = CONNECT_TIMEOUT_SECONDS
) -> AsyncIterator[BaseCheckpointSaver]:
    if not database_url:
        log.warning("DATABASE_URL is empty: conversations are kept in memory only.")
        yield InMemorySaver()
        return

    pool = AsyncConnectionPool(
        conninfo=database_url,
        max_size=POOL_MAX_SIZE,
        # All three are what the LangGraph saver requires of its connections.
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    )
    try:
        await pool.open(wait=True, timeout=timeout)
        saver = AsyncPostgresSaver(pool)
        await saver.setup()  # creates or upgrades the tables
    except psycopg.Error:
        await pool.close()
        # "from None": the original error can include the connection details.
        raise CheckpointStoreError(
            "Could not use PostgreSQL. Check that it is running and that DATABASE_URL is correct."
        ) from None

    try:
        yield saver
    finally:
        await pool.close()
