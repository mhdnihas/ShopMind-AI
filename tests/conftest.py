"""Shared pytest configuration."""

import asyncio
import sys


def pytest_asyncio_loop_factories(config, item):
    # PostgreSQL's async driver cannot run on Windows' default event loop, and this is
    # also the loop the chat API runs on there.
    if sys.platform == "win32":
        return {"selector": asyncio.SelectorEventLoop}
    return None
