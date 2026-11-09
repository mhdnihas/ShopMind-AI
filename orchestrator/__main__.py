"""Run the chat API: `python -m orchestrator` (listens on $PORT, default 8000)."""

import asyncio
import logging
import os
import sys

import uvicorn

from common.settings import get_settings

if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level)
    server = uvicorn.Server(
        uvicorn.Config(
            "orchestrator.api:app",
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8000")),
        )
    )
    if sys.platform == "win32":
        # PostgreSQL's async driver cannot run on Windows' default event loop.
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve())
    else:
        server.run()
