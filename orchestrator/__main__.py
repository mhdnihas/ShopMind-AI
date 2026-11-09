"""Run the chat API: `python -m orchestrator` (listens on $PORT, default 8000)."""

import logging
import os

import uvicorn

from common.settings import get_settings

if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level)
    uvicorn.run(
        "orchestrator.api:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
    )
