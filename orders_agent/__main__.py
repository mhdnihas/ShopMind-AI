"""Run the orders sub-agent: `python -m orders_agent` (listens on $PORT, default 8001)."""

import logging
import os

import uvicorn

from common.settings import get_settings

if __name__ == "__main__":
    logging.basicConfig(level=get_settings().log_level)
    uvicorn.run(
        "orders_agent.server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8001")),
    )
