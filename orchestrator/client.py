"""R6 — A small terminal client for the chat API.

    python -m orchestrator.client --token tok_acme_alice

Type a message and watch the events arrive. When a proposal appears, answer a or r.
"""

from __future__ import annotations

import argparse
import json
import uuid
from collections.abc import Iterator
from typing import Any

import httpx


def parse_events(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Turn the lines of an SSE stream into (event name, data) pairs."""
    name = None
    for line in lines:
        if line.startswith("event:"):
            name = line.removeprefix("event:").strip()
        elif line.startswith("data:") and name:
            yield name, json.loads(line.removeprefix("data:"))
            name = None


def stream(
    http: httpx.Client, path: str, token: str, body: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Print one turn as it streams. Returns the proposal if a decision is needed."""
    headers = {"Authorization": f"Bearer {token}"}
    proposal = None
    with http.stream("POST", path, headers=headers, json=body) as response:
        if response.status_code != 200:
            response.read()
            print(f"[HTTP {response.status_code}] {response.json().get('detail')}")
            return None
        for name, data in parse_events(response.iter_lines()):
            if name == "token":
                print(data["text"], end="", flush=True)
            elif name == "tool_started":
                print(f"[running {data['tool']}]")
            elif name == "tool_finished":
                print(f"[finished {data['tool']}]")
            elif name == "proposal":
                print(f"\nPROPOSAL: {data['summary']}")
                proposal = data
            elif name == "error":
                print(f"\n[error] {data['message']}")
    print()
    return proposal


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the ShopHub assistant.")
    parser.add_argument("--token", required=True, help="Bearer token, e.g. tok_acme_alice")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--thread", default=uuid.uuid4().hex, help="Conversation id")
    args = parser.parse_args()

    print(f"Conversation: {args.thread}. Press Ctrl+C to quit.")
    with httpx.Client(base_url=args.url, timeout=None) as http:
        try:
            while True:
                message = input("\nyou> ").strip()
                if not message:
                    continue
                body = {"message": message, "thread_id": args.thread}
                proposal = stream(http, "/chat", args.token, body)
                while proposal:
                    answer = input("Approve or reject? [a/r] ").strip().lower()
                    decision = "approve" if answer == "a" else "reject"
                    proposal = stream(http, f"/chat/{args.thread}/{decision}", args.token)
        except (KeyboardInterrupt, EOFError):
            print()


if __name__ == "__main__":
    main()
