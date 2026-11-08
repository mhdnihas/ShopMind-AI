"""Client the orchestrator uses to reach remote agents (R2).

The orchestrator never imports the orders sub-agent. It only knows a base URL:
1. `discover()` fetches the agent card and learns the endpoint and skills.
2. `call()` sends a JSON-RPC 2.0 request with the end user's bearer token,
   forwarded exactly as received (R3).

Protocol problems raise `A2AError`; business answers (ok=false) are returned as
data. The orchestrator's tool layer turns A2AError into a message for the model.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

CARD_PATH = "/.well-known/agent-card.json"
TRANSPORT_ERROR = -32000


class SkillCard(BaseModel):
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    mutating: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    version: str
    url: str
    skills: list[SkillCard]

    def skill(self, skill_id: str) -> SkillCard | None:
        return next((s for s in self.skills if s.id == skill_id), None)


class A2AError(Exception):
    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.data = data


class A2AClient:
    def __init__(
        self,
        card: AgentCard,
        *,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.card = card
        self._timeout = timeout
        self._transport = transport

    @classmethod
    async def discover(
        cls,
        base_url: str,
        *,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> A2AClient:
        """Fetch and validate the agent card. The card is public: no token is sent."""
        url = base_url.rstrip("/") + CARD_PATH
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as http:
                resp = await http.get(url)
                resp.raise_for_status()
                card = AgentCard.model_validate(resp.json())
        except (httpx.HTTPError, ValueError, ValidationError) as e:
            raise A2AError(TRANSPORT_ERROR, f"Agent discovery failed at {url}: {e}") from e
        return cls(card, timeout=timeout, transport=transport)

    async def call(self, skill_id: str, params: dict[str, Any], token: str) -> dict[str, Any]:
        """Invoke one skill. `token` is the end user's token, sent unchanged."""
        if self.card.skill(skill_id) is None:
            raise A2AError(-32601, f"Agent '{self.card.name}' has no skill '{skill_id}'")

        req_id = str(uuid.uuid4())
        body = {"jsonrpc": "2.0", "id": req_id, "method": skill_id, "params": params}
        try:
            async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as http:
                resp = await http.post(
                    self.card.url, json=body, headers={"Authorization": f"Bearer {token}"}
                )
                payload = resp.json()
        except httpx.TimeoutException as e:
            raise A2AError(TRANSPORT_ERROR, "The orders agent timed out") from e
        except (httpx.HTTPError, ValueError) as e:
            raise A2AError(TRANSPORT_ERROR, "The orders agent is unavailable") from e

        if not isinstance(payload, dict):
            raise A2AError(TRANSPORT_ERROR, "Malformed JSON-RPC response")
        # Checked before the id: some errors (e.g. parse errors) legitimately carry id=null.
        if isinstance(err := payload.get("error"), dict):
            raise A2AError(
                err.get("code", TRANSPORT_ERROR), err.get("message", ""), err.get("data")
            )
        if payload.get("id") != req_id or not isinstance(payload.get("result"), dict):
            raise A2AError(TRANSPORT_ERROR, "Malformed JSON-RPC response")
        return payload["result"]
