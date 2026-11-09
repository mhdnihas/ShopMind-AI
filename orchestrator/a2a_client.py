"""How the orchestrator reaches remote agents: Google's A2A protocol, through a2a-sdk (R2).

The orchestrator never imports the orders sub-agent. It only knows a base URL:
1. `discover()` fetches the A2A agent card and learns the endpoint and skills.
2. `call()` sends an A2A message with the end user's bearer token, forwarded
   exactly as received (R3).

Protocol problems raise `A2AError`; business answers (ok=false) are returned as
data. The orchestrator's tool layer turns A2AError into a message for the model.
"""

from __future__ import annotations

from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.helpers import get_data_parts, new_data_part, new_message
from a2a.types import AgentCard as SdkAgentCard
from a2a.types import Role, SendMessageRequest
from a2a.utils.errors import A2AError as SdkError
from a2a.utils.errors import InvalidParamsError, MethodNotFoundError
from google.protobuf.json_format import MessageToDict
from pydantic import BaseModel, ValidationError

from common.a2a_contract import SKILL_EXTENSION_URI, restore_integers

CARD_PATH = "/.well-known/agent-card.json"
TRANSPORT_ERROR = -32000
INVALID_PARAMS = -32602
METHOD_NOT_FOUND = -32601


class SkillCard(BaseModel):
    id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    mutating: bool = False


class AgentCard(BaseModel):
    """The parts of an A2A agent card that the orchestrator uses."""

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


def _read_card(card: SdkAgentCard) -> AgentCard:
    extension = next(
        (e for e in card.capabilities.extensions if e.uri == SKILL_EXTENSION_URI), None
    )
    if extension is None:
        raise ValueError("the agent card does not publish skill schemas")
    published = restore_integers(MessageToDict(extension.params))["skills"]
    return AgentCard(
        name=card.name,
        description=card.description,
        version=card.version,
        url=card.supported_interfaces[0].url,
        skills=[
            SkillCard(
                id=skill.id, name=skill.name, description=skill.description, **published[skill.id]
            )
            for skill in card.skills
        ],
    )


def _error_from_sdk(error: SdkError) -> A2AError:
    if isinstance(error, InvalidParamsError):
        return A2AError(INVALID_PARAMS, error.message, error.data)
    if isinstance(error, MethodNotFoundError):
        return A2AError(METHOD_NOT_FOUND, error.message, error.data)
    return A2AError(TRANSPORT_ERROR, "The orders agent is unavailable")


class A2AClient:
    def __init__(
        self,
        card: AgentCard,
        sdk_card: SdkAgentCard,
        *,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.card = card
        self.sdk_card = sdk_card
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
        try:
            async with httpx.AsyncClient(timeout=timeout, transport=transport) as http:
                sdk_card = await A2ACardResolver(http, base_url.rstrip("/")).get_agent_card()
            card = _read_card(sdk_card)
        except (httpx.HTTPError, SdkError, ValueError, KeyError, IndexError, ValidationError) as e:
            url = base_url.rstrip("/") + CARD_PATH
            raise A2AError(TRANSPORT_ERROR, f"Agent discovery failed at {url}: {e}") from e
        return cls(card, sdk_card, timeout=timeout, transport=transport)

    async def call(self, skill_id: str, params: dict[str, Any], token: str) -> dict[str, Any]:
        """Invoke one skill. `token` is the end user's token, sent unchanged."""
        if self.card.skill(skill_id) is None:
            raise A2AError(METHOD_NOT_FOUND, f"Agent '{self.card.name}' has no skill '{skill_id}'")

        message = new_message(
            [new_data_part({"skill": skill_id, "params": params})], role=Role.ROLE_USER
        )
        reply = None
        try:
            async with httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"},
                timeout=self._timeout,
                transport=self._transport,
            ) as http:
                config = ClientConfig(httpx_client=http, streaming=False)
                client = ClientFactory(config).create(self.sdk_card)
                async for response in client.send_message(SendMessageRequest(message=message)):
                    if response.HasField("message"):
                        reply = response.message
        except (httpx.TimeoutException, A2AClientTimeoutError) as e:
            raise A2AError(TRANSPORT_ERROR, "The orders agent timed out") from e
        except (httpx.HTTPError, ValueError) as e:
            raise A2AError(TRANSPORT_ERROR, "The orders agent is unavailable") from e
        except A2AClientError as e:
            raise A2AError(TRANSPORT_ERROR, "The orders agent is unavailable") from e
        except SdkError as e:
            raise _error_from_sdk(e) from e

        parts = get_data_parts(reply.parts) if reply is not None else []
        result = restore_integers(parts[0]) if len(parts) == 1 else None
        if not isinstance(result, dict) or "ok" not in result:
            raise A2AError(TRANSPORT_ERROR, "Malformed A2A response")
        return result
