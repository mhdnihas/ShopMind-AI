"""LangChain tools built from the sub-agent's agent card (R1, R2).

The orchestrator never imports the sub-agent. Each tool's name, description and
argument schema come from the card, and calling it sends JSON-RPC over HTTP.

Skills marked `mutating` are not turned into tools, so the model has no way to
refund or cancel an order, however it is persuaded. They run only after a human
approves (R5).

The user's token comes from LangGraph's run context (R3), for two reasons:
* it is not a tool argument, so the model can neither see nor change it;
* it is not saved. Values in `config["configurable"]` are copied into checkpoint
  metadata, which would put every user's token in the database.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.runtime import get_runtime

from common.untrusted import wrap_untrusted
from orchestrator.a2a_client import A2AClient, A2AError, SkillCard


@dataclass(frozen=True)
class RunContext:
    """Per-request values that must never enter the graph state."""

    auth_token: str = field(repr=False)  # repr=False keeps it out of logs and tracebacks


def _failure(message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "error": message}


def _make_tool(client: A2AClient, skill: SkillCard) -> StructuredTool:
    async def call_skill(**params: Any) -> str:
        context = get_runtime(RunContext).context
        if context is None or not context.auth_token:
            # Fail closed: no token means no call at all.
            result = _failure("This request is not authenticated.")
        else:
            try:
                result = await client.call(skill.id, params, context.auth_token)
            except A2AError as e:
                # Reported as data, so the model can tell the user what went wrong.
                result = _failure(f"{e.message}: {e.data}" if e.data else e.message)
        return wrap_untrusted(json.dumps(result))

    return StructuredTool.from_function(
        coroutine=call_skill,
        name=skill.id,
        description=skill.description,
        args_schema=skill.input_schema,
    )


def build_tools(client: A2AClient) -> list[StructuredTool]:
    """One tool per read-only skill in the card."""
    return [_make_tool(client, skill) for skill in client.card.skills if not skill.mutating]
