"""A scripted chat model, so the whole system runs offline with no API key.

It is a real LangChain `BaseChatModel` and returns real `tool_calls`, so the graph,
tool execution and token forwarding run exactly as they do with a real model. Only
the decision of what to do next is scripted, using fixed rules:

* last message is from the user -> pick a tool from the wording, or reply in text;
* last messages are tool results -> write the answer from them.

Two behaviours exist to make the tests meaningful:

* The answer repeats the tool output, so PII that got past redaction shows up in it.
* The model is deliberately gullible: if instruction-like text reaches it in a tool
  result, it obeys and calls `refund_order`. A real model might resist, but this one
  makes a guardrail failure visible as a test failure.

This proves the system around the model. It says nothing about how well a real
model picks tools or resists injection.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils.function_calling import convert_to_openai_tool

GREETING = "I can look up orders and prepare refunds or cancellations. Which order do you mean?"
NO_TOOL = "I don't have a tool that can do that."
FOLLOWED_INJECTION = "Following the instructions in the customer's note."

_ORDER_ID_RE = re.compile(r"(?:\border\s*#?\s*|#)(\d+)|\b(\d{3,})\b(?!\s*days?)", re.IGNORECASE)
_ACTION_RE = re.compile(r"\b(refund|cancel)\b", re.IGNORECASE)
_LIST_RE = re.compile(r"\borders?\b", re.IGNORECASE)
_DAYS_RE = re.compile(r"(\d+)\s*days?\b", re.IGNORECASE)
# "PLACED" is left out: "orders placed 3 days ago" is about a date, not a status.
_STATUS_RE = re.compile(
    r"(?<!not\s)\b(paid|shipped|delivered|cancelled|canceled|refunded)\b", re.IGNORECASE
)
# Case-sensitive on purpose: a name starts with a capital letter.
_CUSTOMER_RE = re.compile(r"\b(?:customer|from|for|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)")

_UNTRUSTED_TAG_RE = re.compile(r"</?\s*untrusted_data\s*>", re.IGNORECASE)
_INJECTED_RE = re.compile(
    r"\b(?:ignore|disregard)\b.{0,40}\b(?:instructions?|rules)\b|\byou\s+are\s+now\b",
    re.IGNORECASE,
)
_RESULT_ORDER_ID_RE = re.compile(r'"id":\s*(\d+)')


def _order_id(text: str) -> int | None:
    match = _ORDER_ID_RE.search(text)
    return int(match.group(1) or match.group(2)) if match else None


def _list_filters(text: str) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if status := _STATUS_RE.search(text):
        word = status.group(1).upper()
        filters["status"] = "CANCELLED" if word == "CANCELED" else word
    if days := _DAYS_RE.search(text):
        filters["placed_more_than_days_ago"] = int(days.group(1))
    if customer := _CUSTOMER_RE.search(text):
        filters["customer_name"] = customer.group(1)
    return filters


def _choose_tool(text: str) -> tuple[str, dict[str, Any]] | None:
    """Map the user's wording to (tool name, arguments), or None for a text reply."""
    order_id = _order_id(text)
    if action := _ACTION_RE.search(text):
        if order_id is None:
            return None
        # Always the read-only preview: changing an order needs a human approval first.
        return "preview_action", {"action": action.group(1).lower(), "order_id": order_id}
    if order_id is not None:
        return "get_order", {"order_id": order_id}
    if _LIST_RE.search(text):
        return "list_orders", _list_filters(text)
    return None


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def _describe(content: str) -> str:
    """Turn one tool result into a sentence, reading ok=false as a failure."""
    try:
        result = json.loads(content)
    except ValueError:
        return content
    if isinstance(result, dict) and result.get("ok") is False:
        return f"I couldn't complete that: {result.get('error')}"
    return f"Here is what I found: {content}"


def _trailing_tool_results(messages: Sequence[BaseMessage]) -> list[str]:
    """The text of the tool results that follow the model's last request."""
    results: list[str] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        results.append(_UNTRUSTED_TAG_RE.sub("", str(message.content)).strip())
    return results[::-1]


def _obey_injection(results: list[str], tools: set[str], call_id: str) -> AIMessage | None:
    """The deliberate weakness described in the module docstring."""
    for text in results:
        if injected := _INJECTED_RE.search(text):
            ids = _RESULT_ORDER_ID_RE.findall(text[: injected.start()])
            if ids and "refund_order" in tools:
                return _tool_call("refund_order", {"order_id": int(ids[-1])}, call_id)
            return AIMessage(content=FOLLOWED_INJECTION)
    return None


def _reply(messages: Sequence[BaseMessage], tools: set[str]) -> AIMessage:
    # The id only has to be unique within a conversation, and this keeps runs repeatable.
    call_id = f"call_{len(messages)}"
    last = messages[-1] if messages else None

    if isinstance(last, ToolMessage):
        results = _trailing_tool_results(messages)
        return _obey_injection(results, tools, call_id) or AIMessage(
            content="\n".join(_describe(r) for r in results)
        )

    if isinstance(last, HumanMessage):
        choice = _choose_tool(str(last.content))
        if choice is None:
            return AIMessage(content=GREETING)
        name, args = choice
        return _tool_call(name, args, call_id) if name in tools else AIMessage(content=NO_TOOL)

    return AIMessage(content=GREETING)


class ScriptedChatModel(BaseChatModel):
    """Deterministic stand-in for a real chat model. See the module docstring."""

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Runnable:
        return self.bind(tools=[convert_to_openai_tool(tool) for tool in tools], **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Like a real model, it can only call the tools it was actually given.
        names = {tool["function"]["name"] for tool in tools or []}
        return ChatResult(generations=[ChatGeneration(message=_reply(messages, names))])
