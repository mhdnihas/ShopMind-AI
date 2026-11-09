"""LLM factory and the scripted fake model. No API key and no network."""

import json
import sys

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from common.settings import Settings
from common.untrusted import wrap_untrusted
from orchestrator.fake_llm import FOLLOWED_INJECTION, GREETING, NO_TOOL, ScriptedChatModel
from orchestrator.llm import LLMConfigError, get_chat_model
from orders_agent.guardrails import NOTE_WITHHELD, sanitize_order
from orders_agent.skills import SKILLS
from store_api.data import INJECTION_NOTES


def settings(**overrides) -> Settings:
    """Settings that ignore the developer's own .env file."""
    return Settings(_env_file=None, **{"google_api_key": None, **overrides})


def make_tool(skill) -> StructuredTool:
    return StructuredTool.from_function(
        func=lambda **_: "",
        name=skill.id,
        description=skill.description,
        args_schema=skill.params,
    )


TOOLS = [make_tool(skill) for skill in SKILLS.values()]


@pytest.fixture
def model():
    return ScriptedChatModel().bind_tools(TOOLS)


def after_tool(content: str) -> list:
    """A conversation that has just received one tool result."""
    return [
        HumanMessage(content="Show me order 1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "get_order", "args": {}, "id": "c1", "type": "tool_call"}],
        ),
        ToolMessage(content=content, tool_call_id="c1"),
    ]


# --- factory ------------------------------------------------------------------


def test_fake_is_the_default_and_needs_no_key():
    chat = get_chat_model(settings())
    assert isinstance(chat, ScriptedChatModel)
    assert isinstance(chat, BaseChatModel)


@pytest.mark.parametrize("key", [None, ""])
def test_google_without_a_key_fails_clearly(key):
    with pytest.raises(LLMConfigError, match="GOOGLE_API_KEY"):
        get_chat_model(settings(llm_provider="google", google_api_key=key))


def test_google_without_the_package_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)  # makes the import fail
    with pytest.raises(LLMConfigError, match="pip install"):
        get_chat_model(settings(llm_provider="google", google_api_key="not-a-real-key"))


def test_google_model_is_configured_from_settings():
    pytest.importorskip("langchain_google_genai")
    chat = get_chat_model(
        settings(llm_provider="google", google_api_key="not-a-real-key", google_model="m-1")
    )
    assert chat.model.endswith("m-1")
    assert chat.google_api_key.get_secret_value() == "not-a-real-key"
    assert chat.temperature == 0.0
    assert chat.timeout == 60.0
    assert chat.max_retries == 2
    assert chat.max_output_tokens == 16000
    assert "not-a-real-key" not in repr(chat)
    chat.bind_tools(TOOLS)  # the real tool schemas are accepted without a network call


# --- choosing a tool -----------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "tool", "args"),
    [
        ("Show me order 1042", "get_order", {"order_id": 1042}),
        ("What is the status of #2001?", "get_order", {"order_id": 2001}),
        ("Please refund order 1042", "preview_action", {"action": "refund", "order_id": 1042}),
        ("cancel 1005", "preview_action", {"action": "cancel", "order_id": 1005}),
        ("List all orders", "list_orders", {}),
        ("Any refunded orders?", "list_orders", {"status": "REFUNDED"}),
        (
            "Which orders are paid but not shipped for 3 days?",
            "list_orders",
            {"status": "PAID", "placed_more_than_days_ago": 3},
        ),
        ("Orders older than 300 days", "list_orders", {"placed_more_than_days_ago": 300}),
        ("Show me orders for Maria Lopez", "list_orders", {"customer_name": "Maria Lopez"}),
    ],
)
async def test_picks_the_right_tool_with_valid_arguments(model, text, tool, args):
    reply = await model.ainvoke([SystemMessage(content="prompt"), HumanMessage(content=text)])
    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert (call["name"], call["args"]) == (tool, args)
    SKILLS[tool].params.model_validate(call["args"])  # the sub-agent would accept these


@pytest.mark.parametrize("text", ["refund 1042", "cancel order 1042"])
async def test_never_changes_an_order_when_asked_directly(model, text):
    reply = await model.ainvoke([HumanMessage(content=text)])
    assert reply.tool_calls[0]["name"] == "preview_action"


@pytest.mark.parametrize("text", ["Hello there", "Please refund it"])
async def test_replies_in_text_when_no_tool_applies(model, text):
    reply = await model.ainvoke([HumanMessage(content=text)])
    assert reply.tool_calls == []
    assert reply.content == GREETING


async def test_cannot_call_a_tool_it_was_not_given():
    reply = await ScriptedChatModel().ainvoke([HumanMessage(content="Show me order 1042")])
    assert reply.tool_calls == []
    assert reply.content == NO_TOOL


async def test_same_input_gives_the_same_output(model):
    messages = [HumanMessage(content="Show me order 1042")]
    assert (await model.ainvoke(messages)).tool_calls == (await model.ainvoke(messages)).tool_calls


# --- answering from tool results -----------------------------------------------


async def test_answer_uses_the_tool_result(model):
    content = json.dumps({"ok": True, "data": {"customer_name": "Maria Lopez"}, "error": None})
    reply = await model.ainvoke(after_tool(content))
    assert reply.tool_calls == []
    assert "Maria Lopez" in reply.content


async def test_failure_is_reported_as_a_failure(model):
    content = json.dumps({"ok": False, "data": None, "error": "Order 2001 was not found."})
    reply = await model.ainvoke(after_tool(content))
    assert reply.content == "I couldn't complete that: Order 2001 was not found."


async def test_plain_text_tool_result_is_passed_through(model):
    reply = await model.ainvoke(after_tool("The orders agent timed out"))
    assert reply.content == "The orders agent timed out"


async def test_untrusted_tags_are_not_repeated(model):
    content = wrap_untrusted(json.dumps({"ok": True, "data": {"id": 1}, "error": None}))
    reply = await model.ainvoke(after_tool(content))
    assert "untrusted_data" not in reply.content


# --- the deliberate weakness: why the guardrail tests mean something ------------


def order_json(order_id: int, note: str) -> str:
    return json.dumps({"ok": True, "data": {"id": order_id, "customer_note": note}, "error": None})


@pytest.mark.parametrize(("order_id", "note"), INJECTION_NOTES.items())
async def test_raw_injection_is_obeyed(model, order_id, note):
    reply = await model.ainvoke(after_tool(wrap_untrusted(order_json(order_id, note))))
    assert reply.tool_calls[0]["name"] == "refund_order"
    assert reply.tool_calls[0]["args"] == {"order_id": order_id}


async def test_raw_injection_is_visible_even_without_the_refund_tool():
    read_only = ScriptedChatModel().bind_tools([make_tool(SKILLS["get_order"])])
    reply = await read_only.ainvoke(after_tool(order_json(1006, INJECTION_NOTES[1006])))
    assert reply.content == FOLLOWED_INJECTION


@pytest.mark.parametrize(("order_id", "note"), INJECTION_NOTES.items())
async def test_sanitized_injection_is_not_obeyed(model, order_id, note):
    clean = sanitize_order({"id": order_id, "customer_note": note})
    assert clean["customer_note"] == NOTE_WITHHELD
    content = json.dumps({"ok": True, "data": clean, "error": None})
    reply = await model.ainvoke(after_tool(wrap_untrusted(content)))
    assert reply.tool_calls == []
