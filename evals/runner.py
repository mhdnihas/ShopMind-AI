"""R9 — Runs scripted conversations and reports pass or fail for each one.

    python -m evals                      # fake model, no API key
    python -m evals --kind injection     # only one kind of case
    python -m evals --provider google    # the same cases on a real model

Every case goes through the real chat API, graph, sub-agent and store, in-process, with
a fresh store. Fake mode proves the system around the model. It says nothing about how
well a real model behaves, which is what `--provider google` is for.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
import yaml
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from common.settings import get_settings
from orchestrator.a2a_client import A2AClient
from orchestrator.api import create_app
from orchestrator.client import parse_events
from orchestrator.graph import build_graph
from orchestrator.llm import LLMConfigError, get_chat_model
from orders_agent.server import create_app as create_agent
from store_api.data import seed_orders
from store_api.main import create_app as create_store

CASES_FILE = Path(__file__).parent / "cases.yaml"
GOOGLE_PAUSE_SECONDS = 20.0

_AGENT = "http://agent.eval"
_STORE = "http://store.eval"
_CHAT = "http://chat.eval"


class _Strict(BaseModel):
    # A typo in a case must fail loudly, not silently check nothing.
    model_config = ConfigDict(extra="forbid")


class Turn(_Strict):
    say: str | None = None
    decide: Literal["approve", "reject"] | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> Turn:
        if (self.say is None) == (self.decide is None):
            raise ValueError("a turn needs exactly one of `say` or `decide`")
        return self


class ExpectedProposal(_Strict):
    action: Literal["refund", "cancel"]
    order_id: int


class Expect(_Strict):
    http_status: int = 200
    tools: list[str] = []
    no_tools: bool = False
    answer_contains: list[str] = []
    answer_excludes: list[str] = []
    proposal: ExpectedProposal | None = None
    no_proposal: bool = False
    order_status: dict[int, str] = {}


class Case(_Strict):
    name: str
    kind: Literal[
        "tool_choice", "answer", "proposal", "approval", "injection", "cross_merchant", "access"
    ]
    token: str
    turns: list[Turn]
    expect: Expect


@dataclass
class Observed:
    http_status: int = 0
    tools: list[str] = field(default_factory=list)
    answer: str = ""
    stream: str = ""
    proposals: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    case: Case
    failures: list[str]
    seconds: float

    @property
    def passed(self) -> bool:
        return not self.failures


def load_cases(path: Path = CASES_FILE) -> list[Case]:
    cases = [Case.model_validate(raw) for raw in yaml.safe_load(path.read_text(encoding="utf-8"))]
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        raise ValueError("case names must be unique")
    return cases


async def _play(case: Case, http: httpx.AsyncClient) -> Observed:
    observed = Observed()
    headers = {"Authorization": f"Bearer {case.token}"}
    thread = uuid.uuid4().hex
    for turn in case.turns:
        if turn.say is not None:
            body = {"message": turn.say, "thread_id": thread}
            response = await http.post("/chat", json=body, headers=headers)
        else:
            response = await http.post(f"/chat/{thread}/{turn.decide}", headers=headers)

        observed.http_status = response.status_code
        observed.stream += response.text
        observed.answer = ""
        if response.status_code != 200:
            continue
        for name, data in parse_events(iter(response.text.splitlines())):
            if name == "tool_started":
                observed.tools.append(data["tool"])
            elif name == "token":
                observed.answer += data["text"]
            elif name == "proposal":
                observed.proposals.append(data)
            elif name == "error":
                observed.errors.append(data["message"])
    return observed


def _check(expect: Expect, seen: Observed, store: Any) -> list[str]:
    failures = [f"the stream reported an error: {message}" for message in seen.errors]
    if seen.http_status != expect.http_status:
        failures.append(f"expected HTTP {expect.http_status}, got {seen.http_status}")
    failures += [f"tool {t} was not called" for t in expect.tools if t not in seen.tools]
    if expect.no_tools and seen.tools:
        failures.append(f"expected no tools, but called {seen.tools}")

    answer, stream = seen.answer.casefold(), seen.stream.casefold()
    failures += [
        f"answer does not contain {text!r}"
        for text in expect.answer_contains
        if text.casefold() not in answer
    ]
    failures += [
        f"stream contains forbidden text {text!r}"
        for text in expect.answer_excludes
        if text.casefold() in stream
    ]

    raised = [(p.get("action"), p.get("order_id")) for p in seen.proposals]
    wanted = expect.proposal
    if wanted and (wanted.action, wanted.order_id) not in raised:
        failures.append(f"expected a {wanted.action} proposal for {wanted.order_id}, got {raised}")
    if expect.no_proposal and raised:
        failures.append(f"expected no proposal, got {raised}")

    owners = seed_orders()
    for order_id, status in expect.order_status.items():
        actual = store.get(owners[order_id].merchant, order_id).status
        if actual != status:
            failures.append(f"order {order_id} is {actual}, expected {status}")
    return failures


async def run_case(case: Case, model: BaseChatModel) -> CaseResult:
    """Run one case against a brand new store, so cases cannot affect each other."""
    started = time.perf_counter()
    store_app = create_store()
    store = httpx.ASGITransport(app=store_app)
    agent = create_agent(store_url=_STORE, public_url=_AGENT, store_transport=store)
    try:
        a2a = await A2AClient.discover(_AGENT, transport=httpx.ASGITransport(app=agent))
        api = create_app(graph=build_graph(model, a2a), store_url=_STORE, store_transport=store)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api), base_url=_CHAT, timeout=None
        ) as http:
            seen = await _play(case, http)
        failures = _check(case.expect, seen, store_app.state.store)
    except Exception as e:  # noqa: BLE001
        # One broken case must not stop the others from being reported.
        failures = [f"the case crashed: {type(e).__name__}"]
    return CaseResult(case, failures, time.perf_counter() - started)


def _print_result(result: CaseResult) -> None:
    label = "PASS" if result.passed else "FAIL"
    print(f"{label}  {result.case.kind:<15} {result.case.name:<52} {result.seconds:5.1f}s")
    for failure in result.failures:
        print(f"      - {failure}")


async def _run_all(cases: list[Case], model: BaseChatModel, pause: float) -> list[CaseResult]:
    results = []
    for index, case in enumerate(cases):
        if index and pause:
            await asyncio.sleep(pause)
        results.append(await run_case(case, model))
        _print_result(results[-1])
    return results


def main(argv: list[str] | None = None) -> int:
    """Returns the process exit code: 0 all passed, 1 a case failed, 2 could not run."""
    parser = argparse.ArgumentParser(description="Run the ShopHub assistant evals.")
    # The default is fake whatever .env says, so this never spends quota by accident.
    parser.add_argument("--provider", choices=["fake", "google"], default="fake")
    parser.add_argument("--model", help="Gemini model name (default: GOOGLE_MODEL)")
    parser.add_argument("--case", action="append", default=[], help="Only this case (repeatable)")
    parser.add_argument("--kind", action="append", default=[], help="Only this kind (repeatable)")
    parser.add_argument("--pause", type=float, help="Seconds to wait between cases")
    parser.add_argument("--cases", type=Path, default=CASES_FILE, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)
    try:
        cases = [
            case
            for case in load_cases(args.cases)
            if (not args.case or case.name in args.case)
            and (not args.kind or case.kind in args.kind)
        ]
    except (OSError, ValidationError, ValueError, TypeError) as e:
        print(f"Could not read the cases: {e}")
        return 2
    if not cases:
        print("No cases matched.")
        return 2

    settings = get_settings()
    update: dict[str, Any] = {"llm_provider": args.provider}
    if args.model:
        update["google_model"] = args.model
    settings = settings.model_copy(update=update)
    try:
        model = get_chat_model(settings)
    except LLMConfigError as e:
        print(e)
        return 2

    name = settings.google_model if args.provider == "google" else "scripted fake"
    pause = (
        args.pause
        if args.pause is not None
        else (GOOGLE_PAUSE_SECONDS if args.provider == "google" else 0.0)
    )
    print(f"Model: {name}\n")
    results = asyncio.run(_run_all(cases, model, pause))
    failed = sum(not result.passed for result in results)
    print(f"\nPassed: {len(results) - failed}   Failed: {failed}")
    return 1 if failed else 0
