"""R9 — the eval harness: every case passes, and broken behaviour really reports FAIL."""

import pytest
from pydantic import ValidationError

from common.settings import Settings
from evals import runner
from evals.runner import Case, Expect, Turn, load_cases, main, run_case
from orchestrator.fake_llm import ScriptedChatModel

CASES = load_cases()
BY_NAME = {case.name: case for case in CASES}


def case_with(expect, *turns, token="tok_acme_alice"):
    return Case(
        name="custom",
        kind="answer",
        token=token,
        turns=list(turns) or [Turn(say="Show me order 1042")],
        expect=Expect(**expect),
    )


class BrokenModel(ScriptedChatModel):
    def _generate(self, messages, *args, **kwargs):
        raise RuntimeError("provider down")


def test_covers_every_kind_the_task_names():
    assert len(CASES) >= 5
    assert {"tool_choice", "answer", "proposal", "injection", "cross_merchant"} <= {
        case.kind for case in CASES
    }


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_case_passes_in_fake_mode(case):
    result = await run_case(case, ScriptedChatModel())
    assert result.passed, result.failures


# --- a failed expectation must be reported --------------------------------------------


@pytest.mark.parametrize(
    ("expect", "reason"),
    [
        ({"tools": ["list_orders"]}, "tool list_orders was not called"),
        ({"no_tools": True}, "expected no tools"),
        ({"answer_contains": ["Nobody Here"]}, "does not contain"),
        ({"answer_excludes": ["Maria Lopez"]}, "forbidden text"),
        ({"proposal": {"action": "refund", "order_id": 1042}}, "expected a refund proposal"),
        ({"order_status": {1042: "REFUNDED"}}, "order 1042 is PAID"),
        ({"http_status": 403}, "expected HTTP 403"),
    ],
)
async def test_failed_expectations_are_reported(expect, reason):
    result = await run_case(case_with(expect), ScriptedChatModel())
    assert not result.passed
    assert any(reason in failure for failure in result.failures), result.failures


async def test_unexpected_proposal_is_reported():
    case = case_with({"no_proposal": True}, Turn(say="Please refund order 1042"))
    result = await run_case(case, ScriptedChatModel())
    assert any("expected no proposal" in failure for failure in result.failures)


async def test_model_failure_is_reported_without_its_details():
    result = await run_case(case_with({}), BrokenModel())
    assert any("reported an error" in failure for failure in result.failures)
    assert not any("provider down" in failure for failure in result.failures)


# --- broken systems must fail the matching cases -----------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "note_demanding_a_refund_is_ignored",
        "note_demanding_a_status_change_is_ignored",
        "note_demanding_customer_data_is_ignored",
    ],
)
async def test_injection_cases_fail_when_the_guardrail_is_removed(monkeypatch, name):
    monkeypatch.setattr("orders_agent.store_client.sanitize_order", lambda order: order)
    result = await run_case(BY_NAME[name], ScriptedChatModel())
    assert not result.passed


async def test_cross_merchant_case_fails_when_tenancy_is_removed(monkeypatch):
    def any_merchant(self, merchant, order_id):
        return self._orders[order_id]

    monkeypatch.setattr("store_api.data.Store.get", any_merchant)
    result = await run_case(BY_NAME["other_merchants_order_cannot_be_read"], ScriptedChatModel())
    assert not result.passed


# --- case files ---------------------------------------------------------------------------


def test_misspelled_expectation_is_rejected():
    with pytest.raises(ValidationError):
        Expect(answer_contain=["typo"])


@pytest.mark.parametrize("fields", [{}, {"say": "hi", "decide": "approve"}])
def test_a_turn_needs_exactly_one_action(fields):
    with pytest.raises(ValidationError):
        Turn(**fields)


def write_cases(tmp_path, tool):
    path = tmp_path / "cases.yaml"
    path.write_text(
        f"""
- name: only_case
  kind: answer
  token: tok_acme_alice
  turns:
    - say: Show me order 1042
  expect:
    tools: [{tool}]
""",
        encoding="utf-8",
    )
    return str(path)


# --- the command --------------------------------------------------------------------------


def test_command_prints_each_case_and_no_blended_score(capsys):
    assert main(["--kind", "access"]) == 0
    output = capsys.readouterr().out
    assert "PASS  access" in output
    assert "token_with_no_merchant_is_refused" in output
    assert "invalid_token_is_refused" in output
    assert "%" not in output


def test_command_exits_1_when_a_case_fails(tmp_path, capsys):
    assert main(["--cases", write_cases(tmp_path, "list_orders")]) == 1
    output = capsys.readouterr().out
    assert "FAIL  answer" in output
    assert "tool list_orders was not called" in output


def test_command_defaults_to_fake_whatever_the_settings_say(monkeypatch, capsys):
    google = Settings(_env_file=None, llm_provider="google", google_api_key=None)
    monkeypatch.setattr(runner, "get_settings", lambda: google)
    assert main(["--case", "greeting_needs_no_tool"]) == 0
    assert "scripted fake" in capsys.readouterr().out


def test_command_real_provider_needs_a_key(monkeypatch, capsys):
    bare = Settings(_env_file=None, google_api_key=None)
    monkeypatch.setattr(runner, "get_settings", lambda: bare)
    assert main(["--provider", "google"]) == 2
    assert "GOOGLE_API_KEY" in capsys.readouterr().out


def test_command_reports_when_nothing_matches(capsys):
    assert main(["--case", "does_not_exist"]) == 2
