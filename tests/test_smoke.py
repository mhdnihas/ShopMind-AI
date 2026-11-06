"""Phase 0 smoke tests: packages import and config defaults are safe for offline runs."""

import importlib

import pytest

from common.settings import Settings


@pytest.mark.parametrize("pkg", ["common", "store_api", "orders_agent", "orchestrator", "evals"])
def test_packages_import(pkg):
    importlib.import_module(pkg)


def test_default_provider_is_fake(monkeypatch):
    # The graders run offline with no key: the default must be the fake LLM.
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    s = Settings(_env_file=None)
    assert s.llm_provider == "fake"
    assert s.anthropic_api_key is None


def test_api_key_is_not_printed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    s = Settings(_env_file=None)
    assert "sk-test-secret" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-test-secret"
