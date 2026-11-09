"""The one place that creates the chat model.

Everything else calls `get_chat_model()` and receives a LangChain `BaseChatModel`, so
it never knows which provider is behind it. Changing provider is a config change.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from common.settings import Settings, get_settings
from orchestrator.fake_llm import ScriptedChatModel

# A model call takes far longer than a call between our own services, so this is
# separate from HTTP_TIMEOUT_SECONDS.
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 2
MAX_OUTPUT_TOKENS = 16000


class LLMConfigError(RuntimeError):
    """The configured provider cannot be used. Raised at startup, not mid-conversation."""


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return ScriptedChatModel()
    if settings.llm_provider == "anthropic":
        return _anthropic_model(settings)
    raise LLMConfigError(f"Unknown LLM provider: {settings.llm_provider}")


def _anthropic_model(settings: Settings) -> BaseChatModel:
    key = settings.anthropic_api_key
    if key is None or not key.get_secret_value():
        raise LLMConfigError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY to be set.")

    # Imported here because it is an optional dependency: fake mode must work without it.
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as e:
        raise LLMConfigError(
            'LLM_PROVIDER=anthropic requires the extra: pip install -e ".[anthropic]"'
        ) from e

    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=key,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=LLM_MAX_RETRIES,
    )
