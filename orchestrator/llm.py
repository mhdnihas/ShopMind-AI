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
# Support answers should be consistent, so tool choice should not vary between runs.
LLM_TEMPERATURE = 0.0


class LLMConfigError(RuntimeError):
    """The configured provider cannot be used. Raised at startup, not mid-conversation."""


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return ScriptedChatModel()
    if settings.llm_provider == "google":
        return _google_model(settings)
    raise LLMConfigError(f"Unknown LLM provider: {settings.llm_provider}")


def _google_model(settings: Settings) -> BaseChatModel:
    key = settings.google_api_key
    if key is None or not key.get_secret_value():
        raise LLMConfigError("LLM_PROVIDER=google requires GOOGLE_API_KEY to be set.")

    # Imported here because it is an optional dependency: fake mode must work without it.
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise LLMConfigError(
            'LLM_PROVIDER=google requires the extra: pip install -e ".[google]"'
        ) from e

    return ChatGoogleGenerativeAI(
        model=settings.google_model,
        api_key=key,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_OUTPUT_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
        retries=LLM_MAX_RETRIES,
    )
