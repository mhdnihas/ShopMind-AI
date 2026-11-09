"""Central configuration, read from environment variables (and an optional .env file).

Every service imports `get_settings()` instead of reading os.environ directly, so
configuration lives in one place and can be overridden in tests.
"""

from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM: "fake" runs fully offline with a scripted model (the default).
    llm_provider: Literal["fake", "google"] = "fake"
    # SecretStr keeps the key out of repr()/logs.
    google_api_key: SecretStr | None = None
    google_model: str = "gemini-3.1-flash-lite"
    # Other models people can switch to, for example when one reaches its usage limit.
    google_extra_models: str = "gemini-3.5-flash-lite,gemini-3-flash-preview,gemini-3.6-flash"

    store_api_url: str = "http://localhost:8002"
    orders_agent_url: str = "http://localhost:8001"
    # Upper bound for one HTTP call between services; a hung backend must not hang a chat turn.
    http_timeout_seconds: float = 10.0

    database_url: str = "postgresql://shopmind:shopmind@localhost:5432/shopmind"

    log_level: str = "INFO"

    @property
    def selectable_models(self) -> list[str]:
        """The models that can be used, with the default first."""
        if self.llm_provider != "google":
            return [self.llm_provider]
        extras = [name.strip() for name in self.google_extra_models.split(",") if name.strip()]
        return list(dict.fromkeys([self.google_model, *extras]))


@lru_cache
def get_settings() -> Settings:
    return Settings()
