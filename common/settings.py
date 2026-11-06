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
    llm_provider: Literal["fake", "anthropic"] = "fake"
    # SecretStr keeps the key out of repr()/logs.
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"

    store_api_url: str = "http://localhost:8002"
    orders_agent_url: str = "http://localhost:8001"

    database_url: str = "postgresql://shopmind:shopmind@localhost:5432/shopmind"

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
