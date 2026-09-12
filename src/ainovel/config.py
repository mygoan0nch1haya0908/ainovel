from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AINOVEL_", env_file=".env")

    app_name: str = "AI Novel Studio"
    database_url: str = "sqlite+pysqlite:///./ainovel.db"
    session_secret: str | None = None
    ollama_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    allow_real_openai: bool = False
    provider_timeout_seconds: float = 120.0
