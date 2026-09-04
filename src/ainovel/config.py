from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AINOVEL_", env_file=".env")

    app_name: str = "AI Novel Studio"
    database_url: str = "sqlite+pysqlite:///./ainovel.db"
    session_secret: str | None = None
