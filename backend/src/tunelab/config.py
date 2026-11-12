from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TUNELAB_", env_file=".env", extra="ignore")

    data_dir: Path = Path("var")
    base_model: str = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
    base_revision: str | None = "a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3"
    cors_origins: list[str] = ["http://localhost:3104", "http://127.0.0.1:3104"]
    max_upload_mb: int = 20
    # run the job dispatcher thread in the API process (tests start jobs explicitly)
    dispatcher: bool = True

    @field_validator("base_revision", mode="before")
    @classmethod
    def _empty_revision(cls, v):
        """An empty revision means "none" (e.g. a local model directory)."""
        return v or None

    @property
    def db_url(self) -> str:
        return f"sqlite:///{(self.data_dir / 'tunelab.db').resolve()}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
