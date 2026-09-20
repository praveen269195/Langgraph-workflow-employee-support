import os
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote_plus

import dotenv

dotenv.load_dotenv()


@dataclass
class Config:
    # ── Postgres (app DB + checkpointer + PgVector) ────────────────────────
    db_host: str
    db_port: str
    db_name: str
    db_username: str
    db_password: str

    # ── FastAPI server ─────────────────────────────────────────────────────
    port: int
    host: str

    # ── Google / Gemini ────────────────────────────────────────────────────
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    # ── MLflow ─────────────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ops_assistant"

    def db_uri(self) -> str:
        """Sync SQLAlchemy URI (used by LangGraph AsyncPostgresSaver)."""
        return (
            f"postgresql://{self.db_username}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    def async_db_uri(self) -> str:
        """Async SQLAlchemy URI (used by asyncpg / SQLAlchemy async engine)."""
        return (
            f"postgresql+asyncpg://{self.db_username}:{quote_plus(self.db_password)}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def _require(key: str) -> str:
    """Raise immediately if a required env-var is missing — never fall back to a hardcoded secret."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Add it to your .env file and restart the server."
        )
    return value


def get_config() -> Config:
    return Config(
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5432"),
        db_name=os.getenv("DB_NAME", "ops_db"),
        db_username=os.getenv("DB_USERNAME", "postgres"),
        db_password=_require("DB_PASSWORD"),
        port=int(os.getenv("PORT", "8080")),
        host=os.getenv("HOST", "127.0.0.1"),
        gemini_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        mlflow_experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "ops_assistant"),
    )


config = get_config()
