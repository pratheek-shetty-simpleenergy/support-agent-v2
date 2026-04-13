from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from support_agent.db.catalog import DatabaseBinding, default_business_db_catalog


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "support-agent-v2"
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_host: str = Field(default="http://127.0.0.1:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="llama3", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(default="llama3", alias="OLLAMA_EMBEDDING_MODEL")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index: str | None = Field(default=None, alias="PINECONE_INDEX")
    pinecone_namespace: str = Field(default="support", alias="PINECONE_NAMESPACE")
    pinecone_top_k: int = Field(default=5, alias="PINECONE_TOP_K")

    database_server_url: str | None = Field(default=None, alias="DATABASE_SERVER_URL")
    business_db_names: dict[str, str] = Field(
        default_factory=lambda: {
            "dms_stage": "dms-stage",
            "ownership_stage": "ownership-stage",
            "orders_stage": "orders-stage",
            "testride_stage": "testride-stage",
            "users_stage": "users-stage",
            "unified_ticketing_stage": "unified-ticketing-stage",
        },
        alias="BUSINESS_DB_NAMES",
    )

    @property
    def business_db_catalog(self) -> dict[str, DatabaseBinding]:
        catalog = default_business_db_catalog()
        for key, database_name in self.business_db_names.items():
            if key in catalog:
                catalog[key].database_name = database_name
        return catalog

    def business_database_configs(self) -> dict[str, dict[str, str | None]]:
        if not self.database_server_url:
            return {}

        parsed = urlparse(_normalize_database_url(self.database_server_url))
        query_pairs = dict(parse_qsl(parsed.query))
        schema_name = query_pairs.pop("schema", None)
        configs: dict[str, dict[str, str | None]] = {}
        for key, database_name in self.business_db_names.items():
            url = urlunparse(
                parsed._replace(
                    path=f"/{database_name}",
                    query=urlencode(query_pairs),
                )
            )
            configs[key] = {"url": url, "schema_name": schema_name}
        return configs

    def require_pinecone(self) -> None:
        if not self.pinecone_api_key or not self.pinecone_index:
            raise ValueError("Pinecone configuration is incomplete. Set PINECONE_API_KEY and PINECONE_INDEX.")

    def require_database(self) -> None:
        if not self.database_server_url:
            raise ValueError("DATABASE_SERVER_URL is required for business DB-backed investigation tools.")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: Any,
        env_settings: Any,
        dotenv_settings: Any,
        file_secret_settings: Any,
    ) -> tuple[Any, ...]:
        return init_settings, env_settings, dotenv_settings, file_secret_settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _normalize_database_url(raw_url: str) -> str:
    if raw_url.startswith("postgres://"):
        return raw_url.replace("postgres://", "postgresql+psycopg://", 1)
    if raw_url.startswith("postgresql://"):
        return raw_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw_url
