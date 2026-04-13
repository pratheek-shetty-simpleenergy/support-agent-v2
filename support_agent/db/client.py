from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from support_agent.config import Settings
from support_agent.runtime.errors import TransientDependencyError


class BusinessDbManager:
    def __init__(self, settings: Settings) -> None:
        settings.require_database()
        self.settings = settings
        self.engines: dict[str, Engine] = {
            key: create_engine(
                str(config["url"]),
                pool_pre_ping=True,
                pool_size=settings.db_pool_size,
                max_overflow=settings.db_max_overflow,
                pool_recycle=settings.db_pool_recycle_seconds,
                connect_args=_build_connect_args(settings, config.get("schema_name")),
            )
            for key, config in settings.business_database_configs().items()
        }

    @contextmanager
    def connect(self, database_key: str):
        engine = self.engines[database_key]
        with engine.connect() as connection:
            yield connection

    def fetch_one(self, sql: str, params: dict[str, object], database_key: str) -> dict | None:
        rows = self.fetch_all(sql, params, database_key=database_key)
        return rows[0] if rows else None

    def fetch_all(self, sql: str, params: dict[str, object], database_key: str) -> list[dict]:
        try:
            with self.connect(database_key) as connection:
                result = connection.execute(text(sql), params)
                return [dict(row._mapping) for row in result]
        except SQLAlchemyError as exc:
            raise TransientDependencyError(f"DB query failed for database '{database_key}': {exc}") from exc

    def available_databases(self) -> Iterable[str]:
        return self.engines.keys()

    def healthcheck(self) -> dict[str, dict[str, str]]:
        results: dict[str, dict[str, str]] = {}
        for database_key in self.available_databases():
            try:
                with self.connect(database_key) as connection:
                    connection.execute(text("SELECT 1"))
                results[database_key] = {"status": "ok"}
            except SQLAlchemyError as exc:
                results[database_key] = {"status": "error", "error": str(exc)}
        return results


def _build_connect_args(settings: Settings, schema_name: str | None) -> dict[str, str | int]:
    options = ["-cdefault_transaction_read_only=on", f"-cstatement_timeout={settings.db_statement_timeout_ms}"]
    if schema_name:
        options.append(f"-csearch_path={schema_name}")
    return {"options": " ".join(options), "connect_timeout": settings.db_connect_timeout_seconds}
