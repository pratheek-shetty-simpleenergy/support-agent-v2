from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from support_agent.config import Settings


class BusinessDbManager:
    def __init__(self, settings: Settings) -> None:
        settings.require_database()
        self.settings = settings
        self.engines: dict[str, Engine] = {
            key: create_engine(url, pool_pre_ping=True)
            for key, url in settings.business_database_urls().items()
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
            raise RuntimeError(f"DB query failed for database '{database_key}': {exc}") from exc

    def available_databases(self) -> Iterable[str]:
        return self.engines.keys()
