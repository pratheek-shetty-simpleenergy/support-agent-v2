from support_agent.config.settings import Settings
from support_agent.db.repositories import BusinessDbRepository
from support_agent.tools.db_tools import build_business_db_tools


class FakeBusinessDbManager:
    def __init__(self) -> None:
        self.last_call = None

    def fetch_one(self, sql, params, database_key):
        self.last_call = (sql, params, database_key)
        if params.get("user_id") == "missing":
            return None
        return {"id": next(iter(params.values())), "status": "ok"}

    def fetch_all(self, sql, params, database_key):
        self.last_call = (sql, params, database_key)
        return [{"id": "1"}, {"id": "2"}]


def build_repository() -> BusinessDbRepository:
    settings = Settings(DATABASE_SERVER_URL="postgres://readonly:readonly@localhost:5432?schema=public")
    return BusinessDbRepository(FakeBusinessDbManager(), settings)


def test_get_user_profile_tool_returns_structured_payload() -> None:
    tools = build_business_db_tools(build_repository())
    result = tools.run("get_user_profile", {"user_id": "user-1"})
    assert result.success is True
    assert result.payload["user_profile"]["id"] == "user-1"


def test_get_ticket_history_tool_returns_list_payload() -> None:
    tools = build_business_db_tools(build_repository())
    result = tools.run("get_ticket_history", {"user_id": "user-1"})
    assert result.success is True
    assert len(result.payload["ticket_history"]) == 2
