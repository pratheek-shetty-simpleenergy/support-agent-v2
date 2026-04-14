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


def test_get_user_profile_by_mobile_prepends_country_code() -> None:
    repository = build_repository()
    repository.get_user_profile_by_mobile("9480300096")
    _, params, database_key = repository.db.last_call
    assert params["mobile"] == "+919480300096"
    assert database_key == "users_stage"


def test_get_booking_details_uses_uuid_lookup_for_uuid_input() -> None:
    repository = build_repository()
    repository.get_booking_details("3dfb661f-7756-4d06-9d72-a47093112c1a")
    sql, _, _ = repository.db.last_call
    assert '"id" = :order_id' in sql
    assert '"orderNumber" = :order_number' not in sql


def test_get_booking_details_uses_order_number_lookup_for_text_input() -> None:
    repository = build_repository()
    repository.get_booking_details("ORD-1001")
    sql, _, _ = repository.db.last_call
    assert '"orderNumber" = :order_number' in sql
