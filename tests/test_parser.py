import pytest

from pydantic import BaseModel

from support_agent.config.settings import Settings
from support_agent.llm.parser import parse_json_model


class SampleModel(BaseModel):
    name: str


def test_parse_json_model_accepts_plain_json() -> None:
    parsed = parse_json_model('{"name":"ticket"}', SampleModel)
    assert parsed.name == "ticket"


def test_parse_json_model_rejects_invalid_json() -> None:
    with pytest.raises(ValueError):
        parse_json_model("not-json", SampleModel)


class ClassificationModel(BaseModel):
    normalized_issue_summary: str
    issue_category: str
    problem_type: str
    confidence: float


def test_parse_json_model_extracts_key_value_text_fallback() -> None:
    raw = """
    Here is the classification:

    Normalized Issue Summary: Order stuck in pending state despite successful payment.
    Issue Category: delivery
    Problem Type: order_status_update
    Confidence: 1
    """
    parsed = parse_json_model(raw, ClassificationModel)
    assert parsed.issue_category == "delivery"
    assert parsed.confidence == 1.0


def test_business_database_configs_strips_schema_query_and_keeps_search_path_value() -> None:
    settings = Settings(
        DATABASE_SERVER_URL="postgres://user:pass@localhost:7456?schema=public&sslmode=require"
    )
    config = settings.business_database_configs()["orders_stage"]
    assert config["schema_name"] == "public"
    assert str(config["url"]) == "postgresql+psycopg://user:pass@localhost:7456/orders-stage?sslmode=require"
