import pytest

from pydantic import BaseModel

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
