import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError


T = TypeVar("T", bound=BaseModel)


def parse_json_model(raw_text: str, model_type: type[T]) -> T:
    text = raw_text.strip()
    if "```" in text:
        parts = [part.strip() for part in text.split("```") if part.strip()]
        text = next((part.removeprefix("json").strip() for part in parts if "{" in part), text)

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        json_fragment = _extract_json_object(text)
        if json_fragment is not None:
            try:
                payload = json.loads(json_fragment)
            except json.JSONDecodeError:
                payload = _extract_key_value_payload(text)
        else:
            payload = _extract_key_value_payload(text)
        if payload is None:
            raise ValueError(f"Model output is not valid JSON: {raw_text}") from exc

    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Model output failed schema validation: {payload}") from exc


def _extract_key_value_payload(text: str) -> dict | None:
    patterns = {
        "normalized_issue_summary": r"Normalized Issue Summary:\s*(.+)",
        "issue_category": r"Issue Category:\s*(.+)",
        "problem_type": r"Problem Type:\s*(.+)",
        "confidence": r"Confidence:\s*(.+)",
    }
    payload: dict[str, str | float] = {}

    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip().strip("*").strip()
        if key == "confidence":
            try:
                payload[key] = float(value)
            except ValueError:
                continue
        else:
            payload[key] = value

    return payload or None


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]
