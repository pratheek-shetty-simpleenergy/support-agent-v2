from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    success: bool
    payload: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
