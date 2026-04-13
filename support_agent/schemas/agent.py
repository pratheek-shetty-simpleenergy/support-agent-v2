from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class NextAction(str, Enum):
    resolved = "resolved"
    needs_clarification = "needs_clarification"
    escalate = "escalate"
    pending = "pending"


class AgentDecision(BaseModel):
    normalized_issue_summary: str
    issue_category: str
    problem_type: str
    confidence: float = Field(ge=0.0, le=1.0)


class InvestigationPlan(BaseModel):
    rationale: str
    required_tools: list[str] = Field(default_factory=list)
    tool_arguments: dict[str, dict[str, Any]] = Field(default_factory=dict)
    should_stop_after_tools: bool = False


class AgentResult(BaseModel):
    ticket_id: str
    issue_summary: str
    issue_category: str
    problem_type: str
    decision: NextAction
    customer_response: str
    internal_summary: str
    facts: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
