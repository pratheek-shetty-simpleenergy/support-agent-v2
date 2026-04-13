from __future__ import annotations

from typing import Any, TypedDict

from support_agent.schemas.agent import AgentResult, InvestigationPlan, NextAction


class AgentState(TypedDict, total=False):
    ticket_id: str
    raw_user_message: str
    conversation_history: list[dict[str, str]]
    user_id: str | None
    mobile: str | None
    booking_id: str | None
    payment_id: str | None
    order_id: str | None
    order_number: str | None
    vehicle_id: str | None
    normalized_issue_summary: str
    issue_category: str
    problem_type: str
    retrieved_context: str
    retrieved_matches: list[dict[str, Any]]
    facts: dict[str, Any]
    tool_results: list[dict[str, Any]]
    tool_failures: list[dict[str, Any]]
    clarification_requests: list[dict[str, Any]]
    investigation_trace: list[str]
    hypotheses: list[str]
    investigation_plan: InvestigationPlan
    next_action: NextAction | str
    response_summary: str
    confidence: float
    final_result: AgentResult
