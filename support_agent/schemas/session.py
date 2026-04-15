from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from support_agent.schemas.agent import AgentResult
from support_agent.schemas.ticket import SupportTicketInput


class SupportAgentIdentity(BaseModel):
    agent_id: str | None = None
    name: str | None = None


class SupportAiMessage(BaseModel):
    role: Literal["support_agent", "ai_agent"]
    content: str


class SupportAiSessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ticket: SupportTicketInput
    prompt: str | None = None
    support_agent: SupportAgentIdentity | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SupportAiSessionMessageRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: SupportAiMessage
    context_patch: dict[str, Any] = Field(default_factory=dict)


class SupportAiSessionRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str
    status: Literal["running", "completed", "failed"] = "completed"
    ticket: dict[str, Any]
    support_agent: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]] = Field(default_factory=list)
    latest_agent_state: dict[str, Any] = Field(default_factory=dict)
    latest_final_result: dict[str, Any] | None = None
    investigation_trace: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: str
    updated_at: str


class SupportAiSessionCreatedResponse(BaseModel):
    session_id: str
    status: Literal["running", "completed", "failed"]
    updated_at: str


class SupportAiSessionResponse(BaseModel):
    session_id: str
    status: Literal["running", "completed", "failed"]
    message: SupportAiMessage
    final_result: AgentResult | None = None
    investigation_trace: list[str] = Field(default_factory=list)
    clarifications: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str


class SupportAiSessionStatusResponse(BaseModel):
    session_id: str
    status: Literal["running", "completed", "failed"]
    latest_message: SupportAiMessage | None = None
    messages: list[SupportAiMessage] = Field(default_factory=list)
    final_result: AgentResult | None = None
    investigation_trace: list[str] = Field(default_factory=list)
    error: str | None = None
    updated_at: str
