from __future__ import annotations

import time

from support_agent.schemas.agent import AgentResult, NextAction
from support_agent.schemas.session import (
    SupportAiMessage,
    SupportAiSessionCreateRequest,
    SupportAiSessionMessageRequest,
)
from support_agent.schemas.ticket import SupportTicketInput
from support_agent.services.session_store import InMemorySessionStore
from support_agent.services.support_ai_sessions import SupportAiSessionService


class FakeGraph:
    def invoke(self, payload: dict):
        return {
            "ticket_id": payload["ticket_id"],
            "final_result": AgentResult(
                ticket_id=payload["ticket_id"],
                issue_summary="Mobile app is not working",
                issue_category="mobile_app",
                problem_type="app_not_working",
                decision=NextAction.pending,
                customer_response="We are checking the issue.",
                internal_summary="Ownership and VIN linkage are present.",
                facts={},
                confidence=0.9,
            ),
            "investigation_trace": [
                "Resolved user from mobile number.",
                "Matched VIN to ownership record.",
            ],
            "clarification_requests": [],
        }


class FakeLLM:
    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        return "You should escalate this to the telematics team."


class FakeClarificationGraph:
    def invoke(self, payload: dict):
        order_number = payload.get("order_number")
        if order_number == "8332-736785-7815":
            return {
                "ticket_id": payload["ticket_id"],
                "final_result": AgentResult(
                    ticket_id=payload["ticket_id"],
                    issue_summary="Order pending after payment",
                    issue_category="payment",
                    problem_type="payment_processing_delay",
                    decision=NextAction.pending,
                    customer_response="I checked order 8332-736785-7815 and the payment is still being processed.",
                    internal_summary="Confirmed order number and continued investigation.",
                    facts={"order_details": {"order_number": "8332-736785-7815"}},
                    confidence=0.9,
                ),
                "investigation_trace": [
                    "Loaded ticket T-API-CLARIFY",
                    "Confirmed order number from customer clarification.",
                ],
                "clarification_requests": [],
            }
        return {
            "ticket_id": payload["ticket_id"],
            "final_result": AgentResult(
                ticket_id=payload["ticket_id"],
                issue_summary="Order pending after payment",
                issue_category="payment",
                problem_type="payment_processing_delay",
                decision=NextAction.needs_clarification,
                customer_response="I found these likely order numbers for your account: 8332-736785-7815. Please confirm the correct order number.",
                internal_summary="Investigation paused pending customer clarification.",
                facts={"user_enquiries": [{"order_number": "8332-736785-7815"}]},
                confidence=0.6,
            ),
            "investigation_trace": [
                "Loaded ticket T-API-CLARIFY",
                "Finished with clarification request: I found these likely order numbers for your account: 8332-736785-7815. Please confirm the correct order number.",
            ],
            "clarification_requests": [],
        }


def test_create_session_persists_state() -> None:
    service = SupportAiSessionService(
        graph=FakeGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    request = SupportAiSessionCreateRequest(
        ticket=SupportTicketInput(
            ticket_id="T-API-1",
            raw_user_message="App not working for VIN VIN-1",
            mobile="9480300096",
            vehicle_vin="VIN-1",
        ),
        prompt="Debug this ticket.",
    )
    response = service.create_session(request)
    assert response.session_id.startswith("sess_")
    assert response.status == "running"
    stored = service.get_session(response.session_id)
    assert stored is not None
    assert stored.ticket["ticket_id"] == "T-API-1"
    assert stored.messages[0]["role"] == "support_agent"
    completed = _wait_for_completion(service, response.session_id)
    assert completed is not None
    assert completed.latest_final_result is not None
    assert completed.investigation_trace[-1] == "Matched VIN to ownership record."
    assert completed.messages[-1]["content"] == "We are checking the issue."


def test_send_message_uses_session_memory() -> None:
    store = InMemorySessionStore()
    service = SupportAiSessionService(
        graph=FakeGraph(),
        llm_client=FakeLLM(),
        session_store=store,
    )
    created = service.create_session(
        SupportAiSessionCreateRequest(
            ticket=SupportTicketInput(
                ticket_id="T-API-2",
                raw_user_message="App not working for VIN VIN-2",
                mobile="9480300096",
                vehicle_vin="VIN-2",
            ),
            prompt="Debug this ticket.",
        )
    )
    _wait_for_completion(service, created.session_id)
    response = service.send_message(
        created.session_id,
        SupportAiSessionMessageRequest(
            message=SupportAiMessage(role="support_agent", content="Should I escalate this?")
        ),
    )
    assert response.session_id == created.session_id
    assert response.status == "running"
    completed = _wait_for_completion(service, created.session_id)
    assert completed is not None
    stored = service.get_session(created.session_id)
    assert stored is not None
    assert stored.messages[-2]["content"] == "Should I escalate this?"
    assert stored.messages[-1]["role"] == "ai_agent"
    assert stored.messages[-1]["content"] == "You should escalate this to the telematics team."


def test_get_session_status_returns_final_result() -> None:
    service = SupportAiSessionService(
        graph=FakeGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    created = service.create_session(
        SupportAiSessionCreateRequest(
            ticket=SupportTicketInput(
                ticket_id="T-API-STATUS",
                raw_user_message="App not working for VIN VIN-STATUS",
                mobile="9480300096",
                vehicle_vin="VIN-STATUS",
            ),
            prompt="Debug this ticket.",
        )
    )
    _wait_for_completion(service, created.session_id)
    status = service.get_session_status(created.session_id)
    assert status.session_id == created.session_id
    assert status.status == "completed"
    assert status.final_result is not None
    assert status.latest_message is not None
    assert status.latest_message.role == "ai_agent"
    assert status.latest_message.content == "We are checking the issue."
    assert len(status.messages) >= 2


def test_follow_up_answer_is_returned_via_status() -> None:
    service = SupportAiSessionService(
        graph=FakeGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    created = service.create_session(
        SupportAiSessionCreateRequest(
            ticket=SupportTicketInput(
                ticket_id="T-API-FOLLOWUP",
                raw_user_message="App not working for VIN VIN-FOLLOWUP",
                mobile="9480300096",
                vehicle_vin="VIN-FOLLOWUP",
            ),
            prompt="Debug this ticket.",
        )
    )
    _wait_for_completion(service, created.session_id)
    follow_up = service.send_message(
        created.session_id,
        SupportAiSessionMessageRequest(
            message=SupportAiMessage(role="support_agent", content="Should I escalate this?")
        ),
    )
    assert follow_up.status == "running"
    _wait_for_completion(service, created.session_id)
    status = service.get_session_status(created.session_id)
    assert status.latest_message is not None
    assert status.latest_message.content == "You should escalate this to the telematics team."
    assert status.messages[-2].content == "Should I escalate this?"
    assert status.messages[-1].content == "You should escalate this to the telematics team."


def test_follow_up_confirmation_reruns_graph_with_confirmed_order_number() -> None:
    service = SupportAiSessionService(
        graph=FakeClarificationGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    created = service.create_session(
        SupportAiSessionCreateRequest(
            ticket=SupportTicketInput(
                ticket_id="T-API-CLARIFY",
                raw_user_message="Order is pending despite payment",
                mobile="9480300096",
            ),
            prompt="Debug this ticket.",
        )
    )
    _wait_for_completion(service, created.session_id)
    follow_up = service.send_message(
        created.session_id,
        SupportAiSessionMessageRequest(
            message=SupportAiMessage(role="support_agent", content="i confirm this order number")
        ),
    )
    assert follow_up.status == "running"
    _wait_for_completion(service, created.session_id)
    status = service.get_session_status(created.session_id)
    assert status.final_result is not None
    assert status.final_result.decision == NextAction.pending
    assert status.final_result.customer_response == "I checked order 8332-736785-7815 and the payment is still being processed."
    assert status.latest_message is not None
    assert status.latest_message.content == "I checked order 8332-736785-7815 and the payment is still being processed."
    assert any(
        "Re-running investigation with confirmed order number 8332-736785-7815." == entry
        for entry in status.investigation_trace
    )


def test_follow_up_stream_only_emits_new_run_events() -> None:
    service = SupportAiSessionService(
        graph=FakeGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    created = service.create_session(
        SupportAiSessionCreateRequest(
            ticket=SupportTicketInput(
                ticket_id="T-API-STREAM-FOLLOWUP",
                raw_user_message="App not working for VIN VIN-STREAM",
                mobile="9480300096",
                vehicle_vin="VIN-STREAM",
            ),
            prompt="Debug this ticket.",
        )
    )
    _wait_for_completion(service, created.session_id)
    service.send_message(
        created.session_id,
        SupportAiSessionMessageRequest(
            message=SupportAiMessage(role="support_agent", content="Should I escalate this?")
        ),
    )
    events = list(service.stream_session(created.session_id))
    assert any("event: message_started" in item for item in events)
    assert any("event: message_result" in item for item in events)
    assert not any("event: final_result" in item for item in events)


class FakeStreamingGraph:
    def invoke(self, payload: dict):
        return {
            "ticket_id": payload["ticket_id"],
            "final_result": AgentResult(
                ticket_id=payload["ticket_id"],
                issue_summary="App issue",
                issue_category="mobile_app",
                problem_type="app_not_working",
                decision=NextAction.pending,
                customer_response="We are checking the issue.",
                internal_summary="Finished investigation.",
                facts={},
                confidence=0.9,
            ),
            "investigation_trace": [
                "Loaded ticket T-API-3",
                "Ran get_user_profile_by_mobile and received fields: user_profile.",
                "Finished with decision=NextAction.pending confidence=0.9",
            ],
            "clarification_requests": [],
        }


def test_stream_session_emits_sse_events() -> None:
    service = SupportAiSessionService(
        graph=FakeStreamingGraph(),
        llm_client=FakeLLM(),
        session_store=InMemorySessionStore(),
    )
    request = SupportAiSessionCreateRequest(
        ticket=SupportTicketInput(
            ticket_id="T-API-3",
            raw_user_message="App not working",
            mobile="9480300096",
        ),
        prompt="Debug this ticket.",
    )
    created = service.create_session(request)
    events = list(service.stream_session(created.session_id))
    assert events[0].startswith("event: session_loaded")
    assert any("event: session_started" in item for item in events)
    assert any("event: trace" in item and "I completed the get_user_profile_by_mobile check." in item for item in events)
    assert any("raw_message" in item and "Ran get_user_profile_by_mobile and received fields: user_profile." in item for item in events)
    assert events[-1].startswith("event: final_result")


def _wait_for_completion(service: SupportAiSessionService, session_id: str, timeout_seconds: float = 2.0):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        record = service.get_session(session_id)
        if record is not None and record.status in {"completed", "failed"}:
            return record
        time.sleep(0.01)
    return service.get_session(session_id)
