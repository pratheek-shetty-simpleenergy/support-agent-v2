from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from support_agent.agent.investigation_rules import candidate_order_numbers
from support_agent.runtime.errors import PermanentDependencyError
from support_agent.schemas.agent import AgentResult, NextAction
from support_agent.schemas.session import (
    SupportAiMessage,
    SupportAiSessionCreatedResponse,
    SupportAiSessionCreateRequest,
    SupportAiSessionMessageRequest,
    SupportAiSessionRecord,
    SupportAiSessionResponse,
    SupportAiSessionStatusResponse,
)
from support_agent.schemas.ticket import ConversationMessage, SupportTicketInput
from support_agent.services.session_store import SessionStore


class SupportAiSessionService:
    def __init__(self, *, graph: Any, llm_client: Any, session_store: SessionStore) -> None:
        self.graph = graph
        self.llm_client = llm_client
        self.session_store = session_store

    def create_session(self, request: SupportAiSessionCreateRequest) -> SupportAiSessionCreatedResponse:
        session_id = f"sess_{uuid4().hex}"
        timestamp = _utc_now()
        record = SupportAiSessionRecord(
            session_id=session_id,
            status="running",
            ticket=self._build_ticket_for_session(request.ticket, request.prompt).model_dump(mode="json"),
            support_agent=(request.support_agent.model_dump(mode="json") if request.support_agent else {}),
            metadata=request.metadata,
            messages=_initial_messages(request.prompt),
            latest_agent_state={},
            latest_final_result=None,
            investigation_trace=[],
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.session_store.save(session_id, record.model_dump(mode="json"))
        self.session_store.append_event(
            session_id,
            {
                "event": "session_started",
                "data": {
                    "session_id": session_id,
                    "status": "running",
                    "ticket_id": record.ticket.get("ticket_id"),
                    "updated_at": timestamp,
                },
            },
        )
        worker = threading.Thread(
            target=self._run_session_in_background,
            args=(session_id,),
            daemon=True,
            name=f"support-ai-{session_id}",
        )
        worker.start()
        return SupportAiSessionCreatedResponse(
            session_id=session_id,
            status="running",
            updated_at=timestamp,
        )

    def stream_session(self, session_id: str):
        record_payload = self.session_store.load(session_id)
        if record_payload is None:
            raise PermanentDependencyError(f"Session {session_id} was not found.")
        record = SupportAiSessionRecord.model_validate(record_payload)
        yield _sse_event(
            "session_loaded",
            {
                "session_id": session_id,
                "status": record.status,
                "ticket_id": record.ticket.get("ticket_id"),
                "updated_at": record.updated_at,
            },
        )
        next_index = 0
        while True:
            events = self.session_store.load_events(session_id, next_index)
            next_index += len(events)
            for item in events:
                yield _sse_event(item["event"], _humanize_stream_event(item))
            current_payload = self.session_store.load(session_id)
            if current_payload is None:
                break
            current_record = SupportAiSessionRecord.model_validate(current_payload)
            if current_record.status in {"completed", "failed"} and not events:
                break
            time.sleep(0.1)

    def send_message(self, session_id: str, request: SupportAiSessionMessageRequest) -> SupportAiSessionCreatedResponse:
        record_payload = self.session_store.load(session_id)
        if record_payload is None:
            raise PermanentDependencyError(f"Session {session_id} was not found.")
        record = SupportAiSessionRecord.model_validate(record_payload)
        ticket = SupportTicketInput.model_validate(record.ticket | request.context_patch)
        ticket = self._apply_follow_up_context(record, ticket, request.message.content)
        rerun_graph = self._should_rerun_graph(record, ticket)
        updated_at = _utc_now()
        record.messages.append(request.message.model_dump(mode="json"))
        record.status = "running"
        record.error = None
        record.updated_at = updated_at
        record.ticket = ticket.model_dump(mode="json")
        self.session_store.save(session_id, record.model_dump(mode="json"))
        self.session_store.clear_events(session_id)
        self.session_store.append_event(
            session_id,
            {
                "event": "message_started",
                "data": {
                    "session_id": session_id,
                    "status": "running",
                    "message": request.message.model_dump(mode="json"),
                    "updated_at": updated_at,
                },
            },
        )
        if rerun_graph and ticket.order_number:
            rerun_trace = f"Re-running investigation with confirmed order number {ticket.order_number}."
            record.investigation_trace = [*record.investigation_trace, rerun_trace]
            record.updated_at = updated_at
            self.session_store.save(session_id, record.model_dump(mode="json"))
            self.session_store.append_event(
                session_id,
                {
                    "event": "trace",
                    "data": {
                        "session_id": session_id,
                        "message": rerun_trace,
                    },
                },
            )
        worker = threading.Thread(
            target=self._run_follow_up_in_background if not rerun_graph else self._run_session_in_background,
            args=(session_id, request.message.content) if not rerun_graph else (session_id,),
            daemon=True,
            name=f"support-ai-followup-{session_id}",
        )
        worker.start()
        return SupportAiSessionCreatedResponse(
            session_id=session_id,
            status="running",
            updated_at=updated_at,
        )

    def get_session(self, session_id: str) -> SupportAiSessionRecord | None:
        payload = self.session_store.load(session_id)
        if payload is None:
            return None
        return SupportAiSessionRecord.model_validate(payload)

    def get_session_status(self, session_id: str) -> SupportAiSessionStatusResponse:
        record = self.get_session(session_id)
        if record is None:
            raise PermanentDependencyError(f"Session {session_id} was not found.")
        latest_message = None
        if record.messages:
            last = record.messages[-1]
            if isinstance(last, dict) and last.get("role") in {"support_agent", "ai_agent"}:
                latest_message = SupportAiMessage.model_validate(last)
        messages = [
            SupportAiMessage.model_validate(item)
            for item in record.messages
            if isinstance(item, dict) and item.get("role") in {"support_agent", "ai_agent"}
        ]
        final_result = AgentResult.model_validate(record.latest_final_result) if record.latest_final_result else None
        return SupportAiSessionStatusResponse(
            session_id=session_id,
            status=record.status,
            latest_message=latest_message,
            messages=messages,
            final_result=final_result,
            investigation_trace=record.investigation_trace,
            error=record.error,
            updated_at=record.updated_at,
        )

    def _build_ticket_for_session(self, ticket: SupportTicketInput, prompt: str | None) -> SupportTicketInput:
        if not prompt:
            return ticket
        updated_history = [*ticket.conversation_history, ConversationMessage(role="support_agent", content=prompt)]
        return ticket.model_copy(update={"conversation_history": updated_history})

    def _apply_follow_up_context(
        self,
        record: SupportAiSessionRecord,
        ticket: SupportTicketInput,
        message: str,
    ) -> SupportTicketInput:
        updates: dict[str, Any] = {}
        updated_history = [*ticket.conversation_history, ConversationMessage(role="support_agent", content=message)]
        updates["conversation_history"] = updated_history

        explicit_order_number = _extract_order_number(message)
        if explicit_order_number:
            updates["order_number"] = explicit_order_number
            return ticket.model_copy(update=updates)

        latest_final_result = record.latest_final_result or {}
        if latest_final_result.get("decision") != NextAction.needs_clarification.value:
            return ticket.model_copy(update=updates)

        facts = latest_final_result.get("facts", {}) if isinstance(latest_final_result, dict) else {}
        order_numbers = candidate_order_numbers(facts) if isinstance(facts, dict) else []
        if len(order_numbers) == 1 and _looks_like_confirmation(message):
            updates["order_number"] = order_numbers[0]
        return ticket.model_copy(update=updates)

    def _should_rerun_graph(self, record: SupportAiSessionRecord, ticket: SupportTicketInput) -> bool:
        latest_final_result = record.latest_final_result or {}
        if not isinstance(latest_final_result, dict):
            return False
        if latest_final_result.get("decision") != NextAction.needs_clarification.value:
            return False
        prior_ticket = SupportTicketInput.model_validate(record.ticket)
        return (
            (ticket.order_number and ticket.order_number != prior_ticket.order_number)
            or (ticket.order_id and ticket.order_id != prior_ticket.order_id)
            or (ticket.payment_id and ticket.payment_id != prior_ticket.payment_id)
        )

    def _run_session_in_background(self, session_id: str) -> None:
        record_payload = self.session_store.load(session_id)
        if record_payload is None:
            return
        record = SupportAiSessionRecord.model_validate(record_payload)
        ticket = SupportTicketInput.model_validate(record.ticket)
        try:
            existing_trace = list(record.investigation_trace)
            state = self._run_graph_with_events(session_id, ticket.model_dump())
            final_result = state["final_result"]
            message_content = final_result.customer_response or final_result.internal_summary
            record.messages.append({"role": "ai_agent", "content": message_content})
            record.latest_agent_state = _safe_state_dump(state)
            record.latest_final_result = final_result.model_dump(mode="json")
            state_trace = state.get("investigation_trace", [])
            record.investigation_trace = [*existing_trace, *state_trace] if existing_trace else state_trace
            record.status = "completed"
            record.updated_at = _utc_now()
            self.session_store.save(session_id, record.model_dump(mode="json"))
            self.session_store.append_event(
                session_id,
                {
                    "event": "final_result",
                    "data": SupportAiSessionResponse(
                        session_id=session_id,
                        status="completed",
                        message=SupportAiMessage(role="ai_agent", content=message_content),
                        final_result=final_result,
                        investigation_trace=record.investigation_trace,
                        clarifications=state.get("clarification_requests", []),
                        updated_at=record.updated_at,
                    ).model_dump(mode="json"),
                },
            )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = _utc_now()
            self.session_store.save(session_id, record.model_dump(mode="json"))
            self.session_store.append_event(
                session_id,
                {
                    "event": "error",
                    "data": {
                        "session_id": session_id,
                        "message": str(exc),
                        "updated_at": record.updated_at,
                    },
                },
            )

    def _build_follow_up_prompt(self, record: SupportAiSessionRecord, message: str) -> str:
        final_result = record.latest_final_result or {}
        trace = "\n".join(record.investigation_trace[-10:])
        prior_messages = "\n".join(f"{item['role']}: {item['content']}" for item in record.messages[-10:])
        return (
            "You are assisting an internal support agent.\n"
            "Use the prior ticket investigation context to answer the follow-up question.\n"
            "Be concise, operational, and specific.\n\n"
            f"Ticket:\n{record.ticket}\n\n"
            f"Latest final result:\n{final_result}\n\n"
            f"Investigation trace:\n{trace}\n\n"
            f"Conversation so far:\n{prior_messages}\n\n"
            f"Support agent question:\n{message}\n"
        )

    def _run_follow_up_in_background(self, session_id: str, message: str) -> None:
        record_payload = self.session_store.load(session_id)
        if record_payload is None:
            return
        record = SupportAiSessionRecord.model_validate(record_payload)
        try:
            prompt = self._build_follow_up_prompt(record, message)
            answer = self.llm_client.generate_text(prompt)
            content = answer.strip() or "I could not generate a follow-up answer."
            ai_message = SupportAiMessage(role="ai_agent", content=content)
            record.messages.append(ai_message.model_dump(mode="json"))
            record.status = "completed"
            record.updated_at = _utc_now()
            self.session_store.save(session_id, record.model_dump(mode="json"))
            self.session_store.append_event(
                session_id,
                {
                    "event": "message_result",
                    "data": {
                        "session_id": session_id,
                        "status": "completed",
                        "message": ai_message.model_dump(mode="json"),
                        "updated_at": record.updated_at,
                    },
                },
            )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)
            record.updated_at = _utc_now()
            self.session_store.save(session_id, record.model_dump(mode="json"))
            self.session_store.append_event(
                session_id,
                {
                    "event": "error",
                    "data": {
                        "session_id": session_id,
                        "message": str(exc),
                        "updated_at": record.updated_at,
                    },
                },
            )

    def _run_graph_with_events(self, session_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        state: dict[str, Any] = {}
        emitted_trace_count = 0
        if hasattr(self.graph, "stream"):
            for update in self.graph.stream(payload):
                if not isinstance(update, dict):
                    continue
                for node_name, node_state in update.items():
                    self.session_store.append_event(
                        session_id,
                        {
                            "event": "node_update",
                            "data": {
                                "session_id": session_id,
                                "node": node_name,
                                "summary": _summarize_node_update(node_name, node_state),
                            },
                        },
                    )
                    if isinstance(node_state, dict):
                        state.update(node_state)
                        trace_entries = state.get("investigation_trace", [])
                        if isinstance(trace_entries, list):
                            new_entries = trace_entries[emitted_trace_count:]
                            for entry in new_entries:
                                self.session_store.append_event(
                                    session_id,
                                    {
                                        "event": "trace",
                                        "data": {
                                            "session_id": session_id,
                                            "message": entry,
                                        },
                                    },
                                )
                            emitted_trace_count = len(trace_entries)
            return state

        state = self.graph.invoke(payload)
        trace_entries = state.get("investigation_trace", [])
        if isinstance(trace_entries, list):
            for entry in trace_entries:
                self.session_store.append_event(
                    session_id,
                    {
                        "event": "trace",
                        "data": {
                            "session_id": session_id,
                            "message": entry,
                        },
                    },
                )
        return state


def _initial_messages(prompt: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if prompt:
        messages.append({"role": "support_agent", "content": prompt})
    return messages


def _safe_state_dump(state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(state)
    if "final_result" in payload and hasattr(payload["final_result"], "model_dump"):
        payload["final_result"] = payload["final_result"].model_dump(mode="json")
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sse_event(event_name: str, payload: dict[str, Any]) -> str:
    return f"event: {event_name}\ndata: {json.dumps(payload, default=str)}\n\n"


def _humanize_stream_event(item: dict[str, Any]) -> dict[str, Any]:
    event_name = item.get("event")
    payload = dict(item.get("data", {}))
    if event_name == "trace" and isinstance(payload.get("message"), str):
        raw_message = payload["message"]
        payload["raw_message"] = raw_message
        payload["message"] = _humanize_trace_message(raw_message)
    return payload


def _humanize_trace_message(message: str) -> str:
    replacements = {
        "Loaded ticket ": "I loaded ticket ",
        "Normalized issue summary: ": "I understood the issue as: ",
        "Classified as category=": "I classified the case as ",
        "Retrieved 0 context matches from knowledge base.": "I did not find any matching knowledge-base context for this case.",
        "Resolved user from mobile number.": "I found the customer profile from the mobile number.",
        "Matched VIN to ownership record.": "I confirmed that this VIN matches the customer's ownership record.",
        "Loaded vehicle details for the linked VIN.": "I loaded the vehicle details linked to this VIN.",
        "Heartbeat says the vehicle is offline; checking telematics.": "The heartbeat says the vehicle is offline, so I checked live telematics next.",
        "Fresh telematics exists, so this looks like heartbeat inconsistency.": "The vehicle is still sending fresh telematics data, so this looks like a heartbeat-status mismatch.",
        "No recent trip data found for this VIN.": "I could not find any recent trip data for this VIN.",
        "No payment record found for the investigated order.": "I could not find a payment record for the order I checked.",
    }
    if message in replacements:
        return replacements[message]
    if message.startswith("Planned tools: "):
        return "I planned the initial checks needed for this investigation."
    if message.startswith("Ran "):
        tool_name = message[len("Ran ") :].split(" and received fields:", 1)[0]
        return f"I completed the {tool_name} check."
    if message.startswith("Queued follow-up tool "):
        tool_name = message[len("Queued follow-up tool ") :].split(" based on discovered facts.", 1)[0]
        return f"Based on what I found, I queued the next check: {tool_name}."
    if message.startswith("Found ") and " related orders" in message:
        return message.replace("Found", "I found", 1)
    if message.startswith("Loaded order ") and " with status " in message:
        return message.replace("Loaded order", "I checked order", 1)
    if message.startswith("Finished with "):
        return message.replace("Finished with ", "Final diagnosis: ", 1)
    return message


def _extract_order_number(message: str) -> str | None:
    match = re.search(r"\b\d{4}-\d{6}-\d{4}\b", message)
    if match:
        return match.group(0)
    return None


def _looks_like_confirmation(message: str) -> bool:
    lowered = message.lower()
    confirmation_terms = (
        "confirm",
        "confirmed",
        "yes",
        "correct",
        "this order",
        "that order",
        "right order",
    )
    return any(term in lowered for term in confirmation_terms)


def _summarize_node_update(node_name: str, node_state: Any) -> str:
    if not isinstance(node_state, dict):
        return "completed"
    if node_name == "normalize_issue":
        return f"summary={node_state.get('normalized_issue_summary')}"
    if node_name == "classify_issue":
        return (
            f"category={node_state.get('issue_category')} "
            f"problem_type={node_state.get('problem_type')} "
            f"confidence={node_state.get('confidence')}"
        )
    if node_name == "retrieve_context":
        matches = node_state.get("retrieved_matches", [])
        return f"retrieved_matches={len(matches) if isinstance(matches, list) else 0}"
    if node_name == "plan_investigation":
        plan = node_state.get("investigation_plan")
        if plan is not None:
            return f"tools={','.join(plan.required_tools) or 'none'}"
    if node_name == "run_tools":
        results = node_state.get("tool_results", [])
        tool_names = [item.get("name") for item in results if isinstance(item, dict)]
        clarifications = node_state.get("clarification_requests", [])
        return f"executed_tools={','.join(tool_names) or 'none'} clarifications={json.dumps(clarifications, default=str)}"
    if node_name == "finalize":
        result = node_state.get("final_result")
        if result is not None:
            return f"decision={result.decision} confidence={result.confidence}"
    return "completed"
