from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

from support_agent.agent.investigation_rules import (
    candidate_order_numbers,
    follow_up_rule_tools,
    has_multiple_order_candidates,
    initial_rule_tools,
    order_candidates_ready,
    order_candidates,
)
from support_agent.agent.state import AgentState
from support_agent.llm.client import LlamaClient
from support_agent.llm.prompts import (
    build_classification_prompt,
    build_final_response_prompt,
    build_investigation_plan_prompt,
    build_normalize_prompt,
)
from support_agent.runtime import log_event
from support_agent.runtime.errors import SupportAgentError
from support_agent.schemas.agent import AgentDecision, AgentResult, InvestigationPlan, NextAction
from support_agent.tools.base import ToolRegistry


@dataclass
class NodeDependencies:
    settings: Any
    llm_client: LlamaClient
    retriever: Any
    tool_registry: ToolRegistry


def load_ticket(state: AgentState) -> AgentState:
    state.setdefault("facts", {})
    state.setdefault("tool_results", [])
    state.setdefault("tool_failures", [])
    state.setdefault("clarification_requests", [])
    state.setdefault("investigation_trace", [])
    state.setdefault("hypotheses", [])
    return {"investigation_trace": _append_trace(state, f"Loaded ticket {state.get('ticket_id', 'unknown')}")}


def normalize_issue(state: AgentState, deps: NodeDependencies) -> AgentState:
    conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state.get("conversation_history", []))
    prompt = build_normalize_prompt(state["raw_user_message"], conversation)
    try:
        text = deps.llm_client.generate_text(prompt)
        summary = _extract_summary(text)
    except SupportAgentError as exc:
        summary = state["raw_user_message"].strip()
        log_event("node_fallback", node_name="normalize_issue", ticket_id=state.get("ticket_id"), error_type=exc.error_type)
    return {
        "normalized_issue_summary": summary,
        "investigation_trace": _append_trace(state, f"Normalized issue summary: {summary}"),
    }


def classify_issue(state: AgentState, deps: NodeDependencies) -> AgentState:
    prompt = build_classification_prompt(state["normalized_issue_summary"])
    try:
        decision = deps.llm_client.generate_structured(prompt, AgentDecision)
    except SupportAgentError as exc:
        normalized = state["normalized_issue_summary"]
        decision = AgentDecision(
            normalized_issue_summary=normalized,
            issue_category="unknown",
            problem_type="unknown",
            confidence=0.2,
        )
        log_event("node_fallback", node_name="classify_issue", ticket_id=state.get("ticket_id"), error_type=exc.error_type)
    return {
        "normalized_issue_summary": decision.normalized_issue_summary,
        "issue_category": decision.issue_category,
        "problem_type": decision.problem_type,
        "confidence": decision.confidence,
        "investigation_trace": _append_trace(
            state,
            f"Classified as category={decision.issue_category}, problem_type={decision.problem_type}, confidence={decision.confidence}",
        ),
    }


def retrieve_context(state: AgentState, deps: NodeDependencies) -> AgentState:
    retrieved = deps.retriever.retrieve(state["normalized_issue_summary"])
    return {
        "retrieved_context": retrieved["formatted_context"],
        "retrieved_matches": retrieved["matches"],
        "investigation_trace": _append_trace(
            state,
            f"Retrieved {len(retrieved['matches'])} context matches from knowledge base.",
        ),
    }


def plan_investigation(state: AgentState, deps: NodeDependencies) -> AgentState:
    ticket_context = _build_ticket_context(state)
    prompt = build_investigation_plan_prompt(
        issue_summary=state["normalized_issue_summary"],
        issue_category=state["issue_category"],
        problem_type=state["problem_type"],
        available_tools=deps.tool_registry.names(),
        ticket_context=ticket_context,
        rag_context=state.get("retrieved_context", ""),
    )
    try:
        plan = deps.llm_client.generate_structured(prompt, InvestigationPlan)
    except SupportAgentError as exc:
        plan = InvestigationPlan(
            rationale=f"Fallback planning due to {exc.error_type}.",
            required_tools=[],
            tool_arguments={},
            should_stop_after_tools=False,
        )
        log_event("node_fallback", node_name="plan_investigation", ticket_id=state.get("ticket_id"), error_type=exc.error_type)
    return {
        "investigation_plan": plan,
        "investigation_trace": _append_trace(
            state,
            f"Planned tools: {', '.join(plan.required_tools) or 'none'}. Rationale: {plan.rationale}",
        ),
    }


def run_tools(state: AgentState, deps: NodeDependencies) -> AgentState:
    working_state: dict[str, Any] = dict(state)
    plan = state["investigation_plan"]
    facts = dict(state.get("facts", {}))
    raw_results: list[dict[str, Any]] = list(state.get("tool_results", []))
    tool_failures: list[dict[str, Any]] = list(state.get("tool_failures", []))
    clarification_requests: list[dict[str, Any]] = list(state.get("clarification_requests", []))
    trace = list(state.get("investigation_trace", []))
    tool_queue = _dedupe_tool_queue(initial_rule_tools(working_state) + plan.required_tools)
    planned_tools = set(plan.required_tools)
    completed_tools: set[str] = set()

    while tool_queue:
        working_state.update(_propagate_identifiers_from_facts(working_state, facts))
        tool_name = tool_queue.pop(0)
        if tool_name in completed_tools:
            continue
        arguments = dict(plan.tool_arguments.get(tool_name, {}))
        arguments = _hydrate_tool_arguments(working_state, arguments)
        if tool_name in {"get_order_details", "get_booking_details"} and arguments.get("order_number") in (None, "") and working_state.get("order_number"):
            arguments["order_number"] = working_state["order_number"]
        resolved_user_id = _resolve_user_id(working_state, facts)
        if resolved_user_id and arguments.get("user_id") in (None, ""):
            arguments["user_id"] = resolved_user_id
        resolved_vehicle_vin = _resolve_vehicle_vin(working_state, facts)
        if resolved_vehicle_vin and arguments.get("vin") in (None, ""):
            arguments["vin"] = resolved_vehicle_vin
        if tool_name == "get_vehicle_details" and resolved_vehicle_vin and arguments.get("vehicle_id") in (None, ""):
            arguments["vehicle_id"] = resolved_vehicle_vin
        if not deps.tool_registry.has_tool(tool_name):
            if tool_name not in planned_tools:
                trace.append(f"Skipped optional follow-up tool {tool_name} because it is not registered.")
                completed_tools.add(tool_name)
                continue
            clarification_requests.append(
                {
                    "tool_name": tool_name,
                    "missing_arguments": [],
                    "reason": "unknown_tool",
                }
            )
            trace.append(f"Planned tool {tool_name} is not registered.")
            completed_tools.add(tool_name)
            continue
        missing_arguments = [
            parameter
            for parameter in deps.tool_registry.required_parameters(tool_name)
            if arguments.get(parameter) in (None, "")
        ]
        if tool_name in {"get_order_details", "get_booking_details"} and not (
            arguments.get("order_id") or arguments.get("order_number") or arguments.get("booking_id")
        ):
            missing_arguments = ["order_reference"]
        if missing_arguments:
            facts, raw_results, fallback_completed = _run_user_context_fallbacks(
                tool_name=tool_name,
                state=working_state,
                deps=deps,
                facts=facts,
                raw_results=raw_results,
            )
            completed_tools.update(fallback_completed)
            working_state.update(_propagate_identifiers_from_facts(working_state, facts))
            arguments = _hydrate_tool_arguments(working_state, arguments)
            if tool_name in {"get_order_details", "get_booking_details"} and arguments.get("order_number") in (None, "") and working_state.get("order_number"):
                arguments["order_number"] = working_state["order_number"]
            resolved_user_id = _resolve_user_id(working_state, facts)
            if resolved_user_id and arguments.get("user_id") in (None, ""):
                arguments["user_id"] = resolved_user_id
            resolved_vehicle_vin = _resolve_vehicle_vin(working_state, facts)
            if resolved_vehicle_vin and arguments.get("vin") in (None, ""):
                arguments["vin"] = resolved_vehicle_vin
            if tool_name == "get_vehicle_details" and resolved_vehicle_vin and arguments.get("vehicle_id") in (None, ""):
                arguments["vehicle_id"] = resolved_vehicle_vin
            trace = _append_tool_trace(trace, tool_name, facts, working_state)
            missing_arguments = [
                parameter
                for parameter in deps.tool_registry.required_parameters(tool_name)
                if arguments.get(parameter) in (None, "")
            ]
            if tool_name in {"get_order_details", "get_booking_details"} and not (
                arguments.get("order_id") or arguments.get("order_number") or arguments.get("booking_id")
            ):
                missing_arguments = ["order_reference"]
        if missing_arguments and tool_name in {"get_order_details", "get_booking_details"}:
            if has_multiple_order_candidates(facts):
                clarification_requests.append(
                    {
                        "tool_name": tool_name,
                        "missing_arguments": ["order_id"],
                    }
                )
                trace.append(f"Found multiple order candidates before {tool_name}; asking customer to confirm order number.")
                completed_tools.add(tool_name)
                continue
            if facts.get("related_orders") or facts.get("user_enquiries"):
                trace.append(f"Skipped {tool_name} until customer confirms which order candidate to inspect.")
                completed_tools.add(tool_name)
                continue
        if missing_arguments and tool_name == "get_vehicle_details":
            if facts.get("ownership_record") or facts.get("vehicle_details"):
                trace.append(f"Skipped {tool_name} because ownership linkage facts are already available.")
                completed_tools.add(tool_name)
                continue
        if missing_arguments:
            clarification_requests.append(
                {
                    "tool_name": tool_name,
                    "missing_arguments": missing_arguments,
                }
            )
            trace.append(f"Cannot run {tool_name}; missing required inputs: {', '.join(missing_arguments)}.")
            completed_tools.add(tool_name)
            continue
        result = deps.tool_registry.run_safe(tool_name, arguments)
        raw_results.append(result.model_dump())
        if not result.success and result.error:
            tool_failures.append({"tool_name": tool_name, "error": result.error})
            trace.append(f"Tool {tool_name} failed: {result.error}.")
            log_event(
                "tool_failed",
                ticket_id=state.get("ticket_id"),
                tool_name=tool_name,
                error=result.error,
                arguments=arguments,
            )
            completed_tools.add(tool_name)
            continue
        facts.update(result.payload)
        working_state.update(_propagate_identifiers_from_facts(working_state, facts))
        trace.append(f"Ran {tool_name} and received fields: {', '.join(sorted(result.payload.keys()))}.")
        trace = _append_live_summary_trace(trace, tool_name, facts, working_state, deps)
        log_event(
            "tool_succeeded",
            ticket_id=state.get("ticket_id"),
            tool_name=tool_name,
            payload_keys=sorted(result.payload.keys()),
        )
        completed_tools.add(tool_name)
        for follow_up_tool in follow_up_rule_tools(working_state, facts):
            if follow_up_tool not in completed_tools and follow_up_tool not in tool_queue:
                tool_queue.append(follow_up_tool)
                trace.append(f"Queued follow-up tool {follow_up_tool} based on discovered facts.")

    if _needs_payment_reference_clarification(plan.required_tools, facts):
        clarification_requests.append(
            {
                "tool_name": "get_payment_status",
                "missing_arguments": ["payment_id"],
            }
        )
        trace.append("Payment verification still needs a transaction reference from the customer.")

    if _needs_order_candidate_clarification(working_state, facts, clarification_requests):
        clarification_requests.append(
            {
                "tool_name": "get_order_details",
                "missing_arguments": ["order_id"],
            }
        )
        trace.append("Multiple order or enquiry candidates remain; asking customer to confirm the correct order number.")

    return {
        "facts": facts,
        "tool_results": raw_results,
        "tool_failures": tool_failures,
        "clarification_requests": _dedupe_clarification_requests(clarification_requests),
        "investigation_trace": trace,
        **_propagate_identifiers_from_facts(working_state, facts),
    }


def finalize_response(state: AgentState, deps: NodeDependencies) -> AgentState:
    deterministic_mobile_result = _build_mobile_vehicle_linkage_result(state, deps)
    if deterministic_mobile_result is not None:
        deterministic_mobile_result.facts = _prune_final_facts(state, state.get("facts", {}))
        return {
            "next_action": deterministic_mobile_result.decision,
            "final_result": deterministic_mobile_result,
            "investigation_trace": _append_trace(
                state,
                f"Finished with deterministic mobile linkage diagnosis: {deterministic_mobile_result.internal_summary}",
            ),
        }
    deterministic_pending_order_result = _build_pending_order_result(state)
    if deterministic_pending_order_result is not None:
        deterministic_pending_order_result.facts = _prune_final_facts(state, state.get("facts", {}))
        return {
            "next_action": deterministic_pending_order_result.decision,
            "final_result": deterministic_pending_order_result,
            "investigation_trace": _append_trace(
                state,
                f"Finished with deterministic pending-order diagnosis: {deterministic_pending_order_result.internal_summary}",
            ),
        }
    clarification_requests = state.get("clarification_requests", [])
    if clarification_requests:
        final = _build_clarification_result(state, clarification_requests)
        final.facts = _prune_final_facts(state, state.get("facts", {}))
        return {
            "next_action": final.decision,
            "final_result": final,
            "investigation_trace": _append_trace(state, f"Finished with clarification request: {final.customer_response}"),
        }
    if state.get("tool_failures"):
        final = _build_fallback_final_result(state)
        final.facts = _prune_final_facts(state, state.get("facts", {}))
        return {
            "next_action": final.decision,
            "final_result": final,
            "investigation_trace": _append_trace(state, f"Finished with decision={final.decision} confidence={final.confidence}"),
        }

    facts_text = _stringify(state.get("facts", {}))
    tool_results_text = _stringify(state.get("tool_results", []))
    prompt = build_final_response_prompt(
        issue_summary=state["normalized_issue_summary"],
        issue_category=state["issue_category"],
        problem_type=state["problem_type"],
        rag_context=state.get("retrieved_context", ""),
        facts=facts_text,
        tool_results=tool_results_text,
    )
    try:
        final = deps.llm_client.generate_structured(prompt, AgentResult)
    except (SupportAgentError, ValueError):
        final = _build_fallback_final_result(state)
    if not final.customer_response.strip():
        final = _build_fallback_final_result(state)
    final.ticket_id = state["ticket_id"]
    final.facts = _prune_final_facts(state, state.get("facts", {}))
    return {
        "next_action": final.decision,
        "final_result": final,
        "investigation_trace": _append_trace(state, f"Finished with decision={final.decision} confidence={final.confidence}"),
    }


def _extract_summary(raw_text: str) -> str:
    stripped = raw_text.strip()
    try:
        from support_agent.llm.parser import _extract_json_object
        import json

        json_fragment = _extract_json_object(stripped)
        if json_fragment is not None:
            payload = json.loads(json_fragment)
            if "normalized_issue_summary" in payload:
                return str(payload["normalized_issue_summary"]).strip()
    except Exception:
        pass

    marker = "Normalized Issue Summary:"
    if marker.lower() in stripped.lower():
        import re

        match = re.search(r"Normalized Issue Summary:\s*(.+)", stripped, flags=re.IGNORECASE | re.DOTALL)
        if match:
            first_line = match.group(1).strip().splitlines()[0]
            return first_line.strip("* ").strip()
    return stripped


def _build_ticket_context(state: AgentState) -> str:
    identifiers = {
        "ticket_id": state.get("ticket_id"),
        "user_id": state.get("user_id"),
        "mobile": state.get("mobile"),
        "booking_id": state.get("booking_id"),
        "payment_id": state.get("payment_id"),
        "order_id": state.get("order_id"),
        "order_number": state.get("order_number"),
        "vehicle_id": state.get("vehicle_id"),
        "vehicle_vin": state.get("vehicle_vin"),
    }
    history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state.get("conversation_history", []))
    return f"Identifiers: {identifiers}\nConversation:\n{history}"


def _hydrate_tool_arguments(state: AgentState, arguments: dict[str, Any]) -> dict[str, Any]:
    fallback_fields = (
        "ticket_id",
        "user_id",
        "mobile",
        "booking_id",
        "payment_id",
        "order_id",
        "order_number",
        "vehicle_id",
        "vehicle_vin",
    )
    for field in fallback_fields:
        if field not in arguments and state.get(field):
            if field == "ticket_id" and arguments.get("ticket_id") is None:
                arguments["ticket_id"] = state[field]
            if field == "user_id" and arguments.get("user_id") is None:
                arguments["user_id"] = state[field]
            if field == "mobile" and arguments.get("mobile") is None:
                arguments["mobile"] = state[field]
            if field == "booking_id" and arguments.get("booking_id") is None:
                arguments["booking_id"] = state[field]
            if field == "payment_id" and arguments.get("payment_id") is None:
                arguments["payment_id"] = state[field]
            if field == "order_id" and arguments.get("order_id") is None:
                arguments["order_id"] = state[field]
            if field == "order_number" and arguments.get("order_number") is None:
                arguments["order_number"] = state[field]
            if field == "vehicle_id" and arguments.get("vehicle_id") is None:
                arguments["vehicle_id"] = state[field]
            if field == "vehicle_vin" and arguments.get("vin") is None:
                arguments["vin"] = state[field]
    return arguments


def _stringify(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, default=str)


def _append_trace(state: AgentState | dict[str, Any], message: str) -> list[str]:
    return [*state.get("investigation_trace", []), message]


def _append_tool_trace(
    trace: list[str],
    tool_name: str,
    facts: dict[str, Any],
    working_state: dict[str, Any],
) -> list[str]:
    updates: list[str] = list(trace)
    if tool_name == "get_user_profile_by_mobile" and working_state.get("user_id"):
        updates.append(f"Resolved user_id={working_state['user_id']} from mobile number.")
    if tool_name in {"search_related_orders", "get_user_enquiries"}:
        related_count = len(facts.get("related_orders", []) or [])
        enquiry_count = len(facts.get("user_enquiries", []) or [])
        updates.append(f"Current order candidates: {related_count} orders, {enquiry_count} enquiries.")
    return updates


def _append_live_summary_trace(
    trace: list[str],
    tool_name: str,
    facts: dict[str, Any],
    working_state: dict[str, Any],
    deps: NodeDependencies,
) -> list[str]:
    updates: list[str] = list(trace)
    if tool_name == "get_user_profile_by_mobile" and working_state.get("user_id"):
        updates.append("Resolved user from mobile number.")
    elif tool_name in {"search_related_orders", "get_user_enquiries"}:
        related_orders = facts.get("related_orders", []) if isinstance(facts.get("related_orders"), list) else []
        user_enquiries = facts.get("user_enquiries", []) if isinstance(facts.get("user_enquiries"), list) else []
        delivered_count = sum(
            1 for order in related_orders if isinstance(order, dict) and str(order.get("status", "")).upper() == "DELIVERED"
        )
        updates.append(
            f"Found {len(related_orders)} related orders, {delivered_count} delivered, and {len(user_enquiries)} active enquiries."
        )
    elif tool_name == "get_ownership_record":
        ownership_record = facts.get("ownership_record", {}) if isinstance(facts.get("ownership_record"), dict) else {}
        if ownership_record.get("vin"):
            updates.append("Matched VIN to ownership record.")
    elif tool_name == "get_vehicle_details":
        vehicle_details = facts.get("vehicle_details", {}) if isinstance(facts.get("vehicle_details"), dict) else {}
        if vehicle_details:
            updates.append("Loaded vehicle details for the linked VIN.")
    elif tool_name == "get_order_details":
        order_details = facts.get("order_details", {}) if isinstance(facts.get("order_details"), dict) else {}
        if order_details:
            order_number = order_details.get("order_number") or order_details.get("id")
            status = order_details.get("status")
            updates.append(f"Loaded order {order_number} with status {status}.")
    elif tool_name == "get_order_payment_status":
        payment_status = facts.get("payment_status", {}) if isinstance(facts.get("payment_status"), dict) else {}
        if payment_status:
            updates.append("Found payment status for the investigated order.")
        else:
            updates.append("No payment record found for the investigated order.")
    elif tool_name == "get_vehicle_last_seen":
        vehicle_last_seen = facts.get("vehicle_last_seen", {}) if isinstance(facts.get("vehicle_last_seen"), dict) else {}
        is_active = vehicle_last_seen.get("is_active")
        if is_active is False:
            updates.append("Heartbeat says the vehicle is offline; checking telematics.")
        elif is_active is True:
            updates.append("Heartbeat says the vehicle is online.")
        else:
            updates.append("Heartbeat data is missing; checking telematics.")
    elif tool_name == "get_telematics_signal_summary":
        telematics_status = facts.get("telematics_status", {}) if isinstance(facts.get("telematics_status"), dict) else {}
        if telematics_status.get("has_signal_data") is False:
            updates.append("No telematics data found for this VIN.")
        elif _is_recent_epoch_millis(telematics_status.get("latest_event_time"), getattr(deps.settings, "telematics_freshness_minutes", 10)):
            updates.append("Fresh telematics exists, so this looks like heartbeat inconsistency.")
        elif telematics_status:
            updates.append("Telematics data exists for this VIN.")
    elif tool_name == "get_trip_history_summary":
        trip_history_status = facts.get("trip_history_status", {}) if isinstance(facts.get("trip_history_status"), dict) else {}
        if trip_history_status.get("has_trip_data") is False:
            updates.append("No recent trip data found for this VIN.")
        elif trip_history_status:
            updates.append("Trip history exists for this VIN.")
    elif tool_name == "get_charging_history_summary":
        charging_history_status = facts.get("charging_history_status", {}) if isinstance(facts.get("charging_history_status"), dict) else {}
        if charging_history_status.get("has_charging_data") is False:
            updates.append("No charging history found for this VIN.")
        elif charging_history_status:
            updates.append("Charging history exists for this VIN.")
    return updates


def _build_fallback_final_result(state: AgentState) -> AgentResult:
    facts = state.get("facts", {})
    issue_summary = state.get("normalized_issue_summary", state["raw_user_message"])
    issue_category = state.get("issue_category", "unknown")
    problem_type = state.get("problem_type", "unknown")
    has_order_details = bool(facts.get("order_details") or facts.get("booking_details"))
    has_payment_details = bool(facts.get("payment_status"))
    has_ownership = bool(facts.get("ownership_record"))
    has_tool_failures = bool(state.get("tool_failures"))

    if has_tool_failures:
        decision = NextAction.escalate
        customer_response = "We hit a system issue while investigating your request. The support team has been asked to continue this manually."
        internal_summary = "Built fallback escalation result because one or more investigation tools failed."
        confidence = 0.3
    elif _is_mobile_vehicle_linkage_gap(state):
        decision = NextAction.pending
        customer_response = "Your app login appears blocked because no delivered vehicle is fully linked to your account yet. The support team should verify order delivery, ownership mapping, and primary VIN setup."
        internal_summary = "Detected mobile app vehicle-linkage gap from user, order, and ownership facts."
        confidence = 0.8
    elif has_order_details or has_payment_details or has_ownership:
        decision = NextAction.pending
        customer_response = "I found the relevant account records and the support team is reviewing the details."
        internal_summary = "Built fallback final result from retrieved investigation facts after invalid LLM output."
        confidence = min(max(state.get("confidence", 0.7), 0.0), 1.0)
    elif has_multiple_order_candidates(facts):
        decision = NextAction.needs_clarification
        customer_response = _build_order_confirmation_hint(
            facts.get("related_orders", []),
            facts.get("user_enquiries", []),
        ) or "Please confirm the correct order number so I can continue."
        internal_summary = "Built fallback clarification result because multiple order or enquiry candidates were found."
        confidence = 0.5
    else:
        decision = NextAction.needs_clarification
        customer_response = "Please share the missing investigation details so I can continue."
        internal_summary = "Built fallback clarification result because the LLM final output was invalid and investigation facts were incomplete."
        confidence = 0.5

    return AgentResult(
        ticket_id=state["ticket_id"],
        issue_summary=issue_summary,
        issue_category=issue_category,
        problem_type=problem_type,
        decision=decision,
        customer_response=customer_response,
        internal_summary=internal_summary,
        facts=facts,
        confidence=confidence,
    )


def _needs_payment_reference_clarification(required_tools: list[str], facts: dict[str, Any]) -> bool:
    if "get_payment_status" not in required_tools:
        return False
    has_payment_result = bool(facts.get("payment_status"))
    has_orders = bool(facts.get("related_orders"))
    has_enquiries = bool(facts.get("user_enquiries"))
    return not has_payment_result and (has_orders or has_enquiries)


def _needs_order_candidate_clarification(
    state: dict[str, Any],
    facts: dict[str, Any],
    clarification_requests: list[dict[str, Any]],
) -> bool:
    if state.get("order_number"):
        return False
    if state.get("order_id"):
        return False
    if not has_multiple_order_candidates(facts):
        return False
    for request in clarification_requests:
        if request.get("tool_name") in {"get_order_details", "get_booking_details"}:
            return False
    return True


def _run_user_context_fallbacks(
    *,
    tool_name: str,
    state: AgentState,
    deps: NodeDependencies,
    facts: dict[str, Any],
    raw_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    completed_tools: set[str] = set()
    user_id = _resolve_user_id(state, facts)
    if not user_id and state.get("mobile") and deps.tool_registry.has_tool("get_user_profile_by_mobile"):
        result = deps.tool_registry.run_safe("get_user_profile_by_mobile", {"mobile": state["mobile"]})
        raw_results.append(result.model_dump())
        if result.success:
            facts.update(result.payload)
            completed_tools.add("get_user_profile_by_mobile")
        state.update(_propagate_identifiers_from_facts(state, facts))
        user_id = _resolve_user_id(state, facts)
    if not user_id:
        return facts, raw_results, completed_tools

    fallback_tools: list[str] = []
    if tool_name in {"get_order_details", "get_booking_details"}:
        fallback_tools = ["search_related_orders", "get_user_enquiries"]
    elif tool_name == "get_payment_status":
        fallback_tools = ["search_related_orders", "get_user_enquiries"]
    elif tool_name == "get_vehicle_details":
        fallback_tools = []
        if state.get("order_id") and deps.tool_registry.has_tool("get_order_details") and not facts.get("order_details"):
            fallback_tools.append("get_order_details")
        if deps.tool_registry.has_tool("get_ownership_record") and not facts.get("ownership_record"):
            fallback_tools.append("get_ownership_record")

    for fallback_tool in fallback_tools:
        if fallback_tool == "search_related_orders" and facts.get("related_orders") is not None:
            continue
        if fallback_tool == "get_user_enquiries" and facts.get("user_enquiries") is not None:
            continue
        if not deps.tool_registry.has_tool(fallback_tool):
            continue
        fallback_arguments = {"user_id": user_id}
        if fallback_tool == "get_order_details" and state.get("order_id"):
            fallback_arguments = {"order_id": state["order_id"]}
        elif fallback_tool == "get_ownership_record":
            fallback_arguments = {"user_id": user_id}
            if state.get("order_id"):
                fallback_arguments["order_id"] = state["order_id"]
        result = deps.tool_registry.run_safe(fallback_tool, fallback_arguments)
        raw_results.append(result.model_dump())
        if result.success:
            facts.update(result.payload)
            completed_tools.add(fallback_tool)
        state.update(_propagate_identifiers_from_facts(state, facts))
    return facts, raw_results, completed_tools


def _build_clarification_result(state: AgentState, clarification_requests: list[dict[str, Any]]) -> AgentResult:
    missing_arguments = sorted(
        {
            argument
            for request in clarification_requests
            for argument in request.get("missing_arguments", [])
        }
    )
    clarification_lines: list[str] = []
    related_orders = state.get("facts", {}).get("related_orders", [])
    user_enquiries = state.get("facts", {}).get("user_enquiries", [])
    if "payment_id" in missing_arguments:
        clarification_lines.append("Please share your UTR number or transaction ID so I can verify the payment.")
    if "order_id" in missing_arguments or "order_reference" in missing_arguments:
        order_hint = _build_order_confirmation_hint(related_orders, user_enquiries)
        if order_hint:
            clarification_lines.append(order_hint)
        else:
            clarification_lines.append("Please share your order ID or order number.")
    if "ticket_id" in missing_arguments:
        clarification_lines.append("Please share the ticket ID or ticket code.")

    if not clarification_lines:
        clarification_lines.append("Please share the missing investigation details so I can continue.")

    internal_summary = (
        "Investigation paused pending customer clarification. "
        f"Missing inputs: {', '.join(missing_arguments)}."
    )
    return AgentResult(
        ticket_id=state["ticket_id"],
        issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
        issue_category=state.get("issue_category", "unknown"),
        problem_type=state.get("problem_type", "unknown"),
        decision=NextAction.needs_clarification,
        customer_response=" ".join(clarification_lines),
        internal_summary=internal_summary,
        facts=state.get("facts", {}),
        confidence=min(state.get("confidence", 0.0), 0.6),
    )


def _dedupe_clarification_requests(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[dict[str, Any]] = []
    for request in requests:
        key = (
            request.get("tool_name", ""),
            tuple(sorted(request.get("missing_arguments", []))),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(request)
    return deduped


def _build_order_confirmation_hint(related_orders: list[dict[str, Any]], user_enquiries: list[dict[str, Any]]) -> str:
    candidates = candidate_order_numbers({"related_orders": related_orders, "user_enquiries": user_enquiries})
    if not candidates:
        return ""
    preview = ", ".join(candidates[:5])
    return f"I found these likely order numbers for your account: {preview}. Please confirm the correct order number."


def _resolve_user_id(state: AgentState, facts: dict[str, Any]) -> str | None:
    if state.get("user_id"):
        return state["user_id"]
    user_profile = facts.get("user_profile", {})
    if isinstance(user_profile, dict):
        user_id = user_profile.get("id")
        if isinstance(user_id, str) and user_id:
            return user_id
    return None


def _resolve_vehicle_vin(state: AgentState, facts: dict[str, Any]) -> str | None:
    vehicle_vin = state.get("vehicle_vin")
    if isinstance(vehicle_vin, str) and vehicle_vin:
        return vehicle_vin
    vehicle_last_seen = facts.get("vehicle_last_seen", {})
    if isinstance(vehicle_last_seen, dict):
        vin = vehicle_last_seen.get("vin")
        if isinstance(vin, str) and vin:
            return vin
    vehicle_details = facts.get("vehicle_details", {})
    if isinstance(vehicle_details, dict):
        vin = vehicle_details.get("vin")
        if isinstance(vin, str) and vin:
            return vin
    ownership_record = facts.get("ownership_record", {})
    if isinstance(ownership_record, dict):
        vin = ownership_record.get("vin")
        if isinstance(vin, str) and vin:
            return vin
    user_profile = facts.get("user_profile", {})
    if isinstance(user_profile, dict):
        vin = user_profile.get("primary_vin")
        if isinstance(vin, str) and vin:
            return vin
    return None


def _propagate_identifiers_from_facts(state: dict[str, Any], facts: dict[str, Any]) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    user_profile = facts.get("user_profile", {})
    if isinstance(user_profile, dict):
        if not state.get("user_id") and user_profile.get("id"):
            updates["user_id"] = user_profile["id"]
        if not state.get("mobile") and user_profile.get("mobile"):
            updates["mobile"] = user_profile["mobile"]
        if not state.get("vehicle_vin") and user_profile.get("primary_vin"):
            updates["vehicle_vin"] = user_profile["primary_vin"]

    ticket_details = facts.get("ticket_details", {})
    if isinstance(ticket_details, dict):
        if not state.get("user_id") and ticket_details.get("user_id"):
            updates["user_id"] = ticket_details["user_id"]

    order_details = facts.get("order_details", {})
    if isinstance(order_details, dict):
        if not state.get("order_id"):
            order_id = order_details.get("id")
            if order_id:
                updates["order_id"] = order_id
        if not state.get("user_id") and order_details.get("user_id"):
            updates["user_id"] = order_details["user_id"]
        if not state.get("order_number") and order_details.get("order_number"):
            updates["order_number"] = order_details["order_number"]

    booking_details = facts.get("booking_details", {})
    if isinstance(booking_details, dict):
        if not state.get("order_id"):
            order_id = booking_details.get("id")
            if order_id:
                updates["order_id"] = order_id
        if not state.get("user_id") and booking_details.get("user_id"):
            updates["user_id"] = booking_details["user_id"]
        if not state.get("order_number") and booking_details.get("order_number"):
            updates["order_number"] = booking_details["order_number"]

    payment_status = facts.get("payment_status", {})
    if isinstance(payment_status, dict):
        if not state.get("payment_id"):
            payment_id = payment_status.get("id") or payment_status.get("transaction_id")
            if payment_id:
                updates["payment_id"] = payment_id
        if not state.get("order_id") and payment_status.get("order_id"):
            updates["order_id"] = payment_status["order_id"]

    ownership_record = facts.get("ownership_record", {})
    if isinstance(ownership_record, dict):
        if not state.get("vehicle_vin") and ownership_record.get("vin"):
            updates["vehicle_vin"] = ownership_record["vin"]
        if not state.get("vehicle_id"):
            vehicle_id = ownership_record.get("id") or ownership_record.get("registration_number")
            if vehicle_id:
                updates["vehicle_id"] = vehicle_id
        if not state.get("order_id") and ownership_record.get("order_id"):
            updates["order_id"] = ownership_record["order_id"]
        if not state.get("user_id") and ownership_record.get("user_id"):
            updates["user_id"] = ownership_record["user_id"]

    vehicle_details = facts.get("vehicle_details", {})
    if isinstance(vehicle_details, dict):
        if not state.get("vehicle_vin") and vehicle_details.get("vin"):
            updates["vehicle_vin"] = vehicle_details["vin"]
        if not state.get("vehicle_id"):
            vehicle_id = vehicle_details.get("id") or vehicle_details.get("registration_number")
            if vehicle_id:
                updates["vehicle_id"] = vehicle_id
        if not state.get("order_id") and vehicle_details.get("order_id"):
            updates["order_id"] = vehicle_details["order_id"]
        if not state.get("user_id") and vehicle_details.get("user_id"):
            updates["user_id"] = vehicle_details["user_id"]

    vehicle_last_seen = facts.get("vehicle_last_seen", {})
    if isinstance(vehicle_last_seen, dict):
        if not state.get("vehicle_vin") and vehicle_last_seen.get("vin"):
            updates["vehicle_vin"] = vehicle_last_seen["vin"]

    if order_candidates_ready(state, facts):
        candidates = order_candidates(facts)
        if len(candidates) == 1:
            candidate = candidates[0]
            if not state.get("order_id") and candidate.get("id"):
                updates["order_id"] = candidate["id"]
            if not state.get("order_number") and candidate.get("order_number"):
                updates["order_number"] = candidate["order_number"]

    return {key: value for key, value in updates.items() if value not in (None, "")}


def _is_mobile_vehicle_linkage_gap(state: AgentState) -> bool:
    category = str(state.get("issue_category", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    if category not in {"mobile_app", "app_sync", "login"} and problem_type not in {"login", "vehicle_linking", "viewing_order_details"}:
        return False
    facts = state.get("facts", {})
    user_profile = facts.get("user_profile", {}) if isinstance(facts.get("user_profile"), dict) else {}
    ownership_record = facts.get("ownership_record", {}) if isinstance(facts.get("ownership_record"), dict) else {}
    order_details = facts.get("order_details", {}) if isinstance(facts.get("order_details"), dict) else {}
    primary_vin = user_profile.get("primary_vin")
    ownership_vin = ownership_record.get("vin")
    order_status = str(order_details.get("status", "")).upper()
    if order_status == "DELIVERED" and (not ownership_vin or not primary_vin or primary_vin != ownership_vin):
        return True
    if not ownership_vin and primary_vin is None and state.get("order_id"):
        return True
    return False


def _build_mobile_vehicle_linkage_result(state: AgentState, deps: NodeDependencies) -> AgentResult | None:
    category = str(state.get("issue_category", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    if category not in {"mobile_app", "app_sync", "login"} and problem_type not in {"login", "app_sync", "vehicle_linking"}:
        return None

    facts = state.get("facts", {})
    user_profile = facts.get("user_profile", {}) if isinstance(facts.get("user_profile"), dict) else {}
    order_details = facts.get("order_details", {}) if isinstance(facts.get("order_details"), dict) else {}
    ownership_record = facts.get("ownership_record", {}) if isinstance(facts.get("ownership_record"), dict) else {}
    related_orders = facts.get("related_orders", []) if isinstance(facts.get("related_orders"), list) else []
    vehicle_last_seen = facts.get("vehicle_last_seen", {}) if isinstance(facts.get("vehicle_last_seen"), dict) else {}
    telematics_status = facts.get("telematics_status", {}) if isinstance(facts.get("telematics_status"), dict) else {}
    trip_history_status = facts.get("trip_history_status", {}) if isinstance(facts.get("trip_history_status"), dict) else {}
    charging_history_status = facts.get("charging_history_status", {}) if isinstance(facts.get("charging_history_status"), dict) else {}

    if not user_profile:
        return None

    delivered_orders = [
        order for order in related_orders
        if isinstance(order, dict) and str(order.get("status", "")).upper() == "DELIVERED"
    ]
    if related_orders and not delivered_orders:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                "I checked your account and none of the orders linked to it are in DELIVERED state yet. "
                "For the mobile app to attach a vehicle and work fully, at least one order must be delivered and then linked through ownership and primary VIN mapping."
            ),
            internal_summary="Mobile app issue traced to account having no delivered orders yet.",
            facts=facts,
            confidence=0.94,
        )

    order_id = order_details.get("id")
    order_number = order_details.get("order_number") or state.get("order_number")
    order_status = str(order_details.get("status", "")).upper()
    ownership_order_id = ownership_record.get("order_id")
    ownership_vin = ownership_record.get("vin")
    primary_vin = user_profile.get("primary_vin")
    ownership_linked_order = next(
        (
            order
            for order in related_orders
            if isinstance(order, dict) and ownership_order_id and order.get("id") == ownership_order_id
        ),
        {},
    )
    ownership_linked_status = str(ownership_linked_order.get("status", "")).upper()
    ownership_linked_order_number = ownership_linked_order.get("order_number") or ownership_order_id

    if (
        ownership_order_id
        and ownership_vin
        and primary_vin
        and ownership_vin == primary_vin
        and ownership_linked_status == "DELIVERED"
    ):
        order_id = ownership_order_id
        order_number = ownership_linked_order_number
        order_status = "DELIVERED"
    elif order_id and ownership_order_id and ownership_order_id != order_id:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                f"I found that order {order_number or order_id} is not the same order linked to the owned vehicle record. "
                f"The vehicle is linked to order {ownership_linked_order_number or ownership_order_id}, so the support team should verify that delivered order mapping."
            ),
            internal_summary="Ownership record exists but is linked to a different order than the current investigated order.",
            facts=facts,
            confidence=0.88,
        )

    if order_details and order_status and order_status != "DELIVERED":
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                f"The app likely cannot attach a vehicle yet because order {order_number or order_id} is currently {order_status}, "
                "not delivered. Vehicle linkage usually completes only after delivery and ownership mapping."
            ),
            internal_summary="Mobile app issue traced to non-delivered order state before ownership/primary VIN linkage.",
            facts=facts,
            confidence=0.9,
        )

    if order_status == "DELIVERED" and not ownership_record:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response="Your order appears delivered, but I could not find the ownership linkage yet. The support team should verify ownership mapping before app access is expected to work.",
            internal_summary="Delivered order found without ownership mapping.",
            facts=facts,
            confidence=0.82,
        )

    if ownership_vin and not primary_vin:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response="Your ownership record exists, but the app account does not have a primary VIN linked yet. The support team should set the primary VIN on your user profile.",
            internal_summary="Ownership VIN exists but user profile primary VIN is missing.",
            facts=facts,
            confidence=0.86,
        )

    if ownership_vin and primary_vin and ownership_vin != primary_vin:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response="The vehicle linked in ownership records does not match the primary VIN on your app profile. The support team should correct the VIN mapping.",
            internal_summary="Ownership VIN and user primary VIN do not match.",
            facts=facts,
            confidence=0.86,
        )

    if vehicle_last_seen:
        vin = vehicle_last_seen.get("vin")
        last_seen = vehicle_last_seen.get("last_seen")
        is_active = vehicle_last_seen.get("is_active")
        if is_active is False:
            details: list[str] = []
            formatted_last_seen = _format_display_time(last_seen)
            if formatted_last_seen:
                details.append(f"The last heartbeat we found was at {formatted_last_seen}.")
            telematics_latest = _format_display_time(telematics_status.get("latest_event_time"))
            telematics_is_fresh = _is_recent_epoch_millis(
                telematics_status.get("latest_event_time"),
                getattr(deps.settings, "telematics_freshness_minutes", 10),
            )
            if telematics_latest and telematics_latest != last_seen:
                details.append(f"The last telematics signal we found was at {telematics_latest}.")
            elif telematics_status and telematics_status.get("has_signal_data") is False:
                details.append("I could not find telematics signal data for this VIN in our telemetry pipeline.")
            last_trip_end = _format_display_time(trip_history_status.get("last_trip_end_time"))
            if last_trip_end:
                details.append(f"The most recent trip we found ended at {last_trip_end}.")
            elif trip_history_status and trip_history_status.get("has_trip_data") is False:
                details.append("I could not find recent trip data for this VIN.")
            suffix = f" {' '.join(details)}" if details else ""
            if telematics_is_fresh:
                return AgentResult(
                    ticket_id=state["ticket_id"],
                    issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
                    issue_category=state.get("issue_category", "unknown"),
                    problem_type=state.get("problem_type", "unknown"),
                    decision=NextAction.pending,
                    customer_response=(
                        f"Your vehicle {vin or ''} is still sending fresh telematics data, so it does not appear fully offline. "
                        f"The heartbeat status looks inconsistent or stale.{suffix} "
                        "The support team should verify the heartbeat/online-status pipeline for this vehicle."
                    ).strip(),
                    internal_summary="Heartbeat reports offline, but fresh telematics data indicates the vehicle is still actively sending telemetry.",
                    facts=facts,
                    confidence=0.9,
                )
            return AgentResult(
                ticket_id=state["ticket_id"],
                issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
                issue_category=state.get("issue_category", "unknown"),
                problem_type=state.get("problem_type", "unknown"),
                decision=NextAction.pending,
                customer_response=(
                    f"Your vehicle {vin or ''} appears to be offline right now and not connected to the internet.{suffix} "
                    "If app data is not syncing, please switch on the vehicle and ensure it regains connectivity."
                ).strip(),
                internal_summary="Ownership and VIN linkage are present, but vehicle heartbeat shows the vehicle is offline.",
                facts=facts,
                confidence=0.9,
            )
        if is_active is True and telematics_status and telematics_status.get("has_signal_data") is True:
            return AgentResult(
                ticket_id=state["ticket_id"],
                issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
                issue_category=state.get("issue_category", "unknown"),
                problem_type=state.get("problem_type", "unknown"),
                decision=NextAction.pending,
                customer_response=(
                    "Your vehicle appears to be online and connected to the internet, and telematics data exists for it. "
                    "The sync issue is likely not caused by vehicle connectivity or missing telemetry, so the support team should continue investigating the app/data sync path."
                ),
                internal_summary="Vehicle heartbeat is active and telemetry exists; app sync issue is not caused by connectivity or missing telematics.",
                facts=facts,
                confidence=0.88,
            )

    reference_vin = (
        (vehicle_last_seen.get("vin") if isinstance(vehicle_last_seen, dict) else None)
        or ownership_vin
        or primary_vin
        or state.get("vehicle_vin")
        or ""
    )
    if telematics_status and telematics_status.get("has_signal_data") is False:
        heartbeat_prefix = "I could not find a recent heartbeat for your vehicle, and " if not vehicle_last_seen else ""
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                f"{heartbeat_prefix}I could not find telematics signal data for vehicle {reference_vin} in our telemetry pipeline. "
                "The support team should verify telematics ingestion for this VIN."
            ).strip(),
            internal_summary="Vehicle linkage is correct, but no telematics signal data exists for the VIN.",
            facts=facts,
            confidence=0.9,
        )
    if not vehicle_last_seen and telematics_status and telematics_status.get("has_signal_data") is True:
        details: list[str] = []
        telematics_latest = _format_display_time(telematics_status.get("latest_event_time"))
        if telematics_latest:
            details.append(f"The last telematics signal we found was at {telematics_latest}.")
        last_trip_end = _format_display_time(trip_history_status.get("last_trip_end_time"))
        if last_trip_end:
            details.append(f"The most recent trip we found ended at {last_trip_end}.")
        suffix = f" {' '.join(details)}" if details else ""
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                f"I could not find a current heartbeat for vehicle {reference_vin}, but telematics data does exist for it.{suffix} "
                "The support team should verify why heartbeat status is missing even though telemetry is present."
            ).strip(),
            internal_summary="Heartbeat status is missing, but telematics data exists for the VIN.",
            facts=facts,
            confidence=0.88,
        )
    if trip_history_status and trip_history_status.get("has_trip_data") is False:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                "Your vehicle linkage looks correct, but I could not find trip telemetry for this VIN. "
                "The support team should verify trip ingestion and sync for the vehicle."
            ),
            internal_summary="Trip-specific sync issue: no trip history data found for the VIN.",
            facts=facts,
            confidence=0.88,
        )
    if charging_history_status and charging_history_status.get("has_charging_data") is False:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                "Your vehicle linkage looks correct, but I could not find charging telemetry for this VIN. "
                "The support team should verify charging-history ingestion and sync."
            ),
            internal_summary="Charging-specific sync issue: no charging history data found for the VIN.",
            facts=facts,
            confidence=0.88,
        )

    return None


def _format_display_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    ist = ZoneInfo("Asia/Kolkata")
    dt: datetime | None = None
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 1_000_000_000_000:
            timestamp = timestamp / 1000.0
        try:
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    elif isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
        dt = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    else:
        return None
    try:
        local_dt = dt.astimezone(ist)
        now_local = datetime.now(ist)
        local_date = local_dt.date()
        today = now_local.date()
        yesterday = today.fromordinal(today.toordinal() - 1)
        time_text = local_dt.strftime("%I:%M%p").lstrip("0").lower()
        if local_date == today:
            return f"today {time_text}"
        if local_date == yesterday:
            return f"yesterday {time_text}"
        return f"{local_dt.strftime('%d %b %Y')}, {time_text} IST"
    except Exception:
        return None


def _is_recent_epoch_millis(value: Any, freshness_minutes: int) -> bool:
    if not isinstance(value, (int, float)):
        return False
    timestamp = float(value)
    if timestamp > 1_000_000_000_000:
        timestamp = timestamp / 1000.0
    try:
        event_dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return False
    age_seconds = (datetime.now(timezone.utc) - event_dt).total_seconds()
    return age_seconds >= 0 and age_seconds <= max(freshness_minutes, 0) * 60


def _find_target_enquiry(state: AgentState, facts: dict[str, Any]) -> dict[str, Any]:
    target_order_number = state.get("order_number")
    enquiries = facts.get("user_enquiries", [])
    if not isinstance(enquiries, list):
        return {}
    if target_order_number:
        for enquiry in enquiries:
            if isinstance(enquiry, dict) and enquiry.get("order_number") == target_order_number:
                return enquiry
    return {}


def _find_target_related_order(state: AgentState, facts: dict[str, Any]) -> dict[str, Any]:
    target_order_number = state.get("order_number")
    related_orders = facts.get("related_orders", [])
    if not isinstance(related_orders, list):
        return {}
    if target_order_number:
        for order in related_orders:
            if isinstance(order, dict) and order.get("order_number") == target_order_number:
                return order
    return {}


def _prune_final_facts(state: AgentState, facts: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(facts, dict):
        return {}

    target_order_number = state.get("order_number")
    target_order_id = state.get("order_id")
    if not target_order_number and not target_order_id:
        return facts

    pruned = dict(facts)

    related_orders = pruned.get("related_orders")
    if isinstance(related_orders, list):
        filtered_related_orders = [
            order
            for order in related_orders
            if isinstance(order, dict)
            and (
                (target_order_id and order.get("id") == target_order_id)
                or (target_order_number and order.get("order_number") == target_order_number)
            )
        ]
        if filtered_related_orders:
            pruned["related_orders"] = filtered_related_orders
        else:
            pruned.pop("related_orders", None)

    user_enquiries = pruned.get("user_enquiries")
    if isinstance(user_enquiries, list):
        filtered_user_enquiries = [
            enquiry
            for enquiry in user_enquiries
            if isinstance(enquiry, dict)
            and (
                (target_order_id and enquiry.get("id") == target_order_id)
                or (target_order_number and enquiry.get("order_number") == target_order_number)
            )
        ]
        if filtered_user_enquiries:
            pruned["user_enquiries"] = filtered_user_enquiries
        else:
            pruned.pop("user_enquiries", None)

    order_details = pruned.get("order_details")
    if isinstance(order_details, dict):
        order_matches = (
            (target_order_id and order_details.get("id") == target_order_id)
            or (target_order_number and order_details.get("order_number") == target_order_number)
        )
        if not order_matches:
            pruned.pop("order_details", None)

    payment_status = pruned.get("payment_status")
    if isinstance(payment_status, dict) and target_order_id and payment_status.get("order_id") != target_order_id:
        pruned.pop("payment_status", None)

    return pruned


def _build_pending_order_result(state: AgentState) -> AgentResult | None:
    category = str(state.get("issue_category", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    if category not in {"payment", "booking", "delivery"} and "order" not in problem_type and "pending" not in problem_type:
        return None

    facts = state.get("facts", {})
    target_order_number = state.get("order_number")
    if not target_order_number:
        return None

    target_enquiry = _find_target_enquiry(state, facts)
    target_related_order = _find_target_related_order(state, facts)
    order_details = facts.get("order_details", {}) if isinstance(facts.get("order_details"), dict) else {}
    payment_status = facts.get("payment_status", {}) if isinstance(facts.get("payment_status"), dict) else {}

    if order_details and order_details.get("order_number") and order_details.get("order_number") != target_order_number:
        order_details = {}
    if payment_status and order_details and payment_status.get("order_id") and payment_status.get("order_id") != order_details.get("id"):
        payment_status = {}

    if target_enquiry and not target_related_order and not order_details:
        payment_session_id = target_enquiry.get("payment_session_id")
        message = (
            f"I checked order number {target_order_number} and it is still an active enquiry in pending state. "
            "I do not see a converted order or payment record against this enquiry yet."
        )
        if payment_session_id:
            message += " If money was deducted from your end, please share the transaction ID or UTR number so we can verify the payment."
        else:
            message += " This usually means payment has not been completed yet."
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=message,
            internal_summary="Target order number is still an active enquiry and no converted order/payment record was found.",
            facts=facts,
            confidence=0.92,
        )

    if order_details and not payment_status:
        return AgentResult(
            ticket_id=state["ticket_id"],
            issue_summary=state.get("normalized_issue_summary", state["raw_user_message"]),
            issue_category=state.get("issue_category", "unknown"),
            problem_type=state.get("problem_type", "unknown"),
            decision=NextAction.pending,
            customer_response=(
                f"I checked order {target_order_number} and I do not see any payment recorded against it yet. "
                "If money was deducted from your end, please share the transaction ID or UTR number so we can verify the payment."
            ),
            internal_summary="Target order exists but no transaction record was found for its order id.",
            facts=facts,
            confidence=0.9,
        )

    return None


def _dedupe_tool_queue(tools: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for tool in tools:
        if tool in seen:
            continue
        seen.add(tool)
        ordered.append(tool)
    return ordered
