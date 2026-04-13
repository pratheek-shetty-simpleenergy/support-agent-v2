from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from support_agent.agent.state import AgentState
from support_agent.llm.client import LlamaClient
from support_agent.llm.prompts import (
    build_classification_prompt,
    build_final_response_prompt,
    build_investigation_plan_prompt,
    build_normalize_prompt,
)
from support_agent.schemas.agent import AgentDecision, AgentResult, InvestigationPlan
from support_agent.tools.base import ToolRegistry


@dataclass
class NodeDependencies:
    llm_client: LlamaClient
    retriever: Any
    tool_registry: ToolRegistry


def load_ticket(state: AgentState) -> AgentState:
    state.setdefault("facts", {})
    state.setdefault("tool_results", [])
    state.setdefault("clarification_requests", [])
    state.setdefault("hypotheses", [])
    return state


def normalize_issue(state: AgentState, deps: NodeDependencies) -> AgentState:
    conversation = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state.get("conversation_history", []))
    prompt = build_normalize_prompt(state["raw_user_message"], conversation)
    text = deps.llm_client.generate_text(prompt)
    summary = _extract_summary(text)
    return {"normalized_issue_summary": summary}


def classify_issue(state: AgentState, deps: NodeDependencies) -> AgentState:
    prompt = build_classification_prompt(state["normalized_issue_summary"])
    decision = deps.llm_client.generate_structured(prompt, AgentDecision)
    return {
        "normalized_issue_summary": decision.normalized_issue_summary,
        "issue_category": decision.issue_category,
        "problem_type": decision.problem_type,
        "confidence": decision.confidence,
    }


def retrieve_context(state: AgentState, deps: NodeDependencies) -> AgentState:
    retrieved = deps.retriever.retrieve(state["normalized_issue_summary"])
    return {
        "retrieved_context": retrieved["formatted_context"],
        "retrieved_matches": retrieved["matches"],
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
    plan = deps.llm_client.generate_structured(prompt, InvestigationPlan)
    return {"investigation_plan": plan}


def run_tools(state: AgentState, deps: NodeDependencies) -> AgentState:
    plan = state["investigation_plan"]
    facts = dict(state.get("facts", {}))
    raw_results: list[dict[str, Any]] = list(state.get("tool_results", []))
    clarification_requests: list[dict[str, Any]] = list(state.get("clarification_requests", []))

    for tool_name in plan.required_tools:
        arguments = dict(plan.tool_arguments.get(tool_name, {}))
        arguments = _hydrate_tool_arguments(state, arguments)
        missing_arguments = [
            parameter
            for parameter in deps.tool_registry.required_parameters(tool_name)
            if arguments.get(parameter) in (None, "")
        ]
        if missing_arguments:
            clarification_requests.append(
                {
                    "tool_name": tool_name,
                    "missing_arguments": missing_arguments,
                }
            )
            continue
        result = deps.tool_registry.run(tool_name, arguments)
        raw_results.append(result.model_dump())
        facts.update(result.payload)

    if _needs_payment_reference_clarification(plan.required_tools, facts):
        clarification_requests.append(
            {
                "tool_name": "get_payment_status",
                "missing_arguments": ["payment_id"],
            }
        )

    return {
        "facts": facts,
        "tool_results": raw_results,
        "clarification_requests": _dedupe_clarification_requests(clarification_requests),
    }


def finalize_response(state: AgentState, deps: NodeDependencies) -> AgentState:
    clarification_requests = state.get("clarification_requests", [])
    if clarification_requests:
        final = _build_clarification_result(state, clarification_requests)
        return {"next_action": final.decision, "final_result": final}

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
    final = deps.llm_client.generate_structured(prompt, AgentResult)
    final.ticket_id = state["ticket_id"]
    final.facts = state.get("facts", {})
    return {"next_action": final.decision, "final_result": final}


def _extract_summary(raw_text: str) -> str:
    stripped = raw_text.strip()
    if stripped.startswith("{"):
        try:
            import json

            payload = json.loads(stripped)
            return str(payload["normalized_issue_summary"])
        except Exception:
            pass
    return stripped


def _build_ticket_context(state: AgentState) -> str:
    identifiers = {
        "ticket_id": state.get("ticket_id"),
        "user_id": state.get("user_id"),
        "booking_id": state.get("booking_id"),
        "payment_id": state.get("payment_id"),
        "order_id": state.get("order_id"),
        "vehicle_id": state.get("vehicle_id"),
    }
    history = "\n".join(f"{msg['role']}: {msg['content']}" for msg in state.get("conversation_history", []))
    return f"Identifiers: {identifiers}\nConversation:\n{history}"


def _hydrate_tool_arguments(state: AgentState, arguments: dict[str, Any]) -> dict[str, Any]:
    fallback_fields = ("ticket_id", "user_id", "booking_id", "payment_id", "order_id", "vehicle_id")
    for field in fallback_fields:
        if field not in arguments and state.get(field):
            if field == "ticket_id" and arguments.get("ticket_id") is None:
                arguments["ticket_id"] = state[field]
            if field == "user_id" and arguments.get("user_id") is None:
                arguments["user_id"] = state[field]
            if field == "booking_id" and arguments.get("booking_id") is None:
                arguments["booking_id"] = state[field]
            if field == "payment_id" and arguments.get("payment_id") is None:
                arguments["payment_id"] = state[field]
            if field == "order_id" and arguments.get("order_id") is None:
                arguments["order_id"] = state[field]
            if field == "vehicle_id" and arguments.get("vehicle_id") is None:
                arguments["vehicle_id"] = state[field]
    return arguments


def _stringify(value: Any) -> str:
    import json

    return json.dumps(value, indent=2, default=str)


def _needs_payment_reference_clarification(required_tools: list[str], facts: dict[str, Any]) -> bool:
    if "get_payment_status" not in required_tools:
        return False
    has_payment_result = bool(facts.get("payment_status"))
    has_orders = bool(facts.get("related_orders"))
    has_enquiries = bool(facts.get("user_enquiries"))
    return not has_payment_result and (has_orders or has_enquiries)


def _build_clarification_result(state: AgentState, clarification_requests: list[dict[str, Any]]) -> AgentResult:
    missing_arguments = sorted(
        {
            argument
            for request in clarification_requests
            for argument in request.get("missing_arguments", [])
        }
    )
    clarification_lines: list[str] = []
    if "payment_id" in missing_arguments:
        clarification_lines.append("Please share your UTR number or transaction ID so I can verify the payment.")
    if "order_id" in missing_arguments:
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
