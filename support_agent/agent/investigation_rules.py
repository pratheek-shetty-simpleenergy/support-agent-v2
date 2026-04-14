from __future__ import annotations

from typing import Any

from support_agent.agent.state import AgentState


def initial_rule_tools(state: AgentState) -> list[str]:
    tools: list[str] = []
    if state.get("mobile") and not state.get("user_id"):
        tools.append("get_user_profile_by_mobile")
    if _should_fetch_order_candidates(state=state, facts={}):
        tools.extend(["search_related_orders", "get_user_enquiries"])
    if _is_mobile_vehicle_link_issue(state) and state.get("mobile"):
        tools.append("get_user_profile_by_mobile")
    return _dedupe(tools)


def follow_up_rule_tools(state: AgentState, facts: dict[str, Any]) -> list[str]:
    tools: list[str] = []
    if _should_fetch_order_candidates(state=state, facts=facts):
        tools.extend(["search_related_orders", "get_user_enquiries"])

    if _is_order_visibility_issue(state):
        order_reference = single_order_reference(facts)
        if order_reference and order_candidates_ready(state, facts) and not state.get("order_id"):
            tools.append("get_order_details")

    order_details = facts.get("order_details", {})
    if isinstance(order_details, dict):
        status = str(order_details.get("status", "")).upper()
        if state.get("order_id") and not facts.get("payment_status"):
            tools.append("get_order_payment_status")
        if status == "DELIVERED" and state.get("order_id"):
            tools.append("get_ownership_record")
    if _is_mobile_vehicle_link_issue(state):
        if state.get("order_id") and not facts.get("order_details"):
            tools.append("get_order_details")
        if state.get("user_id") and not facts.get("ownership_record"):
            tools.append("get_ownership_record")
        if state.get("vehicle_id") and not facts.get("vehicle_details"):
            tools.append("get_vehicle_details")
    return _dedupe(tools)

def order_candidates(facts: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for payload_key in ("related_orders", "user_enquiries"):
        items = facts.get(payload_key, [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            candidate = {
                "id": _string_or_empty(item.get("id")),
                "order_number": _string_or_empty(item.get("order_number")),
            }
            if not candidate["id"] and not candidate["order_number"]:
                continue
            key = (candidate["id"], candidate["order_number"])
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def single_order_reference(facts: dict[str, Any]) -> str | None:
    candidates = order_candidates(facts)
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    return candidate["id"] or candidate["order_number"] or None


def candidate_order_numbers(facts: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    for candidate in order_candidates(facts):
        order_number = candidate.get("order_number")
        if order_number and order_number not in candidates:
            candidates.append(order_number)
    return candidates


def has_multiple_order_candidates(facts: dict[str, Any]) -> bool:
    return len(order_candidates(facts)) > 1


def order_candidates_ready(state: AgentState, facts: dict[str, Any]) -> bool:
    if not _is_order_visibility_issue(state):
        return True
    return not _should_fetch_order_candidates(state, facts)


def _is_order_visibility_issue(state: AgentState) -> bool:
    category = str(state.get("issue_category", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    raw_message = str(state.get("raw_user_message", "")).lower()
    return (
        category in {"payment", "booking", "delivery"}
        or "order" in problem_type
        or "pending" in problem_type
        or "order" in raw_message
        or "pending" in raw_message
    )


def _is_mobile_vehicle_link_issue(state: AgentState) -> bool:
    category = str(state.get("issue_category", "")).lower()
    problem_type = str(state.get("problem_type", "")).lower()
    raw_message = str(state.get("raw_user_message", "")).lower()
    return (
        category in {"mobile_app", "app_sync", "login"}
        or problem_type in {"login", "app_sync", "vehicle_linking", "vehicle_attachment"}
        or "app" in raw_message
        or "login" in raw_message
        or "vehicle" in raw_message
        or "attached to your phone" in raw_message
    )


def _should_fetch_order_candidates(state: AgentState, facts: dict[str, Any]) -> bool:
    if not _is_order_visibility_issue(state):
        return False
    if not (state.get("user_id") or state.get("mobile")):
        return False
    has_related_orders = isinstance(facts.get("related_orders"), list)
    has_user_enquiries = isinstance(facts.get("user_enquiries"), list)
    return not (has_related_orders and has_user_enquiries)


def _string_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered
