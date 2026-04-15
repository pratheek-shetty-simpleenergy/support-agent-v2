from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from support_agent.runtime import configure_logging
from support_agent.schemas.ticket import SupportTicketInput
from support_agent.services.bootstrap import build_application


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the support investigation agent on a ticket JSON file.")
    parser.add_argument("ticket_file", type=Path, help="Path to a JSON file matching SupportTicketInput.")
    parser.add_argument("--quiet", action="store_true", help="Disable step-by-step progress output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    ticket = SupportTicketInput.model_validate_json(args.ticket_file.read_text())
    app = build_application()
    last_state: dict[str, Any] | None = None
    printed_trace_count = 0

    if not args.quiet:
        print(f"[start] ticket_id={ticket.ticket_id}")

    for update in app.stream(ticket.model_dump(), stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for node_name, node_state in update.items():
            if not args.quiet:
                print(_format_node_update(node_name, node_state))
                if isinstance(node_state, dict):
                    trace = node_state.get("investigation_trace", [])
                    if isinstance(trace, list):
                        new_entries = trace[printed_trace_count:]
                        for entry in new_entries:
                            print(f"[thinking] {entry}")
                        printed_trace_count = max(printed_trace_count, len(trace))
            if isinstance(node_state, dict):
                last_state = node_state

    if last_state is None or "final_result" not in last_state:
        raise RuntimeError("Agent run completed without a final_result.")

    final_payload = last_state["final_result"].model_dump(mode="json")
    final_payload.pop("facts", None)
    print(json.dumps(final_payload, indent=2))


def _format_node_update(node_name: str, node_state: Any) -> str:
    if not isinstance(node_state, dict):
        return f"[{node_name}] completed"

    if node_name == "normalize_issue":
        return f"[{node_name}] summary={node_state.get('normalized_issue_summary')}"
    if node_name == "classify_issue":
        return (
            f"[{node_name}] category={node_state.get('issue_category')} "
            f"problem_type={node_state.get('problem_type')} "
            f"confidence={node_state.get('confidence')}"
        )
    if node_name == "retrieve_context":
        matches = node_state.get("retrieved_matches", [])
        return f"[{node_name}] retrieved_matches={len(matches)}"
    if node_name == "plan_investigation":
        plan = node_state.get("investigation_plan")
        if plan is not None:
            return f"[{node_name}] tools={','.join(plan.required_tools) or 'none'}"
    if node_name == "run_tools":
        results = node_state.get("tool_results", [])
        tool_names = [item.get("name") for item in results if isinstance(item, dict)]
        clarifications = node_state.get("clarification_requests", [])
        return (
            f"[{node_name}] executed_tools={','.join(tool_names) or 'none'} "
            f"clarifications={json.dumps(clarifications, default=str)}"
        )
    if node_name == "finalize":
        result = node_state.get("final_result")
        trace = node_state.get("investigation_trace", [])
        trace_tail = trace[-1] if trace else None
        if result is not None:
            suffix = f" trace={trace_tail}" if trace_tail else ""
            return f"[{node_name}] decision={result.decision} confidence={result.confidence}{suffix}"
    return f"[{node_name}] completed"


if __name__ == "__main__":
    main()
