from support_agent.agent.graph import build_support_graph
from support_agent.schemas.agent import AgentDecision, AgentResult, InvestigationPlan, NextAction
from support_agent.schemas.ticket import SupportTicketInput
from support_agent.schemas.tool import ToolResult
from support_agent.tools.base import ToolRegistry


class FakeLLM:
    def generate_text(self, prompt: str, temperature: float = 0.1) -> str:
        return '{"normalized_issue_summary": "Payment captured but booking still pending."}'

    def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
        if schema is AgentDecision:
            return AgentDecision(
                normalized_issue_summary="Payment captured but booking still pending.",
                issue_category="payment",
                problem_type="payment_booking_mismatch",
                confidence=0.84,
            )
        if schema is InvestigationPlan:
            return InvestigationPlan(
                rationale="Verify payment and booking records.",
                required_tools=["get_payment_status", "get_booking_details"],
                tool_arguments={
                    "get_payment_status": {"payment_id": "PAY-1"},
                    "get_booking_details": {"booking_id": "BKG-1"},
                },
            )
        if schema is AgentResult:
            return AgentResult(
                ticket_id="ignored-by-node",
                issue_summary="Payment captured but booking still pending.",
                issue_category="payment",
                problem_type="payment_booking_mismatch",
                decision=NextAction.pending,
                customer_response="We verified your payment and are checking the booking sync.",
                internal_summary="Payment exists. Booking remains pending.",
                facts={},
                confidence=0.88,
            )
        raise AssertionError(f"Unexpected schema: {schema}")


class FakeRetriever:
    def retrieve(self, normalized_issue: str):
        return {
            "matches": [{"id": "1", "score": 0.9, "metadata": {"source": "sop", "title": "Payment SOP", "text": "Check payment sync lag."}}],
            "formatted_context": "[1] source=sop title=Payment SOP score=0.900\nCheck payment sync lag.",
        }


def test_graph_runs_end_to_end() -> None:
    registry = ToolRegistry()
    registry.register(
        "get_payment_status",
        lambda payment_id: ToolResult(name="get_payment_status", success=True, payload={"payment_status": {"id": payment_id, "status": "captured"}}),
    )
    registry.register(
        "get_booking_details",
        lambda booking_id: ToolResult(name="get_booking_details", success=True, payload={"booking_details": {"id": booking_id, "booking_status": "pending"}}),
    )
    graph = build_support_graph(settings=None, llm_client=FakeLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-1", raw_user_message="Payment issue", payment_id="PAY-1", booking_id="BKG-1")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].ticket_id == "T-1"
    assert result["facts"]["payment_status"]["status"] == "captured"


def test_graph_hydrates_ticket_id_for_ticket_tool() -> None:
    class TicketToolLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Load the current ticket record.",
                    required_tools=["get_ticket_details"],
                    tool_arguments={},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_ticket_details",
        lambda ticket_id: ToolResult(name="get_ticket_details", success=True, payload={"ticket_details": {"id": ticket_id}}),
    )
    graph = build_support_graph(settings=None, llm_client=TicketToolLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-99", raw_user_message="Need ticket details")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["ticket_details"]["id"] == "T-99"


def test_graph_returns_clarification_when_required_tool_id_is_missing() -> None:
    class ClarificationLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need payment verification.",
                    required_tools=["get_payment_status"],
                    tool_arguments={},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_payment_status",
        lambda payment_id: ToolResult(name="get_payment_status", success=True, payload={"payment_status": {"id": payment_id}}),
    )
    graph = build_support_graph(settings=None, llm_client=ClarificationLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-100", raw_user_message="Payment issue", user_id="U-1")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.needs_clarification
    assert "transaction ID" in result["final_result"].customer_response


def test_graph_handles_unknown_tool_as_clarification() -> None:
    class UnknownToolLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Try order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    graph = build_support_graph(settings=None, llm_client=UnknownToolLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-101", raw_user_message="Order pending")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.needs_clarification


def test_graph_fetches_user_context_before_order_clarification() -> None:
    class OrderClarificationLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(name="get_order_details", success=True, payload={"order_details": {"id": order_id}}),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(
            name="search_related_orders",
            success=True,
            payload={"related_orders": [{"order_number": "ORD-1001"}]},
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": [{"order_number": "ENQ-2001"}]},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=OrderClarificationLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-102", raw_user_message="Order pending", user_id="U-2")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["related_orders"][0]["order_number"] == "ORD-1001"
    assert result["final_result"].decision == NextAction.needs_clarification
    assert "ORD-1001" in result["final_result"].customer_response
