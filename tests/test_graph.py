from support_agent.agent.graph import build_support_graph
from support_agent.runtime.errors import TransientDependencyError
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
        lambda order_id=None, order_number=None: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id or "ORDER-ID", "order_number": order_number}},
        ),
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


def test_graph_auto_uses_single_order_candidate() -> None:
    class AutoOrderLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                return AgentResult(
                    ticket_id="ignored-by-node",
                    issue_summary="Customer unable to view order details.",
                    issue_category="booking",
                    problem_type="viewing_order_details",
                    decision=NextAction.pending,
                    customer_response="I found your order details.",
                    internal_summary="Auto-selected the only matching order candidate.",
                    facts={},
                    confidence=0.9,
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id=None, order_number=None: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id or "ORDER-9001", "order_number": order_number or "ORD-9001"}},
        ),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(
            name="search_related_orders",
            success=True,
            payload={"related_orders": [{"id": "ORDER-9001", "order_number": "ORD-9001"}]},
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": []},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=AutoOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-102A", raw_user_message="I cannot see my order", user_id="U-200")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["order_details"]["id"] == "ORDER-9001"


def test_graph_propagates_user_id_before_search_related_orders() -> None:
    class MobileAmbiguityLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is AgentDecision:
                return AgentDecision(
                    normalized_issue_summary="App gets stuck and user cannot view order status.",
                    issue_category="app_sync",
                    problem_type="functionality_issue",
                    confidence=0.85,
                )
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Resolve user, then inspect account orders and ownership.",
                    required_tools=["get_user_profile_by_mobile", "get_ownership_record", "search_related_orders"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                return AgentResult(
                    ticket_id="ignored-by-node",
                    issue_summary="App gets stuck and user cannot view order status.",
                    issue_category="app_sync",
                    problem_type="functionality_issue",
                    decision=NextAction.pending,
                    customer_response="Investigated account-level linkage.",
                    internal_summary="Resolved user and fetched orders without asking for user_id.",
                    facts={},
                    confidence=0.9,
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={"user_profile": {"id": "USER-42", "mobile": mobile, "user_metadata": {"orderId": "ORDER-1"}}},
        ),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(
            name="search_related_orders",
            success=True,
            payload={"related_orders": [{"id": "ORDER-1", "order_number": "ORD-1", "status": "REFUND_SUCCESSFUL"}]},
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": []},
        ),
    )
    registry.register(
        "get_ownership_record",
        lambda user_id=None, order_id=None, vin=None: ToolResult(
            name="get_ownership_record",
            success=True,
            payload={"ownership_record": {}},
        ),
    )
    registry.register(
        "get_order_details",
        lambda order_id=None, order_number=None: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id or "ORDER-1", "order_number": order_number or "ORD-1", "status": "REFUND_SUCCESSFUL"}},
        ),
    )

    graph = build_support_graph(settings=None, llm_client=MobileAmbiguityLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-APP-1", raw_user_message="App stuck and cannot view my order", mobile="9480300096")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["user_profile"]["id"] == "USER-42"
    assert result["facts"]["related_orders"][0]["id"] == "ORDER-1"
    assert all(request.get("tool_name") != "search_related_orders" for request in result.get("clarification_requests", []))


def test_graph_returns_no_payment_message_for_pending_enquiry_order_number() -> None:
    class PendingOrderLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is AgentDecision:
                return AgentDecision(
                    normalized_issue_summary="Order 8339-050394-7692 remains pending.",
                    issue_category="delivery",
                    problem_type="order_status_delay",
                    confidence=0.9,
                )
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Inspect the referenced order.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={"user_profile": {"id": "USER-42", "mobile": mobile}},
        ),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(name="search_related_orders", success=True, payload={"related_orders": []}),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={
                "user_enquiries": [
                    {
                        "id": "ENQ-1",
                        "order_number": "8339-050394-7692",
                        "user_id": user_id,
                        "enquiry_status": "ACTIVE",
                        "status_message": "PENDING",
                        "payment_session_id": "session_123",
                    }
                ]
            },
        ),
    )
    registry.register(
        "get_order_details",
        lambda order_id=None, order_number=None: ToolResult(name="get_order_details", success=False, payload={"order_details": {}}),
    )

    graph = build_support_graph(settings=None, llm_client=PendingOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(
        ticket_id="T-PEND-1",
        raw_user_message="Why is order 8339-050394-7692 pending?",
        mobile="9480300096",
        order_number="8339-050394-7692",
    )
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.pending
    assert "do not see a converted order or payment record" in result["final_result"].customer_response
    assert "transaction ID or UTR" in result["final_result"].customer_response
    assert result.get("clarification_requests", []) == []
    assert result["final_result"].decision == NextAction.pending
    assert "related_orders" not in result["final_result"].facts
    assert len(result["final_result"].facts["user_enquiries"]) == 1
    assert result["final_result"].facts["user_enquiries"][0]["order_number"] == "8339-050394-7692"


def test_graph_returns_confirmation_when_multiple_order_candidates_exist() -> None:
    class MultiOrderLLM(FakeLLM):
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
            payload={
                "related_orders": [
                    {"id": "ORDER-1", "order_number": "ORD-1001"},
                    {"id": "ORDER-2", "order_number": "ORD-1002"},
                ]
            },
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": []},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=MultiOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-102B", raw_user_message="I cannot see my order", user_id="U-201")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.needs_clarification
    assert "ORD-1001" in result["final_result"].customer_response
    assert "ORD-1002" in result["final_result"].customer_response


def test_graph_requests_confirmation_when_rules_find_multiple_candidates() -> None:
    class ProfileOnlyLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Resolve the user account first.",
                    required_tools=["get_user_profile_by_mobile"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                raise ValueError("invalid final output")
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={"user_profile": {"id": "USER-9", "mobile": mobile}},
        ),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(
            name="search_related_orders",
            success=True,
            payload={
                "related_orders": [
                    {"id": "ORDER-11", "order_number": "ORD-1111"},
                    {"id": "ORDER-22", "order_number": "ORD-2222"},
                ]
            },
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": [{"id": "ENQ-33", "order_number": "ENQ-3333"}]},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=ProfileOnlyLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-102C", raw_user_message="I cannot see my order", mobile="9480300096")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.needs_clarification
    assert "ORD-1111" in result["final_result"].customer_response
    assert "ORD-2222" in result["final_result"].customer_response
    assert "ENQ-3333" in result["final_result"].customer_response


def test_graph_resolves_user_by_mobile_before_order_lookup() -> None:
    class MobileOrderLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                return AgentResult(
                    ticket_id="ignored-by-node",
                    issue_summary="Customer unable to view order details.",
                    issue_category="delivery",
                    problem_type="viewing_order_details",
                    decision=NextAction.pending,
                    customer_response="I found your order details and am sharing the linked order references.",
                    internal_summary="Fetched user and related orders from mobile number.",
                    facts={},
                    confidence=0.9,
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(name="get_order_details", success=True, payload={"order_details": {"id": order_id}}),
    )
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={"user_profile": {"id": "USER-1", "mobile": mobile}},
        ),
    )
    registry.register(
        "search_related_orders",
        lambda user_id: ToolResult(
            name="search_related_orders",
            success=True,
            payload={"related_orders": [{"order_number": "ORD-5001"}]},
        ),
    )
    registry.register(
        "get_user_enquiries",
        lambda user_id, active_only=True: ToolResult(
            name="get_user_enquiries",
            success=True,
            payload={"user_enquiries": []},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=MobileOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-103", raw_user_message="Cannot see my order", mobile="9480300096")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["user_profile"]["id"] == "USER-1"
    assert result["facts"]["related_orders"][0]["order_number"] == "ORD-5001"
    assert result["final_result"].decision == NextAction.pending


def test_graph_promotes_order_id_from_user_metadata_and_executes_order_lookup() -> None:
    class MetadataOrderLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need direct order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                return AgentResult(
                    ticket_id="ignored-by-node",
                    issue_summary="Customer unable to view order details.",
                    issue_category="delivery",
                    problem_type="viewing_order_details",
                    decision=NextAction.pending,
                    customer_response="I found your order details.",
                    internal_summary="Resolved order lookup from user metadata.",
                    facts={},
                    confidence=0.9,
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(name="get_order_details", success=True, payload={"order_details": {"id": order_id}}),
    )
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={"user_profile": {"id": "USER-2", "mobile": mobile, "user_metadata": {"orderId": "ORDER-42"}}},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=MetadataOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-104", raw_user_message="Cannot see my order", mobile="9480300096")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["order_details"]["id"] == "ORDER-42"
    assert result["final_result"].decision == NextAction.pending


def test_graph_fetches_ownership_after_delivered_order() -> None:
    class DeliveredOrderLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={"get_order_details": {"order_id": "ORDER-77"}},
                )
            if schema is AgentResult:
                return AgentResult(
                    ticket_id="ignored-by-node",
                    issue_summary="Delivered order needs ownership lookup.",
                    issue_category="delivery",
                    problem_type="ownership_follow_up",
                    decision=NextAction.pending,
                    customer_response="Fetched order and ownership details.",
                    internal_summary="Delivered order linked to ownership.",
                    facts={},
                    confidence=0.9,
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id, "status": "DELIVERED"}},
        ),
    )
    registry.register(
        "get_ownership_record",
        lambda order_id=None, user_id=None, vin=None: ToolResult(
            name="get_ownership_record",
            success=True,
            payload={"ownership_record": {"order_id": order_id, "vin": "VIN-1"}},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=DeliveredOrderLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-105", raw_user_message="Vehicle delivered")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["ownership_record"]["order_id"] == "ORDER-77"
    assert result["final_result"].decision == NextAction.pending


def test_graph_falls_back_when_final_llm_output_is_invalid() -> None:
    class InvalidFinalizeLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need direct order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={"get_order_details": {"order_id": "ORDER-88"}},
                )
            if schema is AgentResult:
                raise ValueError("invalid final output")
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id, "status": "REFUND_SUCCESSFUL"}},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=InvalidFinalizeLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-106", raw_user_message="Unable to view order details")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.pending
    assert result["final_result"].customer_response
    assert "fallback" in result["final_result"].internal_summary.lower()
    assert result["investigation_trace"][-1].startswith("Finished with decision=")


def test_graph_escalates_when_tool_execution_fails() -> None:
    class ToolFailureLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need direct order lookup.",
                    required_tools=["get_order_details"],
                    tool_arguments={"get_order_details": {"order_id": "ORDER-ERR"}},
                )
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()

    def failing_tool(order_id: str):
        raise RuntimeError(f"boom for {order_id}")

    registry.register("get_order_details", failing_tool)
    graph = build_support_graph(settings=None, llm_client=ToolFailureLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-107", raw_user_message="Check my order")
    result = graph.invoke(ticket.model_dump())
    assert result["final_result"].decision == NextAction.escalate
    assert result["tool_failures"][0]["tool_name"] == "get_order_details"


def test_graph_uses_safe_classification_fallback_on_llm_failure() -> None:
    class BrokenClassificationLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is AgentDecision:
                raise TransientDependencyError("classification unavailable")
            if schema is InvestigationPlan:
                return InvestigationPlan(rationale="Fallback plan.", required_tools=[], tool_arguments={})
            return super().generate_structured(prompt, schema, temperature)

    graph = build_support_graph(settings=None, llm_client=BrokenClassificationLLM(), retriever=FakeRetriever(), tool_registry=ToolRegistry())
    ticket = SupportTicketInput(ticket_id="T-108", raw_user_message="My app is broken")
    result = graph.invoke(ticket.model_dump())
    assert result["issue_category"] == "unknown"
    assert result["problem_type"] == "unknown"


def test_graph_investigates_mobile_app_vehicle_linkage_from_user_and_order() -> None:
    class MobileVehicleLLM(FakeLLM):
        def generate_structured(self, prompt: str, schema, temperature: float = 0.1):
            if schema is AgentDecision:
                return AgentDecision(
                    normalized_issue_summary="Mobile app login fails and says no vehicle is attached to the phone number.",
                    issue_category="mobile_app",
                    problem_type="login",
                    confidence=0.9,
                )
            if schema is InvestigationPlan:
                return InvestigationPlan(
                    rationale="Need vehicle linkage lookup.",
                    required_tools=["get_vehicle_details"],
                    tool_arguments={},
                )
            if schema is AgentResult:
                raise ValueError("force fallback finalizer")
            return super().generate_structured(prompt, schema, temperature)

    registry = ToolRegistry()
    registry.register(
        "get_user_profile_by_mobile",
        lambda mobile: ToolResult(
            name="get_user_profile_by_mobile",
            success=True,
            payload={
                "user_profile": {
                    "id": "USER-APP-1",
                    "mobile": mobile,
                    "primary_vin": None,
                    "user_metadata": {"orderId": "ORDER-DELIVERED-1"},
                }
            },
        ),
    )
    registry.register(
        "get_order_details",
        lambda order_id: ToolResult(
            name="get_order_details",
            success=True,
            payload={"order_details": {"id": order_id, "user_id": "USER-APP-1", "status": "DELIVERED", "order_number": "ORD-APP-1"}},
        ),
    )
    registry.register(
        "get_ownership_record",
        lambda order_id=None, user_id=None, vin=None: ToolResult(
            name="get_ownership_record",
            success=True,
            payload={"ownership_record": {"order_id": "ORDER-DELIVERED-1", "user_id": "USER-APP-1", "vin": "VIN-APP-1"}},
        ),
    )
    registry.register(
        "get_vehicle_details",
        lambda vehicle_id: ToolResult(
            name="get_vehicle_details",
            success=True,
            payload={"vehicle_details": {"vin": vehicle_id, "order_id": "ORDER-DELIVERED-1", "user_id": "USER-APP-1"}},
        ),
    )
    graph = build_support_graph(settings=None, llm_client=MobileVehicleLLM(), retriever=FakeRetriever(), tool_registry=registry)
    ticket = SupportTicketInput(ticket_id="T-109", raw_user_message="App says no vehicle attached to my phone number", mobile="9480300096")
    result = graph.invoke(ticket.model_dump())
    assert result["facts"]["order_details"]["status"] == "DELIVERED"
    assert result["facts"]["ownership_record"]["vin"] == "VIN-APP-1"
    assert result["final_result"].decision == NextAction.pending
    assert "primary VIN linked" in result["final_result"].customer_response
