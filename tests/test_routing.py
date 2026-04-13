from support_agent.agent.routing import route_after_plan
from support_agent.schemas.agent import InvestigationPlan


def test_route_after_plan_goes_to_tools_when_tools_exist() -> None:
    state = {"investigation_plan": InvestigationPlan(rationale="test", required_tools=["get_payment_status"])}
    assert route_after_plan(state) == "tools"


def test_route_after_plan_skips_tools_when_no_tools_exist() -> None:
    state = {"investigation_plan": InvestigationPlan(rationale="test")}
    assert route_after_plan(state) == "finalize"
