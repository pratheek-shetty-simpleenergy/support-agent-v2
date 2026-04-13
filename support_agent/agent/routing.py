from support_agent.agent.state import AgentState


def route_after_plan(state: AgentState) -> str:
    plan = state.get("investigation_plan")
    if not plan or not plan.required_tools:
        return "finalize"
    return "tools"
