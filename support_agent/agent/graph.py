from __future__ import annotations

from functools import partial

from langgraph.graph import END, START, StateGraph

from support_agent.agent.nodes import (
    NodeDependencies,
    classify_issue,
    finalize_response,
    load_ticket,
    normalize_issue,
    plan_investigation,
    retrieve_context,
    run_tools,
)
from support_agent.agent.routing import route_after_plan
from support_agent.agent.state import AgentState


def build_support_graph(*, settings, llm_client, retriever, tool_registry):
    deps = NodeDependencies(llm_client=llm_client, retriever=retriever, tool_registry=tool_registry)
    graph = StateGraph(AgentState)

    graph.add_node("load_ticket", load_ticket)
    graph.add_node("normalize_issue", partial(normalize_issue, deps=deps))
    graph.add_node("classify_issue", partial(classify_issue, deps=deps))
    graph.add_node("retrieve_context", partial(retrieve_context, deps=deps))
    graph.add_node("plan_investigation", partial(plan_investigation, deps=deps))
    graph.add_node("run_tools", partial(run_tools, deps=deps))
    graph.add_node("finalize", partial(finalize_response, deps=deps))

    graph.add_edge(START, "load_ticket")
    graph.add_edge("load_ticket", "normalize_issue")
    graph.add_edge("normalize_issue", "classify_issue")
    graph.add_edge("classify_issue", "retrieve_context")
    graph.add_edge("retrieve_context", "plan_investigation")
    graph.add_conditional_edges("plan_investigation", route_after_plan, {"tools": "run_tools", "finalize": "finalize"})
    graph.add_edge("run_tools", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()
