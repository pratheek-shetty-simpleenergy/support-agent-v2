from __future__ import annotations

from support_agent.agent.graph import build_support_graph
from support_agent.config import Settings, get_settings
from support_agent.db.client import BusinessDbManager
from support_agent.db.repositories import BusinessDbRepository
from support_agent.llm.client import LlamaClient
from support_agent.retrieval.embedder import OllamaEmbeddingAdapter
from support_agent.retrieval.retriever import PineconeRetriever
from support_agent.tools.db_tools import build_business_db_tools


def build_application(settings: Settings | None = None):
    app_settings = settings or get_settings()
    llama_client = LlamaClient(app_settings)
    db_manager = BusinessDbManager(app_settings)
    repository = BusinessDbRepository(db_manager, app_settings)
    tool_registry = build_business_db_tools(repository)
    retriever = PineconeRetriever(app_settings, OllamaEmbeddingAdapter(llama_client))
    return build_support_graph(
        settings=app_settings,
        llm_client=llama_client,
        retriever=retriever,
        tool_registry=tool_registry,
    )
