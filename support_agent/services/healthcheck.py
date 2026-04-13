from __future__ import annotations

from typing import Any

from support_agent.config import Settings, get_settings
from support_agent.db.client import BusinessDbManager
from support_agent.llm.client import LlamaClient
from support_agent.retrieval.embedder import OllamaEmbeddingAdapter
from support_agent.retrieval.retriever import PineconeRetriever


def run_healthcheck(settings: Settings | None = None) -> dict[str, Any]:
    app_settings = settings or get_settings()
    llm_client = LlamaClient(app_settings)
    db_manager = BusinessDbManager(app_settings)
    retriever = PineconeRetriever(app_settings, OllamaEmbeddingAdapter(llm_client))
    return {
        "app_name": app_settings.app_name,
        "ollama": llm_client.healthcheck(),
        "databases": db_manager.healthcheck(),
        "pinecone": retriever.healthcheck(),
    }

