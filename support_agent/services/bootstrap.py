from __future__ import annotations

from support_agent.agent.graph import build_support_graph
from support_agent.config import Settings, get_settings
from support_agent.db.client import BusinessDbManager
from support_agent.db.repositories import BusinessDbRepository
from support_agent.llm.client import LlamaClient
from support_agent.retrieval.embedder import OllamaEmbeddingAdapter
from support_agent.retrieval.retriever import PineconeRetriever
from support_agent.runtime import configure_logging, log_event
from support_agent.services.pinot_service import PinotServiceClient
from support_agent.services.vehicle_service import VehicleServiceClient
from support_agent.tools.db_tools import build_business_db_tools


def build_application(settings: Settings | None = None):
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)
    llama_client = LlamaClient(app_settings)
    db_manager = BusinessDbManager(app_settings)
    repository = BusinessDbRepository(db_manager, app_settings)
    vehicle_service = VehicleServiceClient(app_settings)
    pinot_service = PinotServiceClient(app_settings)
    tool_registry = build_business_db_tools(repository, vehicle_service=vehicle_service, pinot_service=pinot_service)
    retriever = PineconeRetriever(app_settings, OllamaEmbeddingAdapter(llama_client))
    log_event(
        "application_bootstrap",
        app_name=app_settings.app_name,
        llm_provider=app_settings.llm_provider,
        llm_model=app_settings.openai_model if app_settings.llm_provider.lower() == "openai" else app_settings.ollama_model,
    )
    return build_support_graph(
        settings=app_settings,
        llm_client=llama_client,
        retriever=retriever,
        tool_registry=tool_registry,
    )
