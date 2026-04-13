from __future__ import annotations

from typing import Any

from support_agent.config import Settings
from support_agent.retrieval.embedder import OllamaEmbeddingAdapter
from support_agent.retrieval.formatter import format_match_context
from support_agent.retrieval.pinecone_client import build_pinecone_index
from support_agent.runtime.errors import SupportAgentError


class PineconeRetriever:
    def __init__(self, settings: Settings, embedder: OllamaEmbeddingAdapter) -> None:
        self.settings = settings
        self.embedder = embedder
        self.index = build_pinecone_index(settings)

    def retrieve(self, normalized_issue: str) -> dict[str, Any]:
        if self.index is None:
            return {
                "matches": [],
                "formatted_context": "Retrieval unavailable. Continuing without Pinecone context.",
            }
        try:
            vector = self.embedder.embed(normalized_issue)
        except SupportAgentError:
            return {
                "matches": [],
                "formatted_context": "Retrieval unavailable. Continuing without Pinecone context.",
            }
        try:
            response = self.index.query(
                namespace=self.settings.pinecone_namespace,
                vector=vector,
                top_k=self.settings.pinecone_top_k,
                include_metadata=True,
            )
        except Exception:
            return {
                "matches": [],
                "formatted_context": "Retrieval unavailable. Continuing without Pinecone context.",
            }
        matches = [self._serialize_match(match) for match in response.get("matches", [])]
        return {
            "matches": matches,
            "formatted_context": format_match_context(matches),
        }

    def healthcheck(self) -> dict[str, Any]:
        if self.index is None:
            return {"status": "degraded", "provider": "pinecone", "message": "Pinecone unavailable or optional."}
        try:
            self.index.describe_index_stats()
            return {"status": "ok", "provider": "pinecone", "index": self.settings.pinecone_index}
        except Exception as exc:
            return {"status": "error", "provider": "pinecone", "error": str(exc)}

    @staticmethod
    def _serialize_match(match: Any) -> dict[str, Any]:
        if isinstance(match, dict):
            return match
        return {
            "id": getattr(match, "id", None),
            "score": getattr(match, "score", None),
            "metadata": getattr(match, "metadata", {}) or {},
        }
