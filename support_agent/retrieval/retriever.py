from __future__ import annotations

from typing import Any

from support_agent.config import Settings
from support_agent.retrieval.embedder import OllamaEmbeddingAdapter
from support_agent.retrieval.formatter import format_match_context
from support_agent.retrieval.pinecone_client import build_pinecone_index


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
        vector = self.embedder.embed(normalized_issue)
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

    @staticmethod
    def _serialize_match(match: Any) -> dict[str, Any]:
        if isinstance(match, dict):
            return match
        return {
            "id": getattr(match, "id", None),
            "score": getattr(match, "score", None),
            "metadata": getattr(match, "metadata", {}) or {},
        }
