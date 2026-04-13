from pinecone import Pinecone

from support_agent.config import Settings
from support_agent.runtime.errors import PermanentDependencyError


def build_pinecone_index(settings: Settings):
    if not settings.pinecone_api_key or not settings.pinecone_index:
        if settings.pinecone_required:
            raise PermanentDependencyError("Pinecone configuration is incomplete.")
        return None
    try:
        client = Pinecone(api_key=settings.pinecone_api_key)
        return client.Index(settings.pinecone_index)
    except Exception:
        return None
