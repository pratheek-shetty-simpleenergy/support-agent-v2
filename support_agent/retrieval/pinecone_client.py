from pinecone import Pinecone

from support_agent.config import Settings


def build_pinecone_index(settings: Settings):
    settings.require_pinecone()
    try:
        client = Pinecone(api_key=settings.pinecone_api_key)
        return client.Index(settings.pinecone_index)
    except Exception:
        return None
