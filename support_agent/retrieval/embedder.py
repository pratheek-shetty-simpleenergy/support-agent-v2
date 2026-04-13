from support_agent.llm.client import LlamaClient


class OllamaEmbeddingAdapter:
    def __init__(self, llama_client: LlamaClient) -> None:
        self.llama_client = llama_client

    def embed(self, text: str) -> list[float]:
        return self.llama_client.embed(text)
