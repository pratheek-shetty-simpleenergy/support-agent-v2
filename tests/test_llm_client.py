from __future__ import annotations

from support_agent.config import Settings
from support_agent.llm.client import LlamaClient


class FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


def test_openai_generate_text_uses_chat_completions(monkeypatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers or {}, json or {}))
        return FakeResponse(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": "hello from gpt"
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr("requests.post", fake_post)
    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-mini",
    )
    client = LlamaClient(settings)
    text = client.generate_text("hi")
    assert text == "hello from gpt"
    assert calls[0][0].endswith("/chat/completions")


def test_openai_embed_uses_embeddings_endpoint(monkeypatch) -> None:
    calls: list[tuple[str, dict, dict]] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, headers or {}, json or {}))
        return FakeResponse(200, {"data": [{"embedding": [0.1, 0.2, 0.3]}]})

    monkeypatch.setattr("requests.post", fake_post)
    settings = Settings(
        LLM_PROVIDER="openai",
        OPENAI_API_KEY="test-key",
        OPENAI_MODEL="gpt-4.1-mini",
        OPENAI_EMBEDDING_MODEL="text-embedding-3-small",
    )
    client = LlamaClient(settings)
    embedding = client.embed("hello")
    assert embedding == [0.1, 0.2, 0.3]
    assert calls[0][0].endswith("/embeddings")
