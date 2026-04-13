from __future__ import annotations

import json
from typing import Any

import requests

from support_agent.config import Settings
from support_agent.llm.parser import parse_json_model


class LLMError(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, settings: Settings, timeout_seconds: int = 60) -> None:
        self.settings = settings
        self.timeout_seconds = timeout_seconds

    def generate_text(self, prompt: str, *, temperature: float = 0.1) -> str:
        url = f"{self.settings.ollama_host}/api/generate"
        payload = {
            "model": self.settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        response = requests.post(url, json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise LLMError(f"Ollama generate call failed: {response.status_code} {response.text}")
        data = response.json()
        return str(data.get("response", "")).strip()

    def generate_structured(self, prompt: str, schema: type[Any], *, temperature: float = 0.1) -> Any:
        structured_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON. Do not add markdown, explanations, headings, or extra text."
        )
        raw = self.generate_text(structured_prompt, temperature=temperature)
        return parse_json_model(raw, schema)

    def embed(self, text: str) -> list[float]:
        for endpoint in ("/api/embed", "/api/embeddings"):
            url = f"{self.settings.ollama_host}{endpoint}"
            payload = {"model": self.settings.ollama_embedding_model, "input": text}
            if endpoint.endswith("embeddings"):
                payload = {"model": self.settings.ollama_embedding_model, "prompt": text}

            response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                continue

            data = response.json()
            if "embeddings" in data and data["embeddings"]:
                return list(data["embeddings"][0])
            if "embedding" in data:
                return list(data["embedding"])

        raise LLMError("Ollama embedding call failed on both /api/embed and /api/embeddings.")
