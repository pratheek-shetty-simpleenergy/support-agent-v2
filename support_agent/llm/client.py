from __future__ import annotations

import json
from typing import Any

import requests

from support_agent.config import Settings
from support_agent.llm.parser import parse_json_model
from support_agent.runtime import log_event
from support_agent.runtime.errors import InvalidModelOutputError, PermanentDependencyError, TransientDependencyError
from support_agent.runtime.retry import run_with_retry


class LLMError(RuntimeError):
    pass


class LlamaClient:
    def __init__(self, settings: Settings, timeout_seconds: int | None = None) -> None:
        self.settings = settings
        self.settings.require_llm()
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds

    def generate_text(self, prompt: str, *, temperature: float = 0.1) -> str:
        provider = self.settings.llm_provider.lower()
        model = self.settings.openai_model if provider == "openai" else self.settings.ollama_model
        log_event("llm_generate_text_start", provider=provider, model=model)
        if provider == "openai":
            result = self._generate_text_openai(prompt, temperature=temperature)
            log_event("llm_generate_text_success", provider=provider, model=model)
            return result
        def call() -> str:
            url = f"{self.settings.ollama_host}/api/generate"
            payload = {
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            }
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
            except requests.Timeout as exc:
                raise TransientDependencyError("Ollama generate call timed out.") from exc
            except requests.RequestException as exc:
                raise TransientDependencyError(f"Ollama generate call failed: {exc}") from exc
            if response.status_code >= 500:
                raise TransientDependencyError(f"Ollama generate call failed: {response.status_code} {response.text}")
            if response.status_code >= 400:
                raise PermanentDependencyError(f"Ollama generate call failed: {response.status_code} {response.text}")
            data = response.json()
            return str(data.get("response", "")).strip()

        result = run_with_retry(
            call,
            attempts=max(1, self.settings.dependency_retry_attempts),
            backoff_seconds=self.settings.dependency_retry_backoff_seconds,
        )
        log_event("llm_generate_text_success", provider=provider, model=model)
        return result

    def generate_structured(self, prompt: str, schema: type[Any], *, temperature: float = 0.1) -> Any:
        structured_prompt = (
            f"{prompt}\n\n"
            "Return only valid JSON. Do not add markdown, explanations, headings, or extra text."
        )
        raw = self.generate_text(structured_prompt, temperature=temperature)
        return parse_json_model(raw, schema)

    def embed(self, text: str) -> list[float]:
        provider = self.settings.llm_provider.lower()
        model = self.settings.openai_embedding_model if provider == "openai" else self.settings.ollama_embedding_model
        log_event("llm_embed_start", provider=provider, model=model)
        if provider == "openai":
            result = self._embed_openai(text)
            log_event("llm_embed_success", provider=provider, model=model)
            return result
        def call() -> list[float]:
            for endpoint in ("/api/embed", "/api/embeddings"):
                url = f"{self.settings.ollama_host}{endpoint}"
                payload = {"model": self.settings.ollama_embedding_model, "input": text}
                if endpoint.endswith("embeddings"):
                    payload = {"model": self.settings.ollama_embedding_model, "prompt": text}
                try:
                    response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                except requests.Timeout as exc:
                    raise TransientDependencyError("Ollama embedding call timed out.") from exc
                except requests.RequestException as exc:
                    raise TransientDependencyError(f"Ollama embedding call failed: {exc}") from exc
                if response.status_code >= 500:
                    raise TransientDependencyError(f"Ollama embedding call failed: {response.status_code} {response.text}")
                if response.status_code >= 400:
                    continue
                data = response.json()
                if "embeddings" in data and data["embeddings"]:
                    return list(data["embeddings"][0])
                if "embedding" in data:
                    return list(data["embedding"])
            raise PermanentDependencyError("Ollama embedding call failed on both /api/embed and /api/embeddings.")

        result = run_with_retry(
            call,
            attempts=max(1, self.settings.dependency_retry_attempts),
            backoff_seconds=self.settings.dependency_retry_backoff_seconds,
        )
        log_event("llm_embed_success", provider=provider, model=model)
        return result

    def healthcheck(self) -> dict[str, Any]:
        try:
            self.generate_text("Reply with JSON: {\"ok\": true}", temperature=0.0)
            provider = self.settings.llm_provider.lower()
            model = self.settings.openai_model if provider == "openai" else self.settings.ollama_model
            return {"status": "ok", "provider": provider, "model": model}
        except (TransientDependencyError, PermanentDependencyError, InvalidModelOutputError) as exc:
            return {"status": "error", "provider": self.settings.llm_provider.lower(), "error": str(exc)}

    def _generate_text_openai(self, prompt: str, *, temperature: float) -> str:
        def call() -> str:
            response = self._openai_post(
                "/chat/completions",
                {
                    "model": self.settings.openai_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                },
            )
            try:
                return str(response["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidModelOutputError(f"OpenAI chat response missing message content: {response}") from exc

        return run_with_retry(
            call,
            attempts=max(1, self.settings.dependency_retry_attempts),
            backoff_seconds=self.settings.dependency_retry_backoff_seconds,
        )

    def _embed_openai(self, text: str) -> list[float]:
        def call() -> list[float]:
            response = self._openai_post(
                "/embeddings",
                {
                    "model": self.settings.openai_embedding_model,
                    "input": text,
                },
            )
            try:
                embedding = response["data"][0]["embedding"]
                return list(embedding)
            except (KeyError, IndexError, TypeError) as exc:
                raise InvalidModelOutputError(f"OpenAI embedding response missing embedding data: {response}") from exc

        return run_with_retry(
            call,
            attempts=max(1, self.settings.dependency_retry_attempts),
            backoff_seconds=self.settings.dependency_retry_backoff_seconds,
        )

    def _openai_post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.openai_base_url.rstrip('/')}{path}"
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout_seconds)
        except requests.Timeout as exc:
            raise TransientDependencyError("OpenAI request timed out.") from exc
        except requests.RequestException as exc:
            raise TransientDependencyError(f"OpenAI request failed: {exc}") from exc
        if response.status_code >= 500:
            raise TransientDependencyError(f"OpenAI request failed: {response.status_code} {response.text}")
        if response.status_code >= 400:
            raise PermanentDependencyError(f"OpenAI request failed: {response.status_code} {response.text}")
        return response.json()
