"""
Ollama generation backend for the Medical RAG application.

Save as:
    models/ollama_generator.py

This module replaces the evaluation notebook's `vllm_chat(...)` helper with
an Ollama-backed equivalent while preserving the same calling pattern:

    text, meta = generator.chat(
        messages=[...],
        max_tokens=1024,
        temperature=0.0,
    )

Returned metadata mirrors the evaluation pipeline as closely as possible:
- latency_s
- input_tokens
- output_tokens
"""

from __future__ import annotations

import time
from typing import Any

import requests


class OllamaGenerator:
    """
    Thin Ollama chat client used by the RAG pipeline.

    Parameters
    ----------
    model:
        Name of a locally installed Ollama model, e.g. "qwen3:8b".
    base_url:
        Ollama API base URL.
    timeout:
        Maximum HTTP request time in seconds.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self._validate_model_available()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate one chat completion through Ollama.

        This intentionally follows the signature of the previous `vllm_chat`
        function so the rest of the evaluation/RAG code can be migrated with
        minimal changes.
        """

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }

        # Qwen-style reasoning models may otherwise expose internal thinking.
        # Ollama supports this field for models that implement thinking control.
        payload["think"] = False

        t0 = time.perf_counter()

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure `ollama serve` "
                "is running on http://127.0.0.1:11434."
            ) from exc

        except requests.Timeout as exc:
            raise RuntimeError(
                f"Ollama generation timed out after {self.timeout} seconds."
            ) from exc

        except requests.HTTPError as exc:
            detail = ""

            try:
                detail = response.json().get("error", "")
            except Exception:
                detail = response.text[:500]

            if detail:
                raise RuntimeError(
                    f"Ollama request failed: {detail}"
                ) from exc

            raise RuntimeError(
                f"Ollama request failed with HTTP {response.status_code}."
            ) from exc

        latency = time.perf_counter() - t0

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "Ollama returned an invalid JSON response."
            ) from exc

        message = data.get("message") or {}
        text = str(message.get("content") or "").strip()

        if not text:
            raise RuntimeError(
                f"Ollama model `{self.model}` returned an empty response."
            )

        meta = {
            "latency_s": float(latency),
            "input_tokens": int(data.get("prompt_eval_count") or 0),
            "output_tokens": int(data.get("eval_count") or 0),
            "model": str(data.get("model") or self.model),
            "done": bool(data.get("done", False)),
        }

        return text, meta

    def simple_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> tuple[str, dict[str, Any]]:
        """
        Convenience helper for the many two-message calls in the original
        evaluation notebook.
        """

        return self.chat(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Ollama availability helpers
    # ------------------------------------------------------------------

    def _validate_model_available(self) -> None:
        """Fail early if the selected Ollama model is not installed."""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()

        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start Ollama before initializing "
                "the generator."
            ) from exc

        data = response.json()
        models = data.get("models", [])

        installed = {
            str(item.get("name") or item.get("model") or "")
            for item in models
        }

        installed.discard("")

        if self.model not in installed:
            available = ", ".join(sorted(installed)) or "none"

            raise RuntimeError(
                f"Ollama model `{self.model}` is not installed. "
                f"Available models: {available}"
            )

    def ping(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            response = requests.get(
                f"{self.base_url}/api/version",
                timeout=3,
            )
            return response.ok
        except requests.RequestException:
            return False
