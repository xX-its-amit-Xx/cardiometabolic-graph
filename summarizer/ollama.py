"""Ollama backend — talks HTTP to a local Ollama server.

Ollama (https://ollama.ai) is the easiest open-weight LLM runtime to
deploy in a regulated environment: a single binary, no Python deps, the
model file lives on local disk, nothing leaves the host. The default
model is ``llama3.1:8b`` which fits in ~5 GB RAM and produces sensible
clinical summaries.

Setup on the host:
    ollama pull llama3.1:8b
    ollama serve   # starts http://localhost:11434

Environment variables:
    OLLAMA_HOST    default ``http://localhost:11434``
    OLLAMA_MODEL   default ``llama3.1:8b``

Falls back transparently to the deterministic backend on connection
errors so the dashboard never goes blank.
"""

from __future__ import annotations

import json
import os
from urllib import error, request

from .base import SummaryRequest
from .deterministic import DeterministicBackend
from .prompts import build_messages

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaBackend:
    name = "ollama"

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", DEFAULT_HOST)).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self._fallback = DeterministicBackend()

    def summarize(self, req: SummaryRequest) -> str:
        body = {
            "model": self.model,
            "messages": build_messages(req),
            "stream": False,
            "options": {
                "temperature": req.temperature,
                # Keep output bounded — open-weight models otherwise
                # ramble. 4 tokens/word is a generous upper bound.
                "num_predict": req.max_words * 4,
            },
        }
        payload = json.dumps(body).encode("utf-8")
        url = f"{self.host}/api/chat"
        http_req = request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with request.urlopen(http_req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()
        except (error.URLError, error.HTTPError, KeyError, json.JSONDecodeError):
            # Quietly degrade — the dashboard's safety story is that it
            # always renders *something* auditable, even if the LLM is
            # down.
            return self._fallback.summarize(req)
