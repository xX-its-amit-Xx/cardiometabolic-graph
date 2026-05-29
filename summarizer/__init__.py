"""Patient-summary backends.

The repo's *default* clinician summary path is the deterministic template
engine in ``dashboard/_summary.py`` — no LLM in the loop, every sentence
traces to a documented rule. That stays the recommended default for
regulated environments.

This package is the *optional* LLM-backed path for teams who want
natural-language summaries with style flexibility (lay-language patient
explanations, pre-visit synopses, prior-auth narratives). All adapters
are **open-weight**: nothing here requires an Anthropic / OpenAI / Google
API key. The available backends are:

* ``DeterministicBackend`` — wraps ``dashboard._summary.build_summary``
  so callers can swap backends without touching the call site.
* ``OllamaBackend`` — POSTs to a local Ollama server
  (https://ollama.ai). Default model is ``llama3.1:8b`` which runs on
  ~5GB RAM. No data leaves the host.
* ``TransformersBackend`` — loads a HuggingFace causal-LM directly
  (default ``microsoft/Phi-3-mini-4k-instruct``, ~2.5GB). Slowest first
  call (download + load), fastest subsequent calls; useful for
  air-gapped deployments where Ollama isn't installable.

Backend selection is via the ``CMG_SUMMARIZER`` environment variable
(``deterministic`` / ``ollama`` / ``transformers``). Default is
``deterministic`` so nothing breaks if dependencies aren't installed.
"""

from __future__ import annotations

from .base import PatientFacts, Summarizer, SummaryRequest
from .factory import get_summarizer

__all__ = ["PatientFacts", "Summarizer", "SummaryRequest", "get_summarizer"]
