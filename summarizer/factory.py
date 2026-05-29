"""Backend factory — pick the summarizer based on env or argument."""

from __future__ import annotations

import os

from .base import Summarizer
from .deterministic import DeterministicBackend

VALID = {"deterministic", "ollama", "transformers"}


def get_summarizer(name: str | None = None) -> Summarizer:
    """Return the requested backend, falling back to deterministic for
    anything unknown or any import error. Order of resolution:
      1. explicit ``name`` argument
      2. ``CMG_SUMMARIZER`` env var
      3. ``"deterministic"``
    """
    chosen = (name or os.environ.get("CMG_SUMMARIZER", "deterministic")).lower()
    if chosen not in VALID:
        return DeterministicBackend()

    if chosen == "deterministic":
        return DeterministicBackend()

    if chosen == "ollama":
        from .ollama import OllamaBackend

        return OllamaBackend()

    if chosen == "transformers":
        from .transformers import TransformersBackend

        return TransformersBackend()

    return DeterministicBackend()
