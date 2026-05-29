"""HuggingFace transformers backend — for air-gapped deployments where
Ollama isn't installable. Loads an open-weight instruct model directly.

Default: ``microsoft/Phi-3-mini-4k-instruct`` (~2.5 GB, 4k context,
permissive MIT license, runs on CPU in ~6 s/summary).

Other tested-good defaults:
    "meta-llama/Llama-3.2-3B-Instruct"  (gated; needs HF login)
    "Qwen/Qwen2.5-3B-Instruct"          (Apache 2.0)
    "google/gemma-2-2b-it"              (Gemma Terms)

Set ``CMG_HF_MODEL`` to override.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .base import SummaryRequest
from .deterministic import DeterministicBackend
from .prompts import build_messages

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str):
    # Import is intentionally lazy and inside the cached helper — the
    # transformers + torch wheels are heavy and we don't want to pay for
    # them at module import time.
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    return pipeline("text-generation", model=model, tokenizer=tok)


class TransformersBackend:
    name = "transformers"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("CMG_HF_MODEL", DEFAULT_MODEL)
        self._fallback = DeterministicBackend()

    def summarize(self, req: SummaryRequest) -> str:
        try:
            pipe = _load_pipeline(self.model_name)
        except Exception:
            # Most likely: transformers not installed, or no internet for
            # the first model download. Fall back gracefully.
            return self._fallback.summarize(req)

        messages = build_messages(req)
        try:
            out = pipe(
                messages,
                max_new_tokens=req.max_words * 4,
                temperature=max(req.temperature, 1e-3),
                do_sample=req.temperature > 0,
                return_full_text=False,
            )
        except Exception:
            return self._fallback.summarize(req)

        text = out[0]["generated_text"]
        if isinstance(text, list):  # newer transformers returns chat list
            text = text[-1].get("content", "")
        return text.strip()
