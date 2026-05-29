"""Summarizer backend tests.

We test:
  * factory returns the right backend by name and env
  * deterministic backend produces the same output as the underlying
    template (no regressions in the safe path)
  * Ollama backend silently falls back to deterministic when the server
    is unreachable (the dashboard's "always render something" invariant)
  * unknown backend names fall back to deterministic
"""

from __future__ import annotations

import pytest

from summarizer import PatientFacts, SummaryRequest, get_summarizer
from summarizer.deterministic import DeterministicBackend


@pytest.fixture
def facts() -> PatientFacts:
    return PatientFacts(
        patient_id="TEST001",
        last_hba1c=7.4,
        pred_hba1c_next=7.7,
        pred_hba1c_low=7.2,
        pred_hba1c_high=8.2,
        last_ldl=145.0,
        last_bp_sys=138.0,
        last_bp_dia=86.0,
        app_opens_30d=42,
        message_responses_30d=12,
        glucose_logs_30d=21,
        dropout_risk=0.31,
        top_factors=[("HbA1c_last", 0.42), ("ldl_last", 0.11)],
    )


def test_factory_default_is_deterministic(monkeypatch):
    monkeypatch.delenv("CMG_SUMMARIZER", raising=False)
    assert get_summarizer().name == "deterministic"


def test_factory_honors_env(monkeypatch):
    monkeypatch.setenv("CMG_SUMMARIZER", "ollama")
    assert get_summarizer().name == "ollama"


def test_factory_unknown_falls_back(monkeypatch):
    monkeypatch.delenv("CMG_SUMMARIZER", raising=False)
    assert get_summarizer("definitely-not-a-backend").name == "deterministic"


def test_deterministic_is_idempotent(facts: PatientFacts):
    summarizer = DeterministicBackend()
    req = SummaryRequest(facts=facts, audience="clinician")
    a = summarizer.summarize(req)
    b = summarizer.summarize(req)
    assert a == b
    assert "trend rising" in a
    assert "moderately engaged" in a


def test_ollama_falls_back_on_unreachable_server(facts: PatientFacts, monkeypatch):
    # Point Ollama at a definitely-dead port so the request fails fast.
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:1")
    from summarizer.ollama import OllamaBackend
    backend = OllamaBackend(timeout=1.0)
    text = backend.summarize(SummaryRequest(facts=facts, max_words=80))
    assert "trend rising" in text


def test_facts_to_dict_is_jsonable(facts: PatientFacts):
    import json
    d = facts.to_dict()
    # top_factors contains tuples which json can handle (as arrays)
    s = json.dumps(d, default=list)
    assert "TEST001" in s
    assert "HbA1c_last" in s
