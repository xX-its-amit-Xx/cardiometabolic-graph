"""Backend-neutral data classes and the ``Summarizer`` protocol.

Adding a new backend means:
  1. Implement ``Summarizer.summarize(request) -> str``.
  2. Register it in ``factory.get_summarizer``.

The structured ``PatientFacts`` payload is the same for every backend so
the inputs are fully auditable — a regulator can ask "what did the
model see?" and the answer is the same JSON regardless of which backend
produced the prose.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

Audience = Literal["clinician", "patient", "care_coordinator", "prior_auth"]


@dataclass
class PatientFacts:
    """The structured payload every backend sees. Designed to be JSON-
    serializable and small enough to fit comfortably in a 4k context."""

    patient_id: str
    last_hba1c: float | None
    pred_hba1c_next: float | None
    pred_hba1c_low: float | None = None
    pred_hba1c_high: float | None = None
    last_ldl: float | None = None
    last_bp_sys: float | None = None
    last_bp_dia: float | None = None
    app_opens_30d: int = 0
    message_responses_30d: int = 0
    glucose_logs_30d: int = 0
    dropout_risk: float | None = None
    top_factors: list[tuple[str, float]] = field(default_factory=list)
    age_years: int | None = None
    sex: str | None = None
    medications: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SummaryRequest:
    facts: PatientFacts
    audience: Audience = "clinician"
    max_words: int = 120
    temperature: float = 0.0
    # When True, backends should refuse to invent numbers not present in
    # ``facts``. The deterministic backend always honors this; LLM
    # backends honor it via system-prompt instruction (best effort).
    facts_only: bool = True


@runtime_checkable
class Summarizer(Protocol):
    """Every backend implements this single method."""

    name: str

    def summarize(self, request: SummaryRequest) -> str: ...
