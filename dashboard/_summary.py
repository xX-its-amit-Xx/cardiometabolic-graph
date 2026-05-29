"""Deterministic, template-driven clinical summary generator.

We intentionally *do not* use an LLM here. In a regulated environment a
clinician-facing summary needs to be:

  * deterministic given the same inputs
  * auditable (every sentence traces to a rule below)
  * free of hallucinated values

So we use plain Jinja-style string templates over a small set of derived
features. Reviewers can map every output sentence to a rule in
``RULES`` below.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PatientSummaryInputs:
    patient_id: str
    last_hba1c: float | None
    pred_hba1c_next: float | None
    pred_hba1c_low: float | None
    pred_hba1c_high: float | None
    last_ldl: float | None
    last_bp_sys: float | None
    last_bp_dia: float | None
    app_opens_30d: int
    message_responses_30d: int
    glucose_logs_30d: int
    dropout_risk: float | None
    top_factors: list[tuple[str, float]]  # (feature_name, signed shap impact)


def _hba1c_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 5.7:
        return "normal"
    if value < 6.5:
        return "pre-diabetic"
    return "diabetic"


def _trend(last: float | None, pred: float | None) -> str:
    if last is None or pred is None:
        return "trend uncertain"
    diff = pred - last
    if abs(diff) < 0.2:
        return "trend stable"
    return "trend rising" if diff > 0 else "trend falling"


def _engagement_phrase(opens: int, responses: int, logs: int) -> str:
    if opens >= 60:
        intensity = "highly engaged"
    elif opens >= 20:
        intensity = "moderately engaged"
    elif opens >= 5:
        intensity = "lightly engaged"
    else:
        intensity = "minimally engaged"
    return (
        f"Over the last 30 days, the patient was {intensity} "
        f"({opens} app opens, {responses} message responses, {logs} glucose logs)."
    )


def _dropout_phrase(p: float | None) -> str:
    if p is None:
        return ""
    if p >= 0.5:
        return (
            f" Model estimates elevated dropout risk ({p:.0%}) — consider a re-engagement outreach."
        )
    if p >= 0.25:
        return f" Dropout risk is moderate ({p:.0%})."
    return f" Dropout risk is low ({p:.0%})."


def _factors_phrase(factors: list[tuple[str, float]]) -> str:
    if not factors:
        return ""
    parts = []
    for name, impact in factors[:3]:
        direction = "raising" if impact > 0 else "lowering"
        parts.append(f"{name} ({direction})")
    return f" Largest contributors to the prediction: {', '.join(parts)}."


def build_summary(inp: PatientSummaryInputs) -> str:
    sentences: list[str] = []

    if inp.last_hba1c is not None:
        sentences.append(
            f"Most recent HbA1c: {inp.last_hba1c:.1f}% ({_hba1c_band(inp.last_hba1c)})."
        )

    if inp.pred_hba1c_next is not None:
        band = _hba1c_band(inp.pred_hba1c_next)
        trend = _trend(inp.last_hba1c, inp.pred_hba1c_next)
        ci = ""
        if inp.pred_hba1c_low is not None and inp.pred_hba1c_high is not None:
            ci = f" (90% CI {inp.pred_hba1c_low:.1f}–{inp.pred_hba1c_high:.1f})"
        sentences.append(
            f"Projected HbA1c at next reading: {inp.pred_hba1c_next:.1f}%{ci} — {band}, {trend}."
        )

    if inp.last_ldl is not None and inp.last_ldl >= 130:
        sentences.append(f"LDL last measured at {inp.last_ldl:.0f} mg/dL — above guideline.")
    if inp.last_bp_sys is not None and inp.last_bp_dia is not None:
        if inp.last_bp_sys >= 140 or inp.last_bp_dia >= 90:
            sentences.append(
                f"BP {inp.last_bp_sys:.0f}/{inp.last_bp_dia:.0f} — stage 2 hypertensive range."
            )
        elif inp.last_bp_sys >= 130 or inp.last_bp_dia >= 80:
            sentences.append(f"BP {inp.last_bp_sys:.0f}/{inp.last_bp_dia:.0f} — elevated.")

    sentences.append(
        _engagement_phrase(inp.app_opens_30d, inp.message_responses_30d, inp.glucose_logs_30d)
    )
    sentences.append(_dropout_phrase(inp.dropout_risk).strip())
    sentences.append(_factors_phrase(inp.top_factors).strip())

    return " ".join(s for s in sentences if s)


# A traceable rule index — every fact above maps to one of these IDs.
RULES = {
    "R1": "Map HbA1c value to ADA band: normal <5.7, pre-diabetic 5.7-6.5, diabetic >=6.5.",
    "R2": "Trend label depends on signed difference predicted - last; stable if |diff|<0.2.",
    "R3": "Engagement intensity buckets on 30-day app opens: 0-4 min, 5-19 light, 20-59 mod, 60+ high.",
    "R4": "Dropout outreach trigger when p_dropout >= 0.5.",
    "R5": "LDL flagged when >=130 mg/dL (per AHA guidance for moderate-high risk).",
    "R6": "BP flagged as stage-2 hypertensive when systolic >=140 or diastolic >=90; elevated at 130/80.",
}
