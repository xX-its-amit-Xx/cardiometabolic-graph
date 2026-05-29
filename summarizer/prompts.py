"""Audience-specific system prompts for LLM backends.

Prompts are intentionally short and instruction-heavy. They emphasize:
  * facts-only constraint (do not invent numbers),
  * audience-appropriate register,
  * structured output (no greetings, no closings).

Editing a prompt here changes the LLM behavior across all open-weight
backends — keep this single source of truth.
"""

from __future__ import annotations

from .base import SummaryRequest

_BASE_RULES = """You are a clinical summarization assistant for a digital
therapeutics platform. You ALWAYS follow these rules:

1. Use ONLY the numbers and facts provided in the JSON payload below.
   Never invent a value. If a field is null, say "not available" or
   simply omit that fact.
2. Never write disclaimers ("I am an AI", "consult your doctor", etc.).
   The platform displays its own disclaimer.
3. No greetings, no signatures, no closings. Write only the summary
   itself.
4. Stay under {max_words} words.
5. Use clinical thresholds: HbA1c <5.7 = normal, 5.7-6.4 = pre-diabetic,
   >=6.5 = diabetic. LDL >=130 = elevated. BP >=140/90 = stage-2 HTN,
   130/80-139/89 = elevated.
"""

_AUDIENCE_BLOCK = {
    "clinician": (
        "Audience: an attending physician reviewing the patient's chart. "
        "Use medical terminology. Lead with the clinical bottom line, "
        "then evidence. Flag actionable abnormalities."
    ),
    "patient": (
        "Audience: the patient themselves. Plain language at a 7th-grade "
        "reading level. No abbreviations (write 'blood pressure' not 'BP', "
        "'long-term blood sugar' not 'HbA1c'). Encouraging tone, but "
        "honest about risk."
    ),
    "care_coordinator": (
        "Audience: a non-clinical care coordinator triaging this week's "
        "outreach calls. Focus on engagement signals and action items. "
        "Short sentences. Bullet-style if it helps."
    ),
    "prior_auth": (
        "Audience: a payer reviewing a prior-authorization request for "
        "intensified diabetes management. Cite the metrics that justify "
        "medical necessity. Neutral, factual tone. No 'I' or 'we'."
    ),
}


def build_messages(request: SummaryRequest) -> list[dict]:
    """Build a chat-style messages list suitable for any open-weight
    instruct model (llama 3, mistral, phi-3 all accept this shape)."""
    system = (
        _BASE_RULES.format(max_words=request.max_words) + "\n\n" + _AUDIENCE_BLOCK[request.audience]
    )
    user = "Patient facts (JSON):\n" f"{request.facts.to_dict()}\n\n" "Write the summary now."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
