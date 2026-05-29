"""Deterministic template backend — wraps ``dashboard._summary`` so the
Summarizer protocol is satisfied without changing the existing template.

This is the default. No network, no model files, no API key, no
randomness. Every output sentence traces to a documented rule in
``dashboard/_summary.RULES``.
"""

from __future__ import annotations

from dashboard._summary import PatientSummaryInputs, build_summary

from .base import SummaryRequest


class DeterministicBackend:
    name = "deterministic"

    def summarize(self, request: SummaryRequest) -> str:
        f = request.facts
        inputs = PatientSummaryInputs(
            patient_id=f.patient_id,
            last_hba1c=f.last_hba1c,
            pred_hba1c_next=f.pred_hba1c_next,
            pred_hba1c_low=f.pred_hba1c_low,
            pred_hba1c_high=f.pred_hba1c_high,
            last_ldl=f.last_ldl,
            last_bp_sys=f.last_bp_sys,
            last_bp_dia=f.last_bp_dia,
            app_opens_30d=f.app_opens_30d,
            message_responses_30d=f.message_responses_30d,
            glucose_logs_30d=f.glucose_logs_30d,
            dropout_risk=f.dropout_risk,
            top_factors=f.top_factors,
        )
        return build_summary(inputs)
