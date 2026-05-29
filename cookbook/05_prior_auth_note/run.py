"""Cookbook 05 — generate a structured prior-authorization note."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from etl._common import log, processed_path, synthetic_path
from summarizer import PatientFacts, SummaryRequest, get_summarizer


HERE = Path(__file__).resolve().parent


def _load_facts(patient_id: str) -> tuple[PatientFacts, pd.DataFrame]:
    labs_path = processed_path() / "labs.parquet"
    feat_path = processed_path() / "features.parquet"
    pred_path = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    drop_path = Path("artifacts/dropout/dropout_predictions.parquet")
    ev_path = synthetic_path() / "engagement_events.parquet"

    if not labs_path.exists() or not feat_path.exists():
        raise FileNotFoundError(
            "Required parquet files missing. Run `make pipeline-parquet` first."
        )

    labs = pd.read_parquet(labs_path)
    labs["taken_ts"] = pd.to_datetime(labs["taken_ts"], utc=True)
    patient_labs = labs[labs["patient_id"] == patient_id]
    if patient_labs.empty:
        raise KeyError(f"Patient {patient_id} has no labs.")
    hba1c = patient_labs[patient_labs["name"] == "HbA1c"].sort_values("taken_ts")
    ldl = patient_labs[patient_labs["name"] == "ldl"].sort_values("taken_ts")

    last_hba1c = float(hba1c["value"].iloc[-1]) if not hba1c.empty else None
    last_ldl = float(ldl["value"].iloc[-1]) if not ldl.empty else None

    pred_next = None
    if pred_path.exists():
        pred = pd.read_parquet(pred_path)
        if patient_id in pred.index:
            pred_next = float(pred.loc[patient_id, "pred"])

    dropout = None
    if drop_path.exists():
        drop = pd.read_parquet(drop_path)
        if patient_id in drop.index:
            dropout = float(drop.loc[patient_id, "p_dropout"])

    opens_30 = resp_30 = glog_30 = 0
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
        pe = ev[ev["patient_id"] == patient_id]
        if not pe.empty:
            cutoff = pe["ts"].max() - pd.Timedelta(days=30)
            recent = pe[pe["ts"] >= cutoff]
            opens_30 = int((recent["kind"] == "app_open").sum())
            resp_30 = int((recent["kind"] == "message_response").sum())
            glog_30 = int((recent["kind"] == "glucose_log").sum())

    facts = PatientFacts(
        patient_id=patient_id,
        last_hba1c=last_hba1c,
        pred_hba1c_next=pred_next,
        pred_hba1c_low=pred_next - 0.5 if pred_next else None,
        pred_hba1c_high=pred_next + 0.5 if pred_next else None,
        last_ldl=last_ldl,
        app_opens_30d=opens_30,
        message_responses_30d=resp_30,
        glucose_logs_30d=glog_30,
        dropout_risk=dropout,
    )
    return facts, hba1c


def build_pa_note(patient_id: str, drug: str, prior_tried: list[str]) -> str:
    facts, hba1c_series = _load_facts(patient_id)
    summarizer = get_summarizer()  # honors CMG_SUMMARIZER

    narrative = summarizer.summarize(
        SummaryRequest(facts=facts, audience="prior_auth", max_words=120)
    )

    six_mo = hba1c_series[
        hba1c_series["taken_ts"] >= hba1c_series["taken_ts"].max() - pd.Timedelta(days=180)
    ]
    delta_6mo = float(six_mo["value"].iloc[-1] - six_mo["value"].iloc[0]) if len(six_mo) >= 2 else None

    lines: list[str] = []
    lines.append(f"# Prior authorization request — `{patient_id}`")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).date().isoformat()}")
    lines.append(f"**Requested therapy:** {drug}")
    lines.append(f"**Summarizer backend:** `{summarizer.name}`")
    lines.append("")
    lines.append("## Clinical narrative")
    lines.append("")
    lines.append(narrative)
    lines.append("")
    lines.append("## Metric table")
    lines.append("")
    lines.append("| Metric | Value | Threshold for indication | Met? |")
    lines.append("|--------|-------|--------------------------|------|")

    if facts.last_hba1c is not None:
        met = "✓" if facts.last_hba1c >= 7.0 else "✗"
        lines.append(f"| Last HbA1c | {facts.last_hba1c:.1f}% | ≥7.0% (uncontrolled T2D) | {met} |")
    if delta_6mo is not None:
        met = "✓" if delta_6mo >= 0.0 else "✗"
        lines.append(f"| 6-month HbA1c change | {delta_6mo:+.2f}% | Not improving on current regimen | {met} |")
    if facts.last_ldl is not None:
        lines.append(f"| Last LDL | {facts.last_ldl:.0f} mg/dL | (CV risk factor — context only) | — |")
    eng_total = facts.app_opens_30d + facts.message_responses_30d + facts.glucose_logs_30d
    met = "✓" if eng_total >= 10 else "✗"
    lines.append(f"| 30-day platform engagement | {eng_total} events | ≥10 (active behavioral participation) | {met} |")
    lines.append("")

    lines.append("## Prior therapies attempted")
    lines.append("")
    if prior_tried:
        for therapy in prior_tried:
            lines.append(f"- {therapy}")
    else:
        lines.append("_None recorded — see attached EHR med history._")
    lines.append("")

    lines.append("## Attestation")
    lines.append("")
    lines.append(
        "I attest that the above information is accurate to the best of my "
        "knowledge based on the patient's chart and the digital "
        "therapeutics engagement record."
    )
    lines.append("")
    lines.append("Prescriber signature: ____________________  Date: __________")
    lines.append("NPI: ________________")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--drug", default="semaglutide 1mg weekly")
    parser.add_argument(
        "--prior",
        nargs="*",
        default=["metformin 1000mg BID (≥3 months, inadequate response)"],
        help="Prior therapies attempted (one per --prior arg).",
    )
    args = parser.parse_args()

    note = build_pa_note(args.patient, args.drug, args.prior)
    out = HERE / f"pa_note_{args.patient}.md"
    out.write_text(note, encoding="utf-8")
    log.info("wrote prior-auth note -> %s", out)


if __name__ == "__main__":
    main()
