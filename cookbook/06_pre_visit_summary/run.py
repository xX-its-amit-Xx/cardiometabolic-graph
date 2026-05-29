"""Cookbook 06 — pre-visit summary worklist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from etl._common import log, processed_path, synthetic_path
from summarizer import PatientFacts, SummaryRequest, get_summarizer


HERE = Path(__file__).resolve().parent


def _gather(pid: str) -> PatientFacts | None:
    labs_path = processed_path() / "labs.parquet"
    pred_path = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    drop_path = Path("artifacts/dropout/dropout_predictions.parquet")
    shap_path = Path("artifacts/gbm/shap_values.parquet")
    ev_path = synthetic_path() / "engagement_events.parquet"

    if not labs_path.exists():
        return None
    labs = pd.read_parquet(labs_path)
    labs["taken_ts"] = pd.to_datetime(labs["taken_ts"], utc=True)
    pl = labs[labs["patient_id"] == pid]
    hba1c = pl[pl["name"] == "HbA1c"].sort_values("taken_ts")
    ldl = pl[pl["name"] == "ldl"].sort_values("taken_ts")

    pred_next = None
    if pred_path.exists():
        p = pd.read_parquet(pred_path)
        if pid in p.index:
            pred_next = float(p.loc[pid, "pred"])

    drop_risk = None
    if drop_path.exists():
        d = pd.read_parquet(drop_path)
        if pid in d.index:
            drop_risk = float(d.loc[pid, "p_dropout"])

    top_factors: list[tuple[str, float]] = []
    if shap_path.exists():
        s = pd.read_parquet(shap_path)
        if pid in s.index:
            row = s.loc[pid]
            top_factors = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:5]

    opens_30 = resp_30 = glog_30 = 0
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
        pe = ev[ev["patient_id"] == pid]
        if not pe.empty:
            cutoff = pe["ts"].max() - pd.Timedelta(days=30)
            r = pe[pe["ts"] >= cutoff]
            opens_30 = int((r["kind"] == "app_open").sum())
            resp_30 = int((r["kind"] == "message_response").sum())
            glog_30 = int((r["kind"] == "glucose_log").sum())

    return PatientFacts(
        patient_id=pid,
        last_hba1c=float(hba1c["value"].iloc[-1]) if not hba1c.empty else None,
        pred_hba1c_next=pred_next,
        pred_hba1c_low=pred_next - 0.5 if pred_next else None,
        pred_hba1c_high=pred_next + 0.5 if pred_next else None,
        last_ldl=float(ldl["value"].iloc[-1]) if not ldl.empty else None,
        app_opens_30d=opens_30,
        message_responses_30d=resp_30,
        glucose_logs_30d=glog_30,
        dropout_risk=drop_risk,
        top_factors=top_factors,
    )


def _questions_to_ask(f: PatientFacts) -> list[str]:
    qs: list[str] = []
    if f.last_hba1c is not None and f.pred_hba1c_next is not None:
        if f.pred_hba1c_next - f.last_hba1c > 0.3:
            qs.append("Predicted HbA1c rise — ask about recent diet, stress, or medication adherence changes.")
        elif f.last_hba1c - f.pred_hba1c_next > 0.3:
            qs.append("Predicted HbA1c improvement — affirm the patient's current self-management strategy.")
    if f.last_ldl is not None and f.last_ldl >= 130:
        qs.append(f"LDL {f.last_ldl:.0f} mg/dL — discuss statin adherence and side effects.")
    if f.dropout_risk is not None and f.dropout_risk >= 0.40:
        qs.append(f"Dropout risk {f.dropout_risk:.0%} — explore barriers to app engagement (time? friction? value?).")
    if f.glucose_logs_30d == 0 and (f.last_hba1c or 0) >= 7.0:
        qs.append("Zero glucose logs in last 30 days despite elevated HbA1c — consider a CGM trial.")
    if not qs:
        qs.append("Routine follow-up — confirm med list and refill timing.")
    return qs


def build_worklist(pids: list[str], audience: str = "clinician") -> str:
    summarizer = get_summarizer()
    lines: list[str] = []
    today = datetime.now(timezone.utc).date().isoformat()
    lines.append(f"# Pre-visit worklist — {today}")
    lines.append("")
    lines.append(f"_Audience: {audience} • Summarizer: `{summarizer.name}`_")
    lines.append("")

    rendered = 0
    for pid in pids:
        facts = _gather(pid)
        if facts is None:
            lines.append(f"## `{pid}` — _data unavailable_\n")
            continue
        summary = summarizer.summarize(
            SummaryRequest(facts=facts, audience=audience, max_words=80)
        )
        qs = _questions_to_ask(facts)
        lines.append(f"## `{pid}`")
        lines.append("")
        lines.append(summary)
        lines.append("")
        lines.append("**Questions to ask:**")
        for q in qs:
            lines.append(f"- {q}")
        lines.append("")
        lines.append("---")
        lines.append("")
        rendered += 1

    lines.append(f"_Rendered {rendered} patient(s)._")
    return "\n".join(lines) + "\n"


def _top_by_delta(top_n: int) -> list[str]:
    pred_path = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    feat_path = processed_path() / "features.parquet"
    if not pred_path.exists() or not feat_path.exists():
        return []
    pred = pd.read_parquet(pred_path)["pred"]
    feats = pd.read_parquet(feat_path)
    if "HbA1c_last" not in feats.columns:
        return list(pred.head(top_n).index)
    df = pd.DataFrame({"pred": pred})
    df["last"] = feats["HbA1c_last"].reindex(df.index)
    df["delta"] = df["pred"] - df["last"]
    return df.sort_values("delta", ascending=False).head(top_n).index.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", nargs="*", default=None,
                        help="Explicit patient IDs. If omitted, uses --top.")
    parser.add_argument("--top", type=int, default=5,
                        help="If --patients not given, take top N by predicted HbA1c rise.")
    parser.add_argument("--audience", choices=("clinician", "patient", "care_coordinator"),
                        default="clinician")
    args = parser.parse_args()

    pids = args.patients or _top_by_delta(args.top)
    if not pids:
        raise SystemExit("No patients available. Run `make pipeline-parquet` first.")

    body = build_worklist(pids, audience=args.audience)
    out = HERE / f"pre_visit_{datetime.now(timezone.utc).date().isoformat()}_{args.audience}.md"
    out.write_text(body, encoding="utf-8")
    log.info("wrote pre-visit worklist (%d patients) -> %s", len(pids), out)


if __name__ == "__main__":
    main()
