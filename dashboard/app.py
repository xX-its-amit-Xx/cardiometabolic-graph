"""Streamlit clinician-facing summary for the cardiometabolic graph demo.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etl._common import processed_path, synthetic_path
from models._features import load_cached

from ._summary import PatientSummaryInputs, build_summary

st.set_page_config(page_title="Cardiometabolic Graph — Clinician View", layout="wide", page_icon="🩺")

st.title("Cardiometabolic patient summary")
st.caption(
    "Reference implementation • not for clinical use • all data shown here is from "
    "the MIMIC-IV demo subset, NHANES public files, and a documented synthetic generator."
)


@st.cache_data(show_spinner=False)
def _load_events() -> pd.DataFrame:
    p = synthetic_path() / "engagement_events.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["patient_id", "kind", "ts", "value"])
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def _load_labs() -> pd.DataFrame:
    p = processed_path() / "labs.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["patient_id", "name", "value", "taken_ts"])
    df = pd.read_parquet(p)
    df["taken_ts"] = pd.to_datetime(df["taken_ts"], utc=True)
    return df


@st.cache_data(show_spinner=False)
def _load_predictions() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (gbm_pred, dropout_pred, shap_values) — empty frames if missing."""
    gbm_p = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    drop_p = Path("artifacts/dropout/dropout_predictions.parquet")
    shap_p = Path("artifacts/gbm/shap_values.parquet")
    return (
        pd.read_parquet(gbm_p) if gbm_p.exists() else pd.DataFrame(),
        pd.read_parquet(drop_p) if drop_p.exists() else pd.DataFrame(),
        pd.read_parquet(shap_p) if shap_p.exists() else pd.DataFrame(),
    )


events_df = _load_events()
labs_df = _load_labs()
ff = load_cached()
gbm_pred, drop_pred, shap_values = _load_predictions()

if ff is None and events_df.empty:
    st.warning(
        "No data found yet. Run `make pipeline` first to generate synthetic data, "
        "run ETL, and train the models."
    )
    st.stop()

available_pids = sorted(set(
    list((ff.patient_ids if ff is not None else pd.Index([])))
    + list(events_df["patient_id"].unique())
    + list(labs_df["patient_id"].unique())
))

# ---- Sidebar ----
with st.sidebar:
    st.header("Patient")
    if not available_pids:
        st.error("No patients available.")
        st.stop()
    pid = st.selectbox("Patient ID", available_pids, index=0)
    st.divider()
    st.caption("Models loaded:")
    st.write(
        {
            "gbm_hba1c": not gbm_pred.empty,
            "dropout":   not drop_pred.empty,
            "shap":      not shap_values.empty,
        }
    )

# ---- Pull this patient's slices ----
ev = events_df[events_df["patient_id"] == pid].sort_values("ts")
labs = labs_df[labs_df["patient_id"] == pid].sort_values("taken_ts")

hba1c_series = labs[labs["name"] == "HbA1c"]
last_hba1c = float(hba1c_series["value"].iloc[-1]) if not hba1c_series.empty else None

pred_next = float(gbm_pred.loc[pid, "pred"]) if not gbm_pred.empty and pid in gbm_pred.index else None
pred_low  = pred_next - 0.5 if pred_next is not None else None
pred_high = pred_next + 0.5 if pred_next is not None else None
dropout_risk = float(drop_pred.loc[pid, "p_dropout"]) if not drop_pred.empty and pid in drop_pred.index else None

last_ldl = None
last_bp_sys = None
last_bp_dia = None
if not labs.empty:
    ldl = labs[labs["name"] == "ldl"]
    if not ldl.empty:
        last_ldl = float(ldl["value"].iloc[-1])

last_30d = ev[ev["ts"] >= ev["ts"].max() - pd.Timedelta(days=30)] if not ev.empty else ev
opens_30 = int((last_30d["kind"] == "app_open").sum())
resp_30 = int((last_30d["kind"] == "message_response").sum())
glog_30 = int((last_30d["kind"] == "glucose_log").sum())

top_factors: list[tuple[str, float]] = []
if not shap_values.empty and pid in shap_values.index:
    row = shap_values.loc[pid]
    top_factors = sorted(row.items(), key=lambda kv: -abs(kv[1]))[:5]

summary = build_summary(PatientSummaryInputs(
    patient_id=pid,
    last_hba1c=last_hba1c,
    pred_hba1c_next=pred_next,
    pred_hba1c_low=pred_low,
    pred_hba1c_high=pred_high,
    last_ldl=last_ldl,
    last_bp_sys=last_bp_sys,
    last_bp_dia=last_bp_dia,
    app_opens_30d=opens_30,
    message_responses_30d=resp_30,
    glucose_logs_30d=glog_30,
    dropout_risk=dropout_risk,
    top_factors=top_factors,
))

# ---- Header KPIs ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Last HbA1c (%)", f"{last_hba1c:.1f}" if last_hba1c is not None else "n/a")
c2.metric("Predicted next HbA1c (%)", f"{pred_next:.1f}" if pred_next is not None else "n/a",
          delta=(f"{pred_next - last_hba1c:+.2f}" if (pred_next is not None and last_hba1c is not None) else None))
c3.metric("30-day app opens", opens_30)
c4.metric("Dropout risk", f"{dropout_risk:.0%}" if dropout_risk is not None else "n/a")

st.markdown("### Clinician summary")
st.info(summary or "No data available to summarize for this patient.")

# ---- Engagement timeline ----
st.markdown("### Engagement timeline (last 180 days)")
if ev.empty:
    st.write("_No engagement events for this patient._")
else:
    daily = (
        ev.groupby([pd.Grouper(key="ts", freq="D"), "kind"])
        .size()
        .reset_index(name="count")
    )
    fig = px.area(daily, x="ts", y="count", color="kind", groupnorm=None, height=320)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend_title=None)
    st.plotly_chart(fig, use_container_width=True)

# ---- HbA1c trajectory with uncertainty band ----
st.markdown("### HbA1c trajectory")
if hba1c_series.empty and pred_next is None:
    st.write("_No HbA1c history available._")
else:
    fig = go.Figure()
    if not hba1c_series.empty:
        fig.add_trace(go.Scatter(
            x=hba1c_series["taken_ts"], y=hba1c_series["value"],
            mode="markers+lines", name="observed",
        ))
    if pred_next is not None and not hba1c_series.empty:
        next_ts = hba1c_series["taken_ts"].iloc[-1] + pd.Timedelta(days=90)
        fig.add_trace(go.Scatter(
            x=[next_ts, next_ts], y=[pred_low, pred_high],
            mode="lines", line=dict(width=10, color="rgba(0,100,200,0.25)"),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[next_ts], y=[pred_next], mode="markers",
            name="predicted",
            marker=dict(size=14, color="rgba(0,100,200,1)", symbol="diamond"),
        ))
    fig.update_yaxes(title="HbA1c (%)")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---- Top SHAP factors ----
st.markdown("### Top model factors (SHAP)")
if not top_factors:
    st.write("_SHAP values not available — run `python -m explain.shap_analysis` first._")
else:
    factor_df = pd.DataFrame(top_factors, columns=["feature", "shap"])
    factor_df["abs"] = factor_df["shap"].abs()
    factor_df = factor_df.sort_values("abs", ascending=True)
    fig = px.bar(factor_df, x="shap", y="feature", orientation="h", height=320,
                 color="shap", color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Summary text is generated by a deterministic template engine "
    "(`dashboard/_summary.py`). Each output sentence traces to a documented rule — "
    "see `RULES` in that module."
)
