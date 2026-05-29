"""Cookbook 08 — pharmacist intervention triggers."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from etl._common import log, processed_path, synthetic_path


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat_p = processed_path() / "features.parquet"
    labs_p = processed_path() / "labs.parquet"
    ev_p = synthetic_path() / "engagement_events.parquet"
    pred_p = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    if not all(p.exists() for p in (feat_p, labs_p, ev_p, pred_p)):
        raise FileNotFoundError("Run `make pipeline-parquet` first.")
    return (
        pd.read_parquet(feat_p),
        pd.read_parquet(labs_p),
        pd.read_parquet(ev_p),
        pd.read_parquet(pred_p),
    )


def _adherence_dropoff(ev: pd.DataFrame) -> pd.Series:
    ev = ev[ev["kind"] == "glucose_log"].copy()
    if ev.empty:
        return pd.Series(dtype=float)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    last_day = ev["ts"].max()
    last_30 = ev[ev["ts"] >= last_day - pd.Timedelta(days=30)].groupby("patient_id").size()
    prev_30 = ev[
        (ev["ts"] >= last_day - pd.Timedelta(days=60))
        & (ev["ts"] < last_day - pd.Timedelta(days=30))
    ].groupby("patient_id").size()
    df = pd.concat([last_30.rename("recent"), prev_30.rename("prior")], axis=1).fillna(0)
    # +1 smoothing avoids divide-by-zero for never-loggers
    df["dropoff"] = ((df["prior"] + 1) - (df["recent"] + 1)).clip(lower=0) / (df["prior"] + 1)
    return df["dropoff"]


def _failing_metric_count(features: pd.DataFrame) -> pd.Series:
    cnt = pd.Series(0, index=features.index)
    if "ldl_last" in features.columns:
        cnt += (features["ldl_last"] >= 130).astype(int)
    if "triglycerides_last" in features.columns:
        cnt += (features["triglycerides_last"] >= 200).astype(int)
    if "hdl_last" in features.columns:
        cnt += (features["hdl_last"] <= 40).astype(int)
    return cnt


def rank(top: int = 30) -> pd.DataFrame:
    features, labs, ev, pred = _load_inputs()
    dropoff = _adherence_dropoff(ev).reindex(features.index, fill_value=0.0)
    failing = _failing_metric_count(features)

    # Pull the actual last-observed HbA1c per patient from the raw labs
    # frame. We can't reuse `features.HbA1c_last` because the feature
    # builder intentionally holds out the most-recent visit as the
    # regression target — leaving `HbA1c_last` zero for patients with
    # only one or two visits. That zero would otherwise inflate the
    # predicted delta to absurd values.
    labs = labs.copy()
    labs["taken_ts"] = pd.to_datetime(labs["taken_ts"], utc=True)
    hba1c_only = labs[labs["name"] == "HbA1c"]
    last_hba1c = (
        hba1c_only.sort_values("taken_ts")
        .groupby("patient_id")["value"]
        .last()
        .astype(float)
    )

    df = pd.DataFrame(index=features.index)
    df["last_hba1c"] = last_hba1c.reindex(features.index)
    df["predicted_hba1c"] = pred["pred"].reindex(features.index)
    df["delta"] = df["predicted_hba1c"] - df["last_hba1c"]
    df["adherence_dropoff"] = dropoff
    df["other_failing"] = failing
    df["leverage_score"] = (
        df["delta"].fillna(0) * 2.0
        + df["adherence_dropoff"] * 1.5
        - df["other_failing"] * 0.5
    )
    # Drop patients without a prediction OR without an observed HbA1c —
    # a pharmacist call without baseline is wasted leverage.
    df = df.dropna(subset=["predicted_hba1c", "last_hba1c"])
    return df.sort_values("leverage_score", ascending=False).head(top).reset_index().rename(
        columns={"index": "patient_id"}
    )


def _script_bullets(row: pd.Series) -> list[str]:
    bullets = []
    if row["adherence_dropoff"] >= 0.4:
        bullets.append("Glucose logging dropped >40% over the last 30 days — confirm meter/CGM is working and reachable.")
    if row["delta"] >= 0.3:
        bullets.append(f"Predicted HbA1c rise (+{row['delta']:.2f}%) — verify last refill date and pill-count.")
    if row["other_failing"] >= 2:
        bullets.append("Multiple lipid panel members out of range — confirm statin dose and adherence.")
    bullets.append("Ask about GI side effects (esp. for GLP-1 / metformin) and CGM tape skin reactions.")
    bullets.append("Schedule a 90-day follow-up call to close the loop.")
    return bullets


def write_report(df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / "call_list.csv", index=False)

    plt.figure(figsize=(6, 3.5))
    plt.barh(df["patient_id"].astype(str), df["leverage_score"], color="#2c7fb8")
    plt.gca().invert_yaxis()
    plt.xlabel("Leverage score")
    plt.title(f"Pharmacist call list — top {len(df)}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "leverage_bar.png", dpi=140)
    plt.close()

    lines: list[str] = []
    lines.append("# Pharmacist intervention — weekly call list")
    lines.append("")
    lines.append(f"_Top {len(df)} patients by leverage score._")
    lines.append("")
    lines.append("![Leverage scores](figures/leverage_bar.png)")
    lines.append("")
    lines.append("## Call queue")
    lines.append("")
    for _, row in df.iterrows():
        lines.append(f"### `{row['patient_id']}` — score {row['leverage_score']:.2f}")
        lines.append("")
        lines.append(
            f"- Last HbA1c **{row['last_hba1c']:.1f}%** → predicted **{row['predicted_hba1c']:.1f}%** "
            f"(Δ {row['delta']:+.2f})"
        )
        lines.append(f"- Glucose-logging dropoff: {row['adherence_dropoff']:.0%}")
        lines.append(f"- Other failing metrics: {int(row['other_failing'])}")
        lines.append("")
        lines.append("**Phone script:**")
        for b in _script_bullets(row):
            lines.append(f"- {b}")
        lines.append("")
    (HERE / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    df = rank(top=args.top)
    write_report(df)
    log.info("wrote pharmacist call list (%d patients) -> %s", len(df), HERE / "report.md")


if __name__ == "__main__":
    main()
