"""Cookbook 02 — re-engagement outreach targeting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from etl._common import log, synthetic_path


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    drop_path = Path("artifacts/dropout/dropout_predictions.parquet")
    ev_path = synthetic_path() / "engagement_events.parquet"
    if not drop_path.exists():
        raise FileNotFoundError(
            f"Missing {drop_path}. Run `python -m models.train --target engagement_dropout --model gbm`."
        )
    if not ev_path.exists():
        raise FileNotFoundError(f"Missing {ev_path}. Run `make synth`.")
    return pd.read_parquet(drop_path), pd.read_parquet(ev_path)


def build_targets(top: int = 200, dropout_thresh: float = 0.40) -> pd.DataFrame:
    drop, events = _load_inputs()
    events["ts"] = pd.to_datetime(events["ts"], utc=True)
    last_day = events["ts"].max()

    recent = events[events["ts"] >= last_day - pd.Timedelta(days=30)]
    prior = events[
        (events["ts"] < last_day - pd.Timedelta(days=30))
        & (events["ts"] >= last_day - pd.Timedelta(days=60))
    ]

    agg_recent = recent.groupby(["patient_id", "kind"]).size().unstack(fill_value=0).add_prefix("recent_")
    agg_prior = prior.groupby(["patient_id", "kind"]).size().unstack(fill_value=0).add_prefix("prior_")

    last_open = events[events["kind"] == "app_open"].groupby("patient_id")["ts"].max()
    days_since_open = (last_day - last_open).dt.days.rename("days_since_last_open")

    df = drop.rename(columns={"p_dropout": "p_dropout"}).copy()
    df = df.join(agg_recent, how="left").join(agg_prior, how="left").join(days_since_open, how="left")
    df = df.fillna({"recent_app_open": 0, "recent_message_response": 0, "recent_glucose_log": 0,
                    "prior_app_open": 0, "days_since_last_open": 999})

    # Lapsing-but-recoverable filter
    mask = (
        (df["p_dropout"] > dropout_thresh)
        & (df.get("recent_app_open", 0) > 0)
        & (df.get("prior_app_open", 0) > 10)
    )
    pool = df[mask].copy()
    if pool.empty:
        log.warning("no patients matched the lapsing-but-recoverable filter; loosening")
        pool = df[df["p_dropout"] > dropout_thresh].copy()

    pool["recoverability"] = (
        pool.get("recent_message_response", 0)
        + 0.5 * pool.get("recent_glucose_log", 0)
        - 0.1 * pool["days_since_last_open"]
    )
    pool = pool.sort_values("recoverability", ascending=False).head(top).reset_index()
    return pool[[
        "patient_id", "p_dropout", "recoverability", "days_since_last_open",
        "recent_app_open", "recent_message_response", "recent_glucose_log", "prior_app_open",
    ]]


def write_report(df: pd.DataFrame, out_dir: Path = HERE) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "outreach_targets.csv", index=False)

    plt.figure(figsize=(6, 3))
    plt.scatter(df["p_dropout"], df["recoverability"], alpha=0.6, c="#3366aa")
    plt.xlabel("p_dropout")
    plt.ylabel("recoverability score")
    plt.title("Outreach targets — dropout risk vs recoverability")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "outreach_scatter.png", dpi=140)
    plt.close()

    md = [
        "# Re-engagement outreach targets",
        "",
        f"_Top {len(df)} lapsing-but-recoverable patients._",
        "",
        "Sort by `recoverability` desc. Suggested workflow: A/B test the top "
        "quartile against an unsent control to measure model uplift.",
        "",
        "![scatter](figures/outreach_scatter.png)",
        "",
        df.head(20).to_markdown(index=False),
    ]
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--dropout-thresh", type=float, default=0.40)
    args = parser.parse_args()
    df = build_targets(top=args.top, dropout_thresh=args.dropout_thresh)
    write_report(df)
    log.info("wrote %d-patient outreach list to %s", len(df), HERE / "report.md")


if __name__ == "__main__":
    main()
