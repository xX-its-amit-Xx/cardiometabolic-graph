"""Cookbook 04 — cohort drift monitor."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etl._common import log, synthetic_path
from models._features import _build_engagement_features, load_cached

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. PSI < 0.1 stable, 0.1-0.2 watch, >0.2 alarm."""
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return float("nan")

    # use reference quantiles as bin edges
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_pct = np.where(ref_counts == 0, 1e-6, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-6, cur_counts / cur_counts.sum())
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _recent_engagement_features(window_days: int) -> pd.DataFrame:
    ev_path = synthetic_path() / "engagement_events.parquet"
    if not ev_path.exists():
        raise FileNotFoundError(f"missing {ev_path}; run `make synth`.")
    ev = pd.read_parquet(ev_path)
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    cutoff = ev["ts"].max() - pd.Timedelta(days=window_days)
    return _build_engagement_features(ev[ev["ts"] >= cutoff], horizon_days=window_days)


def compute_drift(window_days: int = 30, psi_alarm: float = 0.2) -> pd.DataFrame:
    ff = load_cached()
    if ff is None:
        raise FileNotFoundError("no cached features; run `make pipeline`.")

    reference = ff.X
    current = _recent_engagement_features(window_days)
    # Restrict drift comparison to columns that exist on both sides.
    cols = [c for c in current.columns if c in reference.columns]
    rows = []
    for col in cols:
        ref_vals = reference[col].to_numpy(dtype=float)
        cur_vals = current[col].to_numpy(dtype=float)
        psi = _psi(ref_vals, cur_vals)
        rows.append(
            {
                "feature": col,
                "psi": psi,
                "alarm": psi > psi_alarm,
                "ref_mean": float(np.nanmean(ref_vals)),
                "cur_mean": float(np.nanmean(cur_vals)),
                "n_ref": len(ref_vals),
                "n_cur": len(cur_vals),
            }
        )
    df = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    return df


def write_report(df: pd.DataFrame, out_dir: Path = HERE) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "drift_table.csv", index=False)

    top = df.head(15)
    plt.figure(figsize=(7, 4))
    plt.barh(
        top["feature"], top["psi"], color=["#cc4444" if a else "#88aaff" for a in top["alarm"]]
    )
    plt.axvline(0.1, color="#888", linestyle="--", linewidth=0.8, label="watch")
    plt.axvline(0.2, color="#c33", linestyle="--", linewidth=0.8, label="alarm")
    plt.xlabel("PSI")
    plt.title("Top 15 features by drift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "psi_top.png", dpi=140)
    plt.close()

    alarms = df[df["alarm"]]
    md = ["# Cohort drift monitor", ""]
    md.append("_Comparing the most recent window to the cached training reference._")
    md.append("")
    md.append(f"**{len(alarms)} alarming features** (PSI > 0.2).")
    md.append("")
    md.append("![top PSI](figures/psi_top.png)")
    md.append("")
    if alarms.empty:
        md.append("No alarms this run.")
    else:
        md.append("| Feature | PSI | Ref mean | Cur mean | Interpretation |")
        md.append("|---------|-----|----------|----------|----------------|")
        for _, row in alarms.iterrows():
            delta = row["cur_mean"] - row["ref_mean"]
            md.append(
                f"| `{row['feature']}` | {row['psi']:.2f} | {row['ref_mean']:.2f} | "
                f"{row['cur_mean']:.2f} | "
                f"{'up' if delta>0 else 'down'} {abs(delta):.2f} vs reference |"
            )
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--psi-alarm", type=float, default=0.2)
    args = parser.parse_args()

    df = compute_drift(window_days=args.window_days, psi_alarm=args.psi_alarm)
    write_report(df)
    log.info(
        "wrote drift report (%d features, %d alarms) -> %s",
        len(df),
        int(df["alarm"].sum()),
        HERE / "report.md",
    )


if __name__ == "__main__":
    main()
