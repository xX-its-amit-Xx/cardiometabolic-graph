"""Cookbook 01 — generate the weekly at-risk cohort.

See ``cookbook/01_at_risk_cohort/README.md`` for the framing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from etl._common import log
from models._features import load_cached


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"


def _load_predictions_and_shap() -> tuple[pd.Series, pd.DataFrame]:
    pred_path = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    shap_path = Path("artifacts/gbm/shap_values.parquet")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Missing {pred_path}. Run `make train` first (or "
            "`python -m models.train --target hba1c --model gbm`)."
        )
    pred = pd.read_parquet(pred_path)["pred"]
    shap = pd.read_parquet(shap_path) if shap_path.exists() else pd.DataFrame()
    return pred, shap


def _format_reason(pid: str, shap_row: pd.Series | None, last_hba1c: float) -> str:
    if shap_row is None or shap_row.empty:
        return f"HbA1c last observed at {last_hba1c:.1f}%."
    top = shap_row.abs().sort_values(ascending=False).head(3)
    parts = []
    for feat in top.index:
        impact = shap_row.loc[feat]
        direction = "raising" if impact > 0 else "lowering"
        parts.append(f"{feat} ({direction})")
    return (
        f"HbA1c last observed at {last_hba1c:.1f}%. "
        f"Top factors: {', '.join(parts)}."
    )


def build_cohort(top: int = 50, threshold: float = 6.0) -> pd.DataFrame:
    ff = load_cached()
    if ff is None:
        raise FileNotFoundError("No cached features. Run `make pipeline` first.")
    pred, shap = _load_predictions_and_shap()

    last_col = "HbA1c_last"
    if last_col not in ff.X.columns:
        raise KeyError(f"Feature frame missing '{last_col}' — re-run feature build.")

    df = pd.DataFrame({
        "patient_id": pred.index,
        "predicted_next_hba1c": pred.values,
    })
    df = df.merge(ff.X[[last_col]], left_on="patient_id", right_index=True, how="left")
    df["last_hba1c"] = df[last_col]
    df["delta"] = df["predicted_next_hba1c"] - df["last_hba1c"]
    df = df[df["last_hba1c"] >= threshold]
    df = df.sort_values("delta", ascending=False).head(top).reset_index(drop=True)

    reasons = []
    for _, row in df.iterrows():
        shap_row = shap.loc[row["patient_id"]] if row["patient_id"] in shap.index else None
        reasons.append(_format_reason(row["patient_id"], shap_row, row["last_hba1c"]))
    df["reason"] = reasons
    return df[["patient_id", "last_hba1c", "predicted_next_hba1c", "delta", "reason"]]


def write_report(df: pd.DataFrame, out_dir: Path = HERE) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "cohort.csv", index=False)

    plt.figure(figsize=(6, 3))
    plt.hist(df["delta"], bins=20, color="#cc4444", alpha=0.85)
    plt.xlabel("Predicted HbA1c rise (%-points)")
    plt.ylabel("# patients")
    plt.title(f"At-risk cohort — top {len(df)}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cohort_distribution.png", dpi=140)
    plt.close()

    md = ["# Weekly at-risk cohort", ""]
    md.append(f"_Top {len(df)} patients by predicted HbA1c rise over the next 90 days._")
    md.append("")
    md.append("| Rank | Patient | Last HbA1c | Predicted | Δ | Why |")
    md.append("|------|---------|------------|-----------|---|-----|")
    for i, row in df.iterrows():
        md.append(
            f"| {i+1} | `{row['patient_id']}` | {row['last_hba1c']:.1f}% | "
            f"{row['predicted_next_hba1c']:.1f}% | {row['delta']:+.2f} | {row['reason']} |"
        )
    md.append("")
    md.append("![cohort distribution](figures/cohort_distribution.png)")
    (out_dir / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=6.0,
                        help="Minimum last-observed HbA1c to consider for outreach")
    args = parser.parse_args()
    df = build_cohort(top=args.top, threshold=args.threshold)
    write_report(df)
    log.info("wrote %d-patient cohort to %s", len(df), HERE / "report.md")


if __name__ == "__main__":
    main()
