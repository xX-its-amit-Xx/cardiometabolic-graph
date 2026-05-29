"""SHAP analysis for the LightGBM HbA1c regressor.

Produces:
  * docs/figures/shap_summary.png   — beeswarm of global feature impact
  * docs/figures/shap_bar.png       — mean(|SHAP|) bar chart

Per-patient SHAP values are persisted in
``artifacts/gbm/shap_values.parquet`` so the dashboard can render them
without re-running the SHAP computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from models._features import load_cached


def run_shap(
    model_path: Path = Path("artifacts/gbm/gbm_hba1c.joblib"),
    out_dir: Path = Path("docs/figures"),
    max_samples: int = 500,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ff = load_cached()
    if ff is None:
        raise FileNotFoundError("No cached features; run training first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Trained GBM not found: {model_path}")

    model = joblib.load(model_path)
    X = ff.X
    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    sv_arr = sv if isinstance(sv, np.ndarray) else np.asarray(sv)

    pd.DataFrame(sv_arr, index=X.index, columns=X.columns).to_parquet(
        Path("artifacts/gbm/shap_values.parquet")
    )

    plt.figure()
    shap.summary_plot(sv_arr, X, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv_arr, X, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=140, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/gbm/gbm_hba1c.joblib"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()
    run_shap(args.model, args.out_dir, args.max_samples)


if __name__ == "__main__":
    main()
