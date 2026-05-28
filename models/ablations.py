"""GNN ablation study: quantify contribution of lab vs engagement features.

Three conditions:
    (a) no_engagement  — lab features only
    (b) no_labs        — engagement features only
    (c) full           — all features (matches main GNN training)

Results written to artifacts/gnn/ablation_results.json and
docs/figures/gnn_ablations.png.

Usage:
    python -m models.ablations
    python -m models.ablations --epochs 30 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ._features import LAB_FEATURES, load_cached
from .gnn_hba1c import train_gnn


# Prefixes that identify lab-derived columns
_LAB_PREFIXES = tuple(LAB_FEATURES)
_ENGAGEMENT_PREFIXES = ("sum_", "count_", "ev_bucket_")


def _is_lab_col(col: str) -> bool:
    return any(col.startswith(p) or col.split("_")[0] in LAB_FEATURES for p in _LAB_PREFIXES)


def _is_engagement_col(col: str) -> bool:
    return any(col.startswith(p) for p in _ENGAGEMENT_PREFIXES)


def _drop_cols(X: pd.DataFrame, drop_lab: bool, drop_engagement: bool) -> pd.DataFrame:
    cols = X.columns.tolist()
    keep = []
    for c in cols:
        if drop_lab and _is_lab_col(c):
            continue
        if drop_engagement and _is_engagement_col(c):
            continue
        keep.append(c)
    result = X[keep].copy()
    # GNN needs ≥1 column; guard against degenerate zero-col frames
    if result.empty or result.shape[1] == 0:
        result = X.iloc[:, :1].copy()
    return result


def run_ablations(epochs: int = 30, seed: int = 42) -> dict[str, dict]:
    ff = load_cached()
    if ff is None:
        raise RuntimeError("No cached features found — run the pipeline first.")

    train, test = ff.split(test_frac=0.2, seed=seed)

    conditions = {
        "no_engagement": dict(drop_lab=False, drop_engagement=True),
        "no_labs": dict(drop_lab=True, drop_engagement=False),
        "full": dict(drop_lab=False, drop_engagement=False),
    }

    results: dict[str, dict] = {}
    for name, kwargs in conditions.items():
        print(f"\n--- ablation: {name} ---")
        tr_X = _drop_cols(train.X, **kwargs)
        te_X = _drop_cols(test.X, **kwargs)
        n_feat = tr_X.shape[1]
        print(f"    features used: {n_feat}")
        result = train_gnn(
            tr_X, train.y_hba1c,
            te_X, test.y_hba1c,
            epochs=epochs,
            artifact_dir=f"artifacts/gnn/ablation_{name}",
        )
        results[name] = {"pearson_r": result.pearson_r, "mae": result.mae, "n_features": n_feat}
        print(f"    Pearson r={result.pearson_r:.3f}  MAE={result.mae:.3f}")

    return results


def plot_ablations(results: dict[str, dict], out_path: Path) -> None:
    labels = {"no_engagement": "No engagement\n(labs only)",
               "no_labs": "No labs\n(engagement only)",
               "full": "Full\n(all features)"}
    names = list(results.keys())
    r_vals = [results[n]["pearson_r"] for n in names]
    mae_vals = [results[n]["mae"] for n in names]
    x_labels = [labels[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    axes[0].barh(x_labels, r_vals, color=colors)
    axes[0].set_xlabel("Pearson r (↑ better)")
    axes[0].set_title("HbA1c GNN — Pearson r by feature set")
    axes[0].set_xlim(0, 1.05)
    for i, v in enumerate(r_vals):
        axes[0].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    axes[1].barh(x_labels, mae_vals, color=colors)
    axes[1].set_xlabel("MAE HbA1c % (↓ better)")
    axes[1].set_title("HbA1c GNN — MAE by feature set")
    for i, v in enumerate(mae_vals):
        axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved ablation figure -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-fig", type=Path, default=Path("docs/figures/gnn_ablations.png"))
    parser.add_argument("--out-json", type=Path, default=Path("artifacts/gnn/ablation_results.json"))
    args = parser.parse_args()

    results = run_ablations(epochs=args.epochs, seed=args.seed)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(results, indent=2))
    print(f"saved ablation results -> {args.out_json}")

    plot_ablations(results, args.out_fig)


if __name__ == "__main__":
    main()
