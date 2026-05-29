"""Render GNN attention attribution as a horizontal bar chart for a given patient.

Loads the trained HbA1cGNN, re-runs inference for a single patient, maps
GAT attention weights back to feature names, and saves a figure suitable for
embedding in a cookbook evidence trail.

Usage:
    python -m explain.attention_figure --patient SYN000277
    python -m explain.attention_figure --patient SYN000277 --out-dir cookbook/03_pathway_anchored_explanation/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from explain.graph_attention_attribution import attribute
from models._features import load_cached
from models.gnn_hba1c import PYG_AVAILABLE, HbA1cGNN


def _build_named_subgraph(row: pd.Series, y_val: float):
    """Star graph with a feature-name index alongside the Data object."""
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric required")
    from torch_geometric.data import Data

    nonzero = row[row != 0].astype(float)
    feat_names = ["patient_root", *nonzero.index.tolist()]
    n_satellites = len(nonzero)
    feature_dim = 1

    if n_satellites == 0:
        x = torch.zeros((1, feature_dim), dtype=torch.float)
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        patient_feat = torch.tensor([[float(nonzero.mean())]], dtype=torch.float)
        sat_feats = torch.tensor(nonzero.to_numpy(dtype=float).reshape(-1, 1), dtype=torch.float)
        x = torch.cat([patient_feat, sat_feats], dim=0)
        src = torch.arange(1, n_satellites + 1)
        dst = torch.zeros(n_satellites, dtype=torch.long)
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y_val], dtype=torch.float),
    )
    return data, feat_names


def generate_attention_figure(
    patient_id: str,
    model_path: Path = Path("artifacts/gnn/gnn_hba1c.pt"),
    top_k: int = 10,
    out_dir: Path = Path("cookbook/03_pathway_anchored_explanation/figures"),
) -> Path:
    if not PYG_AVAILABLE:
        raise ImportError("torch_geometric required")

    ff = load_cached()
    if ff is None:
        raise RuntimeError("No cached features — run the pipeline first.")

    if patient_id not in ff.X.index:
        raise KeyError(f"{patient_id} not found in feature cache.")

    row = ff.X.loc[patient_id]
    y_val = float(ff.y_hba1c.loc[patient_id])
    data, feat_names = _build_named_subgraph(row, y_val)

    model = HbA1cGNN(in_dim=1, hidden=64)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    attr_df = attribute(model, data, top_k=top_k * 2)

    # Map node indices to feature names (clamp to valid range)
    def _name(idx: int) -> str:
        if idx < len(feat_names):
            return feat_names[idx]
        return f"node_{idx}"

    attr_df["src_name"] = attr_df["src_idx"].apply(_name)
    attr_df["dst_name"] = attr_df["dst_idx"].apply(_name)
    attr_df["edge_label"] = attr_df["src_name"] + " → " + attr_df["dst_name"]

    top = attr_df.head(top_k).copy()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4C72B0" if "patient_root" in e else "#DD8452" for e in top["edge_label"]]
    bars = ax.barh(top["edge_label"][::-1], top["weight"][::-1], color=colors[::-1])
    ax.set_xlabel("Aggregated attention weight (GAT layers 1+2)")
    ax.set_title(f"GNN attention attribution — patient {patient_id}")
    for bar, val in zip(bars, top["weight"][::-1], strict=False):
        ax.text(
            bar.get_width() + 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(0, top["weight"].max() * 1.25)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"attention_{patient_id}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/gnn/gnn_hba1c.pt"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("cookbook/03_pathway_anchored_explanation/figures"),
    )
    args = parser.parse_args()

    out = generate_attention_figure(
        patient_id=args.patient,
        model_path=args.model,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )
    print(f"saved attention figure -> {out}")


if __name__ == "__main__":
    main()
