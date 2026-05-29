"""Attribute GNN predictions back to neighborhood structure via GAT attention.

The HbA1c GNN (``models/gnn_hba1c.HbA1cGNN``) caches its raw attention
weights from the most recent forward pass on the ``_last_attention``
attribute. This module re-runs the model on a target patient's subgraph and
returns a tidy DataFrame of (source_node, target_node, attention_weight) so
the dashboard can render the top contributing edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


@dataclass
class AttributionRow:
    src_idx: int
    dst_idx: int
    weight: float
    layer: int


def attribute(
    model: torch.nn.Module,
    data,  # PyG Data
    top_k: int = 20,
) -> pd.DataFrame:
    """Run a single forward pass on ``data`` and return top-k attention edges
    aggregated across both GAT layers."""
    model.eval()
    with torch.no_grad():
        batch = torch.zeros(data.x.size(0), dtype=torch.long)
        _ = model(data.x, data.edge_index, batch)

    if not hasattr(model, "_last_attention"):
        raise RuntimeError("Model has no attention to attribute.")

    ei1, a1, ei2, a2 = model._last_attention

    rows: list[AttributionRow] = []
    for ei, alpha, layer in [(ei1, a1, 1), (ei2, a2, 2)]:
        a = alpha.mean(dim=-1) if alpha.dim() > 1 else alpha
        for i in range(ei.size(1)):
            rows.append(
                AttributionRow(
                    src_idx=int(ei[0, i]),
                    dst_idx=int(ei[1, i]),
                    weight=float(a[i]),
                    layer=layer,
                )
            )

    df = pd.DataFrame([r.__dict__ for r in rows])
    df = df.groupby(["src_idx", "dst_idx"])["weight"].sum().reset_index()
    df = df.sort_values("weight", ascending=False).head(top_k).reset_index(drop=True)
    return df


def save_attribution(
    df: pd.DataFrame, patient_id: str, out_dir: Path = Path("artifacts/gnn")
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"attribution_{patient_id}.parquet"
    df.to_parquet(path, index=False)
    return path
