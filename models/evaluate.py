"""Aggregate model results into a markdown report.

Reads the per-model result JSON files written under ``artifacts/`` and
emits a single ``docs/figures/model_report.md`` summarizing the head-to-head.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARTIFACTS = Path("artifacts")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_report() -> str:
    gbm_hba1c = _load(ARTIFACTS / "gbm" / "gbm_hba1c_result.json")
    gnn_hba1c = _load(ARTIFACTS / "gnn" / "gnn_hba1c_result.json")
    dropout = _load(ARTIFACTS / "dropout" / "dropout_result.json")

    lines = ["# Model report\n"]
    lines.append("## HbA1c trajectory (regression)\n")
    lines.append("| Model | Pearson r | MAE (HbA1c %) | n_train | n_test |")
    lines.append("|-------|-----------|---------------|---------|--------|")
    if gbm_hba1c:
        lines.append(
            f"| LightGBM | {gbm_hba1c['pearson_r']:.3f} | {gbm_hba1c['mae']:.3f} | "
            f"{gbm_hba1c['n_train']} | {gbm_hba1c['n_test']} |"
        )
    if gnn_hba1c:
        lines.append(
            f"| GAT-GNN  | {gnn_hba1c['pearson_r']:.3f} | {gnn_hba1c['mae']:.3f} | "
            f"{gnn_hba1c['n_train']} | {gnn_hba1c['n_test']} |"
        )

    lines.append("\n## Engagement dropout (binary classification)\n")
    lines.append("| Model | AUROC | AUPRC | n_train | n_test | pos rate (test) |")
    lines.append("|-------|-------|-------|---------|--------|-----------------|")
    if dropout:
        pos_rate = dropout["positives_test"] / max(1, dropout["n_test"])
        lines.append(
            f"| LightGBM | {dropout['auroc']:.3f} | {dropout['auprc']:.3f} | "
            f"{dropout['n_train']} | {dropout['n_test']} | {pos_rate:.2%} |"
        )

    if gbm_hba1c and "feature_importance" in gbm_hba1c:
        lines.append("\n## Top features (GBM HbA1c)\n")
        for k, v in list(gbm_hba1c["feature_importance"].items())[:15]:
            lines.append(f"- `{k}` — {v}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("docs/figures/model_report.md"))
    args = parser.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(build_report(), encoding="utf-8")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
