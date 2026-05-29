"""Cookbook 03 — pathway-anchored per-patient explanation."""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from etl._common import log, processed_path, synthetic_path

HERE = Path(__file__).resolve().parent

# Curated mapping from lab features back to Reactome super-pathways.
# We do not need to query Neo4j for this — the mapping is stable and is
# documented in schema/graph_schema.md.
FEATURE_TO_PATHWAYS: dict[str, list[tuple[str, str]]] = {
    "HbA1c": [("R-HSA-70171", "Glycolysis")],
    "glucose_serum": [("R-HSA-70171", "Glycolysis"), ("R-HSA-74160", "Signaling by insulin")],
    "cholesterol_total": [("R-HSA-556833", "Metabolism of lipids")],
    "ldl": [("R-HSA-556833", "Metabolism of lipids")],
    "hdl": [("R-HSA-556833", "Metabolism of lipids")],
    "triglycerides": [("R-HSA-556833", "Metabolism of lipids")],
}


def _base_feature(name: str) -> str:
    for suffix in ("_last", "_mean", "_min", "_max", "_n"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _pathway_citations(feature: str) -> list[tuple[str, str]]:
    return FEATURE_TO_PATHWAYS.get(_base_feature(feature), [])


def _load_artifacts(patient_id: str) -> dict:
    pred_path = Path("artifacts/gbm/gbm_hba1c_predictions.parquet")
    shap_path = Path("artifacts/gbm/shap_values.parquet")
    drop_path = Path("artifacts/dropout/dropout_predictions.parquet")
    feat_path = processed_path() / "features.parquet"

    out: dict = {}
    if not pred_path.exists():
        raise FileNotFoundError("Run training first; gbm_hba1c_predictions missing.")
    if not feat_path.exists():
        raise FileNotFoundError("Cached features missing; run `make pipeline`.")

    pred = pd.read_parquet(pred_path)
    if patient_id not in pred.index:
        raise KeyError(
            f"patient_id {patient_id} not in predictions. " f"Available example: {pred.index[0]}"
        )

    feats = pd.read_parquet(feat_path)
    out["features"] = feats.loc[patient_id] if patient_id in feats.index else pd.Series(dtype=float)
    out["pred_next"] = float(pred.loc[patient_id, "pred"])
    out["last_hba1c"] = float(out["features"].get("HbA1c_last", 0.0))

    if shap_path.exists():
        shap = pd.read_parquet(shap_path)
        out["shap"] = shap.loc[patient_id] if patient_id in shap.index else None
    else:
        out["shap"] = None

    if drop_path.exists():
        drop = pd.read_parquet(drop_path)
        out["dropout"] = (
            float(drop.loc[patient_id, "p_dropout"]) if patient_id in drop.index else None
        )
    else:
        out["dropout"] = None

    ev_path = synthetic_path() / "engagement_events.parquet"
    if ev_path.exists():
        ev = pd.read_parquet(ev_path)
        out["events"] = ev[ev["patient_id"] == patient_id].copy()
        if not out["events"].empty:
            out["events"]["ts"] = pd.to_datetime(out["events"]["ts"], utc=True)
    else:
        out["events"] = pd.DataFrame()
    return out


def _model_version_hash() -> str:
    """Stable short hash of the trained model file + summary template — gets
    embedded in the report front-matter so reviewers can tell when the
    underlying artifact has changed."""
    h = hashlib.sha256()
    for path in (
        Path("artifacts/gbm/gbm_hba1c.joblib"),
        Path("dashboard/_summary.py"),
    ):
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()[:12]


def build_report(patient_id: str) -> str:
    art = _load_artifacts(patient_id)
    shap = art["shap"]
    top_features = (
        shap.abs().sort_values(ascending=False).head(6).index.tolist() if shap is not None else []
    )

    lines: list[str] = []
    lines.append("---")
    lines.append(f"patient_id: {patient_id}")
    lines.append(f"generated_at: {datetime.now(UTC).isoformat()}")
    lines.append(f"model_version: {_model_version_hash()}")
    lines.append("---")
    lines.append("")
    lines.append(f"# Evidence trail — `{patient_id}`")
    lines.append("")
    lines.append(f"**Last observed HbA1c:** {art['last_hba1c']:.2f}%")
    lines.append(
        f"**Predicted next HbA1c:** {art['pred_next']:.2f}%  "
        f"(Δ {art['pred_next'] - art['last_hba1c']:+.2f})"
    )
    if art["dropout"] is not None:
        lines.append(f"**Engagement dropout risk:** {art['dropout']:.0%}")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")

    if not top_features:
        lines.append("_SHAP values not available — run `python -m explain.shap_analysis`._")
    else:
        for feat in top_features:
            impact = float(shap.loc[feat]) if shap is not None else 0.0
            value = float(art["features"].get(feat, 0.0))
            direction = "raising" if impact > 0 else "lowering"
            lines.append(f"### `{feat}` — SHAP {impact:+.3f} ({direction})")
            lines.append(f"- Observed value: **{value:.3g}**")
            cites = _pathway_citations(feat)
            if cites:
                lines.append("- Linked pathways:")
                for pid, pname in cites:
                    lines.append(
                        f"  - [{pid}](https://reactome.org/PathwayBrowser/#/{pid}) — {pname}"
                    )
            else:
                lines.append("- No direct pathway link in the curated map.")
            lines.append("")

    # GNN attention figure (generated by explain.attention_figure when GNN is trained)
    fig_path = HERE / "figures" / f"attention_{patient_id}.png"
    if fig_path.exists():
        lines.append("## GNN attention attribution (top-10 edges)")
        lines.append("")
        lines.append(f"![GNN attention](figures/attention_{patient_id}.png)")
        lines.append("")
        lines.append(
            "_Aggregated attention weight across both GAT layers. "
            "Edges with higher weight contributed more to the model's prediction "
            "for this patient._"
        )
        lines.append("")

    lines.append("## Behavioral context (last 30 days)")
    ev = art["events"]
    if ev.empty:
        lines.append("_No engagement events for this patient._")
    else:
        cutoff = ev["ts"].max() - pd.Timedelta(days=30)
        recent = ev[ev["ts"] >= cutoff]
        for kind, count in recent.groupby("kind").size().items():
            lines.append(f"- {kind}: {count}")
    lines.append("")

    lines.append("## Reviewer checklist")
    lines.append("")
    for item in [
        "Evidence above is consistent with the patient's known history.",
        "Predicted trajectory is clinically plausible given evidence.",
        "Linked pathways match the clinical reasoning.",
        "No protected-class features appear to be driving the prediction.",
        "Behavioral context does not contradict the recommendation.",
    ]:
        lines.append(f"- [ ] {item}")
    lines.append("")
    lines.append("Reviewer signature: ____________________  Date: __________")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient", required=True, help="Patient ID, e.g. SYN000017 or MIMIC10000032"
    )
    args = parser.parse_args()

    report = build_report(args.patient)
    out = HERE / f"evidence_{args.patient}.md"
    out.write_text(report, encoding="utf-8")
    log.info("wrote evidence trail -> %s", out)


if __name__ == "__main__":
    main()
