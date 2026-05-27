"""Unified training entry-point.

Usage examples:
    python -m models.train --target hba1c --model gbm
    python -m models.train --target hba1c --model gnn --epochs 30
    python -m models.train --target engagement_dropout --model gbm
"""

from __future__ import annotations

import argparse

import pandas as pd

from etl._common import log, processed_path, synthetic_path

from ._features import build_features, load_cached
from .engagement_dropout import train_dropout
from .gbm_baseline import train_regressor


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load labs / events / patients / archetypes from cache if present, else
    fall back to synthetic only (lets ``make pipeline`` work without real
    data). Archetypes are the synthetic ground-truth dropout label and are
    only present in synthetic runs."""
    labs_p = processed_path() / "labs.parquet"
    pats_p = processed_path() / "patients.parquet"
    events_p = synthetic_path() / "engagement_events.parquet"
    arche_p = synthetic_path() / "engagement_archetypes.parquet"

    labs = pd.read_parquet(labs_p) if labs_p.exists() else pd.DataFrame(
        columns=["patient_id", "name", "value", "taken_ts"]
    )
    patients = pd.read_parquet(pats_p) if pats_p.exists() else pd.DataFrame(
        columns=["patient_id", "sex", "birth_year"]
    )
    events = pd.read_parquet(events_p) if events_p.exists() else pd.DataFrame(
        columns=["patient_id", "kind", "ts", "value"]
    )
    archetypes = pd.read_parquet(arche_p) if arche_p.exists() else pd.DataFrame(
        columns=["patient_id", "archetype"]
    )

    if patients.empty and not events.empty:
        patients = pd.DataFrame({
            "patient_id": events["patient_id"].unique(),
            "sex": "U",
            "birth_year": 1980,
        })

    return labs, events, patients, archetypes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["hba1c", "engagement_dropout"], required=True)
    parser.add_argument("--model", choices=["gbm", "gnn"], default="gbm")
    parser.add_argument("--epochs", type=int, default=40, help="(GNN only)")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cached = load_cached()
    if cached is None:
        log.info("no cached features; building from inputs")
        labs, events, patients, archetypes = _load_inputs()
        ff = build_features(labs, events, patients, cache=True, archetypes=archetypes)
    else:
        ff = cached
        log.info("using cached feature frame (%d rows)", len(ff.X))

    train, test = ff.split(test_frac=args.test_frac, seed=args.seed)

    if args.target == "hba1c":
        if args.model == "gbm":
            result = train_regressor(train, test, target="hba1c")
            log.info("HbA1c GBM -> Pearson r=%.3f, MAE=%.3f", result.pearson_r, result.mae)
        else:
            from .gnn_hba1c import train_gnn
            result = train_gnn(train.X, train.y_hba1c, test.X, test.y_hba1c, epochs=args.epochs)
            log.info("HbA1c GNN -> Pearson r=%.3f, MAE=%.3f", result.pearson_r, result.mae)
    else:
        result = train_dropout(train, test)
        log.info("Engagement dropout -> AUROC=%.3f AUPRC=%.3f", result.auroc, result.auprc)


if __name__ == "__main__":
    main()
