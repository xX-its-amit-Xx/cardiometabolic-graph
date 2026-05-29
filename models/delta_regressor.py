"""Delta regressor — predicts HbA1c CHANGE, not level.

The existing ``gbm_baseline`` regressor learns to predict the absolute
next HbA1c value, which it does well (Pearson 0.87 on the held-out
visit). But many of our cookbooks — at-risk cohort triage, pharmacist
intervention — actually need a CHANGE prediction: "who is most likely
to *rise* the most over the next 90 days?"

A level model that scores stably-high patients near the top of its
list is not the same thing as a change model. The evaluation framework
makes this explicit: cookbook 08 NDCG@30 sits near zero when graded
against true delta, because the level model has no way to separate
"patient is high and stable" from "patient is rising".

This module trains a LightGBM regressor whose target is
    y = next_hba1c - prior_last_hba1c
and writes ``artifacts/gbm/delta_predictions.parquet`` so cookbooks 01
and 08 can use it directly.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from etl._common import log, processed_path
from models._features import load_cached


@dataclass
class DeltaResult:
    pearson_r: float
    mae: float
    n_train: int
    n_test: int


def _build_delta_targets() -> tuple[pd.Series, pd.Series]:
    """Per-patient (delta_y, last_hba1c) tuple aligned to feature index."""
    labs_path = processed_path() / "labs.parquet"
    y_path = processed_path() / "y_hba1c.parquet"
    if not labs_path.exists() or not y_path.exists():
        raise FileNotFoundError("Run `make pipeline-parquet` first.")

    labs = pd.read_parquet(labs_path)
    labs["taken_ts"] = pd.to_datetime(labs["taken_ts"], utc=True)
    hba1c = labs[labs["name"] == "HbA1c"].sort_values("taken_ts")
    if hba1c.empty:
        raise RuntimeError("No HbA1c observations available.")
    # The feature builder holds out the LATEST visit as the target. We
    # want the SECOND-to-last (or first) as the "prior baseline" so the
    # delta is computed against pre-prediction history only.
    prior_last = (
        hba1c.groupby("patient_id")["value"].apply(
            lambda s: float(s.iloc[-2]) if len(s) >= 2 else float(s.iloc[-1])
        )
    )
    y_full = pd.read_parquet(y_path)["y"]
    common = y_full.index.intersection(prior_last.index)
    delta = (y_full.loc[common] - prior_last.loc[common]).astype(float)
    return delta, prior_last.loc[common]


def train_delta(
    test_frac: float = 0.2,
    seed: int = 42,
    artifact_dir: Path | str = "artifacts/gbm",
) -> DeltaResult:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ff = load_cached()
    if ff is None:
        raise FileNotFoundError("No cached features. Run `make pipeline-parquet` first.")

    delta, prior_last = _build_delta_targets()
    common = ff.X.index.intersection(delta.index)
    X = ff.X.loc[common].copy()
    y = delta.loc[common].values
    if len(X) < 20:
        raise RuntimeError(f"Too few patients with delta targets: {len(X)}")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    cut = int(len(idx) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y[tr], y[te]

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=8,
        random_state=seed,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    pred = model.predict(X_te)
    r, _ = pearsonr(y_te, pred)
    mae = mean_absolute_error(y_te, pred)

    joblib.dump(model, artifact_dir / "delta_hba1c.joblib")
    pd.Series(pred, index=X_te.index, name="delta_pred").to_frame().to_parquet(
        artifact_dir / "delta_predictions.parquet"
    )

    res = DeltaResult(
        pearson_r=float(r), mae=float(mae),
        n_train=len(X_tr), n_test=len(X_te),
    )
    (artifact_dir / "delta_hba1c_result.json").write_text(
        json.dumps(asdict(res), indent=2)
    )
    log.info("delta GBM -> Pearson r=%.3f, MAE=%.3f", res.pearson_r, res.mae)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_delta(test_frac=args.test_frac, seed=args.seed)


if __name__ == "__main__":
    main()
