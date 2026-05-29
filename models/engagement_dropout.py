"""Engagement-dropout classifier.

Predicts whether a patient will stop engaging with the digital therapeutic
in the next 30 days (label produced by ``models/_features.py``). Uses the
same shared feature frame as the GBM baseline.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ._features import FeatureFrame


@dataclass
class DropoutResult:
    auroc: float
    auprc: float
    n_train: int
    n_test: int
    positives_train: int
    positives_test: int


def train_dropout(
    train: FeatureFrame,
    test: FeatureFrame,
    artifact_dir: Path | str = "artifacts/dropout",
) -> DropoutResult:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pos = int(train.y_dropout.sum())
    if pos == 0 or pos == len(train.y_dropout):
        raise ValueError("Training set has no class diversity for dropout target")

    spw = (len(train.y_dropout) - pos) / max(1, pos)
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=spw,
        random_state=42,
    )
    model.fit(
        train.X,
        train.y_dropout,
        eval_set=[(test.X, test.y_dropout)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    proba = model.predict_proba(test.X)[:, 1]
    auroc = roc_auc_score(test.y_dropout, proba) if test.y_dropout.nunique() > 1 else float("nan")
    auprc = (
        average_precision_score(test.y_dropout, proba)
        if test.y_dropout.nunique() > 1
        else float("nan")
    )

    joblib.dump(model, artifact_dir / "dropout_clf.joblib")
    pd.Series(proba, index=test.patient_ids).to_frame("p_dropout").to_parquet(
        artifact_dir / "dropout_predictions.parquet"
    )

    result = DropoutResult(
        auroc=float(auroc),
        auprc=float(auprc),
        n_train=len(train.X),
        n_test=len(test.X),
        positives_train=int(train.y_dropout.sum()),
        positives_test=int(test.y_dropout.sum()),
    )
    (artifact_dir / "dropout_result.json").write_text(json.dumps(asdict(result), indent=2))
    return result
