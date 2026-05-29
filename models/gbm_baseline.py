"""LightGBM tabular baseline for HbA1c trajectory regression.

Same feature frame is reused by ``engagement_dropout.py`` for classification.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import joblib
import lightgbm as lgb
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

from ._features import FeatureFrame


@dataclass
class GBMResult:
    pearson_r: float
    mae: float
    n_train: int
    n_test: int
    feature_importance: dict[str, float]


def train_regressor(
    train: FeatureFrame,
    test: FeatureFrame,
    target: Literal["hba1c", "dropout"] = "hba1c",
    artifact_dir: Path | str = "artifacts/gbm",
) -> GBMResult:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if target == "hba1c":
        y_tr, y_te = train.y_hba1c, test.y_hba1c
        objective = "regression"
        metric = "l1"
    else:
        y_tr, y_te = train.y_dropout, test.y_dropout
        objective = "binary"
        metric = "binary_logloss"

    model = (
        lgb.LGBMRegressor(
            objective=objective,
            metric=metric,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=10,
            random_state=42,
        )
        if target == "hba1c"
        else lgb.LGBMClassifier(
            objective=objective,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
        )
    )

    model.fit(
        train.X,
        y_tr,
        eval_set=[(test.X, y_te)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    if target == "hba1c":
        pred = model.predict(test.X)
        r, _ = pearsonr(y_te, pred)
        mae = mean_absolute_error(y_te, pred)
    else:
        pred = model.predict_proba(test.X)[:, 1]
        # Treat r / mae as proxies; full metrics in engagement_dropout
        r, mae = float("nan"), float("nan")

    importance = dict(
        sorted(
            zip(train.X.columns, model.feature_importances_, strict=True),
            key=lambda kv: -kv[1],
        )[:50]
    )
    importance = {k: int(v) for k, v in importance.items()}

    joblib.dump(model, artifact_dir / f"gbm_{target}.joblib")
    pd.Series(pred, index=test.patient_ids).to_frame("pred").to_parquet(
        artifact_dir / f"gbm_{target}_predictions.parquet"
    )

    result = GBMResult(
        pearson_r=float(r),
        mae=float(mae),
        n_train=len(train.X),
        n_test=len(test.X),
        feature_importance=importance,
    )
    (artifact_dir / f"gbm_{target}_result.json").write_text(json.dumps(asdict(result), indent=2))
    return result
