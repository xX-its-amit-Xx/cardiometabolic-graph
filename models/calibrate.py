"""Isotonic calibration for the dropout classifier.

Niculescu-Mizil & Caruana 2005: tree ensembles like LightGBM tend to
push probabilities toward the extremes — their predicted probability
distribution is "spread too thin". Isotonic regression on a held-out
calibration split is the standard fix for this. Platt scaling is the
alternative but assumes a sigmoid shape that tree ensembles often
violate.

This module re-fits the dropout classifier with a clean train /
calibrate / test split:
    60% train (LightGBM)
    20% calibrate (isotonic IR on raw probabilities)
    20% test   (report calibrated AUROC + Brier + ECE)

It overwrites the dropout artifacts so downstream cookbooks see the
calibrated probabilities by default.
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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from etl._common import log
from models._features import load_cached


@dataclass
class CalibrationResult:
    auroc_before: float
    auroc_after: float
    auprc_after: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    n_train: int
    n_cal: int
    n_test: int


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    n = len(y)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        err += (mask.sum() / n) * abs(y[mask].mean() - p[mask].mean())
    return float(err)


def calibrate(
    test_frac: float = 0.2,
    cal_frac: float = 0.2,
    seed: int = 42,
    artifact_dir: Path | str = "artifacts/dropout",
) -> CalibrationResult:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ff = load_cached()
    if ff is None:
        raise FileNotFoundError("No cached features. Run `make pipeline-parquet`.")

    X = ff.X
    y = ff.y_dropout.values
    if len(np.unique(y)) < 2:
        raise RuntimeError("Dropout target has no class diversity.")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(n * test_frac)
    n_cal = int(n * cal_frac)
    test_idx = idx[:n_test]
    cal_idx = idx[n_test : n_test + n_cal]
    tr_idx = idx[n_test + n_cal :]

    X_tr, X_cal, X_te = X.iloc[tr_idx], X.iloc[cal_idx], X.iloc[test_idx]
    y_tr, y_cal, y_te = y[tr_idx], y[cal_idx], y[test_idx]

    spw = (len(y_tr) - y_tr.sum()) / max(1, y_tr.sum())
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=spw,
        random_state=seed,
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_cal, y_cal)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )

    # Raw probabilities on the calibration split, then fit IR.
    p_cal = model.predict_proba(X_cal)[:, 1]
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p_cal, y_cal)

    # Test-time numbers, before and after.
    p_test_raw = model.predict_proba(X_te)[:, 1]
    p_test_cal = ir.transform(p_test_raw)

    auroc_b = float(roc_auc_score(y_te, p_test_raw)) if len(np.unique(y_te)) > 1 else float("nan")
    auroc_a = float(roc_auc_score(y_te, p_test_cal)) if len(np.unique(y_te)) > 1 else float("nan")
    auprc_a = (
        float(average_precision_score(y_te, p_test_cal))
        if len(np.unique(y_te)) > 1
        else float("nan")
    )
    brier_b = float(brier_score_loss(y_te, p_test_raw))
    brier_a = float(brier_score_loss(y_te, p_test_cal))
    ece_b = _ece(y_te, p_test_raw)
    ece_a = _ece(y_te, p_test_cal)

    joblib.dump({"model": model, "isotonic": ir}, artifact_dir / "dropout_calibrated.joblib")
    # Write calibrated predictions to a SEPARATE parquet so downstream
    # cookbooks can opt in. The original `dropout_predictions.parquet`
    # produced by `models.train` is the default — calibration is a
    # documented option, not an override, because isotonic regression
    # on a 100-patient calibration split can underfit on this data
    # scale.
    pd.Series(p_test_cal, index=X_te.index, name="p_dropout").to_frame().to_parquet(
        artifact_dir / "dropout_predictions_calibrated.parquet"
    )

    result = CalibrationResult(
        auroc_before=auroc_b,
        auroc_after=auroc_a,
        auprc_after=auprc_a,
        brier_before=brier_b,
        brier_after=brier_a,
        ece_before=ece_b,
        ece_after=ece_a,
        n_train=len(X_tr),
        n_cal=len(X_cal),
        n_test=len(X_te),
    )
    (artifact_dir / "dropout_result.json").write_text(json.dumps(asdict(result), indent=2))

    log.info(
        "calibration — AUROC %.3f -> %.3f, Brier %.3f -> %.3f, ECE %.3f -> %.3f",
        auroc_b,
        auroc_a,
        brier_b,
        brier_a,
        ece_b,
        ece_a,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--cal-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    calibrate(test_frac=args.test_frac, cal_frac=args.cal_frac, seed=args.seed)


if __name__ == "__main__":
    main()
