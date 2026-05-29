"""Clinical-ML metric implementations.

Implementations are intentionally short and dependency-light (numpy +
scikit-learn). Each function carries its field citation in the
docstring so a reviewer can confirm we're using the metric as
intended.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# --- Calibration ----------------------------------------------------------

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier 1950.

    Mean squared error between probabilities and outcomes. Lower is better.
    For a binary outcome with prevalence p, the no-skill Brier is p*(1-p)
    (~0.25 at 50% prevalence, ~0.18 at 25%).
    """
    return float(brier_score_loss(y_true, y_prob))


@dataclass
class ReliabilityCurve:
    bin_centers: np.ndarray   # mean predicted prob per bin
    observed: np.ndarray      # mean actual outcome per bin
    counts: np.ndarray        # patients per bin


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> ReliabilityCurve:
    """Hosmer-Lemeshow style reliability curve.

    Bins by predicted probability decile; plots observed outcome rate per
    bin. Perfect calibration = diagonal line. Deviation above diagonal =
    model under-confident; below = over-confident.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    centers, observed, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        centers.append(float(y_prob[mask].mean()))
        observed.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return ReliabilityCurve(
        bin_centers=np.array(centers),
        observed=np.array(observed),
        counts=np.array(counts),
    )


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Guo et al 2017 "On Calibration of Modern Neural Networks".

    Weighted average of |observed - predicted| over equal-width bins.
    ECE < 0.05 is the commonly cited "well-calibrated" threshold.
    """
    rc = reliability_curve(y_true, y_prob, n_bins=n_bins)
    if len(rc.counts) == 0:
        return float("nan")
    total = rc.counts.sum()
    weights = rc.counts / total
    return float(np.sum(weights * np.abs(rc.observed - rc.bin_centers)))


# --- Operating point at threshold ----------------------------------------

@dataclass
class OperatingPoint:
    threshold: float
    precision: float
    recall: float           # = sensitivity
    specificity: float
    f1: float
    n_called: int


def precision_recall_at_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> OperatingPoint:
    """Confusion-matrix derived metrics at a given operating threshold.

    Healthtech operators usually have a fixed daily capacity (e.g. "30
    calls"). Picking the threshold and reporting precision/recall there
    is more honest than the AUROC point estimate.
    """
    yhat = (y_prob >= threshold).astype(int)
    tp = int(((yhat == 1) & (y_true == 1)).sum())
    fp = int(((yhat == 1) & (y_true == 0)).sum())
    tn = int(((yhat == 0) & (y_true == 0)).sum())
    fn = int(((yhat == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return OperatingPoint(
        threshold=threshold, precision=prec, recall=rec,
        specificity=spec, f1=f1, n_called=int(yhat.sum()),
    )


# --- Ranking / top-K -----------------------------------------------------

def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Järvelin & Kekäläinen 2002 NDCG@K.

    ``scores`` is the model's ranking signal; ``relevance`` is the
    ground-truth gain (e.g. observed HbA1c rise). NDCG@K normalizes
    DCG@K by the ideal-ordering DCG, so 1.0 is perfect and 0.0 is
    information-free.
    """
    order = np.argsort(-scores)[:k]
    gains = relevance[order]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal_order = np.argsort(-relevance)[:k]
    idcg = float(np.sum(relevance[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def lift_at_topk(
    scores: np.ndarray, y_true: np.ndarray, k: int
) -> float:
    """Lift = precision in top K / base rate.

    Lift of 2 means the top K is twice as enriched as random sampling.
    Marketing / DTx convention.
    """
    if len(scores) == 0:
        return float("nan")
    base = float(np.mean(y_true))
    if base <= 0:
        return float("nan")
    order = np.argsort(-scores)[:k]
    top_rate = float(np.mean(y_true[order]))
    return top_rate / base


def capture_at_topk(
    scores: np.ndarray, y_true: np.ndarray, k: int
) -> float:
    """Fraction of all true positives captured in the top-K."""
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum() / y_true.sum())


# --- Decision-curve analysis ---------------------------------------------

@dataclass
class DecisionCurve:
    thresholds: np.ndarray
    net_benefit: np.ndarray         # model
    net_benefit_treat_all: np.ndarray
    net_benefit_treat_none: np.ndarray


def decision_curve(
    y_true: np.ndarray, y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> DecisionCurve:
    """Vickers & Elkin 2006 "Decision-curve analysis".

    Plots net benefit (true positives - weighted false positives) over
    a range of threshold probabilities. The model is clinically useful
    only when its curve sits above both "treat all" and "treat none".
    Standard tool in clinical-decision-support evaluation.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.50, 30)
    n = len(y_true)
    if n == 0:
        empty = np.zeros_like(thresholds)
        return DecisionCurve(thresholds, empty, empty, empty)

    prev = float(y_true.mean())
    nb_model = []
    nb_all = []
    nb_none = np.zeros_like(thresholds)
    for t in thresholds:
        yhat = y_prob >= t
        tp = float(((yhat) & (y_true == 1)).sum())
        fp = float(((yhat) & (y_true == 0)).sum())
        # Net benefit = (TP / n) - (FP / n) * (t / (1-t))
        # See Vickers' formula. Weight (t/(1-t)) is the harm of an
        # unnecessary intervention vs the benefit of a needed one.
        nb_model.append(tp / n - (fp / n) * (t / (1 - t)) if t < 1 else 0)
        nb_all.append(prev - (1 - prev) * (t / (1 - t)) if t < 1 else 0)
    return DecisionCurve(
        thresholds=thresholds,
        net_benefit=np.array(nb_model),
        net_benefit_treat_all=np.array(nb_all),
        net_benefit_treat_none=nb_none,
    )


# --- Quick aggregators ---------------------------------------------------

def auc_pair(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Convenience: (AUROC, AUPRC) on a single pass."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard cardiometabolic regression triple: Pearson r, MAE, R^2."""
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_absolute_error, r2_score
    if len(y_true) == 0:
        return {"pearson_r": float("nan"), "mae": float("nan"), "r2": float("nan")}
    r, _ = pearsonr(y_true, y_pred)
    return {
        "pearson_r": float(r),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
