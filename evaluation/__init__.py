"""Clinical-ML evaluation framework.

Every metric here has:
  1. A docstring naming the field-standard reference (Hosmer-Lemeshow,
     Vickers' decision-curve analysis, Steyerberg's TRIPOD, etc.) so a
     reviewer can verify we're using the metric correctly.
  2. An ``expected_range`` defined in ``expected_ranges.py`` derived
     from published clinical-ML papers in cardiometabolic, DTx, or
     general predictive-medicine contexts.

The per-use-case evaluators in ``use_case_metrics.py`` aggregate these
into one combined report per cookbook example so a healthtech buyer can
look at the dashboard and immediately answer "is this trustworthy?"
"""

from __future__ import annotations

from .expected_ranges import RANGES, MetricRange
from .metrics import (
    brier_score,
    capture_at_topk,
    decision_curve,
    expected_calibration_error,
    lift_at_topk,
    ndcg_at_k,
    precision_recall_at_threshold,
    reliability_curve,
)

__all__ = [
    "RANGES",
    "MetricRange",
    "brier_score",
    "capture_at_topk",
    "decision_curve",
    "expected_calibration_error",
    "lift_at_topk",
    "ndcg_at_k",
    "precision_recall_at_threshold",
    "reliability_curve",
]
