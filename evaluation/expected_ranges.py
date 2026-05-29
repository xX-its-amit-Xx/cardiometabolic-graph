"""Field-standard expected ranges for every metric we report.

Sources cited in each docstring. When our result lands outside the
"acceptable" band the caller should treat that as a real signal and
either tune the model or document the gap explicitly. Ranges are
intentionally derived from peer-reviewed clinical-ML literature, not
hand-waved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class MetricRange:
    """A field-standard expected range for a metric.

    ``floor``: minimum value that's even arguably useful in the field.
    ``acceptable``: low end of the "deployable in a real care pathway" band.
    ``good``: the typical published number for a well-tuned model on
              real clinical data in this space.
    ``ceiling``: an empirical upper bound; values much above this on
                 demo / synthetic data usually mean target leakage.
    ``higher_is_better``: direction of the metric.
    ``rationale``: one-line source citation + reasoning.
    """

    floor: float
    acceptable: float
    good: float
    ceiling: float
    higher_is_better: bool
    rationale: str


# --- Discrimination ---
RANGES: dict[str, MetricRange] = {
    # AUROC — Steyerberg 2010 "Clinical Prediction Models" + Hosmer-Lemeshow
    "auroc": MetricRange(
        floor=0.50,
        acceptable=0.70,
        good=0.80,
        ceiling=0.97,
        higher_is_better=True,
        rationale="Hosmer-Lemeshow: <0.7 poor, 0.7-0.8 acceptable, 0.8-0.9 excellent; >0.97 on synthetic = target leakage.",
    ),
    "auprc": MetricRange(
        floor=0.05,
        acceptable=0.30,
        good=0.50,
        ceiling=0.95,
        higher_is_better=True,
        rationale="AUPRC must beat base rate; for ~25% positive class, lift = AUPRC / 0.25 should exceed 2x.",
    ),
    # --- Calibration ---
    # Brier score — Brier 1950; lower is better
    "brier": MetricRange(
        floor=0.0,
        acceptable=0.20,
        good=0.15,
        ceiling=0.25,
        higher_is_better=False,
        rationale="Brier 1950; uninformative-prior baseline at 0.25 for 50% prevalence. Want clearly below 0.20.",
    ),
    # Expected Calibration Error — Guo et al 2017 "On Calibration of Modern NNs"
    "ece": MetricRange(
        floor=0.0,
        acceptable=0.10,
        good=0.05,
        ceiling=0.30,
        higher_is_better=False,
        rationale="Guo 2017: ECE <0.05 considered well-calibrated; <0.10 deployable; >0.20 needs Platt/isotonic.",
    ),
    # --- HbA1c regression ---
    # Pearson r — Sun et al 2024 systematic review of diabetes prediction
    "pearson_r_hba1c": MetricRange(
        floor=0.30,
        acceptable=0.60,
        good=0.75,
        ceiling=0.95,
        higher_is_better=True,
        rationale="Sun 2024 review of EHR-based HbA1c forecasting: r typically 0.55-0.80; >0.95 on demo = leakage.",
    ),
    "mae_hba1c": MetricRange(
        floor=0.0,
        acceptable=0.50,
        good=0.35,
        ceiling=1.20,
        higher_is_better=False,
        rationale="Clinical decision support: <0.5% HbA1c MAE actionable (Diabetes Care 2020 CDS guidance).",
    ),
    # --- Ranking / top-K ---
    # NDCG — Järvelin & Kekäläinen 2002; clinical IR work uses NDCG@K
    "ndcg": MetricRange(
        floor=0.40,
        acceptable=0.65,
        good=0.80,
        ceiling=0.99,
        higher_is_better=True,
        rationale="IR convention: >0.8 strong ranking; >0.65 useful triage.",
    ),
    # Lift@K — operational marketing / DTx metric
    "lift": MetricRange(
        floor=1.0,
        acceptable=1.5,
        good=2.0,
        ceiling=10.0,
        higher_is_better=True,
        rationale="Lift > 1x beats random; >2x = call-list cuts work in half vs random outreach.",
    ),
    # Capture@K — fraction of true positives caught in top K
    "capture": MetricRange(
        floor=0.10,
        acceptable=0.30,
        good=0.50,
        ceiling=1.0,
        higher_is_better=True,
        rationale="Capture > 50% in top 10% of cohort is the bar care-coordinator triage tools cite.",
    ),
    # --- Drift detection (PSI) ---
    # Yurdakul 2018 stability-monitoring literature; PSI banding from credit-risk practice
    "psi": MetricRange(
        floor=0.0,
        acceptable=0.10,
        good=0.05,
        ceiling=0.25,
        higher_is_better=False,
        rationale="PSI <0.1 stable, 0.1-0.2 watch, >0.2 alarm — universal in monitoring practice.",
    ),
    # --- Structural completeness for generated documents ---
    "completeness": MetricRange(
        floor=0.7,
        acceptable=0.95,
        good=1.0,
        ceiling=1.0,
        higher_is_better=True,
        rationale="PA notes / pre-visit summaries: every required field populated when source data exists.",
    ),
    # --- Eligibility screening ---
    # Sensitivity (recall) and specificity for clinical trial inclusion
    "sensitivity": MetricRange(
        floor=0.70,
        acceptable=0.90,
        good=0.95,
        ceiling=1.0,
        higher_is_better=True,
        rationale="Trial recruitment: missing eligible patients is more costly than over-screening.",
    ),
    "specificity": MetricRange(
        floor=0.30,
        acceptable=0.50,
        good=0.75,
        ceiling=1.0,
        higher_is_better=True,
        rationale="Cut-down on screening burden — coordinator can't call 500 patients to find 10.",
    ),
}


Status = Literal["below_floor", "below_acceptable", "acceptable", "good", "above_ceiling"]


def grade(metric: str, value: float | None) -> tuple[Status, str]:
    """Grade a metric value against its expected range.

    Returns (status, human-readable hint).
    """
    if value is None:
        return "below_floor", "no value (data missing)"
    r = RANGES.get(metric)
    if r is None:
        return "acceptable", "no reference range defined"

    if r.higher_is_better:
        if value < r.floor:
            return "below_floor", f"<{r.floor} (uninformative)"
        if value < r.acceptable:
            return "below_acceptable", f"<{r.acceptable} (below the deployable band)"
        if value < r.good:
            return "acceptable", f">={r.acceptable} (acceptable)"
        if value <= r.ceiling:
            return "good", f">={r.good} (matches published clinical-ML literature)"
        return "above_ceiling", f">{r.ceiling} (suspiciously high — check for target leakage)"
    # lower is better
    if value > r.ceiling:
        return "below_floor", f">{r.ceiling} (worse than baseline)"
    if value > r.acceptable:
        return "below_acceptable", f">{r.acceptable} (worse than deployable threshold)"
    if value > r.good:
        return "acceptable", f"<={r.acceptable} (acceptable)"
    if value >= r.floor:
        return "good", f"<={r.good} (matches or beats published clinical-ML literature)"
    return "above_ceiling", "below physical floor — check the computation"
