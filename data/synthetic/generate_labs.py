"""Generate synthetic cardiometabolic labs that correlate with engagement archetype.

Real-world rationale: app engagement, behavioral self-management, and
cardiometabolic markers are mutually predictive — engaged patients tend
to have lower / more controlled HbA1c, less-engaged patients drift
upward. Without real MIMIC data, we synthesize labs that *encode that
relationship* so the rest of the pipeline can learn meaningful patterns.

Generative model:
    HbA1c per patient ~ Normal(base[archetype], sd[archetype])
    glucose ~ 28.7 * HbA1c - 46.7 + Normal(0, 12)         # Nathan 2008
    LDL ~ 90 + 6 * (HbA1c - 6) + Normal(0, 25)
    HDL ~ 55 - 3 * (HbA1c - 6) + Normal(0, 8)
    cholesterol_total ~ LDL + HDL + 0.2 * triglycerides + Normal(0, 15)
    triglycerides ~ 110 + 18 * (HbA1c - 6) + Normal(0, 35)

Three longitudinal timepoints per patient at ~0, ~90, ~180 days, with
archetype-dependent drift (early_dropout drifts up, power_user drifts down).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARCHETYPE_BASE: dict[str, tuple[float, float, float]] = {
    # archetype -> (mean HbA1c, sd HbA1c, drift per 90 days)
    "power_user":    (6.0, 0.4, -0.10),
    "steady":        (6.8, 0.7, +0.00),
    "episodic":      (7.4, 1.1, +0.15),
    "early_dropout": (8.0, 1.3, +0.30),
}


def _derive_labs(hba1c: float, rng: np.random.Generator) -> dict[str, float]:
    glucose = 28.7 * hba1c - 46.7 + rng.normal(0, 12)
    ldl = 90 + 6 * (hba1c - 6) + rng.normal(0, 25)
    hdl = 55 - 3 * (hba1c - 6) + rng.normal(0, 8)
    triglycerides = 110 + 18 * (hba1c - 6) + rng.normal(0, 35)
    cholesterol = ldl + hdl + 0.2 * triglycerides + rng.normal(0, 15)
    return {
        "HbA1c": float(np.clip(hba1c, 4.0, 14.0)),
        "glucose_serum": float(np.clip(glucose, 50, 400)),
        "ldl": float(np.clip(ldl, 30, 250)),
        "hdl": float(np.clip(hdl, 15, 110)),
        "triglycerides": float(np.clip(triglycerides, 30, 700)),
        "cholesterol_total": float(np.clip(cholesterol, 80, 400)),
    }


def generate_labs(
    archetypes: pd.DataFrame,
    start_date: datetime | None = None,
    days_total: int = 180,
    seed: int = 13,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)

    rows: list[dict] = []
    for _, row in archetypes.iterrows():
        pid, atype = row["patient_id"], row["archetype"]
        base, sd, drift_per_90 = ARCHETYPE_BASE[atype]
        baseline_hba1c = rng.normal(base, sd)
        # Per-patient random-walk volatility on top of the archetype drift —
        # without this, the test-set HbA1c is a near-perfect linear function
        # of prior labs, and the GBM Pearson saturates above 0.99. Real
        # cardiometabolic trajectories don't behave that cleanly.
        rw_sd = 0.55 if atype in ("episodic", "early_dropout") else 0.35
        for visit_idx, offset_days in enumerate((10, 90, days_total - 10)):
            visit_drift = (offset_days / 90.0) * drift_per_90 + rng.normal(0, rw_sd)
            visit_hba1c = baseline_hba1c + visit_drift
            labs_visit = _derive_labs(visit_hba1c, rng)
            ts = (start_date + timedelta(days=int(offset_days))).isoformat()
            for name, value in labs_visit.items():
                rows.append({
                    "patient_id": pid,
                    "name": name,
                    "value": value,
                    "unit": _UNITS[name],
                    "taken_ts": ts,
                    "visit_idx": visit_idx,
                    "source_system": "synthetic-v1",
                })
    return pd.DataFrame(rows)


_UNITS = {
    "HbA1c": "%",
    "glucose_serum": "mg/dL",
    "ldl": "mg/dL",
    "hdl": "mg/dL",
    "triglycerides": "mg/dL",
    "cholesterol_total": "mg/dL",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetypes", type=Path, default=Path("data/synthetic/engagement_archetypes.parquet"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/labs.parquet"))
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    if not args.archetypes.exists():
        raise FileNotFoundError(
            f"Run `python -m data.synthetic.generate_engagement_logs` first to produce {args.archetypes}."
        )
    archetypes = pd.read_parquet(args.archetypes)
    labs = generate_labs(archetypes, days_total=args.days, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labs.to_parquet(args.out, index=False)
    print(f"wrote {len(labs):,} lab rows for {labs['patient_id'].nunique()} patients -> {args.out}")


if __name__ == "__main__":
    main()
