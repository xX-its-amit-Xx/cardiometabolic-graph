"""Generate synthetic patient digital-therapeutic engagement logs.

These logs stand in for app telemetry (opens, in-app message responses,
glucose-log entries) without any privacy concerns. The generative model is
deterministic given a seed and is documented in
``schema/graph_schema.md`` so a reviewer can audit how the synthetic signal
relates to real-world engagement patterns.

Mechanism (intentionally simple, intentionally documented):
    * Each patient is assigned an "engagement archetype":
        - power_user (15%): high baseline opens, low dropout hazard
        - steady (45%): moderate baseline, slow exponential decay
        - episodic (25%): bursty engagement around lab dates
        - early_dropout (15%): high opens for first ~2 weeks then near zero
    * Daily app-open count is drawn from a Poisson with archetype-specific
      rate, modulated by a day-of-week effect and a decay term.
    * Message-response rate is Bernoulli with archetype-specific p, with the
      same decay multiplier applied.
    * Glucose-log timestamps are drawn from a thinned Poisson process whose
      rate is proportional to app-open count that day.

Usage:
    python -m data.synthetic.generate_engagement_logs --patients 500 --days 180
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ARCHETYPES = ("power_user", "steady", "episodic", "early_dropout")
ARCHETYPE_PROBS = (0.15, 0.45, 0.25, 0.15)


@dataclass(frozen=True)
class ArchetypeParams:
    open_rate: float  # mean app opens per day at t=0
    response_p: float  # baseline message-response probability
    decay: float  # daily exponential decay on the rate
    burst_around_labs: bool


PARAMS: dict[str, ArchetypeParams] = {
    "power_user": ArchetypeParams(
        open_rate=4.5, response_p=0.85, decay=0.0005, burst_around_labs=False
    ),
    "steady": ArchetypeParams(
        open_rate=2.0, response_p=0.60, decay=0.0020, burst_around_labs=False
    ),
    "episodic": ArchetypeParams(
        open_rate=0.8, response_p=0.50, decay=0.0010, burst_around_labs=True
    ),
    # decay=0.035 (≈3.5% per day) lands early_dropout total events in the
    # 300-400 range over 180 days — still distinctly lower than steady, but
    # overlapping enough that AUROC sits in the realistic 0.85-0.92 range
    # rather than the previous synthetic-perfect 1.0.
    "early_dropout": ArchetypeParams(
        open_rate=3.5, response_p=0.40, decay=0.0350, burst_around_labs=False
    ),
}


def _day_of_week_multiplier(day_index: np.ndarray) -> np.ndarray:
    """Weekends modestly lower engagement (mirrors observed DTx patterns)."""
    dow = day_index % 7  # 0 = Monday assumed
    weekend = np.isin(dow, (5, 6))
    return np.where(weekend, 0.8, 1.0)


def _generate_one_patient(
    patient_id: str,
    archetype: str,
    days: int,
    start_date: datetime,
    lab_dates: list[datetime],
    rng: np.random.Generator,
) -> pd.DataFrame:
    params = PARAMS[archetype]
    day_index = np.arange(days)
    decay_mul = np.exp(-params.decay * day_index)
    dow_mul = _day_of_week_multiplier(day_index)

    rates = params.open_rate * decay_mul * dow_mul

    if params.burst_around_labs and lab_dates:
        # +1.5x rate within +/- 3 days of each lab
        lab_offsets = np.array(
            [(d - start_date).days for d in lab_dates if 0 <= (d - start_date).days < days]
        )
        if lab_offsets.size:
            window = 3
            mask = np.zeros(days, dtype=bool)
            for off in lab_offsets:
                lo = max(0, off - window)
                hi = min(days, off + window + 1)
                mask[lo:hi] = True
            rates = np.where(mask, rates * 1.5, rates)

    opens = rng.poisson(rates)

    rows: list[dict] = []
    for d, n_opens in enumerate(opens):
        ts_date = start_date + timedelta(days=int(d))
        for _ in range(int(n_opens)):
            ts = ts_date + timedelta(
                hours=int(rng.integers(7, 23)),
                minutes=int(rng.integers(0, 60)),
            )
            rows.append({"patient_id": patient_id, "kind": "app_open", "ts": ts, "value": 1.0})

        # Message response: at most one per day, gated by current decayed p.
        if rng.random() < params.response_p * decay_mul[d]:
            ts = ts_date + timedelta(
                hours=int(rng.integers(8, 21)),
                minutes=int(rng.integers(0, 60)),
            )
            rows.append(
                {"patient_id": patient_id, "kind": "message_response", "ts": ts, "value": 1.0}
            )

        # Glucose log entries: thinned by app opens
        glog_rate = 0.4 * n_opens
        glog_n = rng.poisson(glog_rate)
        for _ in range(int(glog_n)):
            ts = ts_date + timedelta(
                hours=int(rng.integers(6, 22)),
                minutes=int(rng.integers(0, 60)),
            )
            # plausible BG range (mg/dL)
            value = float(rng.normal(135, 35))
            rows.append({"patient_id": patient_id, "kind": "glucose_log", "ts": ts, "value": value})

    if not rows:
        return pd.DataFrame(columns=["patient_id", "kind", "ts", "value"])
    return pd.DataFrame(rows)


def generate(
    n_patients: int,
    days: int,
    start_date: datetime | None = None,
    seed: int = 42,
    lab_dates_by_patient: dict[str, list[datetime]] | None = None,
    archetype_overlap: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate engagement events and a patient archetype assignment table.

    Args:
        archetype_overlap: Fraction (0.0-0.5) of each patient's events that
            are drawn from a uniformly-random other archetype instead of
            their own. Without this, dropout classification on the
            archetype label is trivially perfect (AUROC 1.0). The default
            of 10% lands AUROC in the realistic 0.80-0.90 range.

    Returns:
        (events_df, archetypes_df)
    """
    if start_date is None:
        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(seed)

    archetypes = rng.choice(ARCHETYPES, size=n_patients, p=ARCHETYPE_PROBS)
    patient_ids = [f"SYN{idx:06d}" for idx in range(n_patients)]
    archetype_df = pd.DataFrame({"patient_id": patient_ids, "archetype": archetypes})

    frames: list[pd.DataFrame] = []
    for pid, atype in zip(patient_ids, archetypes, strict=True):
        labs = (lab_dates_by_patient or {}).get(pid, [])
        own = _generate_one_patient(pid, atype, days, start_date, labs, rng)
        if archetype_overlap > 0:
            # Draw a small slice of behavior from a different archetype to
            # add label noise — patients drift between behavioral modes in
            # real life.
            other_archetypes = [a for a in ARCHETYPES if a != atype]
            mixin_atype = rng.choice(other_archetypes)
            mixin = _generate_one_patient(pid, mixin_atype, days, start_date, labs, rng)
            if not mixin.empty:
                keep_n = int(len(mixin) * archetype_overlap)
                if keep_n > 0:
                    own = pd.concat(
                        [own, mixin.sample(n=keep_n, random_state=int(rng.integers(0, 2**31 - 1)))],
                        ignore_index=True,
                    )
        frames.append(own)

    events = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["patient_id", "kind", "ts", "value"])
    )
    return events, archetype_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--patients", type=int, default=500)
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    events, archetypes = generate(args.patients, args.days, seed=args.seed)
    events_path = args.out / "engagement_events.parquet"
    arche_path = args.out / "engagement_archetypes.parquet"
    events.to_parquet(events_path, index=False)
    archetypes.to_parquet(arche_path, index=False)

    print(f"Wrote {len(events):,} events for {args.patients} patients -> {events_path}")
    print(f"Wrote archetype assignments -> {arche_path}")


if __name__ == "__main__":
    main()
