"""Test fixtures shared across the suite."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_synthetic_events() -> pd.DataFrame:
    from data.synthetic.generate_engagement_logs import generate
    events, _ = generate(n_patients=20, days=60, seed=7)
    return events


@pytest.fixture
def tiny_labs(tiny_synthetic_events: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    pids = tiny_synthetic_events["patient_id"].unique()
    rows = []
    base_ts = datetime(2024, 6, 1, tzinfo=timezone.utc)
    for pid in pids:
        for name, mean, sd in (
            ("HbA1c", 6.4, 0.8),
            ("glucose_serum", 110, 25),
            ("ldl", 120, 30),
        ):
            for offset_days in (0, 30, 60):
                rows.append({
                    "patient_id": pid,
                    "name": name,
                    "value": float(rng.normal(mean, sd)),
                    "taken_ts": (base_ts + pd.Timedelta(days=offset_days)).isoformat(),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_patients(tiny_synthetic_events: pd.DataFrame) -> pd.DataFrame:
    pids = tiny_synthetic_events["patient_id"].unique()
    return pd.DataFrame({
        "patient_id": pids,
        "sex": ["M" if i % 2 else "F" for i in range(len(pids))],
        "birth_year": 1970,
    })
