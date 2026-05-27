"""Feature builders shared across baseline and GNN models.

Reading patient state directly from Neo4j is convenient for experimentation
but slow for repeated training. So we materialize a wide-format feature
frame once and cache it under ``data/processed/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from etl._common import log, processed_path


LAB_FEATURES = ["HbA1c", "glucose_serum", "cholesterol_total", "hdl", "ldl", "triglycerides"]


@dataclass
class FeatureFrame:
    """Wide-format per-patient feature matrix + targets."""

    X: pd.DataFrame                 # patient-level features
    y_hba1c: pd.Series              # next-window HbA1c (regression)
    y_dropout: pd.Series            # 1 = patient dropped out in next 30 days
    patient_ids: pd.Index

    def split(self, test_frac: float = 0.2, seed: int = 42) -> tuple["FeatureFrame", "FeatureFrame"]:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.X))
        rng.shuffle(idx)
        cut = int(len(idx) * (1 - test_frac))
        tr, te = idx[:cut], idx[cut:]
        return (
            FeatureFrame(self.X.iloc[tr], self.y_hba1c.iloc[tr], self.y_dropout.iloc[tr], self.patient_ids[tr]),
            FeatureFrame(self.X.iloc[te], self.y_hba1c.iloc[te], self.y_dropout.iloc[te], self.patient_ids[te]),
        )


def _build_lab_aggregates(labs: pd.DataFrame) -> pd.DataFrame:
    """Wide aggregates: last value, mean, min, max, count for each cardio lab."""
    if labs.empty:
        return pd.DataFrame()
    labs = labs.copy()
    labs["taken_ts"] = pd.to_datetime(labs["taken_ts"], utc=True)
    labs = labs[labs["name"].isin(LAB_FEATURES)]

    last = labs.sort_values("taken_ts").groupby(["patient_id", "name"]).tail(1)
    last_w = last.pivot(index="patient_id", columns="name", values="value").add_suffix("_last")
    mean_w = labs.groupby(["patient_id", "name"])["value"].mean().unstack().add_suffix("_mean")
    min_w = labs.groupby(["patient_id", "name"])["value"].min().unstack().add_suffix("_min")
    max_w = labs.groupby(["patient_id", "name"])["value"].max().unstack().add_suffix("_max")
    cnt_w = labs.groupby(["patient_id", "name"])["value"].count().unstack().add_suffix("_n").fillna(0)

    return pd.concat([last_w, mean_w, min_w, max_w, cnt_w], axis=1)


def _build_engagement_features(events: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """Per-patient engagement stats over the most recent ``horizon_days``."""
    if events.empty:
        return pd.DataFrame()
    ev = events.copy()
    ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
    cutoff = ev["ts"].max() - pd.Timedelta(days=horizon_days)
    recent = ev[ev["ts"] >= cutoff]

    pivot = recent.pivot_table(
        index="patient_id", columns="kind", values="value",
        aggfunc=["sum", "count"], fill_value=0,
    )
    pivot.columns = [f"{a}_{b}_30d" for a, b in pivot.columns]

    # Engagement decay: count of events per 14-day window over last 60 days
    last_60 = ev[ev["ts"] >= ev["ts"].max() - pd.Timedelta(days=60)].copy()
    last_60["bucket"] = ((ev["ts"].max() - last_60["ts"]).dt.days // 14).astype(int)
    decay = last_60.pivot_table(
        index="patient_id", columns="bucket", values="value", aggfunc="count", fill_value=0,
    ).add_prefix("ev_bucket_")

    return pivot.join(decay, how="outer").fillna(0)


def build_features(
    labs: pd.DataFrame,
    events: pd.DataFrame,
    patients: pd.DataFrame,
    cache: bool = True,
    archetypes: pd.DataFrame | None = None,
) -> FeatureFrame:
    """Construct the canonical feature frame.

    Targets:
        y_hba1c   = each patient's HbA1c at the LAST recorded visit. Earlier
                    visits feed the feature aggregates; we predict the most
                    recent value as a stand-in for the next-visit target.
                    Uses real values when present; otherwise mean-reverts
                    from synthetic seed.
        y_dropout = 1 for patients in the 'early_dropout' archetype when
                    archetype labels are available (synthetic ground truth);
                    otherwise falls back to a within-series decay heuristic
                    that compares first-vs-last 14-day activity.
    """
    # Hold out the most-recent visit per patient for the regression target,
    # then build aggregates over only the earlier visits so the target isn't
    # trivially leaked.
    last_hba1c_target: pd.Series
    if not labs.empty and "name" in labs.columns:
        labs_sorted = labs.copy()
        labs_sorted["taken_ts"] = pd.to_datetime(labs_sorted["taken_ts"], utc=True)
        hba1c_only = labs_sorted[labs_sorted["name"] == "HbA1c"]
        if not hba1c_only.empty:
            last_idx = hba1c_only.sort_values("taken_ts").groupby("patient_id").tail(1).index
            last_hba1c_target = (
                hba1c_only.loc[last_idx].set_index("patient_id")["value"].astype(float)
            )
            feature_labs = labs_sorted.drop(index=last_idx)
        else:
            last_hba1c_target = pd.Series(dtype=float)
            feature_labs = labs_sorted
    else:
        last_hba1c_target = pd.Series(dtype=float)
        feature_labs = labs

    lab_features = _build_lab_aggregates(feature_labs)
    eng_features = _build_engagement_features(events)
    static = patients.set_index("patient_id")[[c for c in ("sex", "birth_year") if c in patients.columns]]

    X = (
        lab_features.join(eng_features, how="outer")
        .join(static, how="outer")
        .fillna(0)
    )

    if "sex" in X.columns:
        X = pd.concat(
            [X.drop(columns=["sex"]), pd.get_dummies(X["sex"], prefix="sex", dtype=float)],
            axis=1,
        )

    rng = np.random.default_rng(0)
    if not last_hba1c_target.empty:
        y_hba1c = last_hba1c_target.reindex(X.index)
        # For any patient without a held-out HbA1c, mean-impute then add
        # a small per-row jitter so the target isn't pathologically
        # constant for that subset.
        fallback_mean = float(y_hba1c.dropna().mean()) if y_hba1c.notna().any() else 6.5
        missing_mask = y_hba1c.isna()
        if missing_mask.any():
            y_hba1c.loc[missing_mask] = fallback_mean + rng.normal(0, 0.3, size=missing_mask.sum())
    else:
        y_hba1c = pd.Series(rng.normal(6.5, 1.0, size=len(X)), index=X.index)

    if archetypes is not None and not archetypes.empty:
        arch_series = archetypes.set_index("patient_id")["archetype"].reindex(X.index)
        y_dropout = (arch_series == "early_dropout").astype(int)
    elif not events.empty:
        ev = events.copy()
        ev["ts"] = pd.to_datetime(ev["ts"], utc=True)
        last_day = ev["ts"].max()
        first_day = ev["ts"].min()
        # First 14 days vs last 14 days of the series — catches archetypes
        # that drop off early, not just those tailing off at the end.
        first_14 = ev[ev["ts"] <= first_day + pd.Timedelta(days=14)].groupby("patient_id").size()
        last_14 = ev[ev["ts"] >= last_day - pd.Timedelta(days=14)].groupby("patient_id").size()
        joined = pd.concat([last_14.rename("recent"), first_14.rename("baseline")], axis=1).fillna(0)
        joined = joined.reindex(X.index, fill_value=0)
        y_dropout = ((joined["recent"] + 1) / (joined["baseline"] + 1) < 0.25).astype(int)
    else:
        y_dropout = pd.Series(0, index=X.index, dtype=int)

    if cache:
        processed_path().mkdir(parents=True, exist_ok=True)
        X.to_parquet(processed_path() / "features.parquet")
        y_hba1c.to_frame("y").to_parquet(processed_path() / "y_hba1c.parquet")
        y_dropout.to_frame("y").to_parquet(processed_path() / "y_dropout.parquet")
        log.info("cached feature frame -> %s", processed_path())

    return FeatureFrame(X=X, y_hba1c=y_hba1c, y_dropout=y_dropout, patient_ids=X.index)


def load_cached() -> FeatureFrame | None:
    base: Path = processed_path()
    fpath = base / "features.parquet"
    if not fpath.exists():
        return None
    X = pd.read_parquet(fpath)
    y_hba1c = pd.read_parquet(base / "y_hba1c.parquet")["y"]
    y_dropout = pd.read_parquet(base / "y_dropout.parquet")["y"]
    return FeatureFrame(X=X, y_hba1c=y_hba1c, y_dropout=y_dropout, patient_ids=X.index)
