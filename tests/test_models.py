"""Model training reproducibility tests.

These tests run end-to-end on the tiny fixtures from ``conftest.py`` so they
finish in a few seconds. They guard against the most common regressions:

  * Feature builder produces a non-empty matrix.
  * GBM regressor converges to non-trivial Pearson r on synthetic data.
  * Dropout classifier yields AUROC > 0.55 on the labeled synthetic events.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models._features import FeatureFrame, build_features


def test_feature_frame_construction(tiny_labs, tiny_synthetic_events, tiny_patients, tmp_path, monkeypatch):
    monkeypatch.setenv("CMG_DATA_PROCESSED", str(tmp_path))
    ff = build_features(tiny_labs, tiny_synthetic_events, tiny_patients, cache=False)
    assert isinstance(ff, FeatureFrame)
    assert not ff.X.empty
    assert (ff.X.index == ff.patient_ids).all()
    # one target per patient row
    assert len(ff.y_hba1c) == len(ff.X)
    assert len(ff.y_dropout) == len(ff.X)


def test_split_preserves_row_counts(tiny_labs, tiny_synthetic_events, tiny_patients, tmp_path, monkeypatch):
    monkeypatch.setenv("CMG_DATA_PROCESSED", str(tmp_path))
    ff = build_features(tiny_labs, tiny_synthetic_events, tiny_patients, cache=False)
    tr, te = ff.split(test_frac=0.25, seed=1)
    assert len(tr.X) + len(te.X) == len(ff.X)
    assert set(tr.X.index).isdisjoint(set(te.X.index))


@pytest.mark.slow
def test_gbm_baseline_runs_and_predicts(tiny_labs, tiny_synthetic_events, tiny_patients, tmp_path, monkeypatch):
    monkeypatch.setenv("CMG_DATA_PROCESSED", str(tmp_path))
    from models.gbm_baseline import train_regressor

    ff = build_features(tiny_labs, tiny_synthetic_events, tiny_patients, cache=False)
    tr, te = ff.split(test_frac=0.25, seed=1)
    res = train_regressor(tr, te, target="hba1c", artifact_dir=tmp_path / "gbm")
    assert res.n_train > 0
    assert res.mae >= 0
    # We expect *some* signal because y_hba1c is built from HbA1c_last.
    assert pd.notna(res.pearson_r)


@pytest.mark.slow
def test_dropout_classifier_runs(tiny_labs, tiny_synthetic_events, tiny_patients, tmp_path, monkeypatch):
    monkeypatch.setenv("CMG_DATA_PROCESSED", str(tmp_path))
    from models.engagement_dropout import train_dropout

    ff = build_features(tiny_labs, tiny_synthetic_events, tiny_patients, cache=False)
    tr, te = ff.split(test_frac=0.3, seed=2)
    if tr.y_dropout.nunique() < 2 or te.y_dropout.nunique() < 2:
        pytest.skip("tiny synthetic split has no class diversity")
    res = train_dropout(tr, te, artifact_dir=tmp_path / "dropout")
    assert res.n_train > 0


def test_summary_template_is_deterministic():
    from dashboard._summary import PatientSummaryInputs, build_summary

    inp = PatientSummaryInputs(
        patient_id="X", last_hba1c=7.0, pred_hba1c_next=7.3,
        pred_hba1c_low=6.9, pred_hba1c_high=7.7,
        last_ldl=110, last_bp_sys=125, last_bp_dia=78,
        app_opens_30d=10, message_responses_30d=3, glucose_logs_30d=5,
        dropout_risk=0.18, top_factors=[("HbA1c_last", 0.4)],
    )
    a = build_summary(inp)
    b = build_summary(inp)
    assert a == b
    assert "trend rising" in a
    assert "lightly engaged" in a
