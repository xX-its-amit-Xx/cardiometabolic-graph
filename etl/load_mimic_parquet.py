"""Postgres-less variant of load_mimic — writes a curated cardiometabolic
slice of MIMIC-IV directly to parquet under ``data/processed/`` so the
feature builder and dashboard can consume it without standing up Postgres.

Same LOINC subset as ``load_mimic.py``. Suitable for environments where
Docker isn't available (e.g. the demo runs entirely on parquet).

Outputs:
    data/processed/patients.parquet
    data/processed/encounters.parquet
    data/processed/labs.parquet     (merged with synthetic labs if present)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ._common import log, processed_path, raw_path

CARDIO_LOINC = {
    "4548-4": "HbA1c",
    "2345-7": "glucose_serum",
    "2093-3": "cholesterol_total",
    "2085-9": "hdl",
    "2089-1": "ldl",
    "2571-8": "triglycerides",
}

# Direct itemid -> canonical-name map for MIMIC-IV demo (the demo's
# d_labitems table doesn't carry the loinc_code column, so we fall back to
# a hand-curated lookup that resolves the standard cardiometabolic panel.
# Mapping verified against MIMIC-IV demo v2.2 / hosp/d_labitems.csv.gz.)
CARDIO_ITEMID: dict[int, str] = {
    50852: "HbA1c",  # % Hemoglobin A1c
    50931: "glucose_serum",  # Glucose, Blood, Chemistry (serum)
    52569: "glucose_serum",  # Glucose, Blood, Chemistry (alt)
    50907: "cholesterol_total",  # Cholesterol, Total
    50904: "hdl",  # Cholesterol, HDL
    50905: "ldl",  # Cholesterol, LDL, Calculated
    50906: "ldl",  # Cholesterol, LDL, Measured
    51000: "triglycerides",  # Triglycerides
}


def _read_gz(path: Path, **kw) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kw)


def _load_patients(mimic_dir: Path) -> pd.DataFrame:
    p = mimic_dir / "hosp" / "patients.csv.gz"
    df = _read_gz(p)
    df.columns = [c.lower() for c in df.columns]
    # MIMIC-IV date-shifts every patient by a random per-subject offset
    # for privacy, so the anchor_year column lives in the future
    # (2030-2150). anchor_age is the patient's REAL age at the anchor
    # year. We store age_years directly so downstream consumers don't
    # have to know about date shifting; birth_year is filled with
    # 2026 - age_years so it stays internally consistent for the
    # current data year.
    df = df.assign(
        patient_id="MIMIC" + df["subject_id"].astype(str),
        sex=df["gender"],
        age_years=df["anchor_age"].astype("Int64"),
        birth_year=2026 - df["anchor_age"].astype(int),
        cohort="mimic-iv-demo",
        source_system="mimic-iv-demo",
    )
    return df[["patient_id", "sex", "age_years", "birth_year", "cohort", "source_system"]].copy()


def _load_admissions(mimic_dir: Path) -> pd.DataFrame:
    p = mimic_dir / "hosp" / "admissions.csv.gz"
    df = _read_gz(p)
    df.columns = [c.lower() for c in df.columns]
    df = df.assign(
        encounter_id="ENC" + df["hadm_id"].astype(str),
        patient_id="MIMIC" + df["subject_id"].astype(str),
        start_ts=pd.to_datetime(df["admittime"], utc=True),
        end_ts=pd.to_datetime(df["dischtime"], utc=True),
        encounter_type=df["admission_type"].fillna("UNKNOWN"),
    )
    return df[["encounter_id", "patient_id", "start_ts", "end_ts", "encounter_type"]].copy()


def _load_labs(mimic_dir: Path) -> pd.DataFrame:
    target_itemids = set(CARDIO_ITEMID.keys())

    labs_path = mimic_dir / "hosp" / "labevents.csv.gz"
    frames: list[pd.DataFrame] = []
    for chunk in _read_gz(labs_path, chunksize=200_000):
        chunk.columns = [c.lower() for c in chunk.columns]
        chunk = chunk[chunk["itemid"].isin(target_itemids)]
        if chunk.empty:
            continue
        chunk = chunk.dropna(subset=["valuenum"])
        chunk = chunk.assign(
            patient_id="MIMIC" + chunk["subject_id"].astype(str),
            encounter_id=chunk["hadm_id"].apply(lambda x: f"ENC{int(x)}" if pd.notna(x) else None),
            lab_id="LAB" + chunk["labevent_id"].astype(str),
            taken_ts=pd.to_datetime(chunk["charttime"], utc=True),
            name=chunk["itemid"].map(CARDIO_ITEMID),
            value=chunk["valuenum"].astype(float),
            unit=chunk["valueuom"],
            source_system="mimic-iv-demo",
        )
        chunk["loinc"] = None  # demo d_labitems lacks LOINC; left null for traceability
        frames.append(
            chunk[
                [
                    "lab_id",
                    "patient_id",
                    "encounter_id",
                    "loinc",
                    "name",
                    "value",
                    "unit",
                    "taken_ts",
                    "source_system",
                ]
            ]
        )

    if not frames:
        log.warning("No cardiometabolic labs found in MIMIC demo — returning empty")
        return pd.DataFrame(
            columns=[
                "lab_id",
                "patient_id",
                "encounter_id",
                "loinc",
                "name",
                "value",
                "unit",
                "taken_ts",
                "source_system",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, default=raw_path() / "mimic-iv")
    parser.add_argument("--out-dir", type=Path, default=processed_path())
    parser.add_argument(
        "--merge-synthetic-labs",
        action="store_true",
        help="If set, append the synthetic labs parquet to the MIMIC labs output.",
    )
    args = parser.parse_args()

    if not args.mimic_dir.exists():
        raise FileNotFoundError(
            f"MIMIC dir missing: {args.mimic_dir}. Download from "
            "https://physionet.org/content/mimic-iv-demo/2.2/"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = _load_patients(args.mimic_dir)
    log.info("patients: %d", len(patients))
    patients.to_parquet(args.out_dir / "patients.parquet", index=False)

    encounters = _load_admissions(args.mimic_dir)
    log.info("encounters: %d", len(encounters))
    encounters.to_parquet(args.out_dir / "encounters.parquet", index=False)

    labs = _load_labs(args.mimic_dir)
    log.info("MIMIC cardio labs: %d rows for %d patients", len(labs), labs["patient_id"].nunique())

    if args.merge_synthetic_labs:
        synth = args.out_dir / "labs_synthetic.parquet"
        if (args.out_dir / "labs.parquet").exists() and not synth.exists():
            # Rename current synthetic file before overwriting
            (args.out_dir / "labs.parquet").rename(synth)
        if synth.exists():
            synth_df = pd.read_parquet(synth)
            keep = ["patient_id", "name", "value", "unit", "taken_ts", "source_system"]
            for col in keep:
                if col not in synth_df.columns:
                    synth_df[col] = None
            synth_df = synth_df[keep].copy()
            mimic_df = labs[keep].copy() if not labs.empty else pd.DataFrame(columns=keep)
            # Normalize timestamps — synthetic stored as ISO strings, MIMIC
            # as tz-aware datetimes. Coerce both into the same dtype.
            mimic_df["taken_ts"] = pd.to_datetime(mimic_df["taken_ts"], utc=True, errors="coerce")
            synth_df["taken_ts"] = pd.to_datetime(synth_df["taken_ts"], utc=True, errors="coerce")
            combined = pd.concat([mimic_df, synth_df], ignore_index=True)
            combined.to_parquet(args.out_dir / "labs.parquet", index=False)
            log.info(
                "merged labs: %d rows (MIMIC %d + synthetic %d)",
                len(combined),
                len(mimic_df),
                len(synth_df),
            )
            return

    labs.to_parquet(args.out_dir / "labs.parquet", index=False)
    log.info("wrote -> %s", args.out_dir / "labs.parquet")


if __name__ == "__main__":
    main()
