"""Cookbook 07 — clinical trial eligibility screening."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from etl._common import log, processed_path, synthetic_path


HERE = Path(__file__).resolve().parent


@dataclass
class Criterion:
    name: str
    feature: str           # name of the per-patient feature to evaluate
    op: Literal["between", ">=", "<=", "==", "!="]
    threshold: float | tuple[float, float] | str
    direction: Literal["inclusion", "exclusion"] = "inclusion"

    def evaluate(self, value: float | str | None) -> bool:
        if value is None:
            return False
        if self.op == "between":
            lo, hi = self.threshold  # type: ignore[misc]
            return bool(lo <= value <= hi)
        if self.op == ">=":
            return bool(value >= self.threshold)
        if self.op == "<=":
            return bool(value <= self.threshold)
        if self.op == "==":
            return bool(value == self.threshold)
        if self.op == "!=":
            return bool(value != self.threshold)
        return False


@dataclass
class TrialSpec:
    name: str
    criteria: list[Criterion] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: Path) -> "TrialSpec":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            criteria=[
                Criterion(
                    name=c["name"],
                    feature=c["feature"],
                    op=c["op"],
                    threshold=tuple(c["threshold"]) if isinstance(c["threshold"], list) else c["threshold"],
                    direction=c.get("direction", "inclusion"),
                )
                for c in raw["criteria"]
            ],
        )


# Fictional trial inspired by SUSTAIN-6 (semaglutide CV outcomes).
DEFAULT_SPEC = TrialSpec(
    name="Fictional GLP-1 CV outcomes trial (inspired by SUSTAIN-6)",
    criteria=[
        Criterion("HbA1c 7.5–10.5%", "HbA1c_last", "between", (7.5, 10.5)),
        Criterion("Age 30–70", "age_years", "between", (30.0, 70.0)),
        Criterion("LDL not extreme (<200)", "ldl_last", "<=", 200.0),
        Criterion("Triglycerides <500", "triglycerides_last", "<=", 500.0),
        Criterion("Min 30-day app engagement (≥20 opens)",
                  "app_open_count_30d", ">=", 20.0),
    ],
)


def _load_patient_features() -> pd.DataFrame:
    feat_path = processed_path() / "features.parquet"
    labs_path = processed_path() / "labs.parquet"
    if not feat_path.exists() or not labs_path.exists():
        raise FileNotFoundError("Run `make pipeline-parquet` first.")

    df = pd.read_parquet(feat_path)
    # Derive age from birth_year if present
    if "birth_year" in df.columns:
        df["age_years"] = 2026 - df["birth_year"]
    return df


def screen(spec: TrialSpec, top: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = _load_patient_features()

    rows: list[dict] = []
    grid: list[dict] = []
    for pid, row in features.iterrows():
        per_crit = {}
        met = 0
        for c in spec.criteria:
            value = row.get(c.feature)
            ok = c.evaluate(value)
            if c.direction == "exclusion":
                ok = not ok
            per_crit[c.name] = ok
            met += int(ok)

        eng_score = float(row.get("app_open_count_30d", 0.0)) + \
                    float(row.get("glucose_log_count_30d", 0.0))
        rows.append({
            "patient_id": pid,
            "n_criteria_met": met,
            "n_criteria_total": len(spec.criteria),
            "engagement_score": eng_score,
            "all_criteria_met": met == len(spec.criteria),
        })
        per_crit["patient_id"] = pid
        grid.append(per_crit)

    summary = pd.DataFrame(rows).sort_values(
        ["n_criteria_met", "engagement_score"], ascending=[False, False]
    ).head(top).reset_index(drop=True)
    grid_df = pd.DataFrame(grid).set_index("patient_id").reindex(summary["patient_id"])
    return summary, grid_df


def write_report(spec: TrialSpec, summary: pd.DataFrame, grid: pd.DataFrame) -> None:
    summary.to_csv(HERE / "eligible_patients.csv", index=False)

    lines: list[str] = []
    lines.append(f"# Trial eligibility — {spec.name}")
    lines.append("")
    lines.append(f"_Top {len(summary)} candidates ranked by criteria-met then engagement._")
    lines.append("")
    lines.append("## Criteria")
    lines.append("")
    for c in spec.criteria:
        lines.append(f"- **{c.name}** — `{c.feature}` {c.op} {c.threshold} ({c.direction})")
    lines.append("")

    lines.append("## Top candidates")
    lines.append("")
    headers = ["Rank", "Patient", "Met/Total", "Engagement"] + [c.name for c in spec.criteria]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for i, row in summary.iterrows():
        crit_cells = [
            "✓" if grid.loc[row["patient_id"], c.name] else "✗" for c in spec.criteria
        ]
        lines.append(
            f"| {i+1} | `{row['patient_id']}` | {row['n_criteria_met']}/{row['n_criteria_total']} "
            f"| {row['engagement_score']:.0f} | " + " | ".join(crit_cells) + " |"
        )

    (HERE / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None,
                        help="Path to a TrialSpec JSON file. If omitted, uses the default GLP-1 spec.")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    spec = TrialSpec.from_json(args.spec) if args.spec else DEFAULT_SPEC
    summary, grid = screen(spec, top=args.top)
    write_report(spec, summary, grid)
    log.info("wrote eligibility report (%d candidates) -> %s", len(summary), HERE / "report.md")


if __name__ == "__main__":
    main()
