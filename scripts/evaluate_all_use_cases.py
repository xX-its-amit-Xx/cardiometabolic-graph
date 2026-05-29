"""Run every use-case evaluator, generate figures, and write a master report.

Outputs:
    docs/figures/evaluation/reliability_dropout.png
    docs/figures/evaluation/decision_curve_dropout.png
    docs/figures/evaluation/lift_curve_at_risk.png
    docs/figures/evaluation/lift_curve_dropout.png
    docs/figures/evaluation/metric_status.png
    docs/figures/evaluation/metrics_table.csv
    docs/EVALUATION.md
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.expected_ranges import RANGES
from evaluation.metrics import (
    decision_curve,
    lift_at_topk,
    reliability_curve,
)
from evaluation.use_case_metrics import _load_artifacts, all_evaluators

FIG_DIR = Path("docs/figures/evaluation")
GRADE_COLORS = {
    "good": "#1a9850",
    "acceptable": "#91cf60",
    "below_acceptable": "#fdae61",
    "below_floor": "#d73027",
    "above_ceiling": "#542788",   # purple — "suspiciously good"
}


def _plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    rc = reliability_curve(y_true, y_prob, n_bins=10)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(rc.bin_centers, rc.observed, "o-", color="#2c7fb8", label="model")
    for x, y, n in zip(rc.bin_centers, rc.observed, rc.counts, strict=True):
        ax.annotate(f"n={n}", (x, y), fontsize=7, alpha=0.6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted dropout probability")
    ax.set_ylabel("Observed dropout rate")
    ax.set_title("Reliability — engagement dropout (Hosmer-Lemeshow style)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_decision_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    dc = decision_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(dc.thresholds, dc.net_benefit, "-", color="#2c7fb8", label="model")
    ax.plot(dc.thresholds, dc.net_benefit_treat_all, "--", color="#fdae61", label="treat all")
    ax.plot(dc.thresholds, dc.net_benefit_treat_none, ":", color="grey", label="treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis — engagement dropout\n(Vickers & Elkin 2006)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=min(0.0, dc.net_benefit.min() - 0.02))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_lift_curve(
    scores: np.ndarray, y_true: np.ndarray, title: str, path: Path,
    ks: list[int] | None = None,
) -> None:
    if ks is None:
        ks = list(range(5, min(len(scores), 200) + 1, 5))
    lifts = [lift_at_topk(scores, y_true, k) for k in ks]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, lifts, "-o", color="#2c7fb8", markersize=4)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="random")
    ax.axhline(2.0, color="#1a9850", linestyle=":", linewidth=0.8, label="good (≥2x)")
    ax.set_xlabel("K (top-K patients flagged)")
    ax.set_ylabel("Lift")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_metric_status(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return
    df = df.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    palette = [GRADE_COLORS.get(g, "#888") for g in df["grade"]]
    y_pos = np.arange(len(df))
    ax.barh(y_pos, df["value"], color=palette)
    for i, (val, hint) in enumerate(zip(df["value"], df["hint"], strict=True)):
        ax.text(val, i, f"  {val:.3f}", va="center", fontsize=8)
    labels = df["use_case"] + "  —  " + df["metric"]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Metric value")
    ax.set_title("Per-metric value with field-standard grade")
    # Custom legend
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g) for g, c in GRADE_COLORS.items()]
    ax.legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    art = _load_artifacts()
    results = all_evaluators(art)

    df = pd.DataFrame([
        {
            "use_case": r.use_case,
            "metric": r.metric,
            "metric_key": r.metric_key,
            "value": r.value,
            "grade": r.grade,
            "hint": r.hint,
            "detail": r.detail,
        }
        for r in results
    ])
    df.to_csv(FIG_DIR / "metrics_table.csv", index=False)

    # Figures that need raw arrays:
    if art["dropout_pred"] is not None and art["y_dropout"] is not None:
        y_full = art["y_dropout"]["y"]
        p = art["dropout_pred"]["p_dropout"]
        common = p.index.intersection(y_full.index)
        y_arr = y_full.loc[common].to_numpy()
        p_arr = p.loc[common].to_numpy()
        _plot_reliability(y_arr, p_arr, FIG_DIR / "reliability_dropout.png")
        _plot_decision_curve(y_arr, p_arr, FIG_DIR / "decision_curve_dropout.png")
        _plot_lift_curve(p_arr, y_arr, "Lift — engagement-dropout outreach",
                         FIG_DIR / "lift_curve_dropout.png")

    # At-risk lift: predicted delta vs true top-quartile delta
    if art["gbm_pred"] is not None and art["y_hba1c"] is not None and art["features"] is not None:
        y_full = art["y_hba1c"]["y"] if isinstance(art["y_hba1c"], pd.DataFrame) else art["y_hba1c"]
        feats = art["features"]
        pred = art["gbm_pred"]["pred"]
        common = pred.index.intersection(feats.index).intersection(y_full.index)
        last = feats.loc[common].get("HbA1c_last", pd.Series(0.0, index=common))
        actual = y_full.loc[common] - last
        delta_pred = pred.loc[common] - last
        threshold = float(np.quantile(actual, 0.75))
        relevance = (actual >= threshold).astype(int).to_numpy()
        _plot_lift_curve(delta_pred.to_numpy(), relevance,
                         "Lift — at-risk cohort (predicted HbA1c rise)",
                         FIG_DIR / "lift_curve_at_risk.png")

    _plot_metric_status(df, FIG_DIR / "metric_status.png")

    # Write EVALUATION.md
    md = _render_eval_markdown(df)
    Path("docs/EVALUATION.md").write_text(md, encoding="utf-8")
    print(f"wrote evaluation summary -> docs/EVALUATION.md + {FIG_DIR}/")
    print(df.to_string(index=False))


def _render_eval_markdown(df: pd.DataFrame) -> str:
    grade_emoji = {
        "good": "🟢", "acceptable": "🟢",
        "below_acceptable": "🟡", "below_floor": "🔴",
        "above_ceiling": "🟣",
    }
    lines: list[str] = []
    lines.append("# Evaluation — clinical-ML metrics on the cardiometabolic-graph pipeline\n")
    lines.append(
        "Every metric below is computed from the most-recent end-to-end "
        "pipeline run. Each entry shows our value, the field-standard "
        "expected range it's graded against, and the rationale for that "
        "range (with the published source). Generated by "
        "`python -m scripts.evaluate_all_use_cases`.\n"
    )

    lines.append("## Summary status\n")
    lines.append("![Metric status](figures/evaluation/metric_status.png)\n")

    grouped = df.groupby("use_case", sort=False)
    lines.append("## Per-use-case results\n")
    for uc, sub in grouped:
        lines.append(f"### `{uc}`\n")
        lines.append("| Metric | Value | Status | What it means | Expected | Source |")
        lines.append("|--------|-------|--------|----------------|----------|--------|")
        for _, row in sub.iterrows():
            r = RANGES.get(row["metric_key"])
            band = (
                f"floor {r.floor} / acceptable {r.acceptable} / good {r.good} / ceiling {r.ceiling}"
                if r else "—"
            )
            source = r.rationale if r else "—"
            lines.append(
                f"| {row['metric']} | {row['value']:.3f} "
                f"| {grade_emoji.get(row['grade'], '⚪')} {row['grade']} ({row['hint']}) "
                f"| {row['detail']} | {band} | {source} |"
            )
        lines.append("")

    lines.append("## Figures\n")
    lines.append("- [`reliability_dropout.png`](figures/evaluation/reliability_dropout.png) — Hosmer-Lemeshow reliability plot for the engagement-dropout classifier.")
    lines.append("- [`decision_curve_dropout.png`](figures/evaluation/decision_curve_dropout.png) — Vickers & Elkin decision-curve analysis: is acting on the model better than `treat all` / `treat none`?")
    lines.append("- [`lift_curve_at_risk.png`](figures/evaluation/lift_curve_at_risk.png) — Lift as a function of cohort-call list size, for the at-risk cohort cookbook.")
    lines.append("- [`lift_curve_dropout.png`](figures/evaluation/lift_curve_dropout.png) — Same, for the dropout re-engagement cookbook.")
    lines.append("- [`metric_status.png`](figures/evaluation/metric_status.png) — every reported metric color-coded by field-standard grade.")
    lines.append("- [`metrics_table.csv`](figures/evaluation/metrics_table.csv) — raw numbers behind everything above.")
    lines.append("")

    lines.append("## How to interpret a row\n")
    lines.append(
        "* **value** — what we actually measured on the most recent run.\n"
        "* **status** — graded against the field-standard expected range "
        "(🟢 acceptable/good, 🟡 below the deployable threshold, 🔴 uninformative, 🟣 suspiciously good).\n"
        "* **expected** — the floor/acceptable/good/ceiling bands from the "
        "literature. The rationale cites the source.\n"
        "* If a row lands 🟡 or 🔴, we treat it as a real signal — see the "
        "remediation log in `ITERATIONS.md`.\n"
    )

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
