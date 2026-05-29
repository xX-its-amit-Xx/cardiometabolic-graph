"""Unified CLI entry point — see ``cmg.__doc__`` for the command tree.

Implemented with stdlib argparse only so it works on a fresh clone before
any other deps are installed.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MIMIC_DEMO_URL = (
    "https://physionet.org/static/published-projects/" "mimic-iv-demo/mimic-iv-demo-2.2.zip"
)
MIMIC_DIR = REPO / "data" / "raw" / "mimic-iv"


# --- Output helpers -------------------------------------------------------


class _Style:
    OK = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    END = "\033[0m"

    @classmethod
    def disabled(cls) -> bool:
        return os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    return text if _Style.disabled() else f"{code}{text}{_Style.END}"


# Reconfigure stdout/stderr to UTF-8 so the Unicode glyphs below render on
# Windows consoles (cp1252 default). Falls back silently on streams that
# don't expose ``reconfigure`` (some IDE-captured TTYs).
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


# Pick glyphs that any encoding can produce.
_USE_UNICODE = (sys.stdout.encoding or "").lower().startswith("utf")
_OK_GLYPH = "✓" if _USE_UNICODE else "OK"
_WARN_GLYPH = "!" if _USE_UNICODE else "!!"
_FAIL_GLYPH = "✗" if _USE_UNICODE else "X"


def _ok(msg: str) -> None:
    print(f"  {_color(_OK_GLYPH, _Style.OK)} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_color(_WARN_GLYPH, _Style.WARN)} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_color(_FAIL_GLYPH, _Style.FAIL)} {msg}")


def _section(title: str) -> None:
    print(f"\n{_color(title, _Style.BOLD)}")


# --- Doctor: pre-flight health check --------------------------------------


def _has_module(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False


def _doctor() -> int:
    _section("Environment")
    py_ok = sys.version_info >= (3, 11) and sys.version_info < (3, 13)
    if py_ok:
        _ok(f"Python {sys.version.split()[0]}")
    else:
        _fail(f"Python {sys.version.split()[0]} (need 3.11–3.12)")

    _section("Core dependencies")
    core = [
        "pandas",
        "numpy",
        "pyarrow",
        "sklearn",
        "lightgbm",
        "shap",
        "matplotlib",
        "plotly",
        "streamlit",
    ]
    missing_core: list[str] = []
    for mod in core:
        if _has_module(mod):
            _ok(mod)
        else:
            _fail(f"{mod} (run: pip install -e .)")
            missing_core.append(mod)

    _section("Optional integrations")
    for mod, hint in (
        ("torch", "GNN training; install with `pip install torch`"),
        ("torch_geometric", "GNN training; install with `pip install torch-geometric`"),
        ("neo4j", "Postgres+Neo4j graph build; `pip install neo4j`"),
        ("playwright", "Headless dashboard screenshots; `pip install playwright`"),
    ):
        if _has_module(mod):
            _ok(mod)
        else:
            _warn(f"{mod} missing — {hint}")

    _section("Data files")
    files = [
        ("data/synthetic/engagement_events.parquet", "synthetic engagement (run: cmg pipeline)"),
        ("data/processed/labs.parquet", "lab frame (run: cmg pipeline)"),
        ("data/processed/features.parquet", "cached features (run: cmg pipeline)"),
        (
            "artifacts/gbm/gbm_hba1c_predictions.parquet",
            "trained GBM predictions (run: cmg pipeline)",
        ),
        (
            "artifacts/dropout/dropout_predictions.parquet",
            "trained dropout predictions (run: cmg pipeline)",
        ),
    ]
    missing_data: list[str] = []
    for rel, hint in files:
        p = REPO / rel
        if p.exists():
            _ok(rel)
        else:
            _warn(f"{rel} missing — {hint}")
            missing_data.append(rel)

    if MIMIC_DIR.exists():
        _ok("MIMIC-IV demo data present at data/raw/mimic-iv/")
    else:
        _warn(
            "MIMIC-IV demo NOT present — pipeline will run synthetic-only. "
            "Run `cmg fetch-mimic` to download (~16 MB, no credentials)."
        )

    _section("Verdict")
    if missing_core:
        _fail(
            f"{len(missing_core)} core dep(s) missing — run `pip install -e .` "
            "from the repo root."
        )
        return 2
    if missing_data:
        _warn(
            f"{len(missing_data)} data file(s) missing — run `cmg quickstart` "
            "or `cmg pipeline` to build them."
        )
        return 1
    _ok("Ready to go.")
    return 0


# --- Fetch MIMIC demo -----------------------------------------------------


def _fetch_mimic(force: bool = False) -> int:
    if MIMIC_DIR.exists() and not force:
        _ok(f"MIMIC demo already present at {MIMIC_DIR}")
        return 0

    MIMIC_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = MIMIC_DIR.parent / "mimic-iv-demo.zip"
    print(f"Downloading MIMIC-IV demo (~16 MB) from {MIMIC_DEMO_URL}")
    try:
        with urllib.request.urlopen(MIMIC_DEMO_URL, timeout=120) as r:
            zip_path.write_bytes(r.read())
    except (urllib.error.URLError, TimeoutError) as exc:
        _fail(f"Download failed: {exc}")
        return 2
    _ok(f"Saved zip to {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(MIMIC_DIR.parent)
    nested = MIMIC_DIR.parent / "mimic-iv-clinical-database-demo-2.2"
    if nested.exists():
        if MIMIC_DIR.exists():
            shutil.rmtree(MIMIC_DIR)
        nested.rename(MIMIC_DIR)
    zip_path.unlink(missing_ok=True)
    _ok(f"Unzipped to {MIMIC_DIR}")
    return 0


# --- Pipeline / quickstart wrappers --------------------------------------


def _py_module(module: str, *args: str, env: dict | None = None) -> int:
    cmd = [sys.executable, "-m", module, *args]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd, env=env)


def _pipeline(skip_mimic: bool = False) -> int:
    """End-to-end: synth -> labs -> (mimic if available) -> train -> evaluate."""
    _section("1/5 Synthetic engagement + labs")
    if _py_module("data.synthetic.generate_engagement_logs", "--patients", "500", "--days", "180"):
        return 1
    if _py_module("data.synthetic.generate_labs"):
        return 1

    _section("2/5 MIMIC-IV demo ingest (parquet)")
    if MIMIC_DIR.exists() and not skip_mimic:
        if _py_module("etl.load_mimic_parquet", "--merge-synthetic-labs"):
            return 1
    else:
        _warn(
            "Skipping MIMIC ingest — run `cmg fetch-mimic` to download it. "
            "Pipeline will use synthetic labs only."
        )

    # Invalidate cached features so the next step picks up fresh labs.
    for p in (REPO / "data" / "processed").glob("*.parquet"):
        if p.name.startswith(("features", "y_")):
            p.unlink()

    _section("3/5 Train models (HbA1c GBM, dropout, delta regressor)")
    if _py_module("models.train", "--target", "hba1c", "--model", "gbm"):
        return 1
    if _py_module("models.train", "--target", "engagement_dropout", "--model", "gbm"):
        return 1
    if _py_module("models.delta_regressor"):
        return 1

    _section("4/5 SHAP analysis + model report")
    if _py_module("explain.shap_analysis"):
        return 1
    if _py_module("models.evaluate"):
        return 1

    _section("5/5 Use-case evaluation scoreboard")
    runner = REPO / "scripts" / "evaluate_all_use_cases.py"
    if subprocess.call([sys.executable, str(runner)]):
        return 1
    _ok("Pipeline complete.")
    print(
        "\n  Next: `cmg dashboard` to launch the Streamlit clinician view\n"
        "        `cmg cookbook 01` to run a worked example\n"
        "        `cat docs/EVALUATION.md` for the metrics scoreboard"
    )
    return 0


def _quickstart() -> int:
    """First-time setup: install hint -> fetch MIMIC -> pipeline -> ready."""
    _section("cardiometabolic-graph quickstart")
    rc = _doctor()
    if rc == 2:
        _fail("Resolve the core-dep issues above, then re-run `cmg quickstart`.")
        return rc

    if not MIMIC_DIR.exists():
        _section("Fetching MIMIC-IV demo (one-time, ~16 MB)")
        if _fetch_mimic() == 2:
            _warn("MIMIC fetch failed — falling back to synthetic-only.")

    return _pipeline(skip_mimic=not MIMIC_DIR.exists())


def _dashboard() -> int:
    """Launch Streamlit dashboard. Auto-runs the pipeline if missing."""
    feats = REPO / "data" / "processed" / "features.parquet"
    if not feats.exists():
        _warn("No features cached yet — running quickstart first.")
        if _quickstart():
            return 1
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(REPO / "dashboard" / "app.py"),
    ]
    print(f"  $ {' '.join(cmd)}")
    return subprocess.call(cmd)


def _cookbook(name: str, *extra: str) -> int:
    """Run a cookbook example by number or full module name."""
    aliases = {
        "01": "cookbook.01_at_risk_cohort.run",
        "02": "cookbook.02_reengagement_outreach.run",
        "03": "cookbook.03_pathway_anchored_explanation.run",
        "04": "cookbook.04_cohort_drift_monitor.run",
        "05": "cookbook.05_prior_auth_note.run",
        "06": "cookbook.06_pre_visit_summary.run",
        "07": "cookbook.07_trial_eligibility.run",
        "08": "cookbook.08_pharmacist_intervention.run",
    }
    module = aliases.get(name, name)
    return _py_module(module, *extra)


def _evaluate() -> int:
    return subprocess.call([sys.executable, str(REPO / "scripts" / "evaluate_all_use_cases.py")])


def _version() -> int:
    try:
        from importlib.metadata import version

        print(version("cardiometabolic-graph"))
    except Exception:
        print("0.1.0 (dev)")
    return 0


# --- argparse glue --------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cmg",
        description="Unified CLI for the cardiometabolic-graph reference pipeline.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor", help="diagnose env, data, and trained-artifact state")
    sub.add_parser("quickstart", help="one-command setup: fetch + train + evaluate")

    p_fetch = sub.add_parser("fetch-mimic", help="download the MIMIC-IV demo zip (~16 MB)")
    p_fetch.add_argument("--force", action="store_true", help="redownload even if present")

    p_pipe = sub.add_parser("pipeline", help="run synth + ETL + train + evaluate")
    p_pipe.add_argument("--skip-mimic", action="store_true")

    sub.add_parser("dashboard", help="launch Streamlit dashboard (auto-runs pipeline if needed)")

    p_cb = sub.add_parser("cookbook", help="run a cookbook example by number or module path")
    p_cb.add_argument("name", help="01..08 or full module path")
    p_cb.add_argument(
        "rest", nargs=argparse.REMAINDER, help="extra args passed through to the cookbook"
    )

    sub.add_parser("evaluate", help="regenerate the use-case evaluation scoreboard")
    sub.add_parser("version", help="print package version")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "doctor":
        return _doctor()
    if args.cmd == "quickstart":
        return _quickstart()
    if args.cmd == "fetch-mimic":
        return _fetch_mimic(force=args.force)
    if args.cmd == "pipeline":
        return _pipeline(skip_mimic=args.skip_mimic)
    if args.cmd == "dashboard":
        return _dashboard()
    if args.cmd == "cookbook":
        return _cookbook(args.name, *args.rest)
    if args.cmd == "evaluate":
        return _evaluate()
    if args.cmd == "version":
        return _version()
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
