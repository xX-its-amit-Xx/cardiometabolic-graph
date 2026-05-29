"""cardiometabolic-graph — unified CLI namespace.

Most users only need three commands:

    cmg doctor       — diagnose what's installed / missing / broken
    cmg quickstart   — go from a fresh clone to a running dashboard
                       (download MIMIC demo, generate synthetic, train,
                       evaluate, capture screenshots)
    cmg dashboard    — launch the Streamlit dashboard

Power users have:

    cmg pipeline     — the synth -> etl -> train -> evaluate workflow
    cmg cookbook     — run a single cookbook example
    cmg evaluate     — regenerate the metric scoreboard
    cmg version      — print the package version
"""

__all__ = ["cli"]
