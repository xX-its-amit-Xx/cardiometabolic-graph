"""Smoke tests for the unified ``cmg`` CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from cmg import cli


def test_cli_help_exits_zero():
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0


def test_version_subcommand_runs(capsys):
    # _version() may or may not find a package distribution; either way
    # it should exit cleanly and print something resembling a version.
    rc = cli.main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    assert out  # non-empty


def test_doctor_subcommand_runs(monkeypatch, capsys):
    # Doctor uses ANSI codes when stdout is a tty; disable for capture.
    monkeypatch.setenv("NO_COLOR", "1")
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert rc in (0, 1, 2)  # always returns a clean exit code
    assert "Environment" in out
    assert "Verdict" in out


def test_cli_entrypoint_runs_via_python_m():
    """End-to-end: invoke as a subprocess so we exercise the installed
    console script wiring."""
    proc = subprocess.run(
        [sys.executable, "-m", "cmg.cli", "version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout.strip()
