"""Smoke tests for the command line: config loads, key check works, run folder is made."""

import shutil
from pathlib import Path

from typer.testing import CliRunner

from acquirer_engine.cli import app
from acquirer_engine.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.yaml"
runner = CliRunner()


def test_config_loads():
    config = load_config(CONFIG)
    assert config.default_target.ev_mm == 200.0
    assert config.paths.data_csv.exists()


def test_live_without_key_fails_with_hint(monkeypatch):
    # Make sure no key is found, from the environment or from .env.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("acquirer_engine.cli.get_api_key", lambda: None)

    result = runner.invoke(app, ["run", "--mode", "live", "--config", str(CONFIG)])

    assert result.exit_code == 1
    assert "--mode replay" in result.output


def test_replay_with_empty_cache_still_finishes(tmp_path, monkeypatch):
    """No key and nothing recorded: every LLM step falls back, and the run still
    produces 10 evidence-only pages and the full run folder."""
    monkeypatch.setattr("acquirer_engine.cli.get_api_key", lambda: None)
    # Copy config and data into a temp folder so the run folder lands there, not in runs/.
    shutil.copy(CONFIG, tmp_path / "config.yaml")
    shutil.copytree(ROOT / "data", tmp_path / "data")

    result = runner.invoke(app, ["run", "--mode", "replay", "--config", str(tmp_path / "config.yaml")])

    assert result.exit_code == 0, result.output
    assert result.output.count("needs review") == 10
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert {p.name for p in run_dir.iterdir()} == {"report.md", "run_report.json", "state.json", "run.log"}
