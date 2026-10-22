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


def test_replay_runs_without_key(tmp_path, monkeypatch):
    monkeypatch.setattr("acquirer_engine.cli.get_api_key", lambda: None)
    # Copy the config into a temp folder so the run folder lands there, not in runs/.
    shutil.copy(CONFIG, tmp_path / "config.yaml")

    result = runner.invoke(app, ["run", "--mode", "replay", "--config", str(tmp_path / "config.yaml")])

    assert result.exit_code == 0
    assert len(list((tmp_path / "runs").iterdir())) == 1
