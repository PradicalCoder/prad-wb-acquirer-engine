"""Command line entry point: `acquirer-engine run`.

For now this only checks the inputs and creates the run folder. The pipeline
(intake, scoring, agents, report) plugs in where marked below as it is built.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path

import typer

from acquirer_engine.config import get_api_key, load_config

app = typer.Typer(add_completion=False, help="Find the most likely acquirers for a target.")


class Mode(str, Enum):
    """How LLM calls are handled.

    live    - use stored answers when available, otherwise call the API and store the answer
    replay  - stored answers only; no API key needed
    refresh - always call the API and overwrite stored answers
    off     - always call the API, store nothing (for debugging)
    """

    live = "live"
    replay = "replay"
    refresh = "refresh"
    off = "off"


# Modes that may call the API, and therefore need a key.
MODES_NEEDING_KEY = {Mode.live, Mode.refresh, Mode.off}


@app.callback()
def main() -> None:
    """Keeps `run` as a named subcommand (Typer would otherwise drop the name)."""


@app.command()
def run(
    text: str = typer.Option(None, help="The banker's request. Default: the one in config.yaml."),
    mode: Mode = typer.Option(Mode.live, help="live | replay | refresh | off"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Settings file."),
) -> None:
    """Find the 10 most likely acquirers and write a one-page rationale for each."""
    config = load_config(config_path)
    request = text or config.default_banker_text

    if mode in MODES_NEEDING_KEY and not get_api_key():
        typer.echo(
            "No API key found. Set ANTHROPIC_API_KEY, "
            "or run with --mode replay to replay the recorded run.",
            err=True,
        )
        raise typer.Exit(code=1)

    # One folder per run, named by start time, e.g. runs/20261021-143005.
    run_dir = config.paths.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Mode:    {mode.value}")
    typer.echo(f"Request: {request}")
    typer.echo(f"Output:  {run_dir}")

    # Pipeline goes here: intake -> scoring -> coverage agents -> captain
    # -> analysts -> validation -> report.
    typer.echo("Pipeline not built yet.")
