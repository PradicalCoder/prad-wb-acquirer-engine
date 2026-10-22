"""Command line entry point: `acquirer-engine run`."""

import asyncio
from pathlib import Path

import typer

from acquirer_engine.cache import Mode
from acquirer_engine.config import get_api_key, load_config
from acquirer_engine.run import execute

app = typer.Typer(add_completion=False, help="Find the most likely acquirers for a target.")


@app.callback()
def main() -> None:
    """Keeps `run` as a named subcommand (Typer would otherwise drop the name)."""


@app.command()
def run(
    text: str = typer.Option(None, help="The banker's request. Default: the one in config.yaml."),
    mode: Mode = typer.Option(Mode.live, help="live | replay | refresh | off"),
    config_path: Path = typer.Option(Path("config.yaml"), "--config", help="Settings file."),
    cache: Path = typer.Option(None, help="Answer cache file. Default: paths.llm_cache in config."),
    strict: bool = typer.Option(False, help="Stop on unexpected errors instead of falling back."),
) -> None:
    """Find the 10 most likely acquirers and write a one-page rationale for each."""
    config = load_config(config_path)
    request = text or config.default_banker_text

    if mode.calls_api and not get_api_key():
        typer.echo(
            "No API key found. Set ANTHROPIC_API_KEY, "
            "or run with --mode replay to replay the recorded run.",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Mode: {mode.value}\nRequest: {request}\n")
    state, run_dir, report = asyncio.run(
        execute(request, config, mode, cache or config.paths.llm_cache, strict)
    )

    if report["stopped"]:
        typer.echo(f"Stopped: {report['stopped']}")
    for i, row in enumerate(report["final"], 1):
        review = "  (needs review)" if row["needs_review"] else ""
        typer.echo(f"{i:>2}. {row['buyer']:<28} {row['conviction']:<7}{review}")
    for warning in report["warnings"] + report["flags"]:
        typer.echo(f"Warning: {warning}")
    total = report["tokens_total"]
    typer.echo(f"\n{report['seconds']} s | {total['calls']} LLM calls ({total['cache_hits']} from cache) "
               f"| tokens in {total['input']:,} / out {total['output']:,}")
    typer.echo(f"Output: {run_dir}")
