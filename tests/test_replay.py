"""End to end: replaying the recorded run reproduces the committed sample report.

Runs the real pipeline with no API key, reading every answer from
replay_cache/. Only the closing Run line (timings and token counts) may differ.
"""

import asyncio
import re
from pathlib import Path

import pytest

from acquirer_engine.cache import Mode
from acquirer_engine.config import load_config
from acquirer_engine.run import execute

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "config.yaml")
SAMPLE = ROOT / "sample_output" / "report.md"

pytestmark = pytest.mark.skipif(not CONFIG.paths.replay_cache.exists() or not SAMPLE.exists(),
                                reason="no recorded run to replay")


def without_run_stats(markdown: str) -> str:
    """Drop the closing Run section: seconds and tokens differ between recording and replay."""
    return re.split(r"\n### Run\n", markdown)[0].strip()


def test_replay_reproduces_the_sample_report(tmp_path):
    cfg = CONFIG.model_copy(update={"paths": CONFIG.paths.model_copy(update={"runs_dir": tmp_path})})
    state, run_dir, report = asyncio.run(
        execute(cfg.default_banker_text, cfg, Mode.replay, cfg.paths.replay_cache, strict=True))

    assert report["tokens_total"]["calls"] == report["tokens_total"]["cache_hits"] > 0
    assert not any(row["needs_review"] for row in report["final"])
    replayed = (run_dir / "report.md").read_text(encoding="utf-8")
    assert without_run_stats(replayed) == without_run_stats(SAMPLE.read_text(encoding="utf-8"))
