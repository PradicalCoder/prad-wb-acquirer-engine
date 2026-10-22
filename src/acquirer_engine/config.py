"""Load config.yaml into typed settings, and read the API key from the environment.

Each class below mirrors one section of config.yaml. Pydantic checks types when
the file is loaded, so a bad value (e.g. ev_mm: "two hundred") fails at startup
with a clear message instead of deep inside a run. `extra="forbid"` also rejects
unknown keys, so a misspelled setting is caught rather than silently ignored.
"""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from acquirer_engine.data import DealType, TargetProfile


class Section(BaseModel):
    """Base for every config section: unknown keys are an error."""

    model_config = ConfigDict(extra="forbid")


class Paths(Section):
    data_csv: Path
    runs_dir: Path
    llm_cache: Path


class Scoring(Section):
    sector_mult: float
    sector_floor: float
    sector_cap: float
    size_base: float
    size_floor: float
    deal_type_tiers: dict[DealType, float]
    ownership_miss: float
    geography_miss: float
    margin_strong_percentile: float
    margin_medium: float
    margin_weak: float
    recency_steps: list[tuple[int, float]]  # (up to N years old, weight)
    recency_old: float
    min_estimate_peers: int
    top_n_comps: int
    pool_size: int
    mix_pe: int
    mix_strategic: int
    min_per_type: int
    risk_percentile: float
    integration_sector_weight: float
    financing_size_ratio: float
    antitrust_min_deals: int
    max_ability_risks: int
    strong_relevance: float
    recent_year: int


class Models(Section):
    fast: str
    mid: str
    strong: str


class Retries(Section):
    max_retries: int
    base_seconds: float
    cap_seconds: float


class Timeouts(Section):
    intake: float
    coverage: float
    captain: float
    analyst: float


class LLM(Section):
    provider: str
    models: Models
    temperature: float
    max_concurrency: int
    retries: Retries
    timeouts_seconds: Timeouts


class Config(Section):
    paths: Paths
    default_banker_text: str
    default_target: TargetProfile  # same model the intake agent produces
    scoring: Scoring
    llm: LLM


def load_config(path: str | Path = "config.yaml") -> Config:
    """Read config.yaml and return validated settings.

    Relative paths inside the file (e.g. data/ma_transactions_500.csv) are taken
    relative to the config file's own folder, so the engine works no matter
    which folder it is started from.
    """
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = Config.model_validate(raw)

    base = path.parent
    config.paths.data_csv = base / config.paths.data_csv
    config.paths.runs_dir = base / config.paths.runs_dir
    config.paths.llm_cache = base / config.paths.llm_cache
    return config


def get_api_key() -> str | None:
    """Return the Anthropic API key, or None if it is not set.

    Looks in the real environment first, then in a local .env file
    (load_dotenv never overwrites a variable that is already set).
    """
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY") or None
