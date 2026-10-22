"""Load config.yaml into typed settings, and read the API key from the environment.

Each class below mirrors one section of config.yaml. Pydantic checks types when
the file is loaded, so a bad value (e.g. ev_mm: "two hundred") fails at startup
with a clear message instead of deep inside a run. `extra="forbid"` also rejects
unknown keys, so a misspelled setting is caught rather than silently ignored.
"""

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from acquirer_engine.data import BuyerType, DealType, Sector, Tag, TargetProfile


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
    ebitda_mismatch: float
    strong_relevance: float
    recent_year: int


class Thesis(Section):
    """One seed thesis: a buyer story written as deal search filters."""

    buyer_type: BuyerType
    name: str
    story: str
    size: tuple[float, float]            # multiples of the target's EV
    deal_types: list[DealType] | None = None
    sectors: list[Sector] | None = None  # None = the target's sector family
    tags_any: list[Tag] | None = None
    year_min: int | None = None
    min_relevance: float = 0.0


class Coverage(Section):
    sector_families: dict[str, list[Sector]]
    theses: dict[str, Thesis]
    rankings_top_k: int
    top_n: int


class Validation(Section):
    max_repairs: int
    page_words_min: int
    page_words_max: int
    banned_phrases: list[str]
    max_thesis_similarity: float
    rewrite_min_seconds: float


class Profile(Section):
    """How one kind of LLM call is made (see the comments in config.yaml)."""

    model: str
    effort: Literal["low", "medium", "high"]
    thinking: Literal["adaptive", "between_tools"] = "adaptive"
    max_tokens: int


class Retries(Section):
    max_retries: int
    base_seconds: float
    cap_seconds: float
    min_attempt_seconds: float


class Timeouts(Section):
    intake: float
    coverage: float
    captain: float
    analyst: float


class LLM(Section):
    provider: str
    profiles: dict[str, Profile]
    max_concurrency: int
    retries: Retries
    timeouts_seconds: Timeouts
    run_budget_seconds: float


class Config(Section):
    paths: Paths
    default_banker_text: str
    default_target: TargetProfile  # same model the intake agent produces
    scoring: Scoring
    coverage: Coverage
    validation: Validation
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
