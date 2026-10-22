"""Intake: the banker's free text -> a target profile (or a reason to stop).

The model reads the language; fixed rules in the prompt decide policy (what
"private" or "regional" means). Code then checks that every quote the model
cites appears in the request, which catches invented fields. One repair is
allowed. If the model still fails, the run uses intake_fallback().
"""

import re
from typing import get_args

import pandas as pd
from pydantic import BaseModel

from acquirer_engine.agents import load_prompt
from acquirer_engine.config import Config
from acquirer_engine.data import Ownership, Sector, SingleRegion, TargetProfile
from acquirer_engine.llm import LLMClient, LLMOutputInvalid
from acquirer_engine.schemas import IntakeResult


class IntakeOutcome(BaseModel):
    target: TargetProfile | None    # None when the run must stop
    stop_reason: str | None
    result: IntakeResult | None     # the model's answer, with sources (None on fallback)
    warnings: list[str]


async def run_intake(text: str, client: LLMClient, cfg: Config, deals: pd.DataFrame,
                     deadline: float) -> IntakeOutcome:
    """Raises LLMUnavailable, or LLMOutputInvalid if the answer is still wrong
    after one repair; the graph then calls intake_fallback()."""
    version, system = load_prompt(
        "intake",
        sectors=", ".join(get_args(Sector)),
        ownership_types=", ".join(get_args(Ownership)),
        regions=", ".join(get_args(SingleRegion)),
    )
    messages = [{"role": "user", "content": f"Request: {text}"}]
    ask = dict(agent="intake", profile="intake", system=system, schema=IntakeResult,
               prompt_version=version, deadline=deadline)

    for attempt in range(2):  # the first answer, then one repair
        try:
            result = await client.structured(messages=messages, **ask)
            raw, errors = result.model_dump_json(), quote_errors(result, text)
        except LLMOutputInvalid as error:
            result, raw, errors = None, error.raw_text, [str(error)]
        if not errors:
            return IntakeOutcome(target=result.target, stop_reason=result.stop_reason,
                                 result=result, warnings=intake_warnings(result, deals))
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": "Fix these problems and answer again:\n"
                                        + "\n".join(f"- {e}" for e in errors)},
        ]
    raise LLMOutputInvalid("; ".join(errors), raw_text=raw)


def quote_errors(result: IntakeResult, text: str) -> list[str]:
    """Every quote must appear in the request (ignoring case and spacing)."""
    haystack = _normalize(text)
    return [f"the quote for {source.field} ('{source.quote}') does not appear in the request; "
            f"copy the exact words, or leave the field empty"
            for source in result.sources
            if source.quote and _normalize(source.quote) not in haystack]


def intake_warnings(result: IntakeResult, deals: pd.DataFrame) -> list[str]:
    """Things the report should mention, but that don't stop the run."""
    warnings = []
    target = result.target
    if target is not None:
        low, high = deals.deal_size_mm.min(), deals.deal_size_mm.max()
        if not low <= target.ev_mm <= high:
            warnings.append(f"Deal size ${target.ev_mm:,.0f}M is outside the data's range "
                            f"(${low:,.1f}M to ${high:,.1f}M); size matching is less reliable.")
    if result.state_mentioned and (target is None or target.region is None):
        warnings.append(f"{result.state_mentioned} was mentioned, but states are not mapped to "
                        f"regions yet, so geography was not used.")
    return warnings


def intake_fallback(text: str, cfg: Config) -> IntakeOutcome:
    """Used when the model is unavailable or keeps failing. The default request
    has a known answer (default_target in config.yaml); any other request stops
    the run rather than guess a target."""
    if text.strip() == cfg.default_banker_text.strip():
        return IntakeOutcome(target=cfg.default_target, stop_reason=None, result=None,
                             warnings=["Intake model unavailable: used the configured default target."])
    return IntakeOutcome(target=None, result=None, warnings=[],
                         stop_reason="The request could not be read reliably. "
                                     "Please restate sector, deal size and profile.")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
