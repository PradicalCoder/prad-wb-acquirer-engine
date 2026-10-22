"""Tests for intake: quote checks, repair, warnings, stop and fallback (fake client)."""

import time

import pytest

from acquirer_engine.agents.intake import intake_fallback, run_intake
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.llm import LLMOutputInvalid
from acquirer_engine.schemas import IntakeResult

CONFIG = load_config()
TEXT = CONFIG.default_banker_text


@pytest.fixture(scope="module")
def deals():
    return load_deals(CONFIG.paths.data_csv)


def answer(**changes):
    """The correct reading of the default request."""
    fields = {
        "target": CONFIG.default_target.model_dump(),
        "sources": SOURCES,
    }
    return IntakeResult.model_validate(fields | changes)


SOURCES = [
    {"field": "sector", "provenance": "given", "quote": "Sector: Healthcare Services"},
    {"field": "ev_mm", "provenance": "given", "quote": "~$200M EV"},
    {"field": "ownership", "provenance": "derived", "quote": "private"},
    {"field": "margin_band", "provenance": "derived", "quote": "strong EBITDA margins"},
]


def with_quote(field, quote):
    """SOURCES with one field's quote replaced."""
    return [s | {"quote": quote} if s["field"] == field else s for s in SOURCES]


class FakeClient:
    def __init__(self, *script):
        self.script, self.calls = list(script), []

    async def structured(self, **request):
        self.calls.append(request)
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def intake(client, deals, text=TEXT):
    return await run_intake(text, client, CONFIG, deals, deadline=time.monotonic() + 10)


async def test_reads_the_default_request(deals):
    outcome = await intake(FakeClient(answer()), deals)
    assert outcome.target == CONFIG.default_target
    assert outcome.warnings == [] and outcome.stop_reason is None


async def test_invented_quote_is_repaired(deals):
    invented = answer(sources=with_quote("margin_band", "best-in-class margins"))
    client = FakeClient(invented, answer())
    outcome = await intake(client, deals)
    assert outcome.target == CONFIG.default_target
    assert "best-in-class margins" in client.calls[1]["messages"][-1]["content"]


async def test_still_wrong_after_repair_raises(deals):
    invented = answer(sources=with_quote("ev_mm", "$250M"))
    with pytest.raises(LLMOutputInvalid):
        await intake(FakeClient(invented, invented), deals)


async def test_stop_reason_passes_through(deals):
    stop = IntakeResult(stop_reason="Veterinary clinics are not a sector in our deal data.")
    outcome = await intake(FakeClient(stop), deals, text="a $300M veterinary clinic group")
    assert outcome.target is None and "Veterinary" in outcome.stop_reason


async def test_warnings(deals):
    text = "a $20B healthcare services company in Texas"
    huge = answer(
        target=CONFIG.default_target.model_copy(update={"ev_mm": 20000.0, "ownership": None,
                                                         "margin_band": None}).model_dump(),
        sources=[{"field": "sector", "provenance": "given", "quote": "healthcare services"},
                 {"field": "ev_mm", "provenance": "given", "quote": "$20B"}],
        state_mentioned="Texas",
    )
    outcome = await intake(FakeClient(huge), deals, text=text)
    assert any("outside the data's range" in w for w in outcome.warnings)
    assert any("Texas" in w for w in outcome.warnings)


def test_fallback_uses_default_target_only_for_the_default_request():
    assert intake_fallback(TEXT, CONFIG).target == CONFIG.default_target
    other = intake_fallback("a $90M dental group", CONFIG)
    assert other.target is None and other.stop_reason
