"""Tests for the analyst flow with a scripted fake LLM client (no key, no network)."""

import time

import pytest

from acquirer_engine.agents.analyst import fallback_page, run_analyst
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze, build_evidence
from acquirer_engine.llm import LLMOutputInvalid, LLMUnavailable
from acquirer_engine.schemas import AnalystNotes, OnePagerDraft
from acquirer_engine.validate import validate_page

CONFIG = load_config()
# Short test pages: relax only the length rule.
CFG = CONFIG.model_copy(update={"validation": CONFIG.validation.model_copy(
    update={"page_words_min": 1})})


@pytest.fixture(scope="module")
def pack():
    analysis = analyze(load_deals(CONFIG.paths.data_csv), CONFIG.default_target, CONFIG.scoring)
    return build_evidence("KKR", analysis)


@pytest.fixture
def notes(pack):
    return AnalystNotes(
        thesis="Physician-group platform buyer.", featured_deal_ids=[pack.top_deals[0].transaction_id],
        key_numbers=[{"label": "paid", "value": 2.53, "source": "paid.ev_revenue"}],
        risks_to_feature=[r.name for r in pack.risks[:2]],
        conviction_adjustment={"direction": "none"},
    )


@pytest.fixture
def good(pack):
    """A page that passes every check: the evidence-only page, as a draft."""
    return OnePagerDraft(**fallback_page(pack).model_dump(exclude={"needs_review"}))


class FakeClient:
    """Answers come from a script, in order: a model to return, or an exception to raise."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    async def structured(self, **request):
        self.calls.append(request)
        reply = self.script.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


async def run(pack, client):
    return await run_analyst(pack, client, CFG, deadline=time.monotonic() + 30)


def test_fallback_page_passes_the_checks(pack, notes, good):
    assert validate_page(good, pack, notes, CFG.validation) == []


async def test_happy_path(pack, notes, good):
    client = FakeClient(notes, good)
    result = await run(pack, client)
    assert (result.fallback, result.repairs, result.page.needs_review) == (False, 0, False)
    assert [c["profile"] for c in client.calls] == ["analyst_notes", "analyst_write"]
    write_prompt = client.calls[1]["system"][0]["text"]
    assert '"well-positioned"' in write_prompt          # banned phrases come from config


async def test_failed_check_is_repaired(pack, notes, good):
    invented = good.model_copy(update={"valuation_note": "Expect pricing near 16.2x EBITDA."})
    client = FakeClient(notes, invented, good)
    result = await run(pack, client)
    assert (result.fallback, result.repairs) == (False, 1)
    repair_request = client.calls[2]["messages"][-1]["content"]
    assert "16.2" in repair_request                     # the model is told exactly what to fix


async def test_malformed_answer_is_repaired(pack, notes, good):
    client = FakeClient(notes, LLMOutputInvalid("missing field 'risks'", raw_text="{}"), good)
    result = await run(pack, client)
    assert (result.fallback, result.repairs) == (False, 1)


async def test_gives_up_after_max_repairs(pack, notes, good):
    invented = good.model_copy(update={"valuation_note": "Expect pricing near 16.2x EBITDA."})
    client = FakeClient(notes, *[invented] * (CFG.validation.max_repairs + 1))
    result = await run(pack, client)
    assert result.fallback and result.page.needs_review
    assert any("16.2" in e for e in result.errors)


async def test_unavailable_model_falls_back(pack):
    result = await run(pack, FakeClient(LLMUnavailable("out of time")))
    assert result.fallback and result.notes is None and result.page.needs_review
