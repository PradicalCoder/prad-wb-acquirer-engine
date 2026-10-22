"""Tests for the run graph: routing, fallbacks, error handling, set check, budget.
A fake client answers by agent and profile; no key, no network."""

import time

import pytest

from acquirer_engine.agents.analyst import fallback_page
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze, build_evidence
from acquirer_engine.graph import Run, build_graph
from acquirer_engine.llm import LLMUnavailable
from acquirer_engine.schemas import AnalystNotes, IntakeResult, OnePagerDraft

BASE = load_config()
# Fake pages are short: relax only the length rule.
CFG = BASE.model_copy(update={"validation": BASE.validation.model_copy(update={"page_words_min": 1})})
TEXT = CFG.default_banker_text
SAME = "Roll-up of physician groups with cross-sell into behavioral health clinics."


@pytest.fixture(scope="module")
def deals():
    return load_deals(CFG.paths.data_csv)


@pytest.fixture(scope="module")
def analysis(deals):
    return analyze(deals, CFG.default_target, CFG.scoring)


def intake_answer():
    return IntakeResult(target=CFG.default_target, sources=[
        {"field": "sector", "provenance": "given", "quote": "Sector: Healthcare Services"},
        {"field": "ev_mm", "provenance": "given", "quote": "~$200M EV"},
        {"field": "ownership", "provenance": "derived", "quote": "private"},
        {"field": "margin_band", "provenance": "derived", "quote": "strong EBITDA margins"},
    ])


class FakeClient:
    """intake -> `intake`; analysts -> notes and a page built from the pack.
    `fail`: exception raised for every call; `bug_for`: buyer whose call raises a
    ValueError (an unexpected error); `same_thesis`: buyers given identical theses."""

    def __init__(self, analysis, intake=None, fail=None, bug_for=None, same_thesis=()):
        self.analysis, self.intake, self.fail = analysis, intake or intake_answer(), fail
        self.bug_for, self.same_thesis, self.calls, self.usage = bug_for, same_thesis, [], []

    async def structured(self, *, agent, profile, messages, **_):
        self.calls.append((agent, profile))
        if self.fail:
            raise self.fail
        if profile == "intake":
            return self.intake
        buyer = agent.split(":", 1)[1]
        if buyer == self.bug_for:
            raise ValueError("a bug")
        pack = build_evidence(buyer, self.analysis)
        if profile == "analyst_notes":
            return AnalystNotes(thesis="t", featured_deal_ids=[pack.top_deals[0].transaction_id],
                                key_numbers=[{"label": "x", "value": 1, "source": "profile.deals"}],
                                risks_to_feature=[r.name for r in pack.risks[:2]],
                                conviction_adjustment={"direction": "none"})
        page = OnePagerDraft(**fallback_page(pack).model_dump(exclude={"needs_review"}))
        rewrite_asked = "too much like" in messages[0]["content"]
        thesis = SAME if buyer in self.same_thesis and not rewrite_asked else f"{buyer} specific angle."
        return page.model_copy(update={"fit_thesis": thesis})


async def run_graph(client, deals, cfg=CFG, strict=True, text=TEXT):
    graph = build_graph(Run(cfg=cfg, deals=deals, client=client, strict=strict))
    return await graph.ainvoke({"banker_text": text, "events": []})


def statuses(state):
    return {e["node"]: e["status"] for e in state["events"]}


async def test_happy_path(analysis, deals):
    state = await run_graph(FakeClient(analysis), deals)
    assert len(state["final_order"]) == 10
    assert not any(r.fallback for r in state["results"].values())
    s = statuses(state)
    assert s["intake"] == "ok" and s["pe_coverage"] == "rule_based" and s["analyst:KKR"] == "ok"
    levels = [state["results"][b].page.conviction.level for b in state["final_order"]]
    rank = {"High": 0, "Medium": 1, "Low": 2}
    assert levels == sorted(levels, key=rank.get)        # conviction never rises down the list


async def test_unsupported_request_stops(analysis, deals):
    client = FakeClient(analysis, intake=IntakeResult(stop_reason="Veterinary is not in the data."))
    state = await run_graph(client, deals, text="a $300M veterinary group")
    assert "final_order" not in state
    assert client.calls == [("intake", "intake")]          # nothing after intake ran


async def test_api_down_still_finishes(analysis, deals):
    state = await run_graph(FakeClient(analysis, fail=LLMUnavailable("API down")), deals)
    assert statuses(state)["intake"] == "fallback"          # default request -> default target
    assert len(state["final_order"]) == 10
    assert all(r.page.needs_review for r in state["results"].values())


async def test_unexpected_error_raises_when_strict(analysis, deals):
    with pytest.raises(ValueError):
        await run_graph(FakeClient(analysis, bug_for="KKR"), deals, strict=True)


async def test_unexpected_error_is_contained_when_not_strict(analysis, deals):
    state = await run_graph(FakeClient(analysis, bug_for="KKR"), deals, strict=False)
    assert statuses(state)["analyst:KKR"] == "error"
    assert state["results"]["KKR"].page.needs_review
    assert not state["results"]["Blackstone"].fallback     # the others were not cancelled


async def test_similar_theses_rewrite_the_lower_ranked(analysis, deals):
    client = FakeClient(analysis, same_thesis=("Blackstone", "KKR"))
    state = await run_graph(client, deals)
    # KKR is after Blackstone in the selection, so KKR is rewritten.
    assert state["results"]["KKR"].page.fit_thesis == "KKR specific angle."
    assert state["results"]["Blackstone"].page.fit_thesis == SAME
    assert state["flags"] == []


async def test_similar_theses_are_flagged_without_time(analysis, deals):
    no_time = CFG.model_copy(update={"validation": CFG.validation.model_copy(
        update={"rewrite_min_seconds": 1e6})})
    state = await run_graph(FakeClient(analysis, same_thesis=("Blackstone", "KKR")), deals, cfg=no_time)
    assert len(state["flags"]) == 1 and "Blackstone" in state["flags"][0]


def test_deadline_is_capped_by_run_budget(deals):
    run = Run(cfg=CFG, deals=deals, client=None)
    end_of_budget = run.started + CFG.llm.run_budget_seconds
    assert run.deadline(10) <= time.monotonic() + 10
    assert run.deadline(10_000) == pytest.approx(end_of_budget)
