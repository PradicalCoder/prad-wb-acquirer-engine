"""Tests for the run graph: routing, fallbacks, error handling, set check, budget.
A fake client answers by agent and profile; no key, no network."""

import time

import pytest
from fakes import intake_answer, pool_from_seeds, selection_from

from acquirer_engine.agents.analyst import fallback_page
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze, build_evidence
from acquirer_engine.graph import Run, build_graph
from acquirer_engine.cache import Mode
from acquirer_engine.llm import CallRecord, ChatTurn, LLMUnavailable
from acquirer_engine.run import report_markdown, run_report
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


class FakeClient:
    """Answers every agent with a correct reply built from the real data:
    intake -> `intake`; coverage -> no tool calls, then a pool from the seeds;
    captain -> the first 6 PE and 4 strategic candidates; analysts -> notes and a
    page built from the pack.
    `fail`: exception raised for every call; `fail_agent`: one agent whose calls
    fail with LLMUnavailable; `bug_for`: buyer whose analyst call raises a
    ValueError (an unexpected error); `same_thesis`: buyers given identical theses."""

    def __init__(self, analysis, intake=None, fail=None, fail_agent=None, bug_for=None,
                 same_thesis=()):
        self.analysis, self.intake, self.fail = analysis, intake or intake_answer(CFG), fail
        self.fail_agent, self.bug_for, self.same_thesis = fail_agent, bug_for, same_thesis
        self.calls, self.usage, self.seen = [], [], {}   # seen: first message per call
        self.pools = {t: pool_from_seeds(analysis, CFG, t, n=n) for t, n in (("PE", 7), ("Strategic", 5))}

    async def chat(self, *, agent, **_):
        self.calls.append((agent, "chat"))
        self._maybe_fail(agent)
        return ChatTurn(text="Seeds are enough.", tool_calls=[])

    def _maybe_fail(self, agent):
        if self.fail:
            raise self.fail
        if agent == self.fail_agent:
            raise LLMUnavailable(f"{agent} unavailable")

    async def structured(self, *, agent, profile, messages, **_):
        self.calls.append((agent, profile))
        self.seen[(agent, profile)] = messages[0]["content"]
        self._maybe_fail(agent)
        if profile == "intake":
            return self.intake
        if profile == "coverage":
            return self.pools[agent.split(":", 1)[1]]
        if profile == "captain":
            return selection_from(self.pools["PE"], self.pools["Strategic"])
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
    client = FakeClient(analysis)
    state = await run_graph(client, deals)
    assert len(state["final_order"]) == 10
    assert not any(r.fallback for r in state["results"].values())
    s = statuses(state)
    assert s["intake"] == s["pe_coverage"] == s["captain"] == s["analyst:Blackstone"] == "ok"
    levels = [state["results"][b].page.conviction.level for b in state["final_order"]]
    rank = {"High": 0, "Medium": 1, "Low": 2}
    assert levels == sorted(levels, key=rank.get)        # conviction never rises down the list
    assert set(state["final_order"]) == {p.acquirer for p in
                                         selection_from(client.pools["PE"], client.pools["Strategic"]).picks}


async def test_agents_change_the_list_and_say_why(analysis, deals):
    state = await run_graph(FakeClient(analysis), deals)
    added = {d["buyer"] for d in state["baseline_diff"] if d["change"] == "added"}
    assert "Welsh Carson" in added                       # surfaced by a thesis, not by the ranker
    assert all(d["reason"] for d in state["baseline_diff"])


async def test_coverage_notes_reach_the_analysts(analysis, deals):
    client = FakeClient(analysis)
    await run_graph(client, deals)
    # The analyst's notes call sees the thesis, the coverage note and the captain's reason.
    briefing = client.seen[("analyst:Blackstone", "analyst_notes")]
    assert '"theses":["P1: Platform builders"]' in briefing or '"theses":["P2: Add-on acquirers"]' in briefing \
        or '"theses":["P3: Roll-up players"]' in briefing
    assert '"coverage_note":"Fits."' in briefing and '"captain_reason":"Strong fit."' in briefing


async def test_run_report_shows_public_knowledge_and_tokens_per_agent(analysis, deals):
    client = FakeClient(analysis)
    client.pools["Strategic"] = client.pools["Strategic"].model_copy(
        update={"public_knowledge": ["Medtronic is a medical device maker."]})
    state = await run_graph(client, deals)
    client.usage = [CallRecord(agent=a, profile="coverage", model="m", input_tokens=n, output_tokens=1,
                               cache_read_tokens=0, cache_hit=False, attempts=1, seconds=1.0)
                    for a, n in (("coverage:PE", 100), ("coverage:Strategic", 200))]
    report = run_report(state, client, Mode.off, seconds=1.0)
    assert report["public_knowledge_in_screening"] == ["Strategic: Medtronic is a medical device maker."]
    assert report["tokens_by_agent"]["coverage:Strategic"]["input"] == 200
    assert report["tokens_by_call_type"]["coverage"]["input"] == 300
    assert "Public knowledge used in screening" in report_markdown(state, report)


async def test_one_coverage_agent_failing_uses_its_ranking(analysis, deals):
    state = await run_graph(FakeClient(analysis, fail_agent="coverage:Strategic"), deals)
    s = statuses(state)
    assert s["strategic_coverage"] == "fallback" and s["pe_coverage"] == "ok"
    assert state["pool_strategic"].fallback and len(state["final_order"]) == 10


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
        await run_graph(FakeClient(analysis, bug_for="Blackstone"), deals, strict=True)


async def test_unexpected_error_is_contained_when_not_strict(analysis, deals):
    state = await run_graph(FakeClient(analysis, bug_for="Blackstone"), deals, strict=False)
    assert statuses(state)["analyst:Blackstone"] == "error"
    assert state["results"]["Blackstone"].page.needs_review
    assert not state["results"]["New Mountain Capital"].fallback   # the others were not cancelled


async def test_similar_theses_rewrite_the_lower_ranked(analysis, deals):
    client = FakeClient(analysis, same_thesis=("New Mountain Capital", "Blackstone"))
    state = await run_graph(client, deals)
    # Blackstone comes after New Mountain in the selection, so Blackstone is rewritten.
    assert state["results"]["Blackstone"].page.fit_thesis == "Blackstone specific angle."
    assert state["results"]["New Mountain Capital"].page.fit_thesis == SAME
    assert state["flags"] == []


async def test_similar_theses_are_flagged_without_time(analysis, deals):
    no_time = CFG.model_copy(update={"validation": CFG.validation.model_copy(
        update={"rewrite_min_seconds": 1e6})})
    client = FakeClient(analysis, same_thesis=("New Mountain Capital", "Blackstone"))
    state = await run_graph(client, deals, cfg=no_time)
    assert len(state["flags"]) == 1 and "Blackstone" in state["flags"][0]


def test_deadline_is_capped_by_run_budget(deals):
    run = Run(cfg=CFG, deals=deals, client=None)
    end_of_budget = run.started + CFG.llm.run_budget_seconds
    assert run.deadline(10) <= time.monotonic() + 10
    assert run.deadline(10_000) == pytest.approx(end_of_budget)
