"""Tests for the deal captain: pick checks, repair, fallback, baseline diff."""

import time

import pytest
from fakes import pool_from_seeds, selection_from

from acquirer_engine.agents.captain import (
    alternates,
    baseline_diff,
    captain_fallback,
    run_captain,
    validate_selection,
)
from acquirer_engine.agents.coverage import CoverageResult
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze
from acquirer_engine.llm import LLMOutputInvalid

CFG = load_config()


@pytest.fixture(scope="module")
def analysis():
    return analyze(load_deals(CFG.paths.data_csv), CFG.default_target, CFG.scoring)


@pytest.fixture(scope="module")
def pools(analysis):
    removal = {"acquirer": "Medtronic", "reason": "Device buyouts only.", "evidence_ids": ["MA-2022-0317"]}
    pe = pool_from_seeds(analysis, CFG, "PE", n=8)
    strategic = pool_from_seeds(analysis, CFG, "Strategic", n=6, removals=[removal])
    return {t: CoverageResult(buyer_type=t, pool=p, names=[c.acquirer for c in p.candidates],
                              topped_up=[], tool_calls=[], fallback=False)
            for t, p in (("PE", pe), ("Strategic", strategic))}


def default(pools, **kw):
    return selection_from(pools["PE"].pool, pools["Strategic"].pool, **kw)


class FakeClient:
    def __init__(self, *answers):
        self.answers, self.calls = list(answers), []

    async def structured(self, **request):
        self.calls.append(request)
        return self.answers.pop(0)


def errors(selection, pools, analysis):
    return " ".join(validate_selection(selection, pools, analysis, CFG))


def test_default_mix_passes(pools, analysis):
    assert validate_selection(default(pools), pools, analysis, CFG) == []


def test_shift_by_one_needs_a_reason(pools, analysis):
    assert "shift_reason" in errors(default(pools, n_pe=7, n_strategic=3), pools, analysis)
    assert errors(default(pools, n_pe=7, n_strategic=3, shift_reason="4th strategic is thin."),
                  pools, analysis) == ""


def test_shift_by_two_is_rejected(pools, analysis):
    text = errors(default(pools, n_pe=8, n_strategic=2, shift_reason="x"), pools, analysis)
    assert "at most one" in text and "at least 3" in text


def test_pick_outside_pools_and_wrong_ids(pools, analysis):
    s = default(pools)
    first = s.picks[0]
    outsider = first.model_copy(update={"acquirer": "Medtronic", "buyer_type": "Strategic"})
    wrong_ids = s.picks[1].model_copy(update={"evidence_ids": ["MA-2019-0205"]})  # a KKR deal
    bad = s.model_copy(update={"picks": [outsider, wrong_ids, *s.picks[2:]]})
    text = errors(bad, pools, analysis)
    assert "not in either pool" in text
    assert "MA-2019-0205 is not one of its deals" in text or s.picks[1].acquirer == "KKR"


async def test_repair_then_pass(pools, analysis):
    client = FakeClient(default(pools, n_pe=7, n_strategic=3), default(pools))
    result = await run_captain(pools, analysis, client, CFG, deadline=time.monotonic() + 10)
    assert not result.fallback and len(result.names) == 10
    assert "shift_reason" in client.calls[1]["messages"][-1]["content"]


async def test_still_wrong_raises(pools, analysis):
    wrong = default(pools, n_pe=7, n_strategic=3)
    with pytest.raises(LLMOutputInvalid):
        await run_captain(pools, analysis, FakeClient(wrong, wrong), CFG, time.monotonic() + 10)


def test_fallback_takes_the_default_mix_in_ranking_order(pools, analysis):
    result = captain_fallback(pools, analysis, CFG)
    assert result.fallback and len(result.names) == 10
    pe = result.names[:6]
    assert pe == sorted(pe, key=lambda b: analysis.rankings.type_rank[b])


def test_baseline_diff_explains_every_change(pools, analysis):
    result = captain_fallback(pools, analysis, CFG)
    diff = baseline_diff(result.names, pools, result, analysis, CFG)
    dropped = {d["buyer"]: d["reason"] for d in diff if d["change"] == "dropped"}
    assert dropped["Medtronic"].startswith("removed by coverage: Device buyouts")
    assert all(d["reason"] for d in diff)
    assert not set(alternates(result.names, pools, analysis)) & set(result.names)
