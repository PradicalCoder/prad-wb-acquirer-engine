"""Tests for the coverage agents with a scripted fake client (no key, no network)."""

import time

import pytest

from acquirer_engine.agents.coverage import coverage_fallback, run_coverage, validate_pool
from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze
from acquirer_engine.llm import ChatTurn, LLMOutputInvalid
from acquirer_engine.schemas import PECoveragePool, StrategicCoveragePool
from acquirer_engine.tools import CoverageTools, run_seed_theses

CFG = load_config()


@pytest.fixture(scope="module")
def analysis():
    return analyze(load_deals(CFG.paths.data_csv), CFG.default_target, CFG.scoring)


def seeded(analysis, buyer_type):
    """The type's tools after the seeds ran, and {buyer: (thesis, deal IDs)} from the seeds."""
    tools = CoverageTools(buyer_type, analysis, CFG)
    found = {}
    seeds = run_seed_theses(tools, CFG.default_target, CFG)
    for seed in seeds:
        for b in seed["result"]["buyers"]:
            found.setdefault(b["acquirer"], (seed["thesis_id"], b["top_deal_ids"]))
    return tools, seeds, found


def pool_from_seeds(analysis, buyer_type="PE", n=7, removals=(), **changes):
    """A valid pool built only from what the seeds returned."""
    _, seeds, found = seeded(analysis, buyer_type)
    candidates = []
    for name, (thesis, ids) in list(found.items())[:n]:
        c = {"acquirer": name, "theses": [thesis], "evidence_ids": ids[:1], "note": "Fits.",
             "ranker_rank": int(analysis.rankings.type_rank[name]), "score": 1.0}
        if buyer_type == "Strategic":
            c |= {"range_90": (0.03, 0.34), "primary_sector_proxy": None}
        candidates.append(c)
    log = [{"thesis_id": s["thesis_id"], "filters_used": s["filters"], "kept": True, "reason": "ok",
            "buyers_found": [b["acquirer"] for b in s["result"]["buyers"]]} for s in seeds]
    schema = PECoveragePool if buyer_type == "PE" else StrategicCoveragePool
    return schema.model_validate({"candidates": candidates, "removals": list(removals),
                                  "thesis_log": log, "reasoning_summary": "Built from seeds."} | changes)


class FakeClient:
    def __init__(self, turn, *answers):
        self.turn, self.answers, self.calls = turn, list(answers), []

    async def chat(self, **request):
        self.calls.append(("chat", request))
        return self.turn

    async def structured(self, **request):
        self.calls.append(("structured", request))
        reply = self.answers.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


NO_TOOLS = ChatTurn(text="Seeds look sufficient.", tool_calls=[])


async def cover(client, analysis, buyer_type="PE"):
    return await run_coverage(buyer_type, analysis, client, CFG, deadline=time.monotonic() + 20)


async def test_tool_results_reach_the_final_turn(analysis):
    turn = ChatTurn(text="Checking Veritas size.",
                    tool_calls=[{"name": "get_acquirer_profile", "args": {"name": "Veritas Capital"}}])
    client = FakeClient(turn, pool_from_seeds(analysis))
    result = await cover(client, analysis)
    assert len(result.names) == 7 and not result.fallback and result.topped_up == []
    final = client.calls[1][1]["messages"][0]["content"]
    assert "Checking Veritas size." in final and "typical_size_mm" in final   # profile result passed on
    assert "$100M" in client.calls[0][1]["system"][0]["text"]                  # size band filled from EV


async def test_invented_buyer_is_repaired(analysis):
    good = pool_from_seeds(analysis)
    invented = good.model_copy(update={"candidates": good.candidates[:6] + [
        good.candidates[0].model_copy(update={"acquirer": "Imaginary Partners"})]})
    client = FakeClient(NO_TOOLS, invented, good)
    assert not (await cover(client, analysis)).fallback
    assert "did not appear" in client.calls[2][1]["messages"][-1]["content"]


async def test_still_invalid_raises_and_fallback_uses_the_ranking(analysis):
    bad = pool_from_seeds(analysis, thesis_log=pool_from_seeds(analysis).thesis_log[:1])
    with pytest.raises(LLMOutputInvalid):
        await cover(FakeClient(NO_TOOLS, bad, bad), analysis)
    fallback = coverage_fallback("PE", analysis, CFG)
    assert fallback.fallback and fallback.names[:3] == ["Blackstone", "New Mountain Capital", "KKR"]


async def test_thin_pool_is_topped_up_from_the_ranking(analysis):
    removal = {"acquirer": "Blackstone", "reason": "test", "evidence_ids": ["MA-2021-0195"]}
    thin = pool_from_seeds(analysis, n=5, removals=[removal])
    picked = {c.acquirer for c in thin.candidates}
    if "Blackstone" in picked:
        pytest.skip("seed order put Blackstone among the candidates")
    result = await cover(FakeClient(NO_TOOLS, thin), analysis)
    assert len(result.names) == 7 and len(result.topped_up) == 2
    assert "Blackstone" not in result.topped_up            # a removed buyer is never added back


def test_validator_catches_wrong_ids_ranks_and_missing_theses(analysis):
    tools, _, _ = seeded(analysis, "PE")
    pool = pool_from_seeds(analysis)
    c = pool.candidates[0]
    wrong = pool.model_copy(update={
        "candidates": [c.model_copy(update={"evidence_ids": ["MA-2019-0205"], "ranker_rank": 99}),
                       *pool.candidates[1:]],
        "thesis_log": pool.thesis_log[:2]})
    errors = " ".join(validate_pool(wrong, tools, analysis, CFG))
    assert "is a deal by KKR" in errors or c.acquirer == "KKR"
    assert "ranker_rank is" in errors and "missing: ['P3']" in errors


async def test_strategic_agent_can_remove_medtronic_with_its_profile(analysis):
    turn = ChatTurn(text="Profiling the device maker.",
                    tool_calls=[{"name": "get_acquirer_profile", "args": {"name": "Medtronic"}}])
    removal = {"acquirer": "Medtronic", "reason": "Device LBOs only.", "evidence_ids": ["MA-2022-0317"]}
    pool = pool_from_seeds(analysis, "Strategic", n=5, removals=[removal])
    result = await cover(FakeClient(turn, pool), analysis, "Strategic")
    assert "Medtronic" not in result.names


def test_wrong_sector_proxy_is_caught(analysis):
    tools, _, _ = seeded(analysis, "Strategic")
    pool = pool_from_seeds(analysis, "Strategic", n=5)
    first = pool.candidates[0].model_copy(update={"primary_sector_proxy": "Pharma/Biotech"})
    wrong = pool.model_copy(update={"candidates": [first, *pool.candidates[1:]]})
    assert any("primary_sector_proxy is" in e for e in validate_pool(wrong, tools, analysis, CFG))
