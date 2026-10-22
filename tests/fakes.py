"""Test helpers shared by several test files: correct agent answers built from
the real data, and a stand-in LLM client that answers every agent in a run."""

from acquirer_engine.agents.analyst import fallback_page
from acquirer_engine.config import load_config
from acquirer_engine.evidence import build_evidence
from acquirer_engine.graph import Run, build_graph
from acquirer_engine.llm import ChatTurn, LLMUnavailable
from acquirer_engine.schemas import (
    AnalystNotes,
    FinalSelection,
    IntakeResult,
    OnePagerDraft,
    PECoveragePool,
    StrategicCoveragePool,
)
from acquirer_engine.tools import CoverageTools, run_seed_theses

SAME = "Roll-up of physician groups with cross-sell into behavioral health clinics."


def short_page_config():
    """The real config, with only the page-length rule relaxed (fake pages are short)."""
    base = load_config()
    return base.model_copy(update={"validation": base.validation.model_copy(update={"page_words_min": 1})})


def intake_answer(cfg) -> IntakeResult:
    """The correct reading of the default request."""
    return IntakeResult(target=cfg.default_target, sources=[
        {"field": "sector", "provenance": "given", "quote": "Sector: Healthcare Services"},
        {"field": "ev_mm", "provenance": "given", "quote": "~$200M EV"},
        {"field": "ownership", "provenance": "derived", "quote": "private"},
        {"field": "margin_band", "provenance": "derived", "quote": "strong EBITDA margins"},
    ])


def seeded(analysis, buyer_type, cfg):
    """The type's tools after the seeds ran, the seeds, and {buyer: (thesis, deal IDs)}."""
    tools = CoverageTools(buyer_type, analysis, cfg)
    seeds = run_seed_theses(tools, analysis.target, cfg)
    found = {}
    for seed in seeds:
        for b in seed["result"]["buyers"]:
            found.setdefault(b["acquirer"], (seed["thesis_id"], b["top_deal_ids"]))
    return tools, seeds, found


def pool_from_seeds(analysis, cfg, buyer_type="PE", n=7, removals=(), **changes):
    """A valid pool built only from what the seed theses returned."""
    _, seeds, found = seeded(analysis, buyer_type, cfg)
    candidates = []
    for name, (thesis, ids) in list(found.items())[:n]:
        c = {"acquirer": name, "theses": [thesis], "evidence_ids": ids[:1], "note": "Fits.",
             "ranker_rank": int(analysis.rankings.type_rank[name]), "score": 1.0}
        if buyer_type == "Strategic":
            c |= {"range_90": (0.03, 0.34), "primary_sector_proxy": None}
        candidates.append(c)
    log = [{"thesis_id": s["thesis_id"], "kept": True, "reason": "ok"} for s in seeds]
    schema = PECoveragePool if buyer_type == "PE" else StrategicCoveragePool
    return schema.model_validate({"candidates": candidates, "removals": list(removals),
                                  "thesis_log": log, "reasoning_summary": "Built from seeds."} | changes)


class GraphClient:
    """Answers every agent in a run with a correct reply built from the real data:
    intake -> `intake`; coverage -> no tool calls, then a pool from the seeds;
    captain -> the first 6 PE and 4 strategic candidates; analysts -> notes and a
    page built from the pack.
    `fail`: exception raised for every call; `fail_agent`: one agent whose calls
    fail with LLMUnavailable; `bug_for`: buyer whose analyst call raises a
    ValueError (an unexpected error); `same_thesis`: buyers given identical theses."""

    def __init__(self, analysis, cfg, intake=None, fail=None, fail_agent=None, bug_for=None,
                 same_thesis=()):
        self.analysis, self.cfg, self.intake, self.fail = analysis, cfg, intake or intake_answer(cfg), fail
        self.fail_agent, self.bug_for, self.same_thesis = fail_agent, bug_for, same_thesis
        self.calls, self.usage, self.seen = [], [], {}   # seen: first message per call
        self.pools = {t: pool_from_seeds(analysis, cfg, t, n=n) for t, n in (("PE", 7), ("Strategic", 5))}

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


async def run_graph(client, deals, cfg=None, strict=True, text=None):
    """Run the whole graph with a fake client (no key, no network)."""
    cfg = cfg or client.cfg
    graph = build_graph(Run(cfg=cfg, deals=deals, client=client, strict=strict))
    return await graph.ainvoke({"banker_text": text or cfg.default_banker_text, "events": []})


def selection_from(pe_pool, strategic_pool, n_pe=6, n_strategic=4, shift_reason=None) -> FinalSelection:
    """The captain picking the first candidates of each pool."""
    picks = [{"acquirer": c.acquirer, "buyer_type": t, "theses": c.theses, "why_selected": "Strong fit.",
              "evidence_ids": c.evidence_ids}
             for t, pool, n in (("PE", pe_pool, n_pe), ("Strategic", strategic_pool, n_strategic))
             for c in pool.candidates[:n]]
    return FinalSelection(picks=picks, shift_reason=shift_reason)
