"""Test helpers shared by several test files: correct agent answers built from
the real data, and a scripted stand-in for the LLM client."""

from acquirer_engine.schemas import FinalSelection, IntakeResult, PECoveragePool, StrategicCoveragePool
from acquirer_engine.tools import CoverageTools, run_seed_theses


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


def selection_from(pe_pool, strategic_pool, n_pe=6, n_strategic=4, shift_reason=None) -> FinalSelection:
    """The captain picking the first candidates of each pool."""
    picks = [{"acquirer": c.acquirer, "buyer_type": t, "theses": c.theses, "why_selected": "Strong fit.",
              "evidence_ids": c.evidence_ids}
             for t, pool, n in (("PE", pe_pool, n_pe), ("Strategic", strategic_pool, n_strategic))
             for c in pool.candidates[:n]]
    return FinalSelection(picks=picks, shift_reason=shift_reason)
