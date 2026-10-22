"""The deal captain: picks the final 10 from the two coverage pools.

One structured call. Code then checks the picks (validate_selection) and gives
one repair. If the captain is unavailable or still wrong, captain_fallback()
takes the default mix from each pool in ranking order.

Code, not the captain, also works out how the final list differs from the
plain ranking (baseline_diff), so every judgment call is visible in the report.
"""

import structlog
from pydantic import BaseModel

from acquirer_engine.agents import cached_system, load_prompt, repair_turn
from acquirer_engine.agents.coverage import CoverageResult
from acquirer_engine.config import Config
from acquirer_engine.data import BuyerType
from acquirer_engine.evidence import Analysis
from acquirer_engine.llm import LLMClient, LLMOutputInvalid
from acquirer_engine.schemas import FinalSelection
from acquirer_engine.scoring.buyers import default_pick
from acquirer_engine.tools import to_json

log = structlog.get_logger()


class CaptainResult(BaseModel):
    selection: FinalSelection | None   # the captain's answer (None on fallback)
    names: list[str]                   # the final 10
    fallback: bool


async def run_captain(pools: dict[BuyerType, CoverageResult], analysis: Analysis,
                      client: LLMClient, cfg: Config, deadline: float) -> CaptainResult:
    """Raises LLMUnavailable, or LLMOutputInvalid if the picks still fail their
    checks after one repair; the graph then calls captain_fallback()."""
    s = cfg.scoring
    version, system = load_prompt(
        "captain", total=s.mix_pe + s.mix_strategic, mix_pe=s.mix_pe, mix_strategic=s.mix_strategic,
        pe_up=s.mix_pe + 1, st_down=s.mix_strategic - 1, pe_down=s.mix_pe - 1,
        st_up=s.mix_strategic + 1, min_per_type=s.min_per_type)
    briefing = to_json({"target": analysis.target.model_dump(exclude_none=True),
                        "pools": {t: pool_briefing(pools[t], analysis) for t in ("PE", "Strategic")}})
    messages = [{"role": "user", "content": f"Briefing:\n{briefing}\n\nPick the final list."}]
    ask = dict(agent="captain", profile="captain", system=cached_system(system),
               schema=FinalSelection, prompt_version=version, deadline=deadline)

    for attempt in range(2):  # the first answer, then one repair
        try:
            selection = await client.structured(messages=messages, **ask)
            raw, errors = selection.model_dump_json(), validate_selection(selection, pools, analysis, cfg)
        except LLMOutputInvalid as error:
            raw, errors = error.raw_text, [str(error)]
        if not errors:
            return CaptainResult(selection=selection, names=[p.acquirer for p in selection.picks],
                                 fallback=False)
        log.info("captain.repair", errors=errors)
        messages = messages + repair_turn(raw, errors)
    raise LLMOutputInvalid("; ".join(errors), raw_text=raw)


def pool_briefing(pool: CoverageResult, analysis: Analysis) -> list[dict]:
    """What the captain sees about each candidate in one pool."""
    agent_picks = {c.acquirer: c for c in pool.pool.candidates} if pool.pool else {}
    rows = []
    for name in pool.names:
        r, c = analysis.rankings.loc[name], agent_picks.get(name)
        top_ids = list(analysis.scored[analysis.scored.acquirer == name]
                       .nlargest(3, "relevance").transaction_id)
        row = {"acquirer": name, "buyer_type": r.buyer_type, "rank": int(r.type_rank),
               "score": round(r.score, 3), "strong_deals": int(r.strong_deals),
               "best_relevance": round(r.best, 3),
               "evidence_ids": c.evidence_ids if c else top_ids,
               "theses": c.theses if c else [], "note": c.note if c else None,
               "topped_up": name in pool.topped_up}
        if getattr(c, "caveat", None):
            row["caveat"] = c.caveat
        rows.append(row)
    return rows


def validate_selection(selection: FinalSelection, pools: dict[BuyerType, CoverageResult],
                       analysis: Analysis, cfg: Config) -> list[str]:
    """Problems the schema can't see; empty list = the picks pass."""
    errors = []
    s = cfg.scoring
    in_pools = {name: t for t, pool in pools.items() for name in pool.names}
    deals = analysis.scored.set_index("transaction_id")

    for pick in selection.picks:
        name = pick.acquirer
        if name not in in_pools:
            errors.append(f"{name} is not in either pool; pick only from the pools.")
            continue
        if pick.buyer_type != in_pools[name]:
            errors.append(f"{name} is a {in_pools[name]} buyer, not {pick.buyer_type}.")
        for deal_id in pick.evidence_ids:
            if deal_id not in deals.index or deals.at[deal_id, "acquirer"] != name:
                errors.append(f"{name}: {deal_id} is not one of its deals.")

    mix = selection.mix
    if abs(mix["PE"] - s.mix_pe) > 1:
        errors.append(f"the mix may move at most one from {s.mix_pe} PE / {s.mix_strategic} "
                      f"strategic; you have {mix['PE']} / {mix['Strategic']}.")
    if min(mix.values()) < s.min_per_type:
        errors.append(f"keep at least {s.min_per_type} buyers of each type.")
    if mix["PE"] != s.mix_pe and not selection.shift_reason:
        errors.append("you changed the mix: give the reason in shift_reason.")
    return errors


def captain_fallback(pools: dict[BuyerType, CoverageResult], analysis: Analysis,
                     cfg: Config) -> CaptainResult:
    """The default mix, best-ranked first, from each pool."""
    def best(buyer_type: BuyerType, n: int) -> list[str]:
        return sorted(pools[buyer_type].names, key=lambda b: analysis.rankings.type_rank[b])[:n]
    names = best("PE", cfg.scoring.mix_pe) + best("Strategic", cfg.scoring.mix_strategic)
    return CaptainResult(selection=None, names=names, fallback=True)


def baseline_diff(final: list[str], pools: dict[BuyerType, CoverageResult],
                  result: CaptainResult, analysis: Analysis, cfg: Config) -> list[dict]:
    """How the final list differs from the plain ranking's default pick, and why.
    This is the record of what the agents changed."""
    baseline = default_pick(analysis.rankings, cfg.scoring)
    why = {p.acquirer: p.why_selected for p in result.selection.picks} if result.selection else {}
    removed = {r.acquirer: r.reason for pool in pools.values() if pool.pool
               for r in pool.pool.removals}
    diff = [{"buyer": b, "change": "added", "reason": why.get(b, "fallback pick")}
            for b in final if b not in baseline]
    for b in baseline:
        if b not in final:
            reason = (f"removed by coverage: {removed[b]}" if b in removed
                      else "not in the coverage pool" if not any(b in p.names for p in pools.values())
                      else "not picked by the captain")
            diff.append({"buyer": b, "change": "dropped", "reason": reason})
    return diff


def alternates(final: list[str], pools: dict[BuyerType, CoverageResult], analysis: Analysis,
               n: int = 3) -> list[str]:
    """The best-ranked pool members not picked: who would come next."""
    left = [b for pool in pools.values() for b in pool.names if b not in final]
    return sorted(left, key=lambda b: (analysis.rankings.type_rank[b], b))[:n]
