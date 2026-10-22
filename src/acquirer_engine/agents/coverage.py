"""The PE and strategic coverage agents: build each buyer type's candidate pool.

One function serves both; the buyer type picks the prompt, the output schema
and the tool set (locked to that type, see tools.py). A run, per type:

  1. Seed theses run in code (search_deals), plus the type's ranking page.
  2. Turn 1, chat(): the agent reads the briefing and asks for tool calls
     (profiles, refined searches), all at once. Code runs them.
  3. Turn 2, structured(): one message with the briefing, the agent's turn-1
     note and every tool result -> the final pool.
  4. Code checks the pool (validate_pool); one repair with the errors.
  5. If fewer candidates than the pool minimum fit, code tops up from the
     ranking (logged). If the agent is unavailable or the pool still fails,
     the run uses coverage_fallback(): the ranking's top buyers.
"""

import json

import structlog
from pydantic import BaseModel

from acquirer_engine.agents import cached_system, load_prompt, repair_turn
from acquirer_engine.config import Config
from acquirer_engine.data import BuyerType
from acquirer_engine.evidence import Analysis
from acquirer_engine.llm import LLMClient, LLMOutputInvalid
from acquirer_engine.schemas import PECoveragePool, StrategicCoveragePool
from acquirer_engine.tools import CoverageTools, run_seed_theses, to_json

log = structlog.get_logger()

SCHEMAS = {"PE": PECoveragePool, "Strategic": StrategicCoveragePool}
PROMPTS = {"PE": "coverage_pe", "Strategic": "coverage_strategic"}

STEP_1 = ("Step 1: ask for the tool calls you need to check this briefing (all at once). "
          "If nothing needs checking, say so briefly.")
STEP_2 = "Step 2: return the final pool."


class CoverageResult(BaseModel):
    buyer_type: BuyerType
    pool: PECoveragePool | StrategicCoveragePool | None   # the agent's answer (None on fallback)
    names: list[str]          # the final pool: the agent's candidates, then any top-ups
    topped_up: list[str]      # buyers added from the ranking to reach the pool minimum
    tool_calls: list[dict]    # what the agent asked for in turn 1, for the log
    thesis_log: list[dict] = []   # each thesis: filters, buyers found, kept, reason
    fallback: bool


async def run_coverage(buyer_type: BuyerType, analysis: Analysis, client: LLMClient,
                       cfg: Config, deadline: float) -> CoverageResult:
    """Raises LLMUnavailable, or LLMOutputInvalid if the pool still fails its
    checks after one repair; the graph then calls coverage_fallback()."""
    agent, schema = f"coverage:{buyer_type}", SCHEMAS[buyer_type]
    tools = CoverageTools(buyer_type, analysis, cfg)
    low, high = cfg.coverage.size_far
    version, system = load_prompt(PROMPTS[buyer_type], pool_min=schema.POOL_MIN,
                                  pool_max=schema.POOL_MAX,
                                  size_low=f"{low * analysis.target.ev_mm:.0f}",
                                  size_high=f"{high * analysis.target.ev_mm:.0f}")
    seeds = run_seed_theses(tools, analysis.target, cfg)
    briefing = to_json({
        "target": analysis.target.model_dump(exclude_none=True),
        "seed_theses": seeds,
        "ranking": tools.get_buyer_rankings(cfg.coverage.rankings_top_k),
    })
    ask = dict(agent=agent, profile="coverage", system=cached_system(system),
               prompt_version=version, deadline=deadline)

    # Turn 1: the agent asks for tools; code runs them.
    turn = await client.chat(messages=[{"role": "user", "content": f"Briefing:\n{briefing}\n\n{STEP_1}"}],
                             tools=tools.definitions(), **ask)
    calls = [{"tool": c.name, "args": c.args, "result": json.loads(tools.call(c.name, c.args))}
             for c in turn.tool_calls[: cfg.coverage.max_tool_calls]]
    log.info("coverage.tools", agent=agent, calls=[(c["tool"], c["args"]) for c in calls])

    # Turn 2: everything gathered so far, in one message -> the final pool.
    messages = [{"role": "user", "content":
                 f"Briefing:\n{briefing}\n\nYour step 1 note: {turn.text or '(none)'}\n\n"
                 f"Your tool calls and their results:\n{to_json({'calls': calls})}\n\n{STEP_2}"}]
    for attempt in range(2):  # the first answer, then one repair
        try:
            pool = await client.structured(messages=messages, schema=schema, **ask)
            raw, errors = pool.model_dump_json(), validate_pool(pool, tools, analysis, cfg)
        except LLMOutputInvalid as error:
            raw, errors = error.raw_text, [str(error)]
        if not errors:
            names, topped_up = _top_up(pool, tools, schema.POOL_MIN)
            if topped_up:
                log.info("coverage.top_up", agent=agent, added=topped_up)
            return CoverageResult(buyer_type=buyer_type, pool=pool, names=names, topped_up=topped_up,
                                  tool_calls=[{"tool": c["tool"], "args": c["args"]} for c in calls],
                                  thesis_log=full_thesis_log(pool, seeds), fallback=False)
        log.info("coverage.repair", agent=agent, errors=errors)
        messages = messages + repair_turn(raw, errors)
    raise LLMOutputInvalid("; ".join(errors), raw_text=raw)


def validate_pool(pool, tools: CoverageTools, analysis: Analysis, cfg: Config) -> list[str]:
    """Problems with a pool that the schema can't see; empty list = it passes."""
    errors = []
    deals = analysis.scored.set_index("transaction_id")
    rank = analysis.rankings.type_rank

    seeds = {tid for tid, t in cfg.coverage.theses.items() if t.buyer_type == tools.buyer_type}
    missing = seeds - {entry.thesis_id for entry in pool.thesis_log}
    if missing:
        errors.append(f"record every seed thesis in thesis_log; missing: {sorted(missing)}.")

    for entry in [*pool.candidates, *pool.removals]:
        name = entry.acquirer
        if name not in tools.seen_buyers:
            errors.append(f"{name} did not appear in this run's seed or tool results; put forward "
                          f"or remove only buyers from those results.")
            continue
        for deal_id in entry.evidence_ids:
            if deal_id not in deals.index:
                errors.append(f"{name}: {deal_id} is not a deal in the data.")
            elif deals.at[deal_id, "acquirer"] != name:
                errors.append(f"{name}: {deal_id} is a deal by {deals.at[deal_id, 'acquirer']}, not {name}.")
            elif deal_id not in tools.seen_ids:
                errors.append(f"{name}: {deal_id} did not appear in your results; cite deals you were shown.")

    for c in pool.candidates:
        if c.acquirer in tools.seen_buyers and c.ranker_rank != int(rank[c.acquirer]):
            errors.append(f"{c.acquirer}: ranker_rank is {int(rank[c.acquirer])}, not {c.ranker_rank}.")
        proxy = getattr(c, "primary_sector_proxy", None)
        if proxy is not None:
            actual = tools.deals[tools.deals.acquirer == c.acquirer].sector.value_counts().index[0]
            if proxy != actual:
                errors.append(f"{c.acquirer}: primary_sector_proxy is {actual}, not {proxy}.")
    return errors


def full_thesis_log(pool, seeds: list[dict]) -> list[dict]:
    """The agent's decisions, completed with what code already knows: for a seed
    thesis its filters and the buyers it found; for a refinement, its new filters."""
    by_id = {s["thesis_id"]: s for s in seeds}
    log = []
    for entry in pool.thesis_log:
        row = {"thesis_id": entry.thesis_id, "kept": entry.kept, "reason": entry.reason}
        if entry.refined_filters is not None:
            row |= {"refined": True, "filters": entry.refined_filters.model_dump(exclude_defaults=True)}
        elif entry.thesis_id in by_id:
            seed = by_id[entry.thesis_id]
            row |= {"filters": seed["filters"],
                    "buyers_found": [b["acquirer"] for b in seed["result"]["buyers"]]}
        log.append(row)
    return log


def coverage_fallback(buyer_type: BuyerType, analysis: Analysis, cfg: Config) -> CoverageResult:
    """The ranking's top buyers of this type, used when the agent can't deliver."""
    top = analysis.rankings[analysis.rankings.buyer_type == buyer_type].nsmallest(
        cfg.scoring.pool_size, "type_rank")
    return CoverageResult(buyer_type=buyer_type, pool=None, names=list(top.index), topped_up=[],
                          tool_calls=[], fallback=True)


def _top_up(pool, tools: CoverageTools, minimum: int) -> tuple[list[str], list[str]]:
    """The agent's candidates, plus the best-ranked buyers it neither picked nor
    removed, until the pool reaches its minimum size."""
    names = [c.acquirer for c in pool.candidates]
    excluded = set(names) | {r.acquirer for r in pool.removals}
    added = [name for name in tools.rankings.sort_values("type_rank").index
             if name not in excluded][: max(0, minimum - len(names))]
    return names + added, added
