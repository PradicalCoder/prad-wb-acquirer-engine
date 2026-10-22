"""The run as a LangGraph graph: intake -> scoring -> coverage -> captain ->
analysts -> set check -> order.

    intake --(stop?)--> END
      |
    score                       (code: all scoring, once)
      |
    pe_coverage || strategic_coverage
      |
    captain
      |
    analysts                    (10 in parallel)
      |
    set_check                   (near-duplicate theses: rewrite one, or flag)
      |
    order --> END

Every step runs inside node(), which gives it a deadline (the earlier of its
own time limit and the end of the run budget), times it, logs it, and records
an event. Expected failures (model unavailable, out of time, still invalid
after repairs) use the step's rule-based fallback, so the run always finishes.
Anything unexpected is a bug: it raises in strict mode (tests, --strict) and
otherwise is logged at ERROR, falls back, and is flagged in the run report.
"""

import asyncio
import hashlib
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

import pandas as pd
import structlog
from langgraph.graph import END, START, StateGraph

from acquirer_engine.agents.analyst import AnalystResult, fallback_page, revise, run_analyst
from acquirer_engine.agents.intake import IntakeOutcome, intake_fallback, run_intake
from acquirer_engine.config import Config
from acquirer_engine.evidence import Analysis, analyze, build_evidence
from acquirer_engine.llm import LLMClient, LLMOutputInvalid, LLMUnavailable
from acquirer_engine.scoring.buyers import order_final, top_of_type
from acquirer_engine.validate import similar_theses

log = structlog.get_logger()
EXPECTED = (LLMUnavailable, LLMOutputInvalid, TimeoutError)


class RunState(TypedDict, total=False):
    """What the steps hand to each other. Each step writes only its own keys;
    `events` is appended to by every step (the reducer adds the lists)."""

    banker_text: str
    intake: IntakeOutcome
    analysis: Analysis
    pool_pe: list[str]
    pool_strategic: list[str]
    selection: list[str]
    results: dict[str, AnalystResult]
    flags: list[str]
    final_order: list[str]
    events: Annotated[list[dict], operator.add]


@dataclass
class Run:
    """What every step shares: settings, data, the one LLM client, and the clock."""

    cfg: Config
    deals: pd.DataFrame
    client: LLMClient
    strict: bool = False
    started: float = field(default_factory=time.monotonic)

    def deadline(self, limit: float) -> float:
        """The earlier of `limit` seconds from now and the end of the run budget."""
        return min(time.monotonic() + limit, self.started + self.cfg.llm.run_budget_seconds)

    def time_left(self) -> float:
        return self.started + self.cfg.llm.run_budget_seconds - time.monotonic()


def node(run: Run, name: str, work, fallback=None, limit: float | None = None):
    """Wrap a step: deadline, timing, logging, event, and fallback rules.

    `work(state, deadline)` returns the step's state update. It may include
    "_status" (e.g. "rule_based") and "events" (e.g. one per analyst).
    `fallback(state)` returns the update to use instead; None = no fallback.
    """
    async def step(state: RunState) -> dict:
        started = time.monotonic()
        deadline = run.deadline(limit) if limit else None
        log.info("node.start", node=name)
        try:
            update = await work(state, deadline)
            status, reason = update.pop("_status", "ok"), None
        except EXPECTED as error:
            if fallback is None:
                raise
            update, status, reason = fallback(state), "fallback", str(error)
            log.warning("node.fallback", node=name, reason=reason)
        except Exception as error:
            if run.strict or fallback is None:
                raise
            log.error("node.error", node=name, error=repr(error), exc_info=True)
            update, status, reason = fallback(state), "error", repr(error)
        event = {"node": name, "status": status, "reason": reason,
                 "seconds": round(time.monotonic() - started, 2)}
        log.info("node.done", **event)
        return {**update, "events": update.pop("events", []) + [event]}
    return step


def build_graph(run: Run):
    cfg, client, limits = run.cfg, run.client, run.cfg.llm.timeouts_seconds

    # --- intake -----------------------------------------------------------------
    async def intake(state, deadline):
        return {"intake": await run_intake(state["banker_text"], client, cfg, run.deals, deadline)}

    def intake_fb(state):
        return {"intake": intake_fallback(state["banker_text"], cfg)}

    # --- scoring (code) ---------------------------------------------------------
    async def score(state, deadline):
        return {"analysis": analyze(run.deals, state["intake"].target, cfg.scoring)}

    # --- coverage and captain: rule-based until the agents are built ------------
    def pool(buyer_type):
        async def work(state, deadline):
            names = list(top_of_type(state["analysis"].rankings, buyer_type, cfg.scoring.pool_size).index)
            key = "pool_pe" if buyer_type == "PE" else "pool_strategic"
            return {key: names, "_status": "rule_based"}
        return work

    async def captain(state, deadline):
        picks = (state["pool_pe"][:cfg.scoring.mix_pe]
                 + state["pool_strategic"][:cfg.scoring.mix_strategic])
        return {"selection": picks, "_status": "rule_based"}

    # --- analysts: all at once; one failing never stops the others --------------
    async def analysts(state, deadline):
        analysis, selection = state["analysis"], state["selection"]

        async def one(buyer):
            started = time.monotonic()
            pack = build_evidence(buyer, analysis)
            log.info("analyst.start", buyer=buyer, pack_sha256=_sha(pack.model_dump_json()))
            result = await run_analyst(pack, client, cfg, run.deadline(limits.analyst))
            return result, round(time.monotonic() - started, 2)

        outcomes = await asyncio.gather(*(one(b) for b in selection), return_exceptions=True)
        results, events = {}, []
        for buyer, outcome in zip(selection, outcomes):
            if isinstance(outcome, Exception):
                if run.strict:
                    raise outcome
                log.error("analyst.error", buyer=buyer, error=repr(outcome))
                results[buyer] = _fallback_result(buyer, analysis, repr(outcome))
                events.append({"node": f"analyst:{buyer}", "status": "error",
                               "reason": repr(outcome), "seconds": None})
                continue
            result, seconds = outcome
            results[buyer] = result
            events.append({"node": f"analyst:{buyer}", "seconds": seconds,
                           "status": "fallback" if result.fallback else "ok",
                           "repairs": result.repairs, "reason": "; ".join(result.errors) or None})
            log.info("analyst.done", buyer=buyer, **events[-1])
        return {"results": results, "events": events}

    def analysts_fb(state):
        return {"results": {b: _fallback_result(b, state["analysis"], "analyst step failed")
                            for b in state["selection"]}}

    # --- set check: no two pages may read like copies ----------------------------
    async def set_check(state, deadline):
        results, order = dict(state["results"]), state["selection"]
        written = {b: r.page for b, r in results.items() if not r.fallback}
        flags, rewritten = [], set()
        for first, second, score in similar_theses(written, cfg.validation):
            lower = max(first, second, key=order.index)   # later in the list = lower ranked
            other = second if lower == first else first
            if lower in rewritten or run.time_left() < cfg.validation.rewrite_min_seconds:
                flags.append(f"Fit theses of {first} and {second} read alike "
                             f"(similarity {score}); no time to rewrite.")
                continue
            feedback = (f"Your fit thesis reads too much like the one written for {other}: "
                        f"\"{results[other].page.fit_thesis}\" Rewrite the page so its fit "
                        f"thesis rests on what is specific to {lower}.")
            try:
                results[lower] = await revise(results[lower], build_evidence(lower, state["analysis"]),
                                              feedback, client, cfg, run.deadline(limits.analyst))
                rewritten.add(lower)
            except EXPECTED as error:
                flags.append(f"Fit theses of {first} and {second} read alike; rewrite failed ({error}).")
        return {"results": results, "flags": flags}

    def set_check_fb(state):
        return {"flags": ["Set check skipped; theses were not compared."]}

    # --- order: conviction never rises down the list -----------------------------
    async def order(state, deadline):
        conviction = {b: r.page.conviction.level for b, r in state["results"].items()}
        return {"final_order": order_final(state["selection"], state["analysis"].rankings, conviction)}

    graph = StateGraph(RunState)
    graph.add_node("intake", node(run, "intake", intake, intake_fb, limits.intake))
    graph.add_node("score", node(run, "score", score))
    graph.add_node("pe_coverage", node(run, "pe_coverage", pool("PE")))
    graph.add_node("strategic_coverage", node(run, "strategic_coverage", pool("Strategic")))
    graph.add_node("captain", node(run, "captain", captain))
    graph.add_node("analysts", node(run, "analysts", analysts, analysts_fb))
    graph.add_node("set_check", node(run, "set_check", set_check, set_check_fb))
    graph.add_node("order", node(run, "order", order))

    graph.add_edge(START, "intake")
    graph.add_conditional_edges("intake", _after_intake, ["score", END])
    graph.add_edge("score", "pe_coverage")
    graph.add_edge("score", "strategic_coverage")
    graph.add_edge(["pe_coverage", "strategic_coverage"], "captain")   # waits for both
    graph.add_edge("captain", "analysts")
    graph.add_edge("analysts", "set_check")
    graph.add_edge("set_check", "order")
    graph.add_edge("order", END)
    return graph.compile()


def _after_intake(state: RunState) -> str:
    """Stop when intake found no usable target (e.g. sector not in the data)."""
    return "score" if state["intake"].target is not None else END


def _fallback_result(buyer: str, analysis: Analysis, reason: str) -> AnalystResult:
    page = fallback_page(build_evidence(buyer, analysis))
    return AnalystResult(page=page, notes=None, repairs=0, errors=[reason], fallback=True)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
