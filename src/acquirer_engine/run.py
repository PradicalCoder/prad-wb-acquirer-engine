"""Run the engine once and write the run folder.

runs/<timestamp>/
  report.md        the pages, in final order (plain Markdown until the HTML report exists)
  run_report.json  per step: status and seconds; per call type: tokens; fallbacks, flags
  state.json       what each step produced (minus the big scoring tables)
  run.log          every event as one JSON line
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import structlog

from acquirer_engine.agents.analyst import page_markdown
from acquirer_engine.cache import AnswerCache, Mode
from acquirer_engine.config import Config, get_api_key
from acquirer_engine.data import load_deals
from acquirer_engine.graph import Run, RunState, build_graph
from acquirer_engine.llm import LLMClient


async def execute(text: str, cfg: Config, mode: Mode, cache_path: Path, strict: bool = False,
                  client: LLMClient | None = None) -> tuple[RunState, Path, dict]:
    """Run once. Returns (final state, run folder, run report). `client` is for tests."""
    run_dir = cfg.paths.runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = (run_dir / "run.log").open("w", encoding="utf-8")
    _log_to(log_file)
    try:
        if client is None:
            cache = None if mode is Mode.off else AnswerCache(cache_path)
            client = LLMClient(cfg.llm, mode, cache, get_api_key())
        started = time.monotonic()
        graph = build_graph(Run(cfg=cfg, deals=load_deals(cfg.paths.data_csv, cfg.scoring.margin_strong_percentile),
                                client=client, strict=strict))
        state = await graph.ainvoke({"banker_text": text, "events": []})
        seconds = round(time.monotonic() - started, 1)

        report = run_report(state, client, mode, seconds)
        (run_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (run_dir / "report.md").write_text(report_markdown(state, report), encoding="utf-8")
        (run_dir / "state.json").write_text(json.dumps(_snapshot(state), indent=2, default=str),
                                            encoding="utf-8")
        return state, run_dir, report
    finally:
        structlog.reset_defaults()
        log_file.close()


def run_report(state: RunState, client: LLMClient, mode: Mode, seconds: float) -> dict:
    """The numbers a reviewer checks: time per step, tokens per call type, fallbacks."""
    # Tokens two ways: by kind of call (e.g. all analyst write steps together)
    # and by agent (e.g. coverage:PE vs coverage:Strategic, each analyst).
    def empty():
        return {"calls": 0, "cache_hits": 0, "input": 0, "output": 0, "cache_read": 0, "seconds": 0.0}
    tokens, by_agent = defaultdict(empty), defaultdict(empty)
    for call in client.usage:
        for row in (tokens[call.profile], by_agent[call.agent]):
            row["calls"] += 1
            row["cache_hits"] += call.cache_hit
            row["seconds"] = round(row["seconds"] + call.seconds, 2)
            if not call.cache_hit:            # stored answers spend no new tokens
                row["input"] += call.input_tokens
                row["output"] += call.output_tokens
                row["cache_read"] += call.cache_read_tokens
    pools = [p for p in (state.get("pool_pe"), state.get("pool_strategic")) if p]
    intake = state["intake"]
    results = state.get("results", {})
    return {
        "mode": mode.value,
        "seconds": seconds,
        "stopped": intake.stop_reason,
        "target": intake.target.model_dump() if intake.target else None,
        "warnings": intake.warnings,
        "flags": state.get("flags", []) + [f"{e['node']}: {e['reason']}" for e in state["events"]
                                           if e["status"] == "error"],
        "final": [{"buyer": b, "conviction": results[b].page.conviction.level,
                   "needs_review": results[b].page.needs_review, "repairs": results[b].repairs}
                  for b in state.get("final_order", [])],
        "coverage": {pool.buyer_type: {
                         "pool": pool.names, "topped_up": pool.topped_up, "fallback": pool.fallback,
                         "removals": [r.model_dump() for r in pool.pool.removals] if pool.pool else [],
                         "tool_calls": pool.tool_calls, "thesis_log": pool.thesis_log}
                     for pool in pools},
        # Statements the coverage agents made from general knowledge, not from the data.
        "public_knowledge_in_screening": [f"{pool.buyer_type}: {claim}" for pool in pools
                                          if pool.pool for claim in pool.pool.public_knowledge],
        "captain": ({"fallback": state["captain"].fallback,
                     "shift_reason": state["captain"].selection.shift_reason
                     if state["captain"].selection else None}
                    if "captain" in state else None),
        "agent_vs_baseline": state.get("baseline_diff", []),
        "alternates": state.get("alternates", []),
        "steps": state["events"],
        "tokens_by_call_type": dict(tokens),
        "tokens_by_agent": dict(by_agent),
        "tokens_total": {k: sum(row[k] for row in tokens.values())
                         for k in ("calls", "cache_hits", "input", "output", "cache_read")},
    }


def report_markdown(state: RunState, report: dict) -> str:
    lines = ["# Likely acquirers", f"_Request: {state['banker_text']}_"]
    if report["stopped"]:
        return "\n\n".join(lines + [f"**Stopped:** {report['stopped']}"])
    lines += [f"_Run: {report['seconds']} s, mode {report['mode']}_"]
    lines += [f"> Warning: {w}" for w in report["warnings"] + report["flags"]]
    lines.append("| # | Buyer | Conviction | Review |\n|---|---|---|---|")
    lines[-1] += "".join(f"\n| {i} | {row['buyer']} | {row['conviction']} | "
                         f"{'needs review' if row['needs_review'] else ''} |"
                         for i, row in enumerate(report["final"], 1))
    changes = [f"- **{d['buyer']}** {d['change']}: {d['reason']}" for d in report["agent_vs_baseline"]]
    lines.append("### What the agents changed from the plain ranking\n"
                 + ("\n".join(changes) if changes else "Nothing: the agents kept the ranking's picks."))
    if report["alternates"]:
        lines.append(f"_Next in line: {', '.join(report['alternates'])}_")
    if report["public_knowledge_in_screening"]:
        lines.append("### Public knowledge used in screening (not from the data)\n"
                     + "\n".join(f"- {claim}" for claim in report["public_knowledge_in_screening"]))
    lines += [page_markdown(state["results"][b]) for b in state["final_order"]]
    return "\n\n".join(lines)


def _snapshot(state: RunState) -> dict:
    """Everything the steps produced except the scoring tables (rebuilt from the data)."""
    out = {}
    for key, value in state.items():
        if key == "analysis":
            continue
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif key == "results":
            value = {b: r.model_dump() for b, r in value.items()}
        out[key] = value
    return out


def _log_to(file) -> None:
    """Every log event becomes one JSON line in run.log."""
    structlog.configure(
        processors=[structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer()],
        logger_factory=structlog.WriteLoggerFactory(file=file),
    )
