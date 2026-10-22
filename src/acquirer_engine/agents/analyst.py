"""The analyst: one buyer's one-page rationale, in two LLM steps plus code checks.

  1. Notes (reason): plan the page from the evidence pack -> AnalystNotes.
  2. Write: the page from the pack and the notes -> OnePagerDraft.
  3. Check: validate_page() in code. On errors, the draft and the errors go
     back to the model to fix (at most max_repairs times).
  4. If the model is unavailable or the page still fails, ship an
     evidence-only page built from the pack, marked "needs review".
"""

import json

import structlog
from pydantic import BaseModel

from acquirer_engine.agents import cached_system, load_prompt, repair_turn
from acquirer_engine.config import Config
from acquirer_engine.evidence import EvidencePack
from acquirer_engine.llm import LLMClient, LLMOutputInvalid, LLMUnavailable
from acquirer_engine.schemas import AnalystNotes, OnePage, OnePagerDraft
from acquirer_engine.validate import validate_page

log = structlog.get_logger()


class AnalystResult(BaseModel):
    page: OnePage
    notes: AnalystNotes | None
    repairs: int              # how many times the page went back for fixes
    errors: list[str]         # problems left on the shipped page (empty if it passed)
    fallback: bool            # True if this is the evidence-only page


async def run_analyst(
    pack: EvidencePack, client: LLMClient, cfg: Config, deadline: float
) -> AnalystResult:
    agent = f"analyst:{pack.buyer}"
    pack_json = pack.model_dump_json(exclude_none=True)
    try:
        notes = await _plan(agent, pack_json, client, cfg, deadline)
        return await _write(agent, pack, pack_json, notes, client, cfg, deadline)
    except (LLMUnavailable, LLMOutputInvalid) as error:  # expected: model down, or notes unusable
        log.warning("analyst.fallback", agent=agent, reason=str(error))
        return AnalystResult(page=fallback_page(pack), notes=None, repairs=0,
                             errors=[str(error)], fallback=True)


async def _plan(agent, pack_json, client, cfg, deadline) -> AnalystNotes:
    """Step 1. A notes answer that doesn't fit the schema gets one more try."""
    rule = cfg.scoring  # the conviction rule's thresholds, stated in the prompt
    version, system = load_prompt("analyst_notes", strong_relevance=rule.strong_relevance,
                                  recent_year=rule.recent_year,
                                  max_ability_risks=rule.max_ability_risks)
    messages = [{"role": "user", "content": f"Evidence pack:\n{pack_json}"}]
    try:
        return await client.structured(agent=agent, profile="analyst_notes", system=cached_system(system),
                                       messages=messages, schema=AnalystNotes,
                                       prompt_version=version, deadline=deadline)
    except LLMOutputInvalid as error:
        messages += repair_turn(error.raw_text, [str(error)])
        return await client.structured(agent=agent, profile="analyst_notes", system=cached_system(system),
                                       messages=messages, schema=AnalystNotes,
                                       prompt_version=version, deadline=deadline)


async def revise(result: AnalystResult, pack: EvidencePack, feedback: str, client: LLMClient,
                 cfg: Config, deadline: float) -> AnalystResult:
    """Write the page again from the same notes, with one extra instruction
    (used when its fit thesis reads too much like another buyer's)."""
    agent = f"analyst:{pack.buyer}"
    return await _write(agent, pack, pack.model_dump_json(exclude_none=True), result.notes,
                        client, cfg, deadline, extra=feedback)


async def _write(agent, pack, pack_json, notes, client, cfg, deadline, extra=None) -> AnalystResult:
    """Steps 2-4: write, check, repair; fall back if it still fails."""
    rules = cfg.validation
    version, system = load_prompt(
        "analyst_write",
        banned_phrases=", ".join(f'"{p}"' for p in rules.banned_phrases),
        # Ask for the middle of the allowed range, so small misses still pass.
        words_min=rules.page_words_min + 50, words_max=rules.page_words_max - 50,
    )
    instruction = "Write the page." + (f"\n\n{extra}" if extra else "")
    messages = [{"role": "user", "content": f"Evidence pack:\n{pack_json}\n\n"
                                            f"Your notes:\n{notes.model_dump_json()}\n\n"
                                            f"{instruction}"}]
    errors: list[str] = []
    for attempt in range(rules.max_repairs + 1):
        try:
            draft = await client.structured(agent=agent, profile="analyst_write",
                                            system=cached_system(system), messages=messages,
                                            schema=OnePagerDraft, prompt_version=version,
                                            deadline=deadline)
            raw, errors = draft.model_dump_json(), validate_page(draft, pack, notes, rules)
        except LLMOutputInvalid as error:
            draft, raw, errors = None, error.raw_text, [str(error)]
        if not errors:
            return AnalystResult(page=OnePage(**draft.model_dump()), notes=notes,
                                 repairs=attempt, errors=[], fallback=False)
        log.info("analyst.repair", agent=agent, attempt=attempt + 1, errors=errors)
        messages = messages + repair_turn(raw, errors)

    log.warning("analyst.fallback", agent=agent, reason="page still failing checks")
    return AnalystResult(page=fallback_page(pack), notes=notes, repairs=rules.max_repairs,
                         errors=errors, fallback=True)


# --- Evidence-only page ------------------------------------------------------------

def fallback_page(pack: EvidencePack) -> OnePage:
    """A page assembled from the pack alone, with no LLM: every number is
    copied, nothing is interpreted. Marked for review."""
    p, best = pack.profile, pack.top_deals[0]
    sectors = ", ".join(f"{name} ({n})" for name, n in list(p.sectors.items())[:3])
    others = ", ".join(d.transaction_id for d in pack.top_deals[1:3])
    paid = (f"{pack.buyer} paid a median {pack.paid.ev_revenue:.2f}x EV/Revenue on "
            f"{pack.paid.closed_deals} closed deals." if pack.paid.ev_revenue is not None
            else f"{pack.buyer} has no closed deals with a price in the data.")
    w, c = pack.conviction, pack.comps.ev_revenue
    return OnePage(
        acquirer=pack.buyer,
        overview=f"{pack.buyer} ({p.buyer_type}) has {p.deals} deals in the data from "
                 f"{p.first_year} to {p.latest_year}, with a median size of "
                 f"${p.typical_size_mm:.1f}M, mostly in {sectors}.",
        fit_thesis=f"Auto-generated, no written thesis. Closest precedent: {best.transaction_id} "
                   f"({best.sector}, {best.year}, relevance {best.relevance:.2f}).",
        precedent_note=f"{p.strong_deals} strong precedents; next most relevant: {others or 'none'}.",
        valuation_note=f"Comps median {c.median:.2f}x EV/Revenue (middle half {c.q1:.2f}-"
                       f"{c.q3:.2f}x). {paid}",
        risks=[{"name": r.name, "lens": r.lens, "status": r.status, "evidence_ids": r.evidence_ids,
                "text": f"{r.detail}: {_value_text(r)}."} for r in pack.risks],
        conviction={"level": w.level,
                    "rationale": f"Rule result: willingness {w.willingness}, ability {w.ability}."},
        needs_review=True,
    )


def _value_text(risk) -> str:
    if risk.count is not None:
        return f"{risk.count} of {risk.out_of} deals"
    return "no value" if risk.value is None else f"{risk.value:.2f}"


# --- Temporary Markdown view (until the report renderer exists) --------------------

def page_markdown(result: AnalystResult) -> str:
    page = result.page
    lines = [f"## {page.acquirer}" + ("  _(needs review)_" if page.needs_review else ""),
             f"**Overview.** {page.overview}", f"**Strategic fit.** {page.fit_thesis}",
             f"**Precedent activity.** {page.precedent_note}",
             f"**Valuation context.** {page.valuation_note}", "**Risk flags.**"]
    lines += [f"- *{r.lens} ({r.status})*, {r.name}: {r.text} {' '.join(r.evidence_ids)}"
              for r in page.risks]
    lines.append(f"**Conviction: {page.conviction.level}.** {page.conviction.rationale}")
    lines += [f"> Public knowledge, not from the data: {claim}" for claim in page.public_knowledge]
    lines.append(f"\n_repairs: {result.repairs}, errors left: {json.dumps(result.errors)}_")
    return "\n\n".join(lines)
