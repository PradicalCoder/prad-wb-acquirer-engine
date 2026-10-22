"""Turn a finished run into the report: report.html (printable, one page per
buyer) and report.md (the same content in Markdown).

No LLM calls. build_view() gathers everything the report shows into one plain
dictionary (the prose comes from the analysts' finished pages, every table
number from the data); the two Jinja2 templates in templates/ only lay it out.
"""

from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from acquirer_engine.config import Config
from acquirer_engine.graph import RunState, annotated_pack

TEMPLATES = Path(__file__).parent / "templates"
LENS_ORDER = ["Value", "Certainty", "Speed"]
CONVICTION_ORDER = ["High", "Medium", "Low"]


def render(state: RunState, report: dict, cfg: Config) -> tuple[str, str]:
    """Return (html, markdown) for a finished run."""
    view = build_view(state, report, cfg)
    html = _environment(autoescape=True).get_template("report.html.j2").render(**view)
    markdown = _environment(autoescape=False).get_template("report.md.j2").render(**view)
    return html, markdown


def build_view(state: RunState, report: dict, cfg: Config) -> dict:
    """Everything the templates show, computed once."""
    view = {"request": state["banker_text"], "stopped": report["stopped"], "buyers": [],
            "warnings": report["warnings"] + report["flags"], "seconds": report["seconds"]}
    if report["stopped"]:
        return view

    analysis = state["analysis"]
    buyers = [_buyer(i, name, state, cfg) for i, name in enumerate(state["final_order"], 1)]
    convictions = Counter(b["conviction"] for b in buyers)
    types = Counter(b["type"] for b in buyers)
    comps = analysis.comps_summary

    view |= {
        "headline": ", ".join(f"{convictions[c]} {c}" for c in CONVICTION_ORDER if convictions[c])
                    + " conviction · " + f"{types['PE']} PE, {types['Strategic']} strategic · "
                    + f"Comps {comps.ev_revenue.median:.2f}x EV/Revenue "
                    + f"({comps.ev_revenue.q1:.2f}–{comps.ev_revenue.q3:.2f}x)",
        "target": analysis.target,
        "ownership": ownership_label(analysis.target.ownership),
        "estimate": analysis.estimate,
        "comps": comps,
        "comps_rows": analysis.comps.to_dict("records"),
        "buyers": buyers,
        "changes": state.get("baseline_diff", []),
        "alternates": state.get("alternates", []),
        "coverage": [pool for pool in (state.get("pool_pe"), state.get("pool_strategic")) if pool],
        "public_knowledge": report["public_knowledge_in_screening"],
        "steps": [s for s in report["steps"] if not s["node"].startswith("analyst:")],
        "tokens": report["tokens_total"],
    }
    return view


def _buyer(rank: int, name: str, state: RunState, cfg: Config) -> dict:
    """One buyer's page: the analyst's prose plus the evidence tables."""
    pack = annotated_pack(name, state, cfg)
    page = state["results"][name].page
    fired = [r for r in page.risks if r.status == "fired"]
    key_risk = (fired or page.risks)[0]
    return {
        "rank": rank,
        "name": name,
        "type": pack.profile.buyer_type,
        "conviction": page.conviction.level,
        "page": page,
        "pack": pack,
        "best": pack.top_deals[0],
        "key_risk": f"{key_risk.lens}: {key_risk.name.replace('_', ' ')}",
        "risks_by_lens": [(lens, [r for r in page.risks if r.lens == lens])
                          for lens in LENS_ORDER if any(r.lens == lens for r in page.risks)],
    }


# --- formatting --------------------------------------------------------------------

def ownership_label(ownership: list[str] | None) -> str:
    """How the request described ownership: "private" covers both privately held
    owner types, so show it that way rather than as the two data labels."""
    if not ownership:
        return "not stated"
    if set(ownership) == {"Private", "PE-Backed"}:
        return "Private (privately held)"
    return ", ".join(ownership)


def money(value: float | None) -> str:
    """$159.7M; $1.2B from 1,000M up."""
    if value is None:
        return "–"
    return f"${value / 1000:,.1f}B" if value >= 1000 else f"${value:,.1f}M"


def multiple(value: float | None) -> str:
    return "–" if value is None else f"{value:.2f}x"


def share(value: float | None) -> str:
    return "–" if value is None else f"{value:.0%}"


def _environment(autoescape: bool) -> Environment:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=autoescape,
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.filters.update(money=money, multiple=multiple, share=share)
    return env
