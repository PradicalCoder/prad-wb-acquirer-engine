"""Tests for the report: structure, numbers from the data, fallback banner, stop case."""

import pytest
from fakes import GraphClient, run_graph, short_page_config

from acquirer_engine.cache import Mode
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze
from acquirer_engine.graph import annotated_pack
from acquirer_engine.llm import LLMUnavailable
from acquirer_engine.render import money, multiple, ownership_label, render
from acquirer_engine.run import run_report
from acquirer_engine.schemas import IntakeResult

CFG = short_page_config()
SECTIONS = ["Acquirer overview", "Strategic fit thesis", "Precedent activity",
            "Valuation context", "Risk flags", "Conviction"]


@pytest.fixture(scope="module")
def deals():
    return load_deals(CFG.paths.data_csv)


@pytest.fixture(scope="module")
def analysis(deals):
    return analyze(deals, CFG.default_target, CFG.scoring)


async def report_for(client, deals, **kw):
    state = await run_graph(client, deals, **kw)
    report = run_report(state, client, Mode.off, seconds=1.0)
    return state, *render(state, report, CFG)


async def test_one_page_per_buyer_with_all_six_sections(analysis, deals):
    state, html, markdown = await report_for(GraphClient(analysis, CFG), deals)
    pages = html.split('<section class="page buyer">')[1:]
    assert len(pages) == 10
    for page in pages:
        for section in SECTIONS:
            assert section in page
    assert "page-break-after: always" in html                   # one printed page per buyer
    assert markdown.count("\n## ") >= 10 + 4                      # cover sections + 10 buyers


async def test_table_numbers_come_from_the_data(analysis, deals):
    state, html, markdown = await report_for(GraphClient(analysis, CFG), deals)
    for buyer in state["final_order"]:
        pack = annotated_pack(buyer, state, CFG)
        for deal in pack.top_deals:
            for text in (html, markdown):
                assert deal.transaction_id in text
                assert money(deal.size_mm) in text and multiple(deal.ev_revenue) in text
    comps = state["analysis"].comps_summary.ev_revenue
    assert f"{comps.median:.2f}x" in html and f"{comps.q1:.2f}x" in markdown


async def test_cover_summarizes_the_run(analysis, deals):
    state, html, _ = await report_for(GraphClient(analysis, CFG), deals)
    for buyer in state["final_order"]:
        assert buyer in html
    assert "What the agents changed from the plain ranking" in html
    assert "Welsh Carson" in html and " added: " in html          # an agent change, with its reason


async def test_fallback_pages_carry_a_review_banner(analysis, deals):
    _, html, markdown = await report_for(GraphClient(analysis, CFG, fail=LLMUnavailable("API down")), deals)
    assert html.count("<b>Needs review.</b>") == 10
    assert markdown.count("**Needs review.**") == 10


async def test_stopped_run_renders_the_reason(analysis, deals):
    stop = IntakeResult(stop_reason="Veterinary clinics are not a sector in our deal data.")
    _, html, markdown = await report_for(GraphClient(analysis, CFG, intake=stop), deals,
                                         text="a $300M veterinary group")
    assert "Veterinary clinics are not a sector" in html and "**Stopped:**" in markdown
    assert '<section class="page buyer">' not in html


async def test_rendering_is_repeatable(analysis, deals):
    client = GraphClient(analysis, CFG)
    state = await run_graph(client, deals)
    report = run_report(state, client, Mode.off, seconds=1.0)
    assert render(state, report, CFG) == render(state, report, CFG)


def test_formatting():
    assert money(159.7) == "$159.7M" and money(1250.0) == "$1.2B" and money(None) == "–"
    assert multiple(2.4908) == "2.49x"
    assert ownership_label(["Private", "PE-Backed"]) == "Private (privately held)"
    assert ownership_label(["PE-Backed"]) == "PE-Backed" and ownership_label(None) == "not stated"
