"""Tests for risk signals, conviction, and the full pre-agent final 10 (default target)."""

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.scoring.buyers import default_pick, order_final, rank_buyers
from acquirer_engine.scoring.relevance import score_deals
from acquirer_engine.scoring.risk import assess_conviction, risk_flags, risk_table

CONFIG = load_config()
TARGET = CONFIG.default_target
CFG = CONFIG.scoring

# Expected final 10 without agents: conviction and fired risks per buyer, in order.
EXPECTED = {
    "New Mountain Capital": ("High", {"bidders"}),
    "Blackstone": ("High", {"integration"}),
    "TPG Capital": ("High", {"financing"}),
    "KKR": ("Medium", {"execution"}),
    "GTCR": ("Medium", {"execution", "time_to_close"}),
    "Nordic Capital": ("Medium", {"bidders", "price", "financing"}),
    "Medtronic": ("Medium", {"bidders", "time_to_close", "financing"}),
    "Advocate Health": ("Low", {"price", "antitrust"}),
    "Ardent Health Services": ("Low", {"bidders", "price", "financing", "antitrust"}),
    "Aspen Dental (ADMI)": ("Low", {"price", "time_to_close"}),
}


@pytest.fixture(scope="module")
def pipeline():
    scored = score_deals(load_deals(CONFIG.paths.data_csv), TARGET, CFG)
    rankings = rank_buyers(scored, CFG)
    return scored, rankings, risk_table(scored, rankings, TARGET, CFG)


def test_percentiles_are_within_type(pipeline):
    _, _, table = pipeline
    pct = table.percentiles.drop(columns="antitrust")
    assert ((pct >= 0) & (pct <= 1) | pct.isna()).all().all()


def test_antitrust_only_for_strategics(pipeline):
    _, rankings, table = pipeline
    pe = rankings.index[rankings.buyer_type == "PE"]
    assert not table.fired.loc[pe, "antitrust"].any()


@pytest.mark.parametrize("buyer", EXPECTED)
def test_fired_risks(pipeline, buyer):
    _, _, table = pipeline
    fired = {name for name, is_fired in table.fired.loc[buyer].items() if is_fired}
    assert fired == EXPECTED[buyer][1]


@pytest.mark.parametrize("buyer", EXPECTED)
def test_every_buyer_shows_at_least_two_risks(pipeline, buyer):
    _, _, table = pipeline
    flags = risk_flags(buyer, table)
    assert len(flags) >= 2
    assert {f.name for f in flags if f.status == "fired"} == EXPECTED[buyer][1]


def test_conviction_explained(pipeline):
    scored, _, table = pipeline
    kkr = assess_conviction("KKR", scored, table, CFG)
    assert len(kkr.strong_deal_ids) == 2
    assert kkr.latest_strong_year == 2019          # both strong deals before 2021
    assert (kkr.willingness, kkr.ability, kkr.level) == ("Low", "High", "Medium")


def test_final_ten_without_agents(pipeline):
    scored, rankings, table = pipeline
    pick = default_pick(rankings, CFG)
    conviction = {b: assess_conviction(b, scored, table, CFG).level for b in pick}

    assert order_final(pick, rankings, conviction) == list(EXPECTED)
    assert conviction == {b: level for b, (level, _) in EXPECTED.items()}
