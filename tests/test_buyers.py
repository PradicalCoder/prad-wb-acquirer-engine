"""Tests for buyer rankings, the default pick and final ordering (default target)."""

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.scoring.buyers import (
    default_pick,
    order_final,
    rank_buyers,
    shrinkage_prior,
    top_of_type,
)
from acquirer_engine.scoring.relevance import score_deals

CONFIG = load_config()
CFG = CONFIG.scoring


@pytest.fixture(scope="module")
def rankings():
    deals = load_deals(CONFIG.paths.data_csv)
    return rank_buyers(score_deals(deals, CONFIG.default_target, CFG), CFG)


def test_every_buyer_is_ranked(rankings):
    assert len(rankings) == 107
    assert rankings.groupby("buyer_type").type_rank.max().to_dict() == {"PE": 15, "Strategic": 92}


def test_pe_pool(rankings):
    pool = top_of_type(rankings, "PE", CFG.pool_size)
    assert list(pool.index) == [
        "Blackstone", "New Mountain Capital", "KKR", "Nordic Capital",
        "GTCR", "TPG Capital", "Vista Equity", "General Atlantic",
    ]
    assert pool.loc["Blackstone", "score"] == pytest.approx(1.623, abs=1e-3)


def test_shrinkage_prior(rankings):
    mean, k = shrinkage_prior(rankings[rankings.buyer_type == "Strategic"])
    assert mean == pytest.approx(0.114, abs=1e-3)
    assert k == pytest.approx(10.69, abs=0.01)


def test_strategic_pool(rankings):
    pool = top_of_type(rankings, "Strategic", CFG.pool_size)
    assert list(pool.index) == [
        "Ardent Health Services", "Advocate Health", "Medtronic", "Aspen Dental (ADMI)",
        "Oscar Health", "UPMC", "Bon Secours Mercy Health", "Intermountain Health",
    ]
    ardent = pool.loc["Ardent Health Services"]
    assert ardent.score == pytest.approx(0.153, abs=1e-3)
    assert (ardent.low90, ardent.high90) == pytest.approx((0.029, 0.340), abs=1e-3)


def test_one_deal_is_pulled_toward_the_average(rankings):
    oscar = rankings.loc["Oscar Health"]
    assert oscar.deals == 1
    assert oscar.best == pytest.approx(0.520, abs=1e-3)   # raw rate of its single deal
    assert oscar.score == pytest.approx(0.149, abs=1e-3)  # after shrinkage


def test_default_pick(rankings):
    pick = default_pick(rankings, CFG)
    assert len(pick) == CFG.mix_pe + CFG.mix_strategic
    assert pick[:CFG.mix_pe] == [
        "Blackstone", "New Mountain Capital", "KKR", "Nordic Capital", "GTCR", "TPG Capital",
    ]


def test_order_final(rankings):
    conviction = {
        "New Mountain Capital": "High", "Blackstone": "High", "TPG Capital": "High",
        "KKR": "Medium", "GTCR": "Medium", "Nordic Capital": "Medium", "Medtronic": "Medium",
        "Advocate Health": "Low", "Ardent Health Services": "Low", "Aspen Dental (ADMI)": "Low",
    }
    assert order_final(list(conviction), rankings, conviction) == [
        "New Mountain Capital", "Blackstone", "TPG Capital",     # High: 4, 2, 2 strong deals
        "KKR", "GTCR", "Nordic Capital", "Medtronic",           # Medium: 2 each, by best deal
        "Advocate Health", "Ardent Health Services", "Aspen Dental (ADMI)",
    ]
