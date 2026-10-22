"""Tests for the target estimate and the valuation comps (default target)."""

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.scoring.relevance import score_deals
from acquirer_engine.scoring.valuation import estimate_target_financials, valuation_comps

CONFIG = load_config()
TARGET = CONFIG.default_target
CFG = CONFIG.scoring


@pytest.fixture(scope="module")
def deals():
    return load_deals(CONFIG.paths.data_csv)


@pytest.fixture(scope="module")
def comps(deals):
    return valuation_comps(score_deals(deals, TARGET, CFG), CFG)


def test_target_estimate(deals):
    estimate = estimate_target_financials(deals, TARGET, CFG)
    assert len(estimate.peer_ids) == 16
    assert estimate.revenue_mm == pytest.approx(84.4, abs=0.1)
    assert estimate.ebitda_mm == pytest.approx(15.0, abs=0.1)
    assert estimate.margin_pct == pytest.approx(17.8, abs=0.1)
    assert not estimate.low_confidence


def test_estimate_with_few_peers_is_flagged(deals):
    strict = CFG.model_copy(update={"min_estimate_peers": 50})
    assert estimate_target_financials(deals, TARGET, strict).low_confidence


def test_comps_table(comps):
    table, _ = comps
    assert len(table) == CFG.top_n_comps
    assert table.transaction_id.iloc[0] == "MA-2021-0195"
    assert table.relevance.is_monotonic_decreasing


def test_comps_use_closed_deals_only(deals, comps):
    table, _ = comps
    closed = set(deals.loc[deals.outcome == "Closed", "transaction_id"])
    assert set(table.transaction_id) <= closed


def test_comps_summary(comps):
    _, summary = comps
    assert summary.ev_revenue.median == pytest.approx(2.49, abs=0.005)
    assert summary.ev_revenue.q1 == pytest.approx(2.01, abs=0.005)
    assert summary.ev_revenue.q3 == pytest.approx(3.01, abs=0.005)
    assert summary.ev_ebitda_calc.median == pytest.approx(16.59, abs=0.005)
    assert summary.ev_ebitda_stated.median == pytest.approx(12.00, abs=0.005)
    assert summary.ev_revenue_by_type == pytest.approx({"PE": 2.49, "Strategic": 2.38}, abs=0.005)
    assert summary.count_by_type == {"PE": 14, "Strategic": 6}
