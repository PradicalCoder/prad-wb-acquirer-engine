"""Tests for the per-buyer evidence pack (default target)."""

import math

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze, build_evidence
from acquirer_engine.scoring.buyers import default_pick

CONFIG = load_config()


@pytest.fixture(scope="module")
def deals():
    return load_deals(CONFIG.paths.data_csv)


@pytest.fixture(scope="module")
def analysis(deals):
    return analyze(deals, CONFIG.default_target, CONFIG.scoring)


@pytest.fixture(scope="module")
def kkr(analysis):
    return build_evidence("KKR", analysis)


def test_profile(kkr):
    p = kkr.profile
    assert (p.buyer_type, p.deals, p.strong_deals, p.rank_in_type) == ("PE", 17, 2, 3)
    assert p.typical_size_mm == pytest.approx(159.7)
    assert p.range_90 is None                         # PE has no shrinkage range
    assert "Geographic Expansion" not in p.top_tags   # on ~90% of deals: says nothing


def test_top_deal_explains_its_relevance(kkr):
    deal = kkr.top_deals[0]
    assert deal.transaction_id == "MA-2019-0205"
    assert math.prod(deal.factors.values()) == pytest.approx(deal.relevance)
    assert len(kkr.top_deals) == 5


def test_execution_risk_with_its_deals(kkr, deals):
    execution = next(r for r in kkr.risks if r.name == "execution")
    assert (execution.status, execution.count, execution.out_of) == ("fired", 4, 17)
    cited = deals.set_index("transaction_id").loc[execution.evidence_ids]
    assert (cited.acquirer == "KKR").all()
    assert cited.outcome.isin(["Withdrawn", "Terminated"]).all()


def test_what_they_paid_uses_closed_deals(kkr, deals):
    closed = deals[(deals.acquirer == "KKR") & (deals.outcome == "Closed")]
    assert kkr.paid.closed_deals == len(closed) == 11
    assert kkr.paid.ev_revenue == pytest.approx(closed.ev_revenue_calc.median())


def test_conviction_and_anomalies(kkr, analysis):
    assert kkr.conviction.level == "Medium"
    assert any("merger-type" in note for note in kkr.anomalies)
    medtronic = build_evidence("Medtronic", analysis)
    assert medtronic.profile.range_90 is not None
    assert any("LBO/MBO" in note for note in medtronic.anomalies)


def test_every_default_pick_has_two_risks(analysis):
    for buyer in default_pick(analysis.rankings, CONFIG.scoring):
        assert len(build_evidence(buyer, analysis).risks) >= 2


def test_estimate_only_when_not_given(deals, analysis):
    assert analysis.estimate.revenue_mm == pytest.approx(84.4, abs=0.1)
    given = CONFIG.default_target.model_copy(update={"revenue_mm": 90.0, "ebitda_mm": 16.0})
    assert analyze(deals, given, CONFIG.scoring).estimate is None
