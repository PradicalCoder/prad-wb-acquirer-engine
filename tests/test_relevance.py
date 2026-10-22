"""Tests for sector weights and deal relevance (default target: $200M Healthcare Services)."""

import numpy as np
import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.scoring.relevance import FACTORS, score_deals, sector_buyers, sector_weights

CONFIG = load_config()
TARGET = CONFIG.default_target
CFG = CONFIG.scoring


@pytest.fixture(scope="module")
def deals():
    return load_deals(CONFIG.paths.data_csv)


@pytest.fixture(scope="module")
def scored(deals):
    return score_deals(deals, TARGET, CFG).set_index("transaction_id")


def test_sector_buyers(deals):
    assert len(sector_buyers(deals, "Healthcare Services")) == 26


def test_sector_weights(deals):
    table = sector_weights(deals, "Healthcare Services", CFG)
    assert table.loc["Healthcare Services", "weight"] == 1.0
    assert table.loc["Physician Groups", "lift"] == pytest.approx(1.769, abs=1e-3)
    assert table.loc["Physician Groups", "weight"] == pytest.approx(0.884, abs=1e-3)
    assert table.loc["Health IT", "weight"] == pytest.approx(0.275, abs=1e-3)
    # Another sector never reaches the target's own weight.
    assert table.drop("Healthcare Services").weight.max() <= CFG.sector_cap


def test_relevance_is_product_of_factors(scored):
    assert np.allclose(scored.relevance, scored[FACTORS].prod(axis=1))
    assert scored.relevance.between(0, 1).all()


def test_best_precedent(scored):
    # Blackstone's 2021 Healthcare Services bolt-on, $299M.
    assert scored.relevance.idxmax() == "MA-2021-0195"
    deal = scored.loc["MA-2021-0195"]
    assert deal.w_sector == 1.0
    assert deal.w_size == pytest.approx(0.8 ** (np.log2(299 / 200) ** 2))  # 0.928
    assert deal.w_recency == 0.9                                           # 3 years old
    assert deal.relevance == pytest.approx(0.8349, abs=1e-4)


def test_adjacent_sector_deal(scored):
    # KKR's 2019 Physician Groups deal: adjacent sector, 5 years old.
    deal = scored.loc["MA-2019-0205"]
    assert deal.w_sector == pytest.approx(0.884, abs=1e-3)
    assert deal.w_recency == 0.75
    assert deal.relevance == pytest.approx(0.6479, abs=1e-4)


def test_unstated_fields_are_neutral(deals):
    bare = TARGET.model_copy(update={"ownership": None, "margin_band": None, "region": None})
    scored = score_deals(deals, bare, CFG)
    for factor in ["w_ownership", "w_margin", "w_geography"]:
        assert (scored[factor] == 1.0).all()


def test_known_region_discounts_other_regions(deals):
    midwest = TARGET.model_copy(update={"region": "Midwest"})
    scored = score_deals(deals, midwest, CFG)
    by_geo = scored.groupby("geography").w_geography.first()
    assert by_geo["Midwest"] == 1.0
    assert by_geo["National"] == 1.0
    assert by_geo["Multi-Regional"] == 1.0
    assert by_geo["Southeast"] == CFG.geography_miss
