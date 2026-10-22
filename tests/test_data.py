"""Tests for the deal loader and the Deal / TargetProfile models.

Expected numbers were computed independently from the same CSV.
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from acquirer_engine.data import TargetProfile, load_deals, margin_cutoffs

CSV = Path(__file__).resolve().parents[1] / "data" / "ma_transactions_500.csv"


@pytest.fixture(scope="module")
def deals():
    return load_deals(CSV)


def test_loads_every_row(deals):
    assert len(deals) == 500
    assert deals.acquirer.nunique() == 107


def test_buyer_type_mapping(deals):
    buyers = deals.drop_duplicates("acquirer").buyer_type.value_counts()
    assert buyers["PE"] == 15
    assert buyers["Strategic"] == 92
    assert (deals.buyer_type == "PE").sum() == 290


def test_tags_are_split(deals):
    first = deals.set_index("transaction_id").loc["MA-2020-0001"]
    assert first.strategic_rationale_tags == [
        "Technology Capability", "Geographic Expansion", "Vertical Integration",
    ]


def test_calculated_columns(deals):
    deal = deals.set_index("transaction_id").loc["MA-2021-0195"]
    assert deal.ev_revenue_calc == pytest.approx(2.4508, abs=1e-4)
    assert deal.ev_ebitda_calc == pytest.approx(14.7291, abs=1e-4)


def test_healthcare_services_margin_bands(deals):
    cuts = margin_cutoffs(deals).loc["Healthcare Services"]
    assert round(cuts["median"], 1) == 14.5
    assert round(cuts["strong"], 1) == 16.0

    bands = deals[deals.sector == "Healthcare Services"].margin_band.value_counts()
    assert bands["strong"] == 16
    assert bands["weak"] == 23


def test_days_to_close_only_for_closed(deals):
    closed = deals.outcome == "Closed"
    assert deals[closed].days_to_close.notna().all()
    assert deals[~closed].days_to_close.isna().all()


def _write_csv(tmp_path, edit):
    """Copy the first data rows of the real CSV, apply `edit` to the text, save it."""
    lines = CSV.read_text(encoding="utf-8").splitlines()[:3]
    path = tmp_path / "deals.csv"
    path.write_text(edit("\n".join(lines)), encoding="utf-8")
    return path


def test_unknown_sector_is_rejected(tmp_path):
    path = _write_csv(tmp_path, lambda text: text.replace("Health IT,Health IT", "Veterinary,Health IT"))
    with pytest.raises(ValidationError, match="sector"):
        load_deals(path)


def test_closed_deal_without_close_time_is_rejected(tmp_path):
    path = _write_csv(tmp_path, lambda text: text.replace(",157.0,", ",,"))
    with pytest.raises(ValidationError, match="days_to_close"):
        load_deals(path)


def test_target_profile_accepts_the_default_target():
    target = TargetProfile(
        sector="Healthcare Services", ev_mm=200, ownership=["Private", "PE-Backed"],
        region=None, margin_band="strong",
    )
    assert target.revenue_mm is None


@pytest.mark.parametrize("bad", [
    {"sector": "Veterinary"},       # not a sector in the data
    {"ev_mm": 0},                   # size must be positive
    {"region": "National"},         # a regional target sits in one region
    {"ownership": ["Family"]},      # not an ownership type in the data
])
def test_target_profile_rejects_bad_values(bad):
    fields = {"sector": "Healthcare Services", "ev_mm": 200} | bad
    with pytest.raises(ValidationError):
        TargetProfile(**fields)
