"""The deal dataset: allowed values, the Deal and TargetProfile models, and the loader.

The Literal types below are the single source of allowed values. The Deal model
uses them to validate every CSV row; TargetProfile uses the same types, so a
target can only be described in terms that exist in the data.

load_deals() returns a pandas DataFrame (one row per deal) because the scoring
code works on whole columns at once.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

# --- Allowed values (from the dataset) --------------------------------------

Sector = Literal[
    "Behavioral Health", "Dental", "Health IT", "Health Insurance", "Healthcare Services",
    "Home Health/Hospice", "Medical Devices", "Pharma/Biotech", "Physician Groups",
    "Revenue Cycle",
]
DealType = Literal[
    "Bolt-on Acquisition", "Carve-out", "Leveraged Buyout", "Management Buyout",
    "Merger of Equals", "Minority Investment", "Platform Investment", "Recapitalization",
    "SPAC Merger", "Strategic Acquisition",
]
# A "regional" company sits in one of these eight regions.
SingleRegion = Literal[
    "Great Plains", "Mid-Atlantic", "Midwest", "Mountain West", "Northeast", "Southeast",
    "Southwest", "West Coast",
]
Geography = SingleRegion | Literal["Multi-Regional", "National"]
FinancingType = Literal[
    "All Cash", "Cash + Stock", "Earnout", "Leveraged", "Mixed", "Seller Financing", "Stock",
]
Ownership = Literal["Non-Profit", "PE-Backed", "Private", "Public"]
Outcome = Literal["Closed", "Pending", "Rumored", "Terminated", "Withdrawn"]
Quarter = Literal["Q1", "Q2", "Q3", "Q4"]
Tag = Literal[
    "Carve-out Complexity", "Cost Synergies", "Cross-sell", "Diversification",
    "Geographic Expansion", "High Growth", "Margin Improvement", "Market Share",
    "Platform Build", "Scale", "Technology Capability", "Vertical Integration",
]
AcquirerType = Literal["Financial Sponsor", "Strategic"]  # as written in the CSV
BuyerType = Literal["PE", "Strategic"]                     # as used everywhere else
MarginBand = Literal["weak", "medium", "strong"]


# --- One past transaction ---------------------------------------------------

class Deal(BaseModel):
    """One row of the CSV, checked on load."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transaction_id: str = Field(pattern=r"^MA-\d{4}-\d{4}$")
    target_company: str
    acquirer: str
    acquirer_type: AcquirerType
    sector: Sector
    sub_sector: Sector
    deal_year: int
    deal_quarter: Quarter
    deal_type: DealType
    geography: Geography
    financing_type: FinancingType
    target_ownership_pre: Ownership
    # Sizes must be above zero: we divide by revenue and EBITDA below.
    deal_size_mm: float = Field(gt=0)        # enterprise value paid, $M
    target_revenue_mm: float = Field(gt=0)
    target_ebitda_mm: float = Field(gt=0)
    ebitda_margin_pct: float                 # as stated in the CSV
    revenue_growth_pct: float
    ev_ebitda_multiple: float                # as stated in the CSV
    ev_revenue_multiple: float
    synergy_pct_of_deal: float
    strategic_rationale_tags: list[Tag]
    outcome: Outcome
    num_bidders: int = Field(ge=1)
    days_to_close: int | None                # only known for closed deals

    @field_validator("strategic_rationale_tags", mode="before")
    @classmethod
    def split_tags(cls, value):
        """The CSV stores tags as one string: "Scale|Cross-sell" -> ["Scale", "Cross-sell"]."""
        return value.split("|") if isinstance(value, str) else value

    @model_validator(mode="after")
    def close_time_only_when_closed(self):
        """A deal has a time to close if and only if it closed."""
        if (self.outcome == "Closed") != (self.days_to_close is not None):
            raise ValueError("days_to_close must be set for Closed deals and empty otherwise")
        return self

    @computed_field
    @property
    def buyer_type(self) -> BuyerType:
        """The CSV's 'Financial Sponsor' is called 'PE' everywhere else in the code."""
        return "PE" if self.acquirer_type == "Financial Sponsor" else "Strategic"


# --- The company being sold -------------------------------------------------

class TargetProfile(BaseModel):
    """The target, described in the same terms as the deal data.

    A field is None when the request did not state it. None means "don't use
    this to judge similarity", never a guess.
    """

    model_config = ConfigDict(extra="forbid")

    sector: Sector
    ev_mm: float = Field(gt=0)                          # enterprise value, $M
    ownership: list[Ownership] | None = None            # e.g. "private" -> [Private, PE-Backed]
    region: SingleRegion | None = None
    margin_band: MarginBand | None = None               # relative to its own sector
    revenue_mm: float | None = Field(default=None, gt=0)
    ebitda_mm: float | None = Field(default=None, gt=0)


# --- Loading ----------------------------------------------------------------

DEAL_LIST = TypeAdapter(list[Deal])


def load_deals(path: str | Path, strong_percentile: float = 0.65) -> pd.DataFrame:
    """Read the CSV, validate every row, and add the calculated columns.

    Raises pydantic.ValidationError naming the bad row and column if any value
    is outside the allowed values (row numbers count from 0, excluding the header).
    """
    raw = pd.read_csv(path)
    # Empty cells arrive as NaN; the models expect None.
    records = raw.astype(object).where(raw.notna(), None).to_dict("records")
    deals = pd.DataFrame([deal.model_dump() for deal in DEAL_LIST.validate_python(records)])

    # Checks that span rows, so they can't live on a single Deal.
    if deals.transaction_id.duplicated().any():
        raise ValueError("transaction_id values must be unique")
    if (deals.groupby("acquirer").acquirer_type.nunique() > 1).any():
        raise ValueError("each acquirer must have a single acquirer_type")

    # Our own versions of the EBITDA-based figures. The stated margin and
    # EV/EBITDA columns often disagree with revenue and EBITDA in the same row,
    # so all math uses these recalculated columns. EV/Revenue already matches.
    deals["margin_calc"] = deals.target_ebitda_mm / deals.target_revenue_mm * 100
    deals["ev_ebitda_calc"] = deals.deal_size_mm / deals.target_ebitda_mm
    deals["ev_revenue_calc"] = deals.deal_size_mm / deals.target_revenue_mm

    # Margin band of each deal, judged against its own sector.
    cutoffs = margin_cutoffs(deals, strong_percentile)
    median = deals.sector.map(cutoffs["median"])
    strong = deals.sector.map(cutoffs["strong"])
    deals["margin_band"] = np.select(
        [deals.margin_calc >= strong, deals.margin_calc >= median], ["strong", "medium"], "weak"
    )
    return deals


def margin_cutoffs(deals: pd.DataFrame, strong_percentile: float = 0.65) -> pd.DataFrame:
    """Per sector, the margin cut points that define the bands.

    weak   = below the sector median
    medium = median up to the strong cut point
    strong = at or above the strong cut point (65th percentile = top 35%)
    Example (Healthcare Services): median 14.5%, strong from 16.0%.
    """
    cuts = deals.groupby("sector").margin_calc.quantile([0.5, strong_percentile]).unstack()
    cuts.columns = ["median", "strong"]
    return cuts
