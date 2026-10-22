"""What is the target worth, and what does it look like financially?

Two questions, both answered from the deal data:

1. estimate_target_financials(): the request gives only enterprise value, so we
   estimate revenue and EBITDA from similar deals. Used to describe the target,
   never to score deals (that would compare the target with itself).

2. valuation_comps(): the most relevant closed deals, and the multiples they
   were priced at. One comps table per run, shared by every buyer's page.
"""

import pandas as pd
from pydantic import BaseModel

from acquirer_engine.config import Scoring
from acquirer_engine.data import BuyerType, TargetProfile

# Columns shown in the comps table, in display order.
COMPS_COLUMNS = [
    "transaction_id", "target_company", "acquirer", "buyer_type", "sector", "deal_year",
    "deal_size_mm", "ev_revenue_calc", "ev_ebitda_calc", "ev_ebitda_multiple", "relevance",
]


class TargetEstimate(BaseModel):
    """Estimated revenue and EBITDA for the target, with the peers behind them."""

    revenue_mm: float
    ebitda_mm: float
    margin_pct: float
    peer_ids: list[str]        # transaction_ids of the peer deals used
    low_confidence: bool       # True when too few peers to trust the estimate


class MultipleRange(BaseModel):
    """Median and middle half (25th to 75th percentile) of one multiple."""

    median: float
    q1: float
    q3: float


class CompsSummary(BaseModel):
    ev_revenue: MultipleRange                  # main anchor: consistent in the data
    ev_ebitda_calc: MultipleRange              # deal size / EBITDA, our own calculation
    ev_ebitda_stated: MultipleRange            # as stated in the CSV
    ev_revenue_by_type: dict[BuyerType, float] # median EV/Revenue paid by PE vs strategics
    count_by_type: dict[BuyerType, int]


def estimate_target_financials(
    deals: pd.DataFrame, target: TargetProfile, cfg: Scoring
) -> TargetEstimate:
    """Estimate revenue and EBITDA from peer deals.

    Peers = deals in the target's sector with the same margin band (all deals
    in the sector if the band is unknown). Both medians come from the same peers:
        revenue = EV / median EV/Revenue of peers
        EBITDA  = revenue x median margin of peers
    Example ($200M Healthcare Services, strong margins): 16 peers, median
    EV/Revenue 2.37x -> revenue ~$84M; median margin 17.8% -> EBITDA ~$15.0M.

    Only call this when the request did not give revenue or EBITDA.
    """
    peers = deals[deals.sector == target.sector]
    if target.margin_band is not None:
        peers = peers[peers.margin_band == target.margin_band]

    revenue = target.ev_mm / peers.ev_revenue_calc.median()
    margin = peers.margin_calc.median()
    return TargetEstimate(
        revenue_mm=revenue,
        ebitda_mm=revenue * margin / 100,
        margin_pct=margin,
        peer_ids=sorted(peers.transaction_id),
        low_confidence=len(peers) < cfg.min_estimate_peers,
    )


def valuation_comps(scored: pd.DataFrame, cfg: Scoring) -> tuple[pd.DataFrame, CompsSummary]:
    """The top N closed deals by relevance, and a summary of their multiples.

    `scored` is the output of score_deals(). Only closed deals are used: a
    withdrawn or pending deal's price was never actually paid.
    Returns (comps table, summary).
    """
    closed = scored[scored.outcome == "Closed"]
    comps = closed.nlargest(cfg.top_n_comps, "relevance")[COMPS_COLUMNS].reset_index(drop=True)

    summary = CompsSummary(
        ev_revenue=_range(comps.ev_revenue_calc),
        ev_ebitda_calc=_range(comps.ev_ebitda_calc),
        ev_ebitda_stated=_range(comps.ev_ebitda_multiple),
        ev_revenue_by_type=comps.groupby("buyer_type").ev_revenue_calc.median().to_dict(),
        count_by_type=comps.buyer_type.value_counts().to_dict(),
    )
    return comps, summary


def _range(values: pd.Series) -> MultipleRange:
    return MultipleRange(
        median=values.median(), q1=values.quantile(0.25), q3=values.quantile(0.75)
    )
