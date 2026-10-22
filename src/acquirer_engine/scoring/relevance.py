"""How relevant is each past deal to the target?

Every deal gets seven factor weights between 0 and 1, and its relevance is their
product:

    relevance = sector x size x deal type x ownership x margin x geography x recency

Read it as "how much of a perfect precedent is this deal". Example: a deal that is
an 89% sector match, keeps 93% after the size gap and 90% after its age scores
0.89 x 0.93 x 0.90 = 0.75 (if every other factor is 1.0). Multiplying means one
big miss (wrong sector, very different size) drags the score down hard, while
small misses only trim it.

Every deal is scored, whatever its outcome: a withdrawn deal still shows what the
buyer wanted. Only valuation comps are limited to closed deals.
"""

import numpy as np
import pandas as pd

from acquirer_engine.config import Scoring
from acquirer_engine.data import TargetProfile

# Order of the factor columns added by score_deals(): from deal-breakers to tie-breakers.
FACTORS = [
    "w_sector", "w_size", "w_deal_type", "w_ownership", "w_margin", "w_geography", "w_recency",
]


def sector_buyers(deals: pd.DataFrame, sector: str) -> set[str]:
    """Acquirers with at least one deal in the sector."""
    return set(deals.loc[deals.sector == sector, "acquirer"])


def sector_weights(deals: pd.DataFrame, sector: str, cfg: Scoring) -> pd.DataFrame:
    """Weight of every sector relative to the target's sector, based on buyer overlap.

    Lift asks: do buyers of the target's sector buy in sector X more often than
    deals in general fall in X?
      1. Take every acquirer with a deal in the target sector.
      2. Look at their deals in all other sectors.
      3. share_X = fraction of those deals in sector X.
      4. base_X  = fraction of all deals outside the target sector that are in X.
      5. lift_X  = share_X / base_X   (1.0 = no pattern, 2.0 = twice as common)

    Weight = clip(sector_mult x lift, floor, cap); the target's own sector is 1.0.
    Example (Healthcare Services): Physician Groups lift 1.77 -> 0.5 x 1.77 = 0.89.
    A sector these buyers never entered gets lift 0, so the floor weight.

    Returns a table indexed by sector with columns lift and weight.
    """
    outside = deals[deals.sector != sector]
    their_deals = outside[outside.acquirer.isin(sector_buyers(deals, sector))]

    base = outside.sector.value_counts(normalize=True)
    share = their_deals.sector.value_counts(normalize=True).reindex(base.index, fill_value=0)
    lift = share / base

    table = pd.DataFrame({"lift": lift})
    table["weight"] = (cfg.sector_mult * lift).clip(cfg.sector_floor, cfg.sector_cap)
    table.loc[sector] = [np.nan, 1.0]  # own sector: no lift needed, full weight
    return table


def score_deals(deals: pd.DataFrame, target: TargetProfile, cfg: Scoring) -> pd.DataFrame:
    """Return a copy of `deals` with the seven factor columns and `relevance` added.

    A target field that is None (not stated) sets that factor to 1.0 for every
    deal: it is left out of the comparison rather than guessed.
    """
    scored = deals.copy()

    # Sector: lift-based weight, see sector_weights().
    weights = sector_weights(deals, target.sector, cfg)["weight"]
    scored["w_sector"] = scored.sector.map(weights)

    # Size: measured in doublings, so half and double the target's size are equally far.
    #   doublings = |log2(deal size / target EV)|
    #   weight    = max(size_base ^ (doublings ^ 2), size_floor)
    # Example ($200M target, size_base 0.8): $400M deal -> 1 doubling -> 0.8.
    doublings = np.abs(np.log2(scored.deal_size_mm / target.ev_mm))
    scored["w_size"] = np.maximum(cfg.size_base ** (doublings**2), cfg.size_floor)

    # Deal type: whole-company control deals count most (tiers set in config).
    scored["w_deal_type"] = scored.deal_type.map(cfg.deal_type_tiers)
    if scored.w_deal_type.isna().any():
        missing = sorted(scored.loc[scored.w_deal_type.isna(), "deal_type"].unique())
        raise ValueError(f"deal_type_tiers in config has no weight for: {missing}")

    # Ownership: full weight if the deal's target had an ownership type like ours.
    if target.ownership is None:
        scored["w_ownership"] = 1.0
    else:
        same = scored.target_ownership_pre.isin(target.ownership)
        scored["w_ownership"] = np.where(same, 1.0, cfg.ownership_miss)

    # Margin: how many bands apart the deal is from the target (each deal's band
    # is judged within its own sector). Same band -> 1.0, one apart -> margin_medium,
    # two apart -> margin_weak. For a strong-margin target: strong 1.0,
    # medium 0.85, weak 0.7.
    if target.margin_band is None:
        scored["w_margin"] = 1.0
    else:
        rank = {"weak": 0, "medium": 1, "strong": 2}
        gap = (scored.margin_band.map(rank) - rank[target.margin_band]).abs()
        scored["w_margin"] = gap.map({0: 1.0, 1: cfg.margin_medium, 2: cfg.margin_weak})

    # Geography: with a known region, deals in that region or by buyers active
    # nationally / across regions count fully; other single regions are discounted.
    if target.region is None:
        scored["w_geography"] = 1.0
    else:
        near = scored.geography.isin([target.region, "Multi-Regional", "National"])
        scored["w_geography"] = np.where(near, 1.0, cfg.geography_miss)

    # Recency: age counted back from the latest year in the data, not today.
    # recency_steps [[1, 1.0], [3, 0.9], [6, 0.75]] means 0-1 years -> 1.0,
    # 2-3 -> 0.9, 4-6 -> 0.75, older -> recency_old.
    age = deals.deal_year.max() - scored.deal_year
    scored["w_recency"] = np.select(
        [age <= max_age for max_age, _ in cfg.recency_steps],
        [weight for _, weight in cfg.recency_steps],
        cfg.recency_old,
    )

    scored["relevance"] = scored[FACTORS].prod(axis=1)
    return scored
