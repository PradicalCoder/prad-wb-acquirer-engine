"""Rank buyers by the strength of their evidence, separately for PE and strategics.

PE and strategic buyers are ranked with different statistics because their deal
histories look very different in this data: PE firms average about 19 deals
each, strategics about 2. Any score based on deal counts would bury every
strategic under the PE firms.

- PE: score = sum of the buyer's three most relevant deals (0 to 3).
  PE deal counts are similar (14-25), so "how good are your best matches" is a
  fair comparison.
- Strategics: score = average deal relevance, shrunk toward the strategic
  average (0 to 1, with a 90% range). With one or two deals, a single lucky
  deal shouldn't outrank a proven pattern, so thin records are pulled toward
  the group average until they have more evidence.

The two scores are on different scales, so the final 10 is picked as a fixed
mix (default 6 PE / 4 strategic) rather than by merging one list.
"""

import pandas as pd
from scipy.stats import beta as beta_distribution

from acquirer_engine.config import Scoring
from acquirer_engine.data import BuyerType

CONVICTION_ORDER = {"High": 0, "Medium": 1, "Low": 2}


def rank_buyers(scored: pd.DataFrame, cfg: Scoring) -> pd.DataFrame:
    """One row per acquirer with its evidence summary, score and rank within its type.

    `scored` is the output of score_deals(). Columns returned:
      buyer_type, deals, strong_deals (relevance >= strong_relevance), best,
      total (sum of relevance), top3, score, low90 / high90 (strategics only),
      type_rank (1 = best of its type).
    """
    by_buyer = scored.groupby("acquirer").relevance
    buyers = pd.DataFrame({
        "buyer_type": scored.groupby("acquirer").buyer_type.first(),
        "deals": by_buyer.size(),
        "strong_deals": by_buyer.agg(lambda r: (r >= cfg.strong_relevance).sum()),
        "best": by_buyer.max(),
        "total": by_buyer.sum(),
        "top3": by_buyer.agg(lambda r: r.nlargest(3).sum()),
    })

    is_pe = buyers.buyer_type == "PE"
    buyers.loc[is_pe, "score"] = buyers.loc[is_pe, "top3"]
    buyers.loc[~is_pe, ["score", "low90", "high90"]] = _shrink(buyers[~is_pe])

    # Rank within each type; ties keep alphabetical order.
    buyers["type_rank"] = (
        buyers.groupby("buyer_type").score.rank(ascending=False, method="first").astype(int)
    )
    return buyers.sort_values(["buyer_type", "type_rank"])


def shrinkage_prior(strategics: pd.DataFrame) -> tuple[float, float]:
    """The strategic group's average relevance rate and how strongly to pull toward it.

    Each buyer's raw rate = total relevance / number of deals.
      mean = average raw rate across strategics
      k    = mean x (1 - mean) / variance - 1
    k works like a number of "pretend deals" at the group average added to every
    buyer's record. The more alike the buyers are (small variance), the larger k,
    and the harder thin records are pulled to the average. This is the method of
    moments for a beta distribution: the data itself decides how skeptical to be.
    Strategics in this data: mean 0.114, k about 11.
    """
    raw_rate = strategics.total / strategics.deals
    mean, variance = raw_rate.mean(), raw_rate.var()
    return mean, mean * (1 - mean) / variance - 1


def _shrink(strategics: pd.DataFrame) -> pd.DataFrame:
    """Shrunk score and 90% range for each strategic.

    score = (total relevance + k x mean) / (deals + k)
    Example: Oscar Health, 1 deal at 0.52 -> (0.52 + 11 x 0.114) / (1 + 11) = 0.149,
    far below its raw 0.52, because one deal is thin evidence.

    The 90% range comes from a beta distribution with
      alpha = total + k x mean            (relevance "for")
      beta  = deals - total + k x (1 - mean)   (relevance "against")
    whose average is exactly the score above. Few deals -> wide range.
    """
    mean, k = shrinkage_prior(strategics)
    alpha = strategics.total + k * mean
    beta = strategics.deals - strategics.total + k * (1 - mean)
    return pd.DataFrame({
        "score": alpha / (alpha + beta),
        "low90": beta_distribution.ppf(0.05, alpha, beta),
        "high90": beta_distribution.ppf(0.95, alpha, beta),
    }, index=strategics.index)


def top_of_type(rankings: pd.DataFrame, buyer_type: BuyerType, n: int) -> pd.DataFrame:
    """The n best-ranked buyers of one type, best first."""
    return rankings[rankings.buyer_type == buyer_type].nsmallest(n, "type_rank")


def default_pick(rankings: pd.DataFrame, cfg: Scoring) -> list[str]:
    """The final 10 straight from the rankings: top mix_pe PE + top mix_strategic strategics.

    Used as the baseline the deal captain is compared against, and as the
    fallback if the captain fails.
    """
    pe = top_of_type(rankings, "PE", cfg.mix_pe)
    strategic = top_of_type(rankings, "Strategic", cfg.mix_strategic)
    return list(pe.index) + list(strategic.index)


def order_final(names: list[str], rankings: pd.DataFrame, conviction: dict[str, str]) -> list[str]:
    """Order the final list so conviction never rises further down the page.

    Sort by conviction (High, Medium, Low); within a level, by number of strong
    precedents, then by best single deal. These measures mean the same thing for
    PE and strategics, so both types share one order.
    """
    table = rankings.loc[names, ["strong_deals", "best"]].assign(
        level=[CONVICTION_ORDER[conviction[name]] for name in names]
    )
    ordered = table.sort_values(["level", "strong_deals", "best"], ascending=[True, False, False])
    return list(ordered.index)
