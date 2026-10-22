"""Everything an analyst may say about one buyer, computed by code.

analyze() runs the scoring once per run. build_evidence() then packs, for one
buyer, every fact and number the analyst is allowed to use. The analyst copies
numbers from the pack; it never calculates them. That is what lets the
validator check that every number on a page is real.

Shares are stored as fractions (0.12 = 12%); multiples as plain numbers (2.49 = 2.49x).
"""

from typing import NamedTuple

import pandas as pd
from pydantic import BaseModel

from acquirer_engine.config import Scoring
from acquirer_engine.data import BuyerType, TargetProfile
from acquirer_engine.schemas import Lens, RiskName
from acquirer_engine.scoring.buyers import rank_buyers
from acquirer_engine.scoring.relevance import FACTORS, score_deals
from acquirer_engine.scoring.risk import (
    LENS,
    Conviction,
    RiskTable,
    assess_conviction,
    risk_flags,
    risk_table,
)
from acquirer_engine.scoring.valuation import (
    CompsSummary,
    TargetEstimate,
    estimate_target_financials,
    valuation_comps,
)

TOP_DEALS = 5
# A tag on more than this share of all deals says nothing about a buyer
# (e.g. "Geographic Expansion" is on 90% of deals), so it is left out.
COMMON_TAG_SHARE = 0.8


class Analysis(NamedTuple):
    """All scoring results for one target, computed once per run."""

    target: TargetProfile
    estimate: TargetEstimate | None
    scored: pd.DataFrame        # every deal with its factor weights and relevance
    rankings: pd.DataFrame      # every buyer with its score and rank within type
    comps: pd.DataFrame         # the valuation comps table
    comps_summary: CompsSummary
    risks: RiskTable
    cfg: Scoring


def analyze(deals: pd.DataFrame, target: TargetProfile, cfg: Scoring) -> Analysis:
    scored = score_deals(deals, target, cfg)
    rankings = rank_buyers(scored, cfg)
    comps, summary = valuation_comps(scored, cfg)
    # Estimate revenue / EBITDA only if the request didn't give them.
    needs_estimate = target.revenue_mm is None or target.ebitda_mm is None
    estimate = estimate_target_financials(deals, target, cfg) if needs_estimate else None
    return Analysis(target, estimate, scored, rankings, comps, summary,
                    risk_table(scored, rankings, target, cfg), cfg)


# --- The pack -----------------------------------------------------------------

class DealEvidence(BaseModel):
    transaction_id: str
    target_company: str
    year: int
    sector: str
    deal_type: str
    size_mm: float
    outcome: str
    ev_revenue: float
    ev_ebitda_calc: float
    ev_ebitda_stated: float
    relevance: float
    factors: dict[str, float]   # e.g. {"sector": 1.0, "size": 0.93, ...}; relevance = their product


class BuyerProfile(BaseModel):
    buyer_type: BuyerType
    deals: int
    closed_deals: int
    strong_deals: int              # relevance at or above the strong threshold
    best_relevance: float
    typical_size_mm: float         # median deal size
    first_year: int
    latest_year: int
    sectors: dict[str, int]        # deals per sector, most first
    top_tags: dict[str, int]       # most frequent rationale tags (common ones left out)
    rank_in_type: int
    score: float                   # PE: top-3 relevance sum (0-3); strategic: shrunk rate (0-1)
    range_90: tuple[float, float] | None   # strategics only


class WhatTheyPaid(BaseModel):
    """Medians over the buyer's own closed deals."""

    closed_deals: int
    ev_revenue: float | None
    ev_ebitda_calc: float | None
    ev_ebitda_stated: float | None


class RiskEvidence(BaseModel):
    name: RiskName
    lens: Lens
    status: str                  # fired | watch
    value: float | None          # the raw signal (see `detail`)
    percentile: float | None     # vs buyers of the same type, 1.0 = riskiest
    count: int | None = None     # e.g. 2 deals withdrawn ...
    out_of: int | None = None    # ... out of 17
    detail: str                  # what value and count mean, in words
    evidence_ids: list[str] = []


class EvidencePack(BaseModel):
    buyer: str
    target: TargetProfile
    target_estimate: TargetEstimate | None
    profile: BuyerProfile
    top_deals: list[DealEvidence]
    comps: CompsSummary
    type_median_ev_revenue: float | None   # what buyers of this type paid in the comps
    paid: WhatTheyPaid
    risks: list[RiskEvidence]
    conviction: Conviction
    theses: list[str] = []                 # filled once coverage agents run
    coverage_note: str | None = None
    captain_reason: str | None = None
    anomalies: list[str] = []


def build_evidence(buyer: str, analysis: Analysis) -> EvidencePack:
    a = analysis
    deals = a.scored[a.scored.acquirer == buyer].sort_values("relevance", ascending=False)
    ranking = a.rankings.loc[buyer]
    closed = deals[deals.outcome == "Closed"]

    return EvidencePack(
        buyer=buyer,
        target=a.target,
        target_estimate=a.estimate,
        profile=_profile(deals, ranking, a),
        top_deals=[_deal(row) for _, row in deals.head(TOP_DEALS).iterrows()],
        comps=a.comps_summary,
        type_median_ev_revenue=a.comps_summary.ev_revenue_by_type.get(ranking.buyer_type),
        paid=WhatTheyPaid(
            closed_deals=len(closed),
            ev_revenue=_median(closed.ev_revenue_calc),
            ev_ebitda_calc=_median(closed.ev_ebitda_calc),
            ev_ebitda_stated=_median(closed.ev_ebitda_multiple),
        ),
        risks=[_risk(flag.name, flag.status, deals, buyer, a) for flag in risk_flags(buyer, a.risks)],
        conviction=assess_conviction(buyer, a.scored, a.risks, a.cfg),
        anomalies=_anomalies(deals, ranking.buyer_type, a.cfg),
    )


def _profile(deals: pd.DataFrame, ranking: pd.Series, a: Analysis) -> BuyerProfile:
    tag_share = a.scored.strategic_rationale_tags.explode().value_counts() / len(a.scored)
    common = set(tag_share[tag_share > COMMON_TAG_SHARE].index)
    tags = deals.strategic_rationale_tags.explode().value_counts()
    is_strategic = ranking.buyer_type == "Strategic"
    return BuyerProfile(
        buyer_type=ranking.buyer_type,
        deals=len(deals),
        closed_deals=int((deals.outcome == "Closed").sum()),
        strong_deals=int(ranking.strong_deals),
        best_relevance=float(ranking.best),
        typical_size_mm=float(deals.deal_size_mm.median()),
        first_year=int(deals.deal_year.min()),
        latest_year=int(deals.deal_year.max()),
        sectors=deals.sector.value_counts().to_dict(),
        top_tags=tags.drop(labels=list(common & set(tags.index))).head(4).to_dict(),
        rank_in_type=int(ranking.type_rank),
        score=float(ranking.score),
        range_90=(float(ranking.low90), float(ranking.high90)) if is_strategic else None,
    )


def _deal(row: pd.Series) -> DealEvidence:
    return DealEvidence(
        transaction_id=row.transaction_id,
        target_company=row.target_company,
        year=int(row.deal_year),
        sector=row.sector,
        deal_type=row.deal_type,
        size_mm=float(row.deal_size_mm),
        outcome=row.outcome,
        ev_revenue=float(row.ev_revenue_calc),
        ev_ebitda_calc=float(row.ev_ebitda_calc),
        ev_ebitda_stated=float(row.ev_ebitda_multiple),
        relevance=float(row.relevance),
        factors={name.removeprefix("w_"): float(row[name]) for name in FACTORS},
    )


def _risk(name: RiskName, status: str, deals: pd.DataFrame, buyer: str, a: Analysis) -> RiskEvidence:
    """One risk with the counts and deal IDs behind it, in words the writer can use."""
    values = a.risks.values.loc[buyer]
    pct = a.risks.percentiles.loc[buyer, name]
    base = dict(name=name, lens=LENS[name], status=status,
                percentile=None if pd.isna(pct) else float(pct))
    ids = lambda mask: sorted(deals.loc[mask, "transaction_id"])  # noqa: E731

    if name == "execution":
        mask = deals.outcome.isin(["Withdrawn", "Terminated"])
        return RiskEvidence(**base, value=float(values.execution), count=int(mask.sum()),
                            out_of=len(deals), evidence_ids=ids(mask),
                            detail="share of deals withdrawn or terminated")
    if name == "financing":
        mask = deals.financing_type == "Leveraged"
        return RiskEvidence(**base, value=float(values.leverage), count=int(mask.sum()),
                            out_of=len(deals), evidence_ids=ids(mask),
                            detail=f"share of deals financed with leverage; median deal size "
                                   f"${values.median_size:.1f}M vs target EV ${a.target.ev_mm:.0f}M")
    if name == "integration":
        mask = deals.w_sector < a.cfg.integration_sector_weight
        return RiskEvidence(**base, value=float(values.integration), count=int(mask.sum()),
                            out_of=len(deals), evidence_ids=ids(mask),
                            detail="share of deals in sectors far from the target's")
    if name == "antitrust":
        mask = deals.sector == a.target.sector
        return RiskEvidence(**base, value=float(values.same_sector_deals), count=int(mask.sum()),
                            out_of=len(deals), evidence_ids=ids(mask),
                            detail="deals already done in the target's sector")
    if name == "bidders":
        return RiskEvidence(**base, value=float(values.bidders), out_of=len(deals),
                            detail="average number of bidders on its deals (fewer = less auction-tested)")
    if name == "time_to_close":
        value = values.time_to_close
        return RiskEvidence(**base, value=None if pd.isna(value) else float(value),
                            detail="median days from announcement to close, closed deals")
    value = values.price  # price
    return RiskEvidence(**base, value=None if pd.isna(value) else float(value),
                        detail=f"median EV/Revenue paid on closed deals vs comps median "
                               f"{a.comps_summary.ev_revenue.median:.2f}x")


def _anomalies(deals: pd.DataFrame, buyer_type: str, cfg: Scoring) -> list[str]:
    """Data quirks worth a caveat on the page. The analyst decides which to mention."""
    notes = []
    if buyer_type == "PE":
        odd = deals[deals.deal_type.isin(["Merger of Equals", "SPAC Merger"])]
        if len(odd):
            notes.append(f"PE buyer labeled on {len(odd)} merger-type deal(s) "
                         f"({', '.join(sorted(odd.transaction_id))}); labels may be noisy")
    else:
        odd = deals[deals.deal_type.isin(["Leveraged Buyout", "Management Buyout"])]
        if len(odd):
            notes.append(f"Strategic buyer on {len(odd)} LBO/MBO deal(s) "
                         f"({', '.join(sorted(odd.transaction_id))}); unusual for an operating company")
    top = deals.head(TOP_DEALS)
    gap = (top.ev_ebitda_multiple - top.ev_ebitda_calc).abs() / top.ev_ebitda_calc
    mismatched = top[gap > cfg.ebitda_mismatch]
    if len(mismatched):
        notes.append(f"Stated and calculated EV/EBITDA differ by more than "
                     f"{cfg.ebitda_mismatch:.0%} on {', '.join(sorted(mismatched.transaction_id))}; "
                     f"EV/Revenue is the more reliable multiple")
    return notes


def _median(values: pd.Series) -> float | None:
    return None if values.empty else float(values.median())
