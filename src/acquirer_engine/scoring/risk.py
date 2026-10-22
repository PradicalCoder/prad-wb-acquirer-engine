"""Buyer-specific risks and the conviction rule.

Risks: each signal is measured from the buyer's own deals, then compared with
buyers of the same type (PE vs PE, strategic vs strategic), because the two
behave differently (PE uses far more leverage, for example). A risk fires when
the buyer is among the worst 25% of its type. No fixed "magic numbers", so the
thresholds adapt to any dataset.

Risks are grouped the way a sell-side banker weighs a bid:
  Value     - will they pay up?           price
  Certainty - will the deal close?        financing, antitrust, execution, integration
  Speed     - can they move fast and win? bidders, time_to_close

Conviction = willingness x ability:
  willingness High: at least two strong precedents, the latest one recent
  ability High:     at most max_ability_risks Value/Certainty/Speed risks fired
                    (integration is shown but not counted)
  High x High -> High, Low x Low -> Low, otherwise Medium.
"""

from typing import Literal, NamedTuple, get_args

import pandas as pd
from pydantic import BaseModel

from acquirer_engine.config import Scoring
from acquirer_engine.data import TargetProfile
from acquirer_engine.schemas import ConvictionLevel, Lens, RiskName

Level = Literal["High", "Low"]   # willingness and ability

SIGNALS: list[RiskName] = list(get_args(RiskName))
LENS: dict[RiskName, Lens] = {
    "price": "Value",
    "financing": "Certainty", "antitrust": "Certainty",
    "execution": "Certainty", "integration": "Certainty",
    "bidders": "Speed", "time_to_close": "Speed",
}
# Risks that count against ability. Integration is excluded: buying across
# sectors is normal for PE generalists, so it is shown but doesn't lower conviction.
ABILITY = ["financing", "execution", "bidders", "time_to_close", "price", "antitrust"]
# The raw value shown for each risk (column of RiskTable.values).
VALUE_COLUMN = {
    "integration": "integration", "financing": "leverage", "execution": "execution",
    "bidders": "bidders", "time_to_close": "time_to_close", "price": "price",
    "antitrust": "same_sector_deals",
}


class RiskTable(NamedTuple):
    values: pd.DataFrame       # raw signal per buyer (e.g. share of deals withdrawn)
    percentiles: pd.DataFrame  # rank within type, 0 = safest, 1 = riskiest
    fired: pd.DataFrame        # True where the risk is flagged


class RiskFlag(BaseModel):
    name: RiskName
    lens: Lens
    status: Literal["fired", "watch"]  # watch = not flagged, but the buyer's weakest spots
    value: float | None
    percentile: float | None


class Conviction(BaseModel):
    willingness: Level
    ability: Level
    level: ConvictionLevel
    strong_deal_ids: list[str]
    latest_strong_year: int | None
    ability_risks: list[str]


def risk_table(
    scored: pd.DataFrame, rankings: pd.DataFrame, target: TargetProfile, cfg: Scoring
) -> RiskTable:
    """Compute every risk signal for every buyer.

    `scored` comes from score_deals(), `rankings` from rank_buyers().
    """
    by_buyer = scored.groupby("acquirer")
    closed = scored[scored.outcome == "Closed"].groupby("acquirer")
    values = pd.DataFrame({
        # Share of deals in sectors far from the target's (less relevant integration experience).
        "integration": by_buyer.w_sector.agg(lambda w: (w < cfg.integration_sector_weight).mean()),
        "leverage": by_buyer.financing_type.agg(lambda f: (f == "Leveraged").mean()),
        "median_size": by_buyer.deal_size_mm.median(),
        "execution": by_buyer.outcome.agg(lambda o: o.isin(["Withdrawn", "Terminated"]).mean()),
        "bidders": by_buyer.num_bidders.mean(),
        # Only closed deals have a close time and a price actually paid; buyers
        # with no closed deals get no value here (and no risk).
        "time_to_close": closed.days_to_close.median(),
        "price": closed.ev_revenue_calc.median(),
        "same_sector_deals": by_buyer.sector.agg(lambda s: (s == target.sector).sum()),
    })

    # Percentile within type, with every signal pointing the risky way (higher =
    # riskier). Fewer bidders = less tested in auctions; lower price paid = may underbid.
    risky = values[["integration", "leverage", "execution", "time_to_close"]].assign(
        bidders=-values.bidders, price=-values.price
    )
    buyer_type = rankings.buyer_type.reindex(values.index)
    percentiles = risky.groupby(buyer_type).rank(pct=True)
    fired = percentiles >= cfg.risk_percentile

    # Financing: deals typically much smaller than ours, or heavy reliance on leverage.
    too_small = values.median_size < cfg.financing_size_ratio * target.ev_mm
    fired["financing"] = too_small | fired.pop("leverage")
    percentiles["financing"] = percentiles.pop("leverage")

    # Antitrust: a strategic that already owns several businesses in our sector.
    # A fixed count, not a peer comparison: it is about overlap. Not applied to PE.
    overlap = (buyer_type == "Strategic") & (values.same_sector_deals >= cfg.antitrust_min_deals)
    fired["antitrust"] = overlap
    percentiles["antitrust"] = overlap.astype(float)

    return RiskTable(values, percentiles[SIGNALS], fired[SIGNALS])


def risk_flags(buyer: str, table: RiskTable) -> list[RiskFlag]:
    """The risks to show for one buyer: every fired risk, riskiest first.

    Every page shows at least two. If fewer than two fired, the buyer's next
    weakest signals are added as "watch" items (not alarms).
    """
    pct = table.percentiles.loc[buyer]
    fired = table.fired.loc[buyer]

    def flag(name, status):
        value = table.values.loc[buyer, VALUE_COLUMN[name]]
        return RiskFlag(
            name=name, lens=LENS[name], status=status,
            value=None if pd.isna(value) else float(value),
            percentile=None if pd.isna(pct[name]) else float(pct[name]),
        )

    by_risk = pct.sort_values(ascending=False, na_position="last").index
    flags = [flag(name, "fired") for name in by_risk if fired[name]]
    # Antitrust is a yes/no rule, so it is never a watch item.
    watch = [n for n in by_risk if not fired[n] and n != "antitrust" and pd.notna(pct[n])]
    flags += [flag(name, "watch") for name in watch[: max(0, 2 - len(flags))]]
    return flags


def assess_conviction(
    buyer: str, scored: pd.DataFrame, table: RiskTable, cfg: Scoring
) -> Conviction:
    """Willingness x ability -> High / Medium / Low, with the evidence behind it.

    Example: KKR has two strong precedents, but both are from 2019, so
    willingness is Low; one ability risk fired, so ability is High -> Medium.
    """
    deals = scored[scored.acquirer == buyer]
    strong = deals[deals.relevance >= cfg.strong_relevance]
    latest = int(strong.deal_year.max()) if len(strong) else None
    willing = len(strong) >= 2 and latest >= cfg.recent_year

    ability_risks = [name for name in ABILITY if table.fired.loc[buyer, name]]
    able = len(ability_risks) <= cfg.max_ability_risks

    if willing and able:
        level = "High"
    elif not willing and not able:
        level = "Low"
    else:
        level = "Medium"
    return Conviction(
        willingness="High" if willing else "Low",
        ability="High" if able else "Low",
        level=level,
        strong_deal_ids=sorted(strong.transaction_id),
        latest_strong_year=latest,
        ability_risks=ability_risks,
    )
