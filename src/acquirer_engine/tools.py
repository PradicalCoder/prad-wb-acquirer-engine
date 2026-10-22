"""The coverage agents' research tools, each set locked to one buyer type.

CoverageTools("PE", ...) can only ever see PE buyers and their deals; the
strategic set only strategics. The buyer type is fixed when the set is made,
and no tool takes a buyer-type argument, so the agent cannot ask for the other
type. The ranking method is fixed the same way (top-3 sum for PE, shrinkage for
strategics); the agent only reads the results.

Outputs are compact JSON (grouped summaries, top N rows) to keep prompts small.
Every buyer and deal ID a tool returns is remembered in `seen_buyers` /
`seen_ids`: a coverage agent may only put forward buyers and cite deals that
appeared in its tool results this run.
"""

import json

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from acquirer_engine.config import Config, Thesis
from acquirer_engine.data import BuyerType, TargetProfile
from acquirer_engine.evidence import Analysis
from acquirer_engine.schemas import DealSearch

DEAL_COLUMNS = ["transaction_id", "acquirer", "deal_year", "sector", "deal_type",
                "deal_size_mm", "outcome", "relevance", "strategic_rationale_tags",
                "synergy_pct_of_deal"]


class AcquirerName(BaseModel):
    name: str = Field(description="Exact acquirer name, as returned by another tool.")


class RankingsRequest(BaseModel):
    top_k: int = Field(12, ge=1, le=30, description="How many top-ranked buyers to list.")


class CoverageTools:
    """search_deals, get_acquirer_profile and get_buyer_rankings for one buyer type."""

    def __init__(self, buyer_type: BuyerType, analysis: Analysis, cfg: Config):
        self.buyer_type = buyer_type
        self.cfg = cfg
        # The type lock: everything below works on this buyer type's rows only.
        self.deals = analysis.scored[analysis.scored.buyer_type == buyer_type]
        self.rankings = analysis.rankings[analysis.rankings.buyer_type == buyer_type]
        self.type_of = analysis.rankings.buyer_type  # name -> type, only to explain refusals
        self.seen_buyers: set[str] = set()
        self.seen_ids: set[str] = set()

    # --- what the model is told about the tools ---------------------------------

    def definitions(self) -> list[dict]:
        kind, one = (("PE firms", "PE firm") if self.buyer_type == "PE"
                     else ("strategic (operating company) buyers", "strategic buyer"))
        score = ("score = sum of the buyer's three most relevant deals (0-3)" if self.buyer_type == "PE"
                 else "score = shrunk average deal relevance (0-1) with a 90% range; one or two "
                      "deals is normal for a strategic and gives a wide range")
        return [
            {"name": "search_deals",
             "description": f"Search past deals by {kind} with filters. With group_by_buyer, "
                            f"returns one row per buyer: matching deals, best relevance, top deal "
                            f"IDs, rank. Otherwise returns the top deals.",
             "input_schema": _input_schema(DealSearch)},
            {"name": "get_acquirer_profile",
             "description": f"One {one}'s deal history: "
                            f"counts, typical size, sectors, deal types, tags and top deals."
                            + (" Includes primary_sector_proxy, the sector it buys most (an "
                               "inference from deals, not a known fact)." if self.buyer_type == "Strategic" else ""),
             "input_schema": _input_schema(AcquirerName)},
            {"name": "get_buyer_rankings",
             "description": f"The top-ranked {kind} for this target; {score}.",
             "input_schema": _input_schema(RankingsRequest)},
        ]

    def call(self, name: str, args: dict) -> str:
        """Run one tool call from the model. Bad arguments come back as an error
        message the model can read, never as an exception."""
        try:
            if name == "search_deals":
                result = self.search_deals(DealSearch.model_validate(args))
            elif name == "get_acquirer_profile":
                result = self.get_acquirer_profile(AcquirerName.model_validate(args).name)
            elif name == "get_buyer_rankings":
                result = self.get_buyer_rankings(RankingsRequest.model_validate(args).top_k)
            else:
                result = {"error": f"unknown tool '{name}'"}
        except ValidationError as error:
            result = {"error": f"invalid arguments: {error.errors(include_url=False)}"}
        return to_json(result)

    # --- the tools ---------------------------------------------------------------

    def search_deals(self, f: DealSearch) -> dict:
        d = self.deals
        if f.sectors:
            d = d[d.sector.isin(f.sectors)]
        if f.deal_types:
            d = d[d.deal_type.isin(f.deal_types)]
        if f.outcomes:
            d = d[d.outcome.isin(f.outcomes)]
        if f.size_min is not None:
            d = d[d.deal_size_mm >= f.size_min]
        if f.size_max is not None:
            d = d[d.deal_size_mm <= f.size_max]
        if f.year_min is not None:
            d = d[d.deal_year >= f.year_min]
        if f.year_max is not None:
            d = d[d.deal_year <= f.year_max]
        if f.tags_any:
            d = d[d.strategic_rationale_tags.apply(lambda tags: bool(set(tags) & set(f.tags_any)))]
        d = d[d.relevance >= f.min_relevance].sort_values("relevance", ascending=False)

        if not f.group_by_buyer:
            rows = _round(d[DEAL_COLUMNS].head(self.cfg.coverage.top_n))
            return self._seen({"deals_found": len(d), "deals": rows})

        buyers = []
        for name, group in d.groupby("acquirer"):
            buyers.append({"acquirer": name, "matching_deals": len(group),
                           "best_relevance": round(group.relevance.max(), 3),
                           "top_deal_ids": list(group.transaction_id.head(3)),
                           "rank_in_type": int(self.rankings.loc[name, "type_rank"])})
        buyers.sort(key=lambda b: -b["best_relevance"])
        return self._seen({"deals_found": len(d), "buyers_found": len(buyers),
                           "buyers": buyers[: self.cfg.coverage.top_n]})

    def get_acquirer_profile(self, name: str) -> dict:
        if name not in self.rankings.index:
            if name in self.type_of.index:
                return {"error": f"{name} is a {self.type_of[name]} buyer; this tool only "
                                 f"covers {self.buyer_type} buyers"}
            return {"error": f"no {self.buyer_type} buyer named '{name}'"}
        d = self.deals[self.deals.acquirer == name].sort_values("relevance", ascending=False)
        r = self.rankings.loc[name]
        profile = {
            "acquirer": name, "deals": len(d), "closed": int((d.outcome == "Closed").sum()),
            "typical_size_mm": round(d.deal_size_mm.median(), 1),
            "years": [int(d.deal_year.min()), int(d.deal_year.max())],
            "sectors": d.sector.value_counts().to_dict(),
            "deal_types": d.deal_type.value_counts().to_dict(),
            "top_tags": d.strategic_rationale_tags.explode().value_counts().head(5).to_dict(),
            "strong_deals": int(r.strong_deals), "rank_in_type": int(r.type_rank),
            "score": round(r.score, 3),
            "top_deals": _round(d[DEAL_COLUMNS].head(5)),
        }
        if self.buyer_type == "Strategic":
            profile["range_90"] = [round(r.low90, 3), round(r.high90, 3)]
            profile["primary_sector_proxy"] = d.sector.value_counts().index[0]
        return self._seen(profile)

    def get_buyer_rankings(self, top_k: int) -> dict:
        rows = []
        for name, r in self.rankings.nsmallest(top_k, "type_rank").iterrows():
            row = {"acquirer": name, "rank": int(r.type_rank), "score": round(r.score, 3),
                   "deals": int(r.deals), "strong_deals": int(r.strong_deals),
                   "best_relevance": round(r.best, 3)}
            if self.buyer_type == "Strategic":
                row["range_90"] = [round(r.low90, 3), round(r.high90, 3)]
            rows.append(row)
        return self._seen({"buyer_type": self.buyer_type, "rankings": rows})

    # --- helpers -------------------------------------------------------------------

    def _seen(self, result: dict) -> dict:
        """Remember every buyer name and deal ID in a result (for the validator)."""
        def walk(value, key=None):
            if isinstance(value, dict):
                for k, v in value.items():
                    walk(v, k)
            elif isinstance(value, list):
                for v in value:
                    walk(v, key)
            elif key == "acquirer":
                self.seen_buyers.add(value)
            elif isinstance(value, str) and value.startswith("MA-"):
                self.seen_ids.add(value)
        walk(result)
        return result


def to_json(result: dict) -> str:
    """Compact JSON; numpy numbers from pandas become plain numbers."""
    return json.dumps(result, separators=(",", ":"),
                      default=lambda v: v.item() if hasattr(v, "item") else str(v))


def run_seed_theses(tools: CoverageTools, target: TargetProfile, cfg: Config) -> list[dict]:
    """Run this buyer type's seed theses in code, before the agent's first turn.
    Their results count as tool results (they pass through search_deals)."""
    family = next((sectors for sectors in cfg.coverage.sector_families.values()
                   if target.sector in sectors), [target.sector])
    seeds = []
    for thesis_id, thesis in cfg.coverage.theses.items():
        if thesis.buyer_type != tools.buyer_type:
            continue
        filters = thesis_filters(thesis, target, family)
        seeds.append({"thesis_id": thesis_id, "name": thesis.name, "story": thesis.story,
                      "filters": filters.model_dump(exclude_defaults=True),
                      "result": tools.search_deals(filters)})
    return seeds


def thesis_filters(thesis: Thesis, target: TargetProfile, family: list[str]) -> DealSearch:
    """A thesis's filters for this target: size bands are multiples of the target's EV."""
    low, high = thesis.size
    return DealSearch(
        sectors=thesis.sectors or family, deal_types=thesis.deal_types, tags_any=thesis.tags_any,
        size_min=round(low * target.ev_mm, 1), size_max=round(high * target.ev_mm, 1),
        year_min=thesis.year_min, min_relevance=thesis.min_relevance,
    )


def _input_schema(model: type[BaseModel]) -> dict:
    schema = model.model_json_schema()
    schema.pop("title", None)
    return schema


def _round(frame: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
            for row in frame.to_dict("records")]
