"""What each agent must return. Every LLM call is parsed into one of these models.

Pydantic enforces the rules that need no data: allowed values, counts, no
duplicates, "a reason is required when...". Rules that need the data (does this
deal ID exist and belong to this buyer? is every number in the prose found in
the evidence?) are checked by the validator after parsing.

Order follows the run: intake -> coverage -> deal captain -> analyst.
"""

from typing import ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from acquirer_engine.data import DealType, Outcome, Sector, Tag, TargetProfile

# Shared with the risk scoring code, which imports them from here.
RiskName = Literal[
    "integration", "financing", "execution", "bidders", "time_to_close", "price", "antitrust",
]
Lens = Literal["Value", "Certainty", "Speed"]      # how a seller weighs a bid
ConvictionLevel = Literal["High", "Medium", "Low"]

PEThesis = Literal["P1", "P2", "P3"]       # platform builders, add-on acquirers, roll-ups
StrategicThesis = Literal["S1", "S3"]      # health systems expanding, cross-selling consolidators
ThesisId = PEThesis | StrategicThesis


class Model(BaseModel):
    """Base for every schema: unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")


def _no_duplicates(names: list[str], what: str) -> None:
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate {what}: each buyer may appear once")


# --- Intake -----------------------------------------------------------------

TargetField = Literal[
    "sector", "ev_mm", "ownership", "region", "margin_band", "revenue_mm", "ebitda_mm",
]


class FieldSource(Model):
    """Where one target field came from.

    Sources are a list of these (not a dict keyed by field name): structured
    outputs only allow objects with fixed keys, so a dict could only ever be {}."""

    field: TargetField
    provenance: Literal["given", "derived", "estimated"]
    # given     = stated in the request ("$200M")
    # derived   = a stated word mapped by a fixed rule ("private" -> [Private, PE-Backed])
    # estimated = computed from peer deals (revenue, EBITDA); no quote
    quote: str | None = None   # exact phrase from the request behind this field

    @model_validator(mode="after")
    def quote_matches_provenance(self):
        if self.provenance == "estimated" and self.quote is not None:
            raise ValueError("an estimated field has no quote")
        if self.provenance != "estimated" and not self.quote:
            raise ValueError("a given or derived field needs the exact quote behind it")
        return self


class IntakeResult(Model):
    """The banker's request as a target, with the source of every filled field,
    or the reason the request can't be run (e.g. a sector not in the data, or
    no deal size). Exactly one of target and stop_reason is set."""

    target: TargetProfile | None = None
    stop_reason: str | None = None
    sources: list[FieldSource] = []       # one per filled target field
    state_mentioned: str | None = None    # e.g. "Texas"; kept for the report, not mapped to a region

    @model_validator(mode="after")
    def target_or_stop_reason(self):
        if (self.target is None) == (self.stop_reason is None):
            raise ValueError("set exactly one of target and stop_reason")
        return self

    @model_validator(mode="after")
    def every_filled_field_has_a_source(self):
        if self.target is None:
            return self
        named = [source.field for source in self.sources]
        _no_duplicates(named, "source field")
        filled = {name for name in TargetProfile.model_fields if getattr(self.target, name) is not None}
        missing = filled - set(named)
        if missing:
            raise ValueError(f"no source given for: {sorted(missing)}")
        return self

    def source(self, field: str) -> "FieldSource | None":
        return next((s for s in self.sources if s.field == field), None)


# --- Coverage agents ----------------------------------------------------------

class DealSearch(Model):
    """Filters for the search_deals tool, and how a thesis's filters are logged.
    Every field is optional; the buyer type is never a filter here (each agent's
    tools are locked to its own type)."""

    sectors: list[Sector] | None = None
    deal_types: list[DealType] | None = None
    size_min: float | None = None           # $M
    size_max: float | None = None
    year_min: int | None = None
    year_max: int | None = None
    tags_any: list[Tag] | None = None       # deals with at least one of these tags
    outcomes: list[Outcome] | None = None
    min_relevance: float = 0.0
    group_by_buyer: bool = True             # one row per buyer instead of per deal


class Candidate(Model):
    acquirer: str
    evidence_ids: list[str] = Field(min_length=1)   # transaction_ids behind the pick
    ranker_rank: int = Field(ge=1)                  # rank within its type from the ranking
    score: float
    note: str                                       # why this buyer fits, in a sentence or two


class PECandidate(Candidate):
    buyer_type: Literal["PE"] = "PE"                # a PE pool can only hold PE buyers
    theses: list[PEThesis] = Field(min_length=1)


class StrategicCandidate(Candidate):
    buyer_type: Literal["Strategic"] = "Strategic"
    theses: list[StrategicThesis] = Field(min_length=1)
    range_90: tuple[float, float]                   # 90% range of the shrunk score
    primary_sector_proxy: Sector | None             # most common sector it bought in (an inference)
    caveat: str | None = None                       # kept, but with a stated doubt

    @model_validator(mode="after")
    def range_in_order(self):
        if self.range_90[0] > self.range_90[1]:
            raise ValueError("range_90 must be (low, high)")
        return self


class Removal(Model):
    """A ranked buyer the agent screened out, and why."""

    acquirer: str
    reason: str
    evidence_ids: list[str] = Field(min_length=1)


class ThesisLog(Model):
    """The agent's decision on one thesis: kept or dropped, and why. For a seed
    thesis, code adds the filters and the buyers found (it already has them), so
    the model doesn't spend time re-typing them. A refinement adds its filters."""

    thesis_id: ThesisId
    kept: bool
    reason: str
    refined_filters: DealSearch | None = None   # only when this entry refines the thesis


class CoveragePool(Model):
    """Shared shape of both pools. Each subclass narrows the candidate type,
    the pool size and the thesis IDs it may use."""

    THESES: ClassVar[tuple[str, ...]] = ()

    candidates: list[Candidate]
    removals: list[Removal] = []
    thesis_log: list[ThesisLog] = Field(min_length=1)
    # Claims from general knowledge rather than the data (e.g. "Medtronic is a
    # device maker"), listed so the report can label them.
    public_knowledge: list[str] = []
    reasoning_summary: str

    @model_validator(mode="after")
    def each_buyer_once(self):
        _no_duplicates([c.acquirer for c in self.candidates], "candidate")
        picked = {c.acquirer for c in self.candidates}
        if picked & {r.acquirer for r in self.removals}:
            raise ValueError("a buyer cannot be both a candidate and removed")
        wrong = {log.thesis_id for log in self.thesis_log} - set(self.THESES)
        if wrong:
            raise ValueError(f"theses not available to this pool: {sorted(wrong)}")
        return self


# The agent may return fewer candidates than POOL_MIN rather than pad the list
# with misfits; code then tops the pool up from the ranking, so the final pool
# always holds POOL_MIN to POOL_MAX buyers.
class PECoveragePool(CoveragePool):
    THESES = get_args(PEThesis)
    POOL_MIN: ClassVar[int] = 7
    POOL_MAX: ClassVar[int] = 8
    candidates: list[PECandidate] = Field(min_length=1, max_length=8)


class StrategicCoveragePool(CoveragePool):
    THESES = get_args(StrategicThesis)
    POOL_MIN: ClassVar[int] = 5
    POOL_MAX: ClassVar[int] = 8
    candidates: list[StrategicCandidate] = Field(min_length=1, max_length=8)


# --- Deal captain -------------------------------------------------------------

class Pick(Model):
    acquirer: str
    buyer_type: Literal["PE", "Strategic"]
    theses: list[ThesisId] = []    # empty for a buyer topped up from the ranking
    why_selected: str
    evidence_ids: list[str] = Field(min_length=1)


class FinalSelection(Model):
    """The final 10. The mix is counted from the picks, so it can't disagree with them.
    Mix limits (default +/- 1, minimum per type) come from config and are checked
    by the validator."""

    picks: list[Pick] = Field(min_length=10, max_length=10)
    shift_reason: str | None = None   # required if the mix differs from the default

    @model_validator(mode="after")
    def each_buyer_once(self):
        _no_duplicates([p.acquirer for p in self.picks], "pick")
        return self

    @property
    def mix(self) -> dict[str, int]:
        pe = sum(p.buyer_type == "PE" for p in self.picks)
        return {"PE": pe, "Strategic": len(self.picks) - pe}


# --- Analyst ------------------------------------------------------------------

class KeyNumber(Model):
    """A number the analyst plans to use, and where in the evidence pack it is."""

    label: str        # e.g. "comps median EV/Revenue"
    value: float      # e.g. 2.49
    source: str       # path in the evidence pack, e.g. "comps.ev_revenue.median"


class ConvictionAdjustment(Model):
    direction: Literal["none", "up", "down"]   # at most one level from the rule's result
    reason: str | None = None

    @model_validator(mode="after")
    def reason_when_changed(self):
        if self.direction != "none" and not self.reason:
            raise ValueError("a conviction change needs a reason citing the data")
        return self


class AnalystNotes(Model):
    """Step A: the analyst's plan for one buyer, written before the page."""

    thesis: str                                              # why this buyer, 1-2 sentences
    featured_deal_ids: list[str] = Field(min_length=1)
    key_numbers: list[KeyNumber] = Field(min_length=1)
    risks_to_feature: list[RiskName] = Field(min_length=2, max_length=3)
    conviction_adjustment: ConvictionAdjustment
    anomalies_to_mention: list[str] = []                     # data quirks worth a caveat


class RiskItem(Model):
    name: RiskName
    lens: Lens
    status: Literal["fired", "watch"]   # fired = flagged; watch = weakest spot, not an alarm
    text: str
    evidence_ids: list[str] = []


class ConvictionCall(Model):
    level: ConvictionLevel
    rationale: str   # 1-2 sentences citing numbers


class OnePagerDraft(Model):
    """Step B: one buyer's page as the model writes it. The precedent and comps
    tables are added by code from the data, so the model writes only the prose."""

    acquirer: str
    overview: str           # who they are as a buyer
    fit_thesis: str         # why this target for this buyer, phrased as a hypothesis
    precedent_note: str     # the pattern in their relevant deals
    valuation_note: str     # expected pricing vs what they have paid
    risks: list[RiskItem] = Field(min_length=2)
    conviction: ConvictionCall
    public_knowledge: list[str] = []   # claims not from the data, labeled on the page


class OnePage(OnePagerDraft):
    """The page as shipped: the draft plus fields only code sets. Kept separate so
    the model never sees (or fills) them."""

    needs_review: bool = False   # True on a fallback page or one that failed repair
