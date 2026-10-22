"""Tests for the agent output schemas: each rule rejects a bad answer."""

import pytest
from pydantic import ValidationError

from acquirer_engine.data import TargetProfile
from acquirer_engine.schemas import (
    AnalystNotes,
    ConvictionAdjustment,
    FinalSelection,
    IntakeResult,
    OnePage,
    OnePagerDraft,
    PECoveragePool,
    StrategicCandidate,
)

# --- Intake -----------------------------------------------------------------

TARGET = TargetProfile(sector="Healthcare Services", ev_mm=200, ownership=["Private", "PE-Backed"])
SOURCES = {
    "sector": {"provenance": "given", "quote": "Sector: Healthcare Services"},
    "ev_mm": {"provenance": "given", "quote": "~$200M EV"},
    "ownership": {"provenance": "derived", "quote": "private"},
}


def test_intake_accepts_sourced_fields():
    assert IntakeResult(target=TARGET, sources=SOURCES).sources["ownership"].provenance == "derived"


def test_intake_sets_exactly_one_of_target_and_stop_reason():
    assert IntakeResult(stop_reason="Veterinary clinics are not a sector in the data.").target is None
    with pytest.raises(ValidationError):
        IntakeResult()                                                 # neither
    with pytest.raises(ValidationError):
        IntakeResult(target=TARGET, sources=SOURCES, stop_reason="x")  # both


@pytest.mark.parametrize("sources", [
    {k: v for k, v in SOURCES.items() if k != "ownership"},               # filled field, no source
    SOURCES | {"ev_mm": {"provenance": "given"}},                          # given, but no quote
    SOURCES | {"revenue_mm": {"provenance": "estimated", "quote": "x"}},   # estimate with a quote
    SOURCES | {"buyer": {"provenance": "given", "quote": "x"}},            # not a target field
])
def test_intake_rejects_bad_sources(sources):
    with pytest.raises(ValidationError):
        IntakeResult(target=TARGET, sources=sources)


# --- Coverage pools -----------------------------------------------------------

def pe_candidate(name, **extra):
    return {"acquirer": name, "theses": ["P1"], "evidence_ids": ["MA-2021-0195"],
            "ranker_rank": 1, "score": 1.5, "note": "Platform buyer."} | extra


def pe_pool(candidates, thesis="P1"):
    return {
        "candidates": candidates,
        "thesis_log": [{"thesis_id": thesis, "filters_used": {}, "buyers_found": [],
                        "kept": True, "reason": "Found 12 buyers."}],
        "reasoning_summary": "Summary.",
    }


SEVEN = [pe_candidate(f"Firm {i}") for i in range(7)]


def test_pe_pool_accepts_seven():
    assert len(PECoveragePool(**pe_pool(SEVEN)).candidates) == 7


@pytest.mark.parametrize("pool", [
    pe_pool(SEVEN[:6]),                                                     # too few
    pe_pool(SEVEN[:6] + [pe_candidate("Firm 0")]),                          # duplicate buyer
    pe_pool(SEVEN[:6] + [pe_candidate("Firm 9", buyer_type="Strategic")]),  # wrong buyer type
    pe_pool(SEVEN[:6] + [pe_candidate("Firm 9", theses=["S1"])]),           # strategic thesis
    pe_pool(SEVEN, thesis="S3"),                                            # strategic thesis logged
])
def test_pe_pool_rejects(pool):
    with pytest.raises(ValidationError):
        PECoveragePool(**pool)


def test_strategic_range_must_be_low_high():
    with pytest.raises(ValidationError):
        StrategicCandidate(
            acquirer="UPMC", theses=["S1"], evidence_ids=["MA-2020-0001"], ranker_rank=6,
            score=0.148, note="Health system.", range_90=(0.3, 0.03), primary_sector_proxy=None,
        )


# --- Deal captain -------------------------------------------------------------

def pick(name, buyer_type="PE"):
    return {"acquirer": name, "buyer_type": buyer_type, "theses": ["P1"],
            "why_selected": "Strong precedents.", "evidence_ids": ["MA-2021-0195"]}


def test_final_selection_counts_mix():
    picks = [pick(f"PE {i}") for i in range(6)] + [pick(f"S {i}", "Strategic") for i in range(4)]
    assert FinalSelection(picks=picks).mix == {"PE": 6, "Strategic": 4}


@pytest.mark.parametrize("picks", [
    [pick(f"PE {i}") for i in range(9)],                     # nine picks
    [pick(f"PE {i}") for i in range(9)] + [pick("PE 0")],    # duplicate
])
def test_final_selection_rejects(picks):
    with pytest.raises(ValidationError):
        FinalSelection(picks=picks)


# --- Analyst ------------------------------------------------------------------

def test_conviction_change_needs_reason():
    with pytest.raises(ValidationError):
        ConvictionAdjustment(direction="up")


def test_notes_reject_unknown_risk():
    with pytest.raises(ValidationError):
        AnalystNotes(
            thesis="t", featured_deal_ids=["MA-2021-0195"],
            key_numbers=[{"label": "median", "value": 2.49, "source": "comps.ev_revenue.median"}],
            risks_to_feature=["execution", "reputation"],
            conviction_adjustment={"direction": "none"},
        )


RISK = {"name": "execution", "lens": "Certainty", "status": "fired", "text": "2 of 17 withdrawn."}
PAGE = {"acquirer": "KKR", "overview": "o", "fit_thesis": "f", "precedent_note": "p",
        "valuation_note": "v", "conviction": {"level": "Medium", "rationale": "r"}}


def test_one_pager_needs_two_risks():
    with pytest.raises(ValidationError):
        OnePagerDraft(**PAGE, risks=[RISK])


def test_code_only_fields_are_hidden_from_the_model():
    assert "needs_review" not in OnePagerDraft.model_json_schema()["properties"]
    with pytest.raises(ValidationError):
        OnePagerDraft(**PAGE, risks=[RISK, RISK], needs_review=True)

    draft = OnePagerDraft(**PAGE, risks=[RISK, RISK])
    page = OnePage(**draft.model_dump(), needs_review=True)
    assert page.needs_review
