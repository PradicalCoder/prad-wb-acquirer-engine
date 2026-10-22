"""Tests for page validation: a good page passes; each rule catches its own bad page."""

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze, build_evidence
from acquirer_engine.schemas import AnalystNotes, OnePagerDraft
from acquirer_engine.validate import similar_theses, validate_page

CONFIG = load_config()
# Length is tested on its own; other tests use short pages.
SHORT_OK = CONFIG.validation.model_copy(update={"page_words_min": 1})


@pytest.fixture(scope="module")
def pack():
    analysis = analyze(load_deals(CONFIG.paths.data_csv), CONFIG.default_target, CONFIG.scoring)
    return build_evidence("KKR", analysis)


def notes(direction="none"):
    return AnalystNotes(
        thesis="Physician-group platform buyer.", featured_deal_ids=["MA-2019-0205"],
        key_numbers=[{"label": "best relevance", "value": 0.65, "source": "top_deals[0].relevance"}],
        risks_to_feature=["execution", "time_to_close"],
        conviction_adjustment={"direction": direction, "reason": "test" if direction != "none" else None},
    )


def good_page(pack, **changes):
    deal, execution = pack.top_deals[0], pack.risks[0]
    page = {
        "acquirer": "KKR",
        "overview": f"KKR has done {pack.profile.deals} deals since {pack.profile.first_year}, "
                    f"typically around ${pack.profile.typical_size_mm:.1f}M.",
        "fit_thesis": f"Its pattern suggests appetite for physician groups: {deal.transaction_id} "
                      f"scored {deal.relevance:.2f} relevance.",
        "precedent_note": f"{pack.profile.strong_deals} strong precedents, latest in {deal.year}.",
        "valuation_note": f"It paid a median {pack.paid.ev_revenue:.2f}x EV/Revenue versus a comps "
                          f"median of {pack.comps.ev_revenue.median:.2f}x.",
        "risks": [
            {"name": "execution", "lens": "Certainty", "status": "fired",
             "text": f"Withdrew from or terminated {execution.count} of {execution.out_of} deals.",
             "evidence_ids": execution.evidence_ids[:2]},
            {"name": "time_to_close", "lens": "Speed", "status": "watch",
             "text": "Slower than most sponsors to close."},
        ],
        "conviction": {"level": "Medium", "rationale": "Appetite is clear but dated; execution is a concern."},
    }
    return OnePagerDraft.model_validate(page | changes)


def test_good_page_passes(pack):
    assert validate_page(good_page(pack), pack, notes(), SHORT_OK) == []


@pytest.mark.parametrize("field, text, expected", [
    ("valuation_note", "Expect pricing near 16.2x EBITDA.", "16.2"),                # invented number
    ("fit_thesis", "Its deal MA-2015-0001 fits.", "MA-2015-0001"),                  # deal not in the pack
    ("overview", "KKR is a leading healthcare investor.", "leading"),               # generic phrase
])
def test_prose_errors(pack, field, text, expected):
    errors = validate_page(good_page(pack, **{field: text}), pack, notes(), SHORT_OK)
    assert len(errors) == 1 and expected in errors[0]


def test_rounded_numbers_are_accepted(pack):
    rounded = f"Median {pack.paid.ev_revenue:.1f}x paid; {pack.risks[0].value:.0%} of deals fell away."
    assert validate_page(good_page(pack, valuation_note=rounded), pack, notes(), SHORT_OK) == []


def test_risk_must_match_the_pack(pack):
    page = good_page(pack)
    wrong_status = page.model_copy(update={"risks": [
        page.risks[0].model_copy(update={"status": "watch"}), page.risks[1]]})
    assert "status fired" in validate_page(wrong_status, pack, notes(), SHORT_OK)[0]

    not_a_risk_here = page.model_copy(update={"risks": [
        page.risks[0], page.risks[1].model_copy(update={"name": "antitrust", "lens": "Certainty"})]})
    assert "not one of this buyer's risks" in validate_page(not_a_risk_here, pack, notes(), SHORT_OK)[0]


def test_conviction_moves_only_with_a_plan(pack):
    high = good_page(pack, conviction={"level": "High", "rationale": "Strong fit."})
    assert validate_page(high, pack, notes("none"), SHORT_OK)          # no plan -> error
    assert validate_page(high, pack, notes("up"), SHORT_OK) == []      # planned, one level -> ok


def test_length(pack):
    errors = validate_page(good_page(pack), pack, notes(), CONFIG.validation)
    assert len(errors) == 1 and "words of prose" in errors[0]


def test_similar_theses_are_flagged(pack):
    base = good_page(pack)
    pages = {
        "A": base.model_copy(update={"fit_thesis": "Roll-up of physician groups in the Southeast with cross-sell."}),
        "B": base.model_copy(update={"fit_thesis": "Roll-up of physician groups in the Southeast with cross-sell."}),
        "C": base.model_copy(update={"fit_thesis": "Health system adding outpatient imaging capacity."}),
    }
    assert [(a, b) for a, b, _ in similar_theses(pages, CONFIG.validation)] == [("A", "B")]
