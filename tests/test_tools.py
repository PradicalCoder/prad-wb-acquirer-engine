"""Tests for the type-locked coverage tools and the seed theses (default target)."""

import json

import pytest

from acquirer_engine.config import load_config
from acquirer_engine.data import load_deals
from acquirer_engine.evidence import analyze
from acquirer_engine.tools import CoverageTools, run_seed_theses

CONFIG = load_config()


@pytest.fixture(scope="module")
def analysis():
    return analyze(load_deals(CONFIG.paths.data_csv), CONFIG.default_target, CONFIG.scoring)


def tools(analysis, buyer_type):
    return CoverageTools(buyer_type, analysis, CONFIG)


def call(t, tool, **args):
    return json.loads(t.call(tool, args))


def test_seed_theses_match_their_design(analysis):
    found = {}
    for buyer_type in ("PE", "Strategic"):
        for seed in run_seed_theses(tools(analysis, buyer_type), CONFIG.default_target, CONFIG):
            r = seed["result"]
            found[seed["thesis_id"]] = (r["deals_found"], r["buyers_found"],
                                        {b["acquirer"] for b in r["buyers"]})
    counts = {k: v[:2] for k, v in found.items()}
    assert counts == {"P1": (25, 12), "P2": (11, 7), "P3": (17, 10), "S1": (8, 7), "S3": (19, 16)}
    assert "Welsh Carson" in found["P1"][2]                         # missed by the ranker's top 10
    assert not any("Medtronic" in found[s][2] for s in ("S1", "S3"))  # device LBOs fit no thesis


def test_size_bands_scale_with_target_ev(analysis):
    p1 = next(s for s in run_seed_theses(tools(analysis, "PE"), CONFIG.default_target, CONFIG)
              if s["thesis_id"] == "P1")
    assert (p1["filters"]["size_min"], p1["filters"]["size_max"]) == (75.0, 500.0)


def test_search_only_returns_own_type(analysis):
    pe = tools(analysis, "PE")
    names = {b["acquirer"] for b in call(pe, "search_deals", min_relevance=0.0)["buyers"]}
    types = set(analysis.rankings.loc[list(names), "buyer_type"])
    assert types == {"PE"}


def test_profile_refuses_the_other_type(analysis):
    assert "is a Strategic buyer" in call(tools(analysis, "PE"), "get_acquirer_profile", name="Medtronic")["error"]
    assert "is a PE buyer" in call(tools(analysis, "Strategic"), "get_acquirer_profile", name="KKR")["error"]
    assert "no PE buyer" in call(tools(analysis, "PE"), "get_acquirer_profile", name="Nobody Inc")["error"]


def test_bad_arguments_come_back_as_an_error(analysis):
    assert "invalid arguments" in call(tools(analysis, "PE"), "search_deals", size_min="big")["error"]
    assert "unknown tool" in call(tools(analysis, "PE"), "delete_everything")["error"]


def test_strategic_profile_has_sector_proxy_and_range(analysis):
    medtronic = call(tools(analysis, "Strategic"), "get_acquirer_profile", name="Medtronic")
    assert medtronic["primary_sector_proxy"] == "Medical Devices"
    assert medtronic["deal_types"] == {"Leveraged Buyout": 2, "SPAC Merger": 1}
    assert len(medtronic["range_90"]) == 2


def test_rankings_use_each_types_method(analysis):
    pe = call(tools(analysis, "PE"), "get_buyer_rankings", top_k=3)["rankings"]
    assert pe[0]["acquirer"] == "Blackstone" and "range_90" not in pe[0]
    strategic = call(tools(analysis, "Strategic"), "get_buyer_rankings", top_k=3)["rankings"]
    assert strategic[0]["acquirer"] == "Ardent Health Services" and "range_90" in strategic[0]


def test_results_are_remembered_for_validation(analysis):
    pe = tools(analysis, "PE")
    call(pe, "get_acquirer_profile", name="KKR")
    assert "KKR" in pe.seen_buyers and "MA-2019-0205" in pe.seen_ids


def test_no_tool_takes_a_buyer_type(analysis):
    for definition in tools(analysis, "PE").definitions():
        assert not {"buyer_type", "acquirer_type"} & set(definition["input_schema"]["properties"])
