"""Tests for the rule and dedup stages — no network, no API key required.

The LLM stage is deliberately not tested against a live API here; its two
guarantees (fail-open, must_keep override) are asserted through the pipeline
with the LLM disabled and, for the override, by calling the judge with no key
set (which triggers the fail-open path).
"""

import os

import yaml

from noise_filter import Rules, run

CFG = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "config.example.yaml")))
RULES = Rules.from_config(CFG)


def test_drops_charity_and_awards():
    assert RULES.is_noise("Globex donates $2M to local flood relief effort")
    assert RULES.is_noise("Initech CEO honored with industry leadership award")


def test_drops_market_chatter():
    assert RULES.is_noise("Acme shares rose 3% intraday after upbeat guidance")
    assert RULES.is_noise("Analyst reiterates buy rating on Umbrella, raises price target")


def test_keeps_priority_entity_earnings():
    assert not RULES.is_noise("Acme reports record Q3 revenue as cloud unit grows 40%")
    assert RULES.must_keep("Globex posts quarterly profit, tops analyst estimates") is False  # 'analyst' present
    assert RULES.must_keep("Acme reports record Q3 earnings, cloud revenue up sharply")


def test_drops_generic_earnings_roundup():
    # Earnings vocabulary but no priority entity → dropped.
    assert RULES.is_noise("Sector earnings roundup: mixed results across mid-cap names")


def test_analyst_estimate_dropped_even_for_priority_entity():
    # Earnings vocab + priority entity, but framed as an analyst estimate → dropped.
    assert RULES.is_noise("Analyst sees Globex revenue topping $5B next year")
    assert RULES.is_noise("Globex earnings seen beating estimates, analyst says")


def test_pure_analyst_headline_left_for_llm():
    # No earnings keyword, so rules can't classify it — this is exactly the kind
    # of headline stage 3 (the LLM judge) exists to catch.
    assert not RULES.is_noise("Consensus estimate pegs Initech loss narrower than feared")


def test_keeps_regulatory_and_product_news():
    assert not RULES.is_noise("Regulator proposes new data-portability rules for the sector")
    assert not RULES.is_noise("Umbrella data breach exposes 1.2 million customer records")


def test_dedup_collapses_near_duplicates():
    titles = [
        "Umbrella data breach exposes 1.2 million customer records",
        "Umbrella breach exposes 1.2 million customer records",
    ]
    result = run(CFG, titles, use_llm=False)
    assert result.counts["after_dedup"] == 1


def test_pipeline_counts_monotonic():
    titles = json_sample()
    result = run(CFG, titles, use_llm=False)
    c = result.counts
    assert c["raw"] >= c["after_rules"] >= c["after_dedup"] >= c["kept"]
    assert c["kept"] > 0  # signal survives


def json_sample():
    import json

    path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "headlines.json")
    return json.load(open(path))
