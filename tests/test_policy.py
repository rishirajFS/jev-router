import pytest

from jev_router.features import RoutingFeatures
from jev_router.policy import Policy, Tier, route


def feats(difficulty=0.5, conf=0.95, multistep=0.1):
    return RoutingFeatures("chat", 0.95, difficulty, conf, multistep)


def test_easy_confident_prompt_goes_to_small():
    assert route(feats(0.3), Policy()).tier is Tier.SMALL


def test_mid_difficulty_goes_to_medium():
    assert route(feats(1.5), Policy()).tier is Tier.MEDIUM


def test_hard_prompt_goes_to_large():
    assert route(feats(2.6), Policy()).tier is Tier.LARGE


def test_low_confidence_escalates_one_tier():
    decision = route(feats(0.3, conf=0.2), Policy(min_confidence=0.5))
    assert decision.tier is Tier.MEDIUM
    assert decision.escalated_for_low_confidence


def test_low_confidence_cannot_exceed_large():
    assert route(feats(2.9, conf=0.1), Policy()).tier is Tier.LARGE


def test_high_multistep_probability_escalates():
    decision = route(feats(0.3, multistep=0.95), Policy(multistep_threshold=0.8))
    assert decision.tier is Tier.MEDIUM


def test_policy_is_frozen():
    with pytest.raises(Exception):
        Policy().min_confidence = 0.1  # type: ignore[misc]


def test_higher_threshold_routes_more_to_small():
    strict = Policy(small_max=0.5, medium_max=1.5)
    lenient = Policy(small_max=1.2, medium_max=2.2)
    f = feats(1.0)
    assert route(f, strict).tier is Tier.MEDIUM
    assert route(f, lenient).tier is Tier.SMALL


def test_policy_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        Policy(small_max=2.0, medium_max=1.0)
